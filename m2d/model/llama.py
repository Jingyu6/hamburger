import re
from typing import Dict, List, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.modeling_flash_attention_utils as utils
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import \
    RepetitionPenaltyLogitsProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast

from m2d.config import GenConfig
from m2d.model.fa2_monkey_patch import prepare_fa2_from_position_ids
from m2d.model.m2d_modules import (CompositionalEmbedder,
                                   ConditionalMicroStepDecoder)
from m2d.model.teacher import DistillTeacher

# apply a monkey patch here
utils.prepare_fa2_from_position_ids = prepare_fa2_from_position_ids


class SpeedupReport:
    def __init__(self):
        self.total_queries = 0
        self.speedup = 0
        self.reset()

    def add_query_stats(self, macro, micro):
        self.total_queries += 1
        self.speedup += (1.0 * micro / macro)
    
    def reset(self):
        self.total_queries = 0
        self.speedup = 0

    def get_speedup(self):
        if self.total_queries == 0:
            print("No records yet.")
            return None
        avg_speedup = self.speedup / self.total_queries * 100
        print(f"Total number of queries: {self.total_queries}. Avg speedup: {avg_speedup:.2f}%")
        return avg_speedup


class M2DLlama(L.LightningModule):
    def __init__(
        self, 
        base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
        max_steps: int = 4, 
        distill_kl: Optional[float] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True) # for generation

        # this is for optimization
        self.model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            # device_map="auto"
        )
        self.comp_embedder = CompositionalEmbedder(
            embedding=self.model.model.embed_tokens, 
            max_steps=max_steps
        )
        # this is a reserved token in the model '<|reserved_special_token_0|>'
        self.micro_stop_token_id = 128002
        # special init as the average
        self.model.model.embed_tokens.weight.data[self.micro_stop_token_id] = \
            self.model.model.embed_tokens.weight.data.mean(0)
        # create conditional micro step decoder
        self.micro_step_decoder = ConditionalMicroStepDecoder(
            config=self.model.config, 
            max_steps=max_steps
        )
        self.max_steps = max_steps
        self.train()

        self.distill_kl = distill_kl
        if distill_kl is not None:
            self.teacher = DistillTeacher(teacher_model_name=base_model_name)
            self.kl_loss = nn.KLDivLoss(reduction="none")

        self.report = SpeedupReport()

    @torch.inference_mode
    def generate(
        self, 
        prompt: Optional[str] = None, 
        conversation: Optional[List[Dict]] = None, 
        config: Optional[GenConfig] = None
    ) -> str:
        self.eval()

        if config is None:
            config = GenConfig()

        if prompt is not None:
            conversation = [{"role": "user", "content": prompt}]
        else:
            assert conversation is not None
        
        if config.system_message is not None:
            if conversation[0]["role"] == "system":
                print("Warning: Input already has a system message while attemping to add another one.")
            conversation = [{"role": "system", "content": config.system_message}] + conversation

        input_ids = self.tokenizer.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            return_tensors='pt', 
            return_dict=True
        )["input_ids"][0].to(self.model.device)

        # logits processor
        logit_processor = None
        if config.repetition_penalty is not None:
            logit_processor = RepetitionPenaltyLogitsProcessor(
                penalty=config.repetition_penalty
            )
        
        # create a cache object
        macro_past_key_values = DynamicCache()

        seq_len = input_ids.shape[-1]
        # TODO: refactor later
        history_ids = input_ids.clone()
        output_token_ids = []
        output_token_probs = []
        total_len = seq_len

        # MACRO STEP
        for macro_idx in range(config.decode_steps):
            token_embeds = self.comp_embedder.single_forward(
                input_ids=input_ids, 
                disable_merge=(macro_idx == 0)
            )[None, ]

            position_ids = torch.arange(0, token_embeds.shape[1], device=self.model.device)[None, ]
            if macro_idx != 0:
                # correct position ids
                position_ids += total_len

            base_output: BaseModelOutputWithPast = self.model.model.forward(
                inputs_embeds=token_embeds, 
                position_ids=position_ids, 
                use_cache=True, 
                output_hidden_states=True, 
                return_dict=True, 
                past_key_values=macro_past_key_values
            )

            # always takes the last one
            hidden_states = torch.concat([
                base_output.hidden_states[layer_idx + 1][:, -1:, :]
                for layer_idx in self.micro_step_decoder.feature_layer_indices
            ], dim=1)

            # MICRO STEP
            # TODO: refactor this to encapsulate everything
            micro_past_key_values = DynamicCache()
            hiddens = hidden_states
            input_ids = []

            for micro_idx in range(self.max_steps):
                past_seen_tokens = micro_past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + hiddens.shape[1], device=hiddens.device
                )
                position_ids = cache_position.unsqueeze(0)

                for decoder_layer in self.micro_step_decoder.decoders:
                    position_embeddings = self.micro_step_decoder.rotary_emb(
                        hiddens, 
                        position_ids
                    )

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        hiddens = decoder_layer.forward(
                            hiddens, 
                            use_cache=True, 
                            past_key_value=micro_past_key_values,
                            cache_position=cache_position,  
                            position_embeddings=position_embeddings, 
                        )[0]
                
                logits = self.model.lm_head.forward(hiddens[:, -1:, :])

                # apply penalty
                if logit_processor is not None and micro_idx == 0:
                    logits = logit_processor(
                        input_ids=history_ids[None, ], 
                        scores=logits[0],
                    )

                pred_token = logits.argmax(dim=-1).view(-1)

                # stop if we hit micro stop
                if pred_token[0] == self.micro_stop_token_id:
                    break

                prob = self._get_prob(logits).item()
                output_token_probs.append(prob)

                if micro_idx > 0 and config.micro_step_confidence is not None:
                    if prob < config.micro_step_confidence:
                        break
                
                # update hidden
                hiddens = self.comp_embedder.embedding.forward(
                    pred_token
                )[None, ]

                # update next macro step values
                input_ids.append(pred_token)

            input_ids = torch.concat(input_ids).flatten()
            total_len += len(input_ids)
            output_token_ids.append(input_ids)
            history_ids = torch.concat([history_ids, input_ids], dim=-1)

            if any(input_ids == self.tokenizer.eos_token_id):
                break

            if (total_len - seq_len) >= config.max_gen_len:
                break
        
        all_token_ids = torch.concat(output_token_ids, dim=0).cpu()
        output = self.tokenizer.decode(all_token_ids, skip_special_tokens=True)
        token_str_list = self.tokenizer.batch_decode(all_token_ids.view(-1))
        token_output = "\033[42m \033[0m".join(self._color_output(
            token_str_list, 
            output_token_probs, 
            [len(x) for x in output_token_ids]
        ))

        if config.remove_think:
            # remove the think block in the output
            output = re.sub(r"<think>.*?</think>[\s\r\n]*", "", output, flags=re.DOTALL)

        self.report.add_query_stats(
            macro=len(output_token_ids), 
            micro=all_token_ids.shape[0]
        )

        return {
            "output": output, 
            "micro_token_output": token_output, 
            "speedup": all_token_ids.shape[0] / len(output_token_ids)
        }
    
    def _color_output(self, token_list, prob_list, steps):
        colors = [None, 231, 159, 51, 48]
        colored_token_list = []
        for token, prob in zip(token_list, prob_list):
            color_idx = min(int(prob / 0.2), 4)
            color = colors[color_idx]
            if color is None:
                colored_token_list.append(token)
            else:
                colored_token_list.append(f"\033[38;5;{color}m{token}\033[0m")
        
        concat_colored_token_list = []
        idx = 0
        for step in steps:
            concat_colored_token_list.append("".join(colored_token_list[idx:idx + step]))
            idx += step

        return concat_colored_token_list
    
    def _get_prob(self, logits: torch.Tensor, token_id: Optional[torch.Tensor] = None):
        logits = logits.view(-1)
        if token_id is None:
            token_id = logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        return probs[token_id]

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        seq_lens: List[int], 
        inst_lens: List[int], 
        steps: List[List[int]] 
    ):
        # composition embedding
        token_embeds, position_ids, comp_seq_lens, unmerged_embeds \
            = self.comp_embedder.forward(
                input_ids, 
                seq_lens, 
                inst_lens, 
                steps, 
                return_unmerged=True
            )

        # get hidden state of the base llama
        base_output: BaseModelOutputWithPast = self.model.model.forward(
            inputs_embeds=token_embeds, 
            position_ids=position_ids, 
            use_cache=False, 
            output_hidden_states=True, 
            return_dict=True, 
        )

        # take off the embeddings, number of layers
        hidden_states = base_output.hidden_states[1:]

        # micro step decoding [num_of_decodes, max_steps, model_size]
        micro_step_outputs = self.micro_step_decoder.forward(
            hidden_states=hidden_states, 
            token_embeds=unmerged_embeds, # not sure if we want to use detach here
            comp_seq_lens=comp_seq_lens, 
            inst_lens=inst_lens
        )

        micro_step_outputs = self._trim_micro_steps(micro_step_outputs, steps)

        # calculate logits
        assert self.model.config.pretraining_tp == 1
        logits = self.model.lm_head.forward(micro_step_outputs)

        return logits

    def _trim_micro_steps(
        self,
        data: torch.Tensor, 
        steps: List[List[int]]
    ) -> torch.Tensor:
        """
            We remove unnecessary tokens here
            e.g.: 
            [---max_step---]
            [t0][t1][t2][pd][pd]
            [m0][m1][m2][m3][m4]
            
            [t0][t1][t2]
            [m0][m1][m2]
            This will improve loss accuracy and memory usage
        """
        step = []
        for s in steps:
            step.extend(s)
        trim_data = []
        for s, d in zip(step, data):
            trim_data.append(d[:s+1])
        return torch.concat(trim_data, dim=0)

    def _get_targets(
        self,
        input_ids: torch.LongTensor, 
        seq_lens: List[int], 
        inst_lens: List[int], 
        steps: List[List[int]] 
    ):
        targets = []
        offset = 0
        for seq_len, inst_len, step in zip(seq_lens, inst_lens, steps):
            targets.extend(
                input_ids[offset + inst_len:offset + seq_len].split(
                    split_size=step, 
                    dim=0
                )
            )
            offset += seq_len
        # add a dummy tensor at the end to make sure all tensors are of size max_steps
        targets.append(torch.arange(0, self.max_steps + 1).to(input_ids.device))
        targets = pad_sequence(targets, batch_first=True, padding_value=self.micro_stop_token_id)
        return self._trim_micro_steps(targets[:-1], steps)

    def _calc_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ):
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.float(), targets)
        return loss

    def _calc_accuracy(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ):
        pred = logits.argmax(dim=-1)
        correct_mask = (pred == targets)
        non_stop_mask = (targets != self.micro_stop_token_id)

        # total accuracy
        acc_all = torch.sum(correct_mask) / len(targets.view(-1))

        # excluded accuracy (normalized)
        acc_exc = torch.sum(correct_mask & non_stop_mask) / torch.sum(non_stop_mask)

        metrics = {
            "eval_token_acc_all": acc_all, 
            "eval_token_acc_excluded": acc_exc
        }

        def _calc_step_mask(step):
            mask = torch.zeros_like(non_stop_mask, dtype=torch.bool)
            last = -1
            for i in range(mask.shape[0]):
                if non_stop_mask[i] == False:
                    last = i
                elif i - last == step:
                    mask[i] = True
            return mask

        # per-step metrics
        for step in range(1, self.max_steps + 1):
            step_mask = _calc_step_mask(step)
            acc_step = torch.sum(correct_mask & step_mask) / (torch.sum(step_mask) + 1e-9)
            metrics[f"eval_token_acc_s{step}"] = acc_step

        return metrics

    def training_step(self, batch, batch_idx):
        """
        batch: {
            "input_ids": input_ids, 
            "seq_lens": seq_lens, 
            "inst_lens": inst_lens, 
            "steps": steps
        }
        """
        logits = self.forward(**batch)
        targets = self._get_targets(**batch)
        loss = self._calc_loss(logits, targets)

        log_dict = {
            "train_loss": loss, 
            "train_perplexity": torch.exp(loss)
        }

        if self.distill_kl is not None:
            with torch.no_grad():
                teacher_logits, mask = self.teacher.get_logits(**batch)

            # put into the correct space
            teacher_logits = F.softmax(teacher_logits, dim=-1)
            logits = F.log_softmax(logits, dim=-1)

            kl_loss = self.kl_loss(logits, teacher_logits)

            # apply mask and avg
            kl_loss = (kl_loss * mask.unsqueeze(-1)).sum() / mask.sum()

            loss += self.distill_kl * kl_loss

            log_dict["kl_loss"] = kl_loss

        self.log_dict(log_dict, on_step=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(**batch)
        targets = self._get_targets(**batch)
        loss = self._calc_loss(logits, targets)

        log_dict = {
            "eval_loss": loss, 
            "eval_perplexity": torch.exp(loss)
        }

        log_dict.update(self._calc_accuracy(logits, targets))

        self.log_dict(
            log_dict, 
            prog_bar=True, 
            logger=True, 
            sync_dist=True, 
            batch_size=len(batch["seq_lens"])
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
                # smaller learning rate for the main model
                {"params": self.model.model.embed_tokens.parameters(), "lr": 5e-5}, 
                # slightly larger lr for the embedding and lm head (tied)
                {"params": [p for n, p in self.model.named_parameters() if "embed_tokens" not in n], "lr": 5e-5}, 
                # larger lr for grafted modules
                {"params": self.comp_embedder.merger.parameters(), "lr": 1e-3}, 
                {"params": self.comp_embedder.out_proj.parameters(), "lr": 1e-3}, 
                {"params": self.micro_step_decoder.parameters()}, 
            ], lr=1e-4
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.0 * total_steps)
        stable_steps = int(0.7 * total_steps)
        decay_steps = total_steps - warmup_steps - stable_steps

        def lr_lambda_wsd(step):
            if step < warmup_steps:
                return step / warmup_steps
            elif step < (warmup_steps + stable_steps):
                return 1.0
            else:
                return (total_steps - step) / decay_steps
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_wsd)

        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step"
            }
        }


if __name__ == "__main__":
    model = M2DLlama()
