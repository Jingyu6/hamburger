from typing import List

import lightning as L
import torch
import transformers.modeling_flash_attention_utils as utils
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from m2d.model.fa2_monkey_patch import prepare_fa2_from_position_ids
from m2d.model.m2d_modules import (CompositionalEmbedder,
                                   ConditionalMicroStepDecoder)

# apply a monkey patch here
utils.prepare_fa2_from_position_ids = prepare_fa2_from_position_ids


class M2DLlama(L.LightningModule):
    def __init__(
        self, 
        base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
        max_steps: int = 4
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
        self.micro_step_decoder = ConditionalMicroStepDecoder(
            config=self.model.config, 
            micro_stop_token_id=self.micro_stop_token_id, 
            max_steps=max_steps
        )
        self.max_steps = max_steps
        self.train()

    @torch.inference_mode
    def generate(
        self, 
        prompt: str, 
        max_gen_len: int = 128
    ) -> str:
        self.eval()

        conversation = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            return_tensors='pt', 
            return_dict=True
        )["input_ids"][0].to(self.model.device)

        # create a cache object
        macro_past_key_values = DynamicCache()

        seq_len = input_ids.shape[-1]
        output_token_ids = []
        total_len = seq_len

        # MACRO STEP
        for macro_idx in range(max_gen_len):
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
                return_dict=True,
                past_key_values=macro_past_key_values
            )

            # always takes the last one
            hidden_states = base_output.last_hidden_state[:, -1:, :]

            # MICRO STEP
            # TODO: refactor this to encapsulate everything
            micro_past_key_values = DynamicCache()
            hiddens = hidden_states
            input_ids = []

            for micro_idx in range(self.max_steps):
                position_embeddings = self.micro_step_decoder.rotary_emb(
                    hiddens, 
                    torch.arange(micro_idx, micro_idx + 1)[None, ].to(hiddens.device)
                )
                past_seen_tokens = micro_past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + 1, device=hiddens.device
                )
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = self.micro_step_decoder.decoder.forward(
                        hiddens, 
                        use_cache=True, 
                        past_key_value=micro_past_key_values,
                        cache_position=cache_position,  
                        position_embeddings=position_embeddings, 
                    )[0]
                
                logits = self.model.lm_head.forward(out)
                pred_token = logits.argmax(dim=-1).view(-1)

                # stop if we hit micro stop
                if pred_token[0] == self.micro_stop_token_id:
                    break
                
                # update hidden
                hiddens = self.comp_embedder.single_forward(
                    input_ids=pred_token, 
                    disable_merge=True
                )[None, ]

                # update next macro step values
                input_ids.append(pred_token)

            input_ids = torch.concat(input_ids).flatten()
            total_len += len(input_ids)
            output_token_ids.append(input_ids.cpu())

            if any(input_ids == self.tokenizer.eos_token_id):
                break
        
        all_token_ids = torch.concat(output_token_ids, dim=0)
        output = self.tokenizer.decode(all_token_ids, skip_special_tokens=True)
        micro_token_output = "\033[42m \033[0m".join(self.tokenizer.batch_decode(output_token_ids))
        token_output = "\033[42m \033[0m".join(self.tokenizer.batch_decode(all_token_ids.view(-1)))
        
        return {
            "output": output, 
            "token_output": token_output, 
            "micro_token_output": micro_token_output, 
            "speedup": all_token_ids.shape[0] / len(output_token_ids)
        }

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
            return_dict=True, 
        )

        hidden_states = base_output.last_hidden_state

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

        self.log_dict({
            "train_loss": loss, 
            "train_perplexity": torch.exp(loss)
        }, on_step=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(**batch)
        targets = self._get_targets(**batch)
        loss = self._calc_loss(logits, targets)

        self.log_dict({
                "eval_loss": loss, 
                "eval_perplexity": torch.exp(loss)
            },  
            prog_bar=True, 
            logger=True, 
            sync_dist=True, 
            batch_size=len(batch["seq_lens"])
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
                # smaller learning rate for the main model
                {"params": self.model.parameters(), 
                    "lr": 1e-5, "weight_decay": 1e-6}, 
                {"params": self.comp_embedder.pe}, 
                {"params": self.micro_step_decoder.parameters()}, 
            ], 
            lr=5e-4, 
            weight_decay=1e-5
        )

        return optimizer


if __name__ == "__main__":
    model = M2DLlama()
