from turtle import position
from typing import List

import lightning as L
import torch
from deepspeed.ops.adam import FusedAdam
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast

from m2d.model.m2d_modules import CompositionalEmbedder, MicroStepDecoder


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
        self.model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(base_model_name)
        self.model.train()
        self.comp_embedder = CompositionalEmbedder(
            embedding=self.model.model.embed_tokens, 
            max_steps=max_steps
        )
        # this is a reserved token in the model '<|reserved_special_token_0|>'
        self.micro_stop_token_id = 128002
        self.micro_step_decoder = MicroStepDecoder(
            config=self.model.config, 
            micro_stop_token_id=self.micro_stop_token_id, 
            max_steps=max_steps
        )
        self.max_steps = max_steps

    @torch.inference_mode
    def generate(
        self, 
        prompt: str, 
        max_gen_len: int = 128, 
        include_micro_stop_token: bool = False
    ) -> str:
        conversation = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            conversation, 
            return_tensors='pt', 
            return_dict=True
        )["input_ids"][0]
        seq_len = input_ids.shape[-1]

        # create a cache object
        past_key_values = {}

        output_token_ids = []

        # main loop
        for idx in range(max_gen_len):
            token_embeds = self.comp_embedder.single_forward(
                input_ids=input_ids, 
                is_prefill=(idx == 0)
            )[None, ]

            position_ids = torch.arange(0, token_embeds.shape[1])[None, ]
            if idx != 0:
                # correct position ids
                position_ids += seq_len

            base_output: BaseModelOutputWithPast = self.model.model.forward(
                inputs_embeds=token_embeds, 
                position_ids=position_ids, 
                use_cache=True, 
                return_dict=True,
                past_key_values=past_key_values
            )

            # update cache
            past_key_values = base_output.past_key_values
            # always takes the last one
            hidden_states = base_output.last_hidden_state[:, -1:, :]

            # micro steps
            micro_step_outputs = self.micro_step_decoder.single_forward(hidden_states)
            
            # lm head
            assert self.model.config.pretraining_tp == 1
            logits = self.model.lm_head.forward(micro_step_outputs)
            
            # greedy tokens
            pred_token = logits.argmax(dim=-1).view(-1)
            
            micro_stop = len(pred_token)
            for i in range(len(pred_token)):
                if pred_token[i] == self.micro_stop_token_id:
                    micro_stop = i
                    break
            
            input_ids = pred_token[:micro_stop]
            seq_len = micro_stop

            output_token_ids.append(input_ids)

            if any(input_ids == self.tokenizer.eos_token_id):
                break
        
        output_token_ids = torch.concat(output_token_ids, dim=0)
        return self.tokenizer.decode(output_token_ids.cpu())

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        seq_lens: List[int], 
        inst_lens: List[int], 
        steps: List[List[int]] 
    ):
        # composition embedding
        token_embeds, position_ids, comp_seq_lens = self.comp_embedder.forward(
            input_ids, 
            seq_lens, 
            inst_lens, 
            steps
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
        optimizer = FusedAdam([
                # smaller learning rate for the main model
                {"params": self.model.parameters(), 
                    "lr": 5e-6, "weight_decay": 5e-7}, 
                {"params": self.micro_step_decoder.parameters()}
            ], 
            lr=1e-5, 
            weight_decay=1e-6
        )

        return optimizer


if __name__ == "__main__":
    model = M2DLlama()
