from typing import List

import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel
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

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # this is for optimization
        self.model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(base_model_name)
        self.base_model: LlamaModel = self.model.base_model
        self.lm_head = self.model.lm_head
        self.comp_embedder = CompositionalEmbedder(
            embedding=self.base_model.embed_tokens, 
            max_steps=max_steps
        )
        # TODO: we will prob change to a reserved token later
        self.micro_stop_token_id = tokenizer.eos_token_id
        self.micro_step_decoder = MicroStepDecoder(
            config=self.base_model.config, 
            micro_stop_token_id=self.micro_stop_token_id, 
            max_steps=max_steps
        )
        self.max_steps = max_steps

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
        base_output: BaseModelOutputWithPast = self.base_model.forward(
            inputs_embeds=token_embeds, 
            position_ids=position_ids, 
            use_cache=False, 
            return_dict=True, 
        )

        hidden_states = base_output.last_hidden_state

        # micro step decoding [num_of_decodes * max_steps, model_size]
        micro_step_outputs = self.micro_step_decoder.forward(
            hidden_states=hidden_states, 
            comp_seq_lens=comp_seq_lens, 
            inst_lens=inst_lens
        )

        # calculate logits
        assert self.base_model.config.pretraining_tp == 1
        logits = self.lm_head.forward(micro_step_outputs)

        return logits

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
        targets = pad_sequence(targets, batch_first=True, padding_value=self.micro_stop_token_id)
        return targets.view(-1)

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
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
                # smaller learning rate for the main model
                {"params": self.model.parameters(), "lr": 1e-6}, 
                {"params": self.micro_step_decoder.parameters()}
            ], 
            lr=1e-5
        )
        return optimizer

if __name__ == "__main__":
    model = M2DLlama()
