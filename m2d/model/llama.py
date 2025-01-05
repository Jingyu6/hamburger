from typing import List

import lightning as L
import torch
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

        model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model: LlamaModel = model.base_model
        self.lm_head = model.lm_head
        self.comp_embedder = CompositionalEmbedder(
            embedding=self.base_model.embed_tokens, 
            max_steps=max_steps
        )
        self.micro_step_decoder = MicroStepDecoder(
            config=self.base_model.config, 
            # TODO: we will prob change to a reserved token later
            micro_stop_token_id=tokenizer.eos_token_id, 
            max_steps=max_steps
        )

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

        # micro step decoding
        micro_step_outputs = self.micro_step_decoder.forward(
            hidden_states=hidden_states, 
            comp_seq_lens=comp_seq_lens, 
            inst_lens=inst_lens
        )

        exit()

    def training_step(self, batch, batch_idx):
        """
        batch: {
            "input_ids": input_ids, 
            "seq_lens": seq_lens, 
            "inst_lens": inst_lens, 
            "steps": steps
        }
        """
        output = self.forward(**batch)
        exit()

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass


if __name__ == "__main__":
    model = M2DLlama()
