from typing import List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import (LlamaDecoderLayer,
                                                      LlamaRotaryEmbedding)


class CompositionalEmbedder(nn.Module):
    """
        A simple average composition for now
    """
    def __init__(
        self, 
        embedding: nn.Embedding, 
        max_steps: int
    ):
        super().__init__()
        self.embedding = embedding
        self.max_steps = max_steps
        self.pe = nn.Parameter(
            torch.zeros(
                (self.max_steps, embedding.weight.shape[-1]), 
                dtype=self.embedding.weight.dtype
            ), 
            requires_grad=True
        )
    
    def _merge_fn(self, embeddings: torch.Tensor):
        emb_len = embeddings.shape[0]
        return (embeddings + self.pe[:emb_len]).mean(dim=0, keepdim=True)

    def single_forward(
        self,
        input_ids: torch.Tensor, 
        disable_merge: bool
    ):
        if disable_merge:
            # TODO: need to refactor this part later
            return self.embedding.forward(input_ids)
        else:
            return self._merge_fn(self.embedding.forward(input_ids))

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        seq_lens: List[int], 
        inst_lens: List[int], 
        steps: List[List[int]], 
        return_unmerged: bool = False
    ):
        # normal embeddings
        token_embeds = self.embedding.forward(input_ids)
        # composition
        offset = 0
        result_tokens = []
        comp_seq_lens = []
        position_ids = []
        unmerged_embeds = [] if return_unmerged else None

        for seq_len, inst_len, step in zip(seq_lens, inst_lens, steps):
            cur_seq_len = inst_len
            # instruction
            result_tokens.append(token_embeds[offset:offset + inst_len])
            position_ids.extend(range(inst_len))
            # response
            result_tokens.extend([
                self._merge_fn(embs) for embs in torch.split(
                    token_embeds[offset + inst_len:offset + seq_len], 
                    split_size_or_sections=step, 
                    dim=0
                )
            ])
            if return_unmerged:
                unmerged_embeds.extend(
                    torch.split(
                        token_embeds[offset + inst_len:offset + seq_len], 
                        split_size_or_sections=step, 
                        dim=0
                    )
                )
            for s in step:
                position_ids.append(position_ids[-1] + s)
            cur_seq_len += len(step)
            comp_seq_lens.append(cur_seq_len)
            offset += seq_len

        token_embeds = torch.concat(result_tokens, dim=0)[None, ]
        position_ids = torch.LongTensor(position_ids)[None, ].to(token_embeds.device)
        return token_embeds, position_ids, comp_seq_lens, unmerged_embeds


class ConditionalMicroStepDecoder(nn.Module):
    def __init__(
        self, 
        config: LlamaConfig, 
        micro_stop_token_id: int, 
        max_steps: int
    ):
        """
            Currently we're just using a single transformer decoder layer
        """
        super().__init__()
        self.micro_stop_token_id = micro_stop_token_id
        self.max_steps = max_steps
        assert self.max_steps >= 1
        self.decoder = LlamaDecoderLayer(config=config, layer_idx=0)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self, 
        hidden_states: torch.Tensor, # [1, total_seq_len, model_size]
        token_embeds: List[torch.Tensor], 
        comp_seq_lens: List[int], 
        inst_lens: List[int]
    ):
        """
            Since we're just doing SFT for now, we just need to
            figure out how to do micro step decoding
        """

        # extract macro step hidden
        macro_step_hiddens = []
        offset = 0
        for seq_len, inst_len in zip(comp_seq_lens, inst_lens):
            macro_step_hiddens.append(
                hidden_states[0, offset + inst_len - 1: offset + seq_len - 1]
            )
            offset += seq_len
        macro_step_hiddens = torch.concat(macro_step_hiddens, dim=0).unsqueeze(1)

        # pad token embeds
        token_embeds.append(torch.zeros(self.max_steps, hidden_states.shape[-1]).to(hidden_states.device))
        token_embeds = pad_sequence(token_embeds, batch_first=True, padding_value=0)[:-1]
        
        # concate together
        hiddens = torch.concat([macro_step_hiddens, token_embeds], dim=1)
        position_embeddings = self.rotary_emb(
            hiddens, 
            torch.arange(0, self.max_steps + 1)[None, ].to(hiddens.device)
        )

        out = self.decoder.forward(
            hiddens, 
            position_embeddings=position_embeddings, 
        )[0]

        # [total_seq_len, max_steps, model_size]
        return out
