from typing import List

import torch
import torch.nn as nn
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
    
    def single_forward(
        self,
        input_ids: torch.Tensor, 
        is_prefill: bool
    ):
        if is_prefill:
            return self.embedding.forward(input_ids)
        else:
            return self.embedding.forward(input_ids).mean(0, keepdim=True)

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        seq_lens: List[int], 
        inst_lens: List[int], 
        steps: List[List[int]] 
    ):
        # normal embeddings
        token_embeds = self.embedding.forward(input_ids)
        # composition
        offset = 0
        result_tokens = []
        comp_seq_lens = []
        position_ids = []
        for seq_len, inst_len, step in zip(seq_lens, inst_lens, steps):
            cur_seq_len = inst_len
            # instruction
            result_tokens.append(token_embeds[offset:offset + inst_len])
            position_ids.extend(range(inst_len))
            # response
            result_tokens.extend([
                embs.mean(dim=0, keepdim=True) for embs in torch.split(
                    token_embeds[offset + inst_len:offset + seq_len], 
                    split_size_or_sections=step, 
                    dim=0
                )
            ])
            for s in step:
                position_ids.append(position_ids[-1] + s)
            cur_seq_len += len(step)
            comp_seq_lens.append(cur_seq_len)
            offset += seq_len

        token_embeds = torch.concat(result_tokens, dim=0)[None, ]
        position_ids = torch.LongTensor(position_ids)[None, ].to(token_embeds.device)
        return token_embeds, position_ids, comp_seq_lens


class MicroStepDecoder(nn.Module):
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
    
    def single_forward(
        self,
        hidden_states: torch.Tensor, # [1, 1, model_size]
    ):
        hiddens = hidden_states
        out = None
        for idx in range(self.max_steps + 1):
            position_embeddings = self.rotary_emb(
                hiddens, 
                torch.arange(0, idx + 1)[None, ].to(hiddens.device)
            )

            out = self.decoder.forward(
                hiddens, 
                position_embeddings=position_embeddings, 
            )[0]

            hiddens = torch.concat(
                [hiddens, out[:, -1:, :]], 
                dim=1
            )

        # [1, max_steps, model_size]
        return out

    def forward(
        self, 
        hidden_states: torch.Tensor, # [1, total_seq_len, model_size]
        comp_seq_lens: List[int], 
        inst_lens: List[int], 
        take_off_last: bool = True
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
                hidden_states[0, offset + inst_len - 1: offset + seq_len - 1 if take_off_last else 0]
            )
            offset += seq_len
        macro_step_hiddens = torch.concat(macro_step_hiddens, dim=0).unsqueeze(1)

        # micro step decoding
        hiddens = macro_step_hiddens
        out = None
        for idx in range(self.max_steps + 1):
            position_embeddings = self.rotary_emb(
                hiddens, 
                torch.arange(0, idx + 1)[None, ].to(hiddens.device)
            )

            out = self.decoder.forward(
                hiddens, 
                position_embeddings=position_embeddings, 
            )[0]

            hiddens = torch.concat(
                [hiddens, out[:, -1:, :]], 
                dim=1
            )

        # [total_seq_len, max_steps, model_size]
        return out
