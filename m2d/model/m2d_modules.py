from typing import List

import torch
import torch.nn as nn


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
    def __init__(self):
        super().__init__()

    
