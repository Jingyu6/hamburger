from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.emb_size = embedding.weight.shape[-1]
        self.emb_dtype = self.embedding.weight.dtype

        self.gate = nn.Sequential(
            nn.Linear(
                in_features=self.emb_size + self.emb_size // 2, 
                out_features=self.emb_size, 
                dtype=self.emb_dtype
            ), 
            nn.SiLU(), 
            nn.Linear(
                in_features=self.emb_size, 
                out_features=self.emb_size, 
                dtype=self.emb_dtype
            )
        )

        self.pos_weight = nn.Parameter(
            torch.zeros(self.max_steps, self.emb_size // 2, dtype=self.emb_dtype), 
            requires_grad=True
        )

        nn.init.xavier_normal_(self.pos_weight)

    def _merge_fn(self, embeddings: torch.Tensor):
        emb_len = embeddings.shape[0]
        if emb_len == 1:
            # we dont do merging
            return embeddings
        # apply gating
        gates = F.softmax(self.gate.forward(
            torch.concat([embeddings, self.pos_weight[:emb_len]], dim=-1)
        ), dim=0)

        return embeddings.mean(dim=0, keepdim=True) + \
            (embeddings * gates).sum(dim=0, keepdim=True).to(self.emb_dtype)

    def single_forward(
        self,
        input_ids: torch.Tensor, 
        disable_merge: bool
    ):
        embeddings = self.embedding.forward(input_ids)
        if not disable_merge:
            embeddings = self._merge_fn(embeddings)
        return embeddings

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
        max_steps: int
    ):
        """
            Currently we're just using a single transformer decoder layer
        """
        super().__init__()
        self.max_steps = max_steps
        assert self.max_steps >= 1
        self.num_layers = 4
        self.decoders = nn.ModuleList([
            LlamaDecoderLayer(config=config, layer_idx=layer_idx) 
            for layer_idx in range(self.num_layers)])
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.feature_layer_indices = [3, 7, 11, 15]

    def forward(
        self, 
        hidden_states: Tuple[torch.Tensor], # [1, total_seq_len, model_size]
        token_embeds: List[torch.Tensor], 
        comp_seq_lens: List[int], 
        inst_lens: List[int]
    ):
        """
            Since we're just doing SFT for now, we just need to
            figure out how to do micro step decoding
        """

        num_features = len(self.feature_layer_indices)

        # extract macro step hidden
        macro_step_hiddens = []

        for layer_idx in self.feature_layer_indices:
            per_layer_hiddens = []
            offset = 0
            for seq_len, inst_len in zip(comp_seq_lens, inst_lens):
                per_layer_hiddens.append(
                    hidden_states[layer_idx][0, offset + inst_len - 1: offset + seq_len - 1]
                )
                offset += seq_len
            per_layer_hiddens = torch.concat(per_layer_hiddens, dim=0).unsqueeze(1)
            macro_step_hiddens.append(per_layer_hiddens)

        # [batched macro step, num of features, model size]
        macro_step_hiddens = torch.concat(macro_step_hiddens, dim=1)

        # pad token embeds
        token_embeds.append(torch.zeros(self.max_steps, macro_step_hiddens.shape[-1]).to(macro_step_hiddens.device))
        token_embeds = pad_sequence(token_embeds, batch_first=True, padding_value=0)[:-1]
        
        # concate together
        hiddens = torch.concat([macro_step_hiddens, token_embeds], dim=1)
        position_ids = torch.arange(0, self.max_steps + num_features)[None, ].to(hiddens.device)

        for decoder_layer in self.decoders:
            position_embeddings = self.rotary_emb(hiddens, position_ids)

            hiddens = decoder_layer.forward(
                hiddens, 
                position_embeddings=position_embeddings, 
            )[0]

        # [total_seq_len, max_steps, model_size]
        return hiddens[:, num_features - 1:, :]
