from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import (LlamaDecoderLayer,
                                                      LlamaRotaryEmbedding)


class AttentionMerger(nn.Module):
    def __init__(
        self, 
        emb_size: int, 
        num_heads: int, 
        max_steps: int, 
        emb_dtype: torch.dtype
    ):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = self.emb_size // self.num_heads
        self.max_steps = max_steps
        self.emb_dtype = emb_dtype
        
        emb_plus_pe_size = self.emb_size + self.emb_size // 2
        self.q_proj = nn.Linear(self.emb_size, self.emb_size, dtype=self.emb_dtype)
        self.k_proj = nn.Linear(emb_plus_pe_size, self.emb_size, dtype=self.emb_dtype)
        self.v_proj = nn.Linear(emb_plus_pe_size, self.emb_size, dtype=self.emb_dtype)
        self.scale = self.head_size ** 0.5
        self.pos = nn.Parameter(
            torch.zeros(self.max_steps, self.emb_size // 2, dtype=self.emb_dtype), 
            requires_grad=True
        )
        nn.init.xavier_normal_(self.pos)

    def forward(self, embeddings: torch.Tensor):
        # embeddings [S, D]
        emb_len = embeddings.shape[0]
        # QKV
        hidden_shape = (-1, self.num_heads, self.head_size)
        q = self.q_proj.forward(embeddings.mean(dim=0, keepdim=True)).view(hidden_shape)
        # add positional information
        embeddings = torch.concat([embeddings, self.pos[:emb_len]], dim=-1)
        k = self.k_proj.forward(embeddings).view(hidden_shape)
        v = self.v_proj.forward(embeddings).view(hidden_shape)
        # cross attention with mean
        q = q.permute(1, 0, 2) # [H, 1, D]
        k = k.permute(1, 2, 0) # [H, D, S]
        v = v.permute(1, 0, 2) # [H, S, D]
        scores = torch.bmm(q, k) / self.scale # [H, 1, S]
        attention = F.softmax(scores, dim=-1) # [H, 1, S]
        output = torch.bmm(attention, v) # [H, 1, D]

        # [H, 1, D] -> [1, H, D] -> [1, H * D]
        return output.permute(1, 0, 2).view(-1, self.emb_size).contiguous()


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

        self.merger = AttentionMerger(
            emb_size=self.emb_size, 
            num_heads=64, # same as llama 
            max_steps=self.max_steps, 
            emb_dtype=self.emb_dtype
        )

        # TODO: will probably refactor this into merger later
        self.out_proj = nn.Linear(
            self.emb_size, 
            self.emb_size, 
            dtype=self.emb_dtype
        )

    def _merge_fn(self, embeddings: torch.Tensor):
        emb_len = embeddings.shape[0]
        if emb_len == 1:
            # we dont do merging
            return embeddings
        merge_emb = self.merger.forward(embeddings)
        merge_emb = self.out_proj.forward(merge_emb)

        return embeddings.mean(dim=0, keepdim=True) + merge_emb

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
        # decoder
        self.num_layers = 4
        self.decoders = nn.ModuleList([
            LlamaDecoderLayer(config=config, layer_idx=layer_idx) 
            for layer_idx in range(self.num_layers)])
        # stop prediction
        hidden_size = config.hidden_size
        model_dtype = config.torch_dtype
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=model_dtype), 
            nn.SiLU(), 
            nn.Linear(hidden_size, hidden_size, dtype=model_dtype), 
            nn.SiLU(), 
            nn.Linear(hidden_size, 1, dtype=model_dtype), 
        )
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

        # skip hidden for the first tokens: -1 means the original hidden output
        skip_hiddens = macro_step_hiddens[:, -1:, :]

        # first trim off the last label
        for i in range(len(token_embeds)):
            token_embeds[i] = token_embeds[i][:-1, :]
        # pad token embeds
        token_embeds.append(torch.zeros(self.max_steps - 1, macro_step_hiddens.shape[-1]).to(macro_step_hiddens.device))
        token_embeds = pad_sequence(token_embeds, batch_first=True, padding_value=0)[:-1]
        
        # concate together
        hiddens = torch.concat([macro_step_hiddens, token_embeds], dim=1)
        position_ids = torch.arange(0, num_features + self.max_steps - 1)[None, ].to(hiddens.device)

        for decoder_layer in self.decoders:
            position_embeddings = self.rotary_emb(hiddens, position_ids)

            hiddens = decoder_layer.forward(
                hiddens, 
                position_embeddings=position_embeddings, 
            )[0]

        # [total_seq_len, max_steps, model_size]
        micro_step_outputs = torch.concat([
            skip_hiddens, 
            hiddens[:, num_features:, :]
        ],dim=1)
        stop_outputs = self.stop_head.forward(micro_step_outputs)
        
        return micro_step_outputs, stop_outputs
