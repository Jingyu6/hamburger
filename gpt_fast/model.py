"""
    Modified from https://github.com/pytorch-labs/gpt-fast
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (BlockMask, _mask_mod_signature,
                                               create_block_mask,
                                               flex_attention)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_mask_mod(mask_mod: _mask_mod_signature, offset: int):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)

    return _mask_mod


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
            
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),

    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
    "llama-3.1-8b": dict(block_size=131072, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "llama-3.1-70b": dict(block_size=131072, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "llama-3.1-405b": dict(block_size=131072, n_layer=126, n_head=128, n_local_heads=8, dim=16384, intermediate_size=53248, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "Llama-3.2-1B-Instruct": dict(block_size=131072, n_layer=16, n_head=32, n_local_heads=8, dim=2048, intermediate_size=8192, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192)
    ), 
    # the same as Llama-3.2-1B-Instruct
    "hamburger": dict(block_size=131072, n_layer=16, n_head=32, n_local_heads=8, dim=2048, intermediate_size=8192, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192)
    )
}


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.get_mask_mod = get_mask_mod

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype, self.config.rope_scaling)

    def forward(
        self, 
        mask: BlockMask, 
        idx: Tensor, 
        input_pos: Optional[Tensor] = None, 
        skip_embedding: bool = False, 
        feature_layer_indices: List[int] = [] # for hamburger
    ) -> Tensor | List[Tensor]:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])
        freqs_cis = self.freqs_cis[input_pos]
        if skip_embedding:
            x = idx
        else:
            x = self.tok_embeddings(idx)

        return_features = len(feature_layer_indices) > 0
        if return_features:
            features = []

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
            # we store the last one after norm
            if i in feature_layer_indices and i != len(self.layers) - 1:
                features.append(x[:, -1:, :])
        x = self.norm(x)
        
        if len(self.layers) - 1 in feature_layer_indices:
            features.append(x[:, -1:, :])
        if return_features:
            return features
        
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class _AttentionMerger(nn.Module):
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


class _CompositionalEmbedder(nn.Module):
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

        self.merger = _AttentionMerger(
            emb_size=self.emb_size, 
            num_heads=64, # same as llama 
            max_steps=self.max_steps, 
            emb_dtype=self.emb_dtype
        )

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

    def forward(
        self,
        input_ids: torch.Tensor, 
        is_prefill: bool
    ):
        embeddings = self.embedding.forward(input_ids)
        if not is_prefill:
            # TODO: later change it to have bs > 1
            embeddings = self._merge_fn(embeddings[0])[None, ]
        return embeddings


class _ConditionalMicroStepDecoder(nn.Module):
    def __init__(
        self, 
        config: ModelArgs, 
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
            TransformerBlock(config)
            for _ in range(self.num_layers)])
        # stop prediction
        hidden_size = config.dim
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.SiLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.SiLU(), 
            nn.Linear(hidden_size, 1), 
        )
        self.feature_layer_indices = [3, 7, 11, 15]
        
        # kv cache related info
        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_seq_length = -1
        self.max_batch_size = -1
        self.get_mask_mod = get_mask_mod
        self.mask = None

    def setup_caches(self, max_batch_size, config, dtype):
        # no more check for seq_len since its fixed
        if self.max_batch_size >= max_batch_size:
            return

        # setup cache for micro step decoder
        head_dim = config.dim // config.n_head
        self.max_seq_length = find_multiple(self.max_steps + len(self.feature_layer_indices), 8)
        self.max_batch_size = max_batch_size
        for b in self.decoders:
            b.attention.kv_cache = KVCache(
                self.max_batch_size, 
                self.max_seq_length, 
                config.n_local_heads, 
                head_dim, 
                dtype
            )

        self.freqs_cis = precompute_freqs_cis(
            config.block_size, 
            config.dim // config.n_head, 
            config.rope_base, 
            dtype, 
            config.rope_scaling
        )

        self.mask = create_block_mask(
            lambda b, h, q, kv: q >= kv, 
            1, 1, self.max_seq_length, self.max_seq_length
        )


class HAMburger(nn.Module):
    def __init__(self, model: Transformer):
        super().__init__()
        self.model = model
        self.config = self.model.config
        # extra module here
        self.max_steps = 4 # TODO: hard coded for now
        self.comp_embedder = _CompositionalEmbedder(
            embedding=self.model.tok_embeddings, 
            max_steps=self.max_steps
        )
        self.micro_step_decoder = _ConditionalMicroStepDecoder(
            config=self.model.config, 
            max_steps=self.max_steps
        )
        self.micro_step_confidence = None

        # gpt-fast related
        self.max_batch_size = -1
        self.max_seq_length = -1

    @classmethod
    def from_transformer(cls, model: Transformer) -> "HAMburger":
        return cls(model)

    def setup_caches(self, max_batch_size, max_seq_length):
        dtype = self.model.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.model.output, "scales"):
            dtype = self.model.output.scales.dtype
        elif hasattr(self.model.output, "scales_and_zeros"):
            dtype = self.model.output.scales_and_zeros.dtype
        
        # base model setup
        self.model.setup_caches(max_batch_size, max_seq_length)
        self.micro_step_decoder.setup_caches(max_batch_size, self.config, dtype)

        self.max_batch_size = max(self.max_batch_size, self.model.max_batch_size)
        self.max_seq_length = max(self.max_seq_length, self.model.max_seq_length)

    def forward(
        self, 
        mask: BlockMask, 
        idx: Tensor, 
        input_pos: Optional[Tensor] = None, 
        is_prefill: bool = False
    ) -> Tensor:
        """
            Input shapes:
                idx: [bs, seqlen]
                input_pos: [seqlen]
        """

        # Compositional embedder [bs, l, d]
        x = self.comp_embedder.forward(idx, is_prefill)

        # Base models list of [bs, 1, d] since we just want the last one
        features = self.model.forward(
            mask=mask, 
            idx=x, 
            input_pos=input_pos, 
            skip_embedding=True, 
            feature_layer_indices=self.micro_step_decoder.feature_layer_indices
        )

        hiddens = torch.concat(features, dim=1)

        output_ids = []
        micro_input_pos = torch.arange(0, hiddens.shape[-2] + 1, device=hiddens.device)

        for i in range(self.max_steps):
            if i > 0:
                # TODO: Look at how cache is done
                self.micro_step_decoder.mask.mask_mod = \
                    self.micro_step_decoder.get_mask_mod(
                        self.micro_step_decoder.mask.mask_mod, micro_input_pos[0])
                micro_freqs_cis = self.micro_step_decoder.freqs_cis[micro_input_pos]
                for decoder in self.micro_step_decoder.decoders:
                    hiddens = decoder(hiddens, micro_input_pos, micro_freqs_cis, self.micro_step_decoder.mask)
            
                micro_input_pos = torch.arange(
                    micro_input_pos[-1] + 1, 
                    micro_input_pos[-1] + 2, 
                    device=hiddens.device
                )

            last_hidden = hiddens[:, -1:, :]
            logits = self.model.output.forward(last_hidden)
            # TODO: later try sampling
            pred_token = logits.argmax(dim=-1).view(-1)
            pred_stop = F.sigmoid(self.micro_step_decoder.stop_head.forward(last_hidden)).view(-1)

            if i > 0:
                hiddens = self.comp_embedder.embedding.forward(pred_token).view(1, 1, -1)
            else:
                hiddens = torch.concat([
                    hiddens, 
                    self.comp_embedder.embedding.forward(pred_token).view(1, 1, -1)
                ], dim=1)

            output_ids.append(pred_token)

            if self.micro_step_confidence is not None:
                # we stop if our continue confident is less than X
                if (1.0 - pred_stop) < self.micro_step_confidence:
                    break
            elif pred_stop > 0.5:
                break

        return torch.concat(output_ids, dim=0)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: BlockMask) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: BlockMask, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        y = flex_attention(q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads))

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)