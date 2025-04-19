import argparse
import random
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from m2d.config import GenConfig
from m2d.model.llama import M2DLlama

GEN_CONFIG = GenConfig(
    micro_step_confidence=None
)

# Adopted from https://github.com/feifeibear/LLMSpeculativeSampling
def _max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    return x_max / x_max_sum


# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def _top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """
    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


# Adopted from https://github.com/feifeibear/LLMSpeculativeSampling
def _norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """
    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = _top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


# Adopted from https://github.com/feifeibear/LLMSpeculativeSampling
def _sample(probs : torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    if (idx_next.item() == 0):
        # in case of invalid values, we simply use greedy
        return torch.argmax(probs, dim=-1, keepdim=True)
    return idx_next


# Adopted from https://github.com/feifeibear/LLMSpeculativeSampling
class KVCacheModel():
    def __init__(
        self, 
        model: nn.Module, 
        model_type: str, 
        temperature: float = 1, 
        top_k: int = 0, 
        top_p: float = 0
    ) -> None:
        self._model = model
        self._model_type = model_type
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

        # for m2d models
        self._steps = None

    def _ar_forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model.speculate(input_ids, use_cache=True)
            self._prob_history = outputs["logits"]
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = _norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs["past_key_values"]
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
                
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            outputs = self._model.speculate(
                last_input_id, 
                past_key_values=self._past_key_values, 
                use_cache=True
            )
            
            not_cached_q = outputs["logits"]
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = _norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
            
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs["past_key_values"]
        
        return torch.argmax(last_q, dim=-1, keepdim=True)
    
    def _m2d_forward_with_kvcache(self, input_ids: torch.Tensor, idx: int) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # prefill
            prefix_len = input_ids.shape[-1]
            outputs = self._model.speculate(input_ids, is_prefill=True, config=GEN_CONFIG)
            # add dummy logits since we don't go through the lm head
            self._prob_history = torch.concat([
                torch.zeros(
                    (1, prefix_len - 1, outputs["logits"].shape[-1]), 
                    dtype=outputs["logits"].dtype, 
                    device=outputs["logits"].device), 
                outputs["logits"]
            ], dim=1)
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = _norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs["past_key_values"]
            
            self._steps = [outputs["ids"].shape[-1]]
            return outputs["ids"]
        else:
            # decode
            prefix_len = self._prob_history.shape[-2]
            if idx == 0:
                last_input_ids = input_ids[:, prefix_len:]
            else:
                last_input_ids = input_ids[:, prefix_len - self._steps[-1]:]

            if last_input_ids.dim() == 1:
                last_input_ids = torch.unsqueeze(last_input_ids, 0)
                
            outputs = self._model.speculate(
                last_input_ids, 
                is_prefill=False, 
                prefix_len=prefix_len, 
                past_key_values=self._past_key_values, 
                config=GEN_CONFIG
            )
            
            not_cached_q = outputs["logits"]
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = _norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
            
            if idx == 0:
                # pad non-computed probabilities
                # because we received extra tokens from the base model
                self._prob_history = torch.cat([
                    self._prob_history, 
                    torch.zeros(
                        (1, last_input_ids.shape[-1] - 1, self._prob_history.shape[-1]), 
                        dtype=self._prob_history.dtype, 
                        device=self._prob_history.device), 
                ], dim=1)

            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            self._past_key_values = outputs["past_key_values"]

            if idx == 0:
                self._steps = [outputs["ids"].shape[-1]]
            else:
                self._steps.append(outputs["ids"].shape[-1])
            return outputs["ids"]
        
    def _generate_with_kvcache(
        self, prefix : torch.Tensor, 
        gamma : int, 
    ) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix
        for idx in range(gamma):
            if self._model_type == "ar":
                next_toks = self._ar_forward_with_kvcache(x)
            elif self._model_type == "m2d":
                # we might end up draft more than gamma due to multi-step decode
                next_toks = self._m2d_forward_with_kvcache(x, idx=idx)
            else:
                raise ValueError
            x = torch.cat((x, next_toks), dim=1)
        return x

    @torch.inference_mode
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output
    
    @torch.inference_mode
    def rollback(self, end_pos: int, add_new: bool = False):
        # change to align with new HF API
        if self._model_type == "ar":
            assert end_pos is not None
            self._past_key_values._seen_tokens = end_pos
            for layer_idx in range(len(self._past_key_values.value_cache)):
                self._past_key_values.key_cache[layer_idx] = \
                    self._past_key_values.key_cache[layer_idx][:, :, :end_pos, :]
                self._past_key_values.value_cache[layer_idx] = \
                    self._past_key_values.value_cache[layer_idx][:, :, :end_pos, :]
            self._prob_history = self._prob_history[:, :end_pos, :]

        elif self._model_type == "m2d":
            # determine the number of KVs we need to evict
            prefix_len = self._prob_history.shape[-2]

            delete_kv_cnt = 0
            for step in reversed(self._steps):
                prefix_len -= step
                if end_pos <= prefix_len:
                    delete_kv_cnt += 1

            if delete_kv_cnt > 0:
                # we only remove the last macro KV if all tokens are rejected
                kv_len = self._past_key_values._seen_tokens
                self._past_key_values._seen_tokens -= delete_kv_cnt
                for layer_idx in range(len(self._past_key_values.value_cache)):
                    self._past_key_values.key_cache[layer_idx] = \
                        self._past_key_values.key_cache[layer_idx][:, :, :kv_len - delete_kv_cnt, :]
                    self._past_key_values.value_cache[layer_idx] = \
                        self._past_key_values.value_cache[layer_idx][:, :, :kv_len - delete_kv_cnt, :]
            
            self._prob_history = self._prob_history[:, :end_pos, :]        


# Adopted from https://github.com/feifeibear/LLMSpeculativeSampling
@torch.inference_mode
def speculative_sampling(
    prefix: torch.Tensor, 
    base_model: torch.nn.Module, 
    draft_model: torch.nn.Module, 
    draft_model_type: str, 
    max_gen_len: int, 
    gamma: int = 4,
    temperature: float = 1, 
    top_k: int = 0, 
    top_p: float = 0, 
    eos_token_id: Optional[int] = None, 
    print_latency: bool = False
):
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        draft_model (torch.nn.Module): approx model, the small one
        base_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_gen_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert draft_model.device == base_model.device
    
    device = base_model.device
    
    draft_model_cache = KVCacheModel(draft_model, draft_model_type, temperature, top_k, top_p)
    base_model_cache = KVCacheModel(base_model, "ar", temperature, top_k, top_p)
    
    resample_count = 0
    base_sample_count = 0
    accepted_count = 0
    draft_count = 0
    decode_steps = 0

    decode_start = None
    decode_end = None
    
    while prefix.shape[1] < T:
        prefix_len = prefix.shape[1]

        draft_start = time.time()
        x = draft_model_cache.generate(prefix, gamma)
        draft_end = time.time()
        _ = base_model_cache.generate(x, 1)

        draft_cnt = x.shape[1] - prefix_len
        draft_count += draft_cnt
        n = prefix_len + draft_cnt - 1
        decode_steps += 1

        if print_latency:
            print(f"Drafting {draft_cnt} tokens in {(draft_end - draft_start):.2f} seconds.")

        for i in range(draft_cnt):
            r = torch.rand(1, device=device)
            j = x[:, prefix_len + i]
            
            if r > (base_model_cache._prob_history[:, prefix_len + i - 1, j]) / (draft_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                break
            
            accepted_count += 1
        
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]

        # check for stop during acceptance
        if torch.any(prefix[:, prefix_len:n + 1] == eos_token_id):
            # this might give a few tokens off but does not
            # change the final stats significantly
            break
        
        draft_model_cache.rollback(end_pos=n + 1)
        
        assert draft_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {draft_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + draft_cnt - 1:
            # reject someone, sample from the pos n
            t = _sample(_max_fn(base_model_cache._prob_history[:, n, :] - draft_model_cache._prob_history[:, n, :]))
            resample_count += 1
            base_model_cache.rollback(end_pos=n + 1)
        else:
            # all approx model decoding accepted
            assert n == base_model_cache._prob_history.shape[1] - 1
            t = _sample(base_model_cache._prob_history[:, -1, :])
            base_sample_count += 1
            base_model_cache.rollback(end_pos=n + 2)

        prefix = torch.cat((prefix, t), dim=1)

        # check again for stop during acceptance
        if torch.any(t == eos_token_id):
            break

        if decode_start is None:
            decode_start = time.time()

    if decode_start is None:
        print(f"Warning: stop at the first token")
        decode_start = time.time()
    decode_end = time.time()

    return {
        "token_ids": prefix, 
        "gen_len": prefix.shape[-1] - seq_len, 
        "decode_steps": decode_steps, 
        "draft_count": draft_count, 
        "accepted_count": accepted_count, 
        "base_sample_count": base_sample_count, 
        "resample_count": resample_count, 
        "decode_time": (decode_end - decode_start)
    }


def prepare_data(
    dataset_name: str, 
    subset: str, 
    split: str, 
    prompt_key: str, 
    max_samples: int, 
    tokenizer: AutoTokenizer
):
    dataset = load_dataset(dataset_name, name=subset, split=split).take(max_samples)
    
    prompt_ids = []
    for prompt in tqdm(dataset[prompt_key], desc="Tokenize data"):
        conversation = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            conversation, 
            # this is used to make the model not output the gen prompt
            add_generation_prompt=True, 
            return_tensors='pt', 
            return_dict=True
        )["input_ids"]
        prompt_ids.append(input_ids)
    
    return prompt_ids


def parse_args():
    parser = argparse.ArgumentParser(description='Run speculative decoding benchmark')
    
    parser.add_argument('--dataset_name', type=str, 
        default="openai/gsm8k",
        help='HuggingFace dataset to use for benchmarking')
    parser.add_argument('--subset', type=str, default=None,
        help='Subset name in HF dataset')
    parser.add_argument('--split', type=str, default="train",
        help='Split name in HF dataset')
    parser.add_argument('--prompt_key', type=str, default="prompt",
        help='The key in the dataset for prompts')
    parser.add_argument('--max_samples', type=int, default=256,
        help='Max number of samples to evaluate')
    
    parser.add_argument('--base_model', type=str, 
        default='meta-llama/Llama-3.2-3B-Instruct',
        help='HuggingFace model ID or path for base model')
    parser.add_argument('--draft_model', type=str, 
        default='meta-llama/Llama-3.2-1B-Instruct',
        help='HuggingFace model ID or path for draft model')
    parser.add_argument('--draft_model_type', type=str, 
        default='ar', choices=["ar", "m2d"], 
        help='What model type is used as the draft model')
    parser.add_argument("--device", type=str, default="cuda:0", 
        help='Device name')
    
    parser.add_argument('--max_gen_len', type=int, default=256,
        help='Max number of tokens to generate')
    parser.add_argument('--gamma', type=int, default=4,
        help='Max number of drafting tokens')
    
    parser.add_argument('--print_output', action="store_true", default=False, 
        help="Whether to print the generated output")
    parser.add_argument('--print_latency', action="store_true", default=False, 
        help="Whether to print the latency of drafting")
    parser.add_argument('--seed', type=int, default=227,
        help='Random seed for experiments')
    
    parser.add_argument('--compile', action="store_true", default=False, 
        help="Whether to torch.compile the model")
    
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)    
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    data = prepare_data(
        dataset_name=args.dataset_name, 
        subset=args.subset, 
        split=args.split, 
        prompt_key=args.prompt_key, 
        max_samples=args.max_samples, 
        tokenizer=tokenizer
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        device_map=args.device
    )

    base_model.speculate = base_model.forward

    if args.draft_model_type == "ar":
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            device_map=args.device
        )
        draft_model.speculate = draft_model.forward
    elif args.draft_model_type == "m2d":
        draft_model = M2DLlama.load_from_checkpoint(
            args.draft_model, 
            map_location='cpu'
        ).to(args.device)

        draft_model.max_steps -= 1
    else:
        raise ValueError(f"Unsupported draft model type {args.draft_model_type}.")

    if args.compile:
        print("Compiling models.")
        base_model = torch.compile(base_model)
        draft_model = torch.compile(draft_model)

    print(f"Start evaluating dataset {args.dataset_name} with {len(data)} samples.")
    print(f"Base model: {args.base_model}")
    print(f"Draft model: {args.draft_model}")

    accepted_ratios = []
    draft_efficiencies = []
    gammas = []
    gen_lens = []
    total_decode_time = 0

    for prompt_ids in tqdm(data, desc="Evaluate speculative decoding"):
        prompt_len = prompt_ids.shape[-1]
        outputs = speculative_sampling(
            prefix=prompt_ids.to(base_model.device), 
            draft_model=draft_model, 
            draft_model_type=args.draft_model_type, 
            base_model=base_model, 
            max_gen_len=args.max_gen_len, 
            gamma=args.gamma, 
            eos_token_id=tokenizer.eos_token_id, 
            print_latency=args.print_latency
        )
        if args.print_output:
            print(tokenizer.decode(
                outputs["token_ids"].cpu().view(-1).tolist()[prompt_len:]
            ))
        accepted_ratios.append(outputs["accepted_count"] / outputs["gen_len"])
        draft_efficiencies.append(outputs["accepted_count"] / outputs["draft_count"])
        gammas.append(outputs["draft_count"] / outputs["decode_steps"])
        gen_lens.append(outputs["gen_len"])
        total_decode_time += outputs["decode_time"]

    assert len(accepted_ratios) > 0
    print(f"Average accepted ratio: {sum(accepted_ratios) / len(accepted_ratios) * 100:.2f}%")
    print(f"Average draft efficiency: {sum(draft_efficiencies) / len(draft_efficiencies) * 100:.2f}%")
    print(f"Average gamma: {sum(gammas) / len(gammas):.2f} tokens / step")
    print(f"Average generation length: {sum(gen_lens) / len(gen_lens):.2f} tokens")
    print(f"Average decoding TPS: {sum(gen_lens) / (total_decode_time):.2f} tokens / sec")


if __name__ == "__main__":
    main()
