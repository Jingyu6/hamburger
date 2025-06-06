"""
    Modified from https://github.com/pytorch-labs/gpt-fast
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from transformers import AutoTokenizer


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

create_block_mask = torch.compile(create_block_mask)

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import HAMburger, Transformer
from tokenizer import get_tokenizer


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def roundup(val, multiplier):
    return ((val - 1) // multiplier + 1) * multiplier


def causal_mask(b, h, q, kv):
    return q >= kv


def prefill(
    model: Transformer, 
    x: torch.Tensor, 
    input_pos: torch.Tensor, 
    **sampling_kwargs
) -> torch.Tensor:
    # TODO: probably need to separate later for compiling
    mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def prefill_hamburger(
    model: HAMburger, 
    x: torch.Tensor, 
    input_pos: torch.Tensor, 
    **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [B, S]
    mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    # we need to sample ourselves
    return model(mask, x, input_pos, is_prefill=True)


def decode_one_token(
    model: Transformer, 
    x: torch.Tensor, 
    input_pos: torch.Tensor, 
    block_mask: BlockMask, 
    **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_tokens_hamburger(
    model: HAMburger, 
    x: torch.Tensor, 
    input_pos: torch.Tensor, 
    block_mask: BlockMask, 
    **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [1, N]
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    return model(mask, x, input_pos, is_prefill=False)


def decode_n_tokens(
    model: Transformer | HAMburger, 
    cur_token: torch.Tensor, 
    input_pos: torch.Tensor, 
    num_new_tokens: int, 
    callback=lambda _: _, 
    is_hamburger: bool = False, 
    **sampling_kwargs
):
    block_mask = create_block_mask(causal_mask, 1, 1, model.max_seq_length, model.max_seq_length, device=cur_token.device)

    if is_hamburger:
        total_gen = 0
        new_tokens = [] # We don't use hamburger for draft so no need to output probs
        while total_gen < num_new_tokens:
            next_token = decode_tokens_hamburger(
                model, cur_token, input_pos, block_mask, **sampling_kwargs
            )
            output_len = len(next_token)
            input_pos += 1
            total_gen += output_len
            new_tokens.append(next_token.clone())
            if torch.any(next_token == 128009):
                break
            # TODO: ignore callback for now
            cur_token = next_token[None, ].clone()

        return torch.cat(new_tokens, dim=-1)[:min(total_gen, num_new_tokens)], None
    else:
        new_tokens, new_probs = [], []
        for i in range(num_new_tokens):
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, block_mask, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.clone()

        return new_tokens, new_probs


def model_forward(model, x, input_pos, mask):
    return model(mask, x, input_pos)


def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens, dim=1)
    # parallel inference on target model using draft tokens
    block_mask = create_block_mask(causal_mask, 1, 1, model.max_seq_length, model.max_seq_length, device=device)
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1, -1), draft_tokens], dim=-1).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device), 
        block_mask
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.cat(draft_probs)

    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]

    accept_draft_prob = torch.minimum(torch.ones_like(q[:speculate_k]), q[:speculate_k] / p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        block_mask = create_block_mask(causal_mask, 1, 1, 1, model.max_seq_length, device=device)
        model_forward(
            draft_model,
            draft_tokens[0, -1].view(1, -1),
            orig_input_pos + speculate_k, 
            block_mask
        )
        return torch.cat([draft_tokens.view(-1), last_token])
    else:
        accept_length = rejected_locations[0][-1].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[0, :accept_length].view(-1), next_token])


@torch.no_grad()
def generate(
    model: Transformer | HAMburger,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback = lambda x: x, 
    is_hamburger: bool = False, 
    device_main = default_device, 
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            assert not is_hamburger, "HAMburger does not support spec decoding"
            draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    device_sync(device=device_main)
    t_prefill_start = time.perf_counter()
    # From now on, we might output more than one token per step for HAMburger
    if is_hamburger:
        next_token = prefill_hamburger(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
    else:
        next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
    
    if is_speculative:
        assert not is_hamburger, "HAMburger does not support spec decoding"
        prefill(draft_model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
    
    device_sync(device=device_main)
    t_prefill_end = time.perf_counter()

    if is_hamburger:
        output_len = len(next_token)
        prefill_end = min(T + output_len, max_seq_length)
        next_token = next_token[:prefill_end - T]
        seq[:, T:prefill_end] = next_token
    else:
        seq[:, T] = next_token.squeeze()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)    
    
    accept_counts = [0] * (speculate_k + 1)

    device_sync(device=device_main)
    t_decode_start = time.perf_counter()
    if is_speculative:
        assert not is_hamburger, "HAMburger does not support spec decoding"
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(1, -1)

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[:, input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            for i in next_tokens[: num_added,]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        generated_tokens, _ = decode_n_tokens(
            model, 
            next_token.view(batch_size, -1), 
            input_pos, 
            max_new_tokens - (output_len if is_hamburger else 1), 
            callback=callback,
            is_hamburger=is_hamburger,  
            **sampling_kwargs
        )
        # TODO: it might hurt performance for HAMburger if we continue
        # generation after EOS
        if is_hamburger:
            seq = seq[:, :prefill_end + len(generated_tokens)]
            print(prefill_end, seq.shape, len(generated_tokens))
            seq[:, prefill_end:] = generated_tokens[None, ]
        else:
            seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)
    
    t_decode_end = time.perf_counter()

    generate_stats = {
        'accept_counts': accept_counts, 
        'prefill_time': t_prefill_end - t_prefill_start, 
        'decode_time': t_decode_end - t_decode_start
    }
    return seq, generate_stats


def encode_tokens(tokenizer, string, bos=True, device=default_device): 
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": string}], 
            add_generation_prompt=True, 
            return_tensors='pt', 
            return_dict=True
        )["input_ids"][0].to(device)
    else:
        tokens = tokenizer.encode(string)
        if bos:
            tokens = [tokenizer.bos_id()] + tokens
        return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(checkpoint_path, device, precision, use_tp, is_hamburger=False):
    use_cuda = 'cuda' in device
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    # do patching for hambuger model
    if is_hamburger:
        model = HAMburger.from_transformer(model)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]

    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params


# TODO: Fix this part latter to apply the right chat template
B_INST, E_INST = "[INST]", "[/INST]"


def main(
    prompt: Union[int, str] = "Hello, my name is", 
    prompt_file: Optional[Path] = None, 
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5, 
    is_hamburger: bool = False, 
    device=default_device,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    global print
    from tp import maybe_init_dist
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None

    # some hamburger related check
    if is_hamburger: 
        assert draft_checkpoint_path is None, "Currently hamburger doesn't support spec decoding"
        assert batch_size == 1, "Currently hamburger doesn't support batch size > 1"
        assert not use_tp, "Currently hambuger doesn't support TP"
        assert not compile_prefill, "Currenly hamburger doesn't support compile prefill"
        assert not interactive, "Currently hambuger doesn't support interactive"

    if is_speculative:
        assert batch_size == 1, "Currently speculative decoding is fixed with bs = 1"

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp, is_hamburger)

    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
    else:
        draft_model = None

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path, is_hamburger)

    if prompt_file is not None:
        with open(prompt_file, 'r') as f:
            prompt = f.read()
    
    if isinstance(prompt, str):
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    else:
        # generate a fully synthetic prompt
        encoded = torch.randint(0, 1024, (prompt,), device=device, dtype=torch.int64)
    prompt_length = encoded.size(-1)

    torch.manual_seed(1234)
    if compile:
        if is_speculative and use_tp: # and ("cuda" in device):
            torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        if is_hamburger:
            global decode_tokens_hamburger
            decode_tokens_hamburger = torch.compile(decode_tokens_hamburger, mode="reduce-overhead")
        else:
            global decode_one_token, prefill
            decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

            # Uncomment to squeeze more perf out of prefill
            if compile_prefill:
                prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    aggregate_metrics = {
        'prefill_tps': [],
        'decode_tps': [], 
        'accept_counts': [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        device_sync(device=device) # MKG
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0]
            done_generating = False
            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                
                assert x.numel() == 1, "Original code doesn't work with bs > 1 in interactive"
                buffer.append(tokenizer.decode([period_id, x.item()])[1:])
                
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print(''.join(buffer), end='', flush=True)
                    buffer.clear()
        else:
            callback = lambda x : x
        t0 = time.perf_counter()
        import contextlib
        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                batch_size=batch_size,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                is_hamburger=is_hamburger, 
                device_main=device, 
                temperature=temperature,
                top_k=top_k, 
            )
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
            aggregate_metrics['prefill_tps'].append(
                prompt_length / metrics["prefill_time"]
            )
            aggregate_metrics['decode_tps'].append(
                (y.size(-1) - prompt_length) / metrics["decode_time"]
            )
        if i == -1:
            device_sync(device=device) # MKG
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device) # MKG

        if not interactive:
            # Just displaying the first generation
            if batch_size > 1:
                print("Only displaying the first generation of the batch")
            print(tokenizer.decode(y[0].tolist()))
        else:
            print()
        
        print(f"Prefill TPS {i + 1}: {aggregate_metrics['prefill_tps'][-1]:.02f}")
        print(f"Decode TPS {i + 1}: {aggregate_metrics['decode_tps'][-1]:.02f}")
        print()
    print("==========")
    if is_speculative:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
        acceptance_probs = [i/sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

    print(f"Batch Size: {batch_size}")
    print(f"Prompt Length: {prompt_length}")
    print(f"Average prefill TPS: {torch.mean(torch.tensor(aggregate_metrics['prefill_tps'])).item():.2f}")
    print(f"Average decode TPS: {torch.mean(torch.tensor(aggregate_metrics['decode_tps'])).item():.2f}")    
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    def int_or_str(x):
        try:
            return int(x)
        except:
            return x

    parser.add_argument('--prompt', type=int_or_str, default="Hello, my name is", help="Input prompt. If it's an integer, will instead generate a synthetic prompt.")
    parser.add_argument('--prompt_file', type=Path, help="Input prompt, in the form of a file")
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to benchmark with')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')

    # hamburger specific arguments
    parser.add_argument('--is_hamburger', action='store_true', help="Is the model type HAMburger")

    args = parser.parse_args()
    main(
        args.prompt, args.prompt_file, args.interactive, args.num_samples, args.max_new_tokens, args.batch_size, args.top_k,
        args.temperature, args.checkpoint_path, args.compile, args.compile_prefill, args.profile, args.draft_checkpoint_path,
        args.speculate_k, args.is_hamburger, args.device
    )
