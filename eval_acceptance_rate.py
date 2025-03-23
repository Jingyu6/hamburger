import argparse

import torch
import tqdm
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          RepetitionPenaltyLogitsProcessor, TopPLogitsWarper)


def acceptance_rate(
    target_logits, 
    draft_logits, 
    temp=1.0, 
    top_p=1.0, 
    rep_penalty=1.0, 
    input_ids=None
):
    # `target_logits` and `draft_logits` should have shape `(batch_size, seq_len, vocab_size)`.
    assert len(target_logits.shape) == 3
    assert len(draft_logits.shape) == 3
    draft_logits = draft_logits.to(target_logits.device)

    if rep_penalty != 1.0:
        processor = RepetitionPenaltyLogitsProcessor(rep_penalty)
        for i in range(1, target_logits.size(1)):
            processor(input_ids[:, :i], target_logits[:, i])
            processor(input_ids[:, :i], draft_logits[:, i])

    if top_p != 1.0:
        target_logits = TopPLogitsWarper(top_p)(input_ids, target_logits)
        draft_logits = TopPLogitsWarper(top_p)(input_ids, draft_logits)

    if temp != 0.0:
        target_logits = target_logits / temp
        draft_logits = draft_logits / temp

        # Compute softmax distributions
        target_p = torch.softmax(target_logits, dim=-1)
        draft_p = torch.softmax(draft_logits, dim=-1)

        # L1 distance between distributions => sum of absolute differences
        # normalized so that if distributions are identical, acceptance = 1,
        # if they're completely disjoint, acceptance rate = 0
        # shape: (batch_size, seq_len)
        acc = 1.0 - (target_p - draft_p).abs().sum(dim=-1) / 2.0

    else:
        target_p = torch.argmax(target_logits, dim=-1, keepdim=False)
        draft_p = torch.argmax(draft_logits, dim=-1, keepdim=False)
        acc = 1.0 - (target_p != draft_p).float()

    assert len(acc.shape) == 2
    return acc

def get_prompt_response(item, tokenizer):
    input1 = item["turn_1_input"]
    output1 = item[args.output_key]

    #assign user and assistant roles
    chat = [{"role": "user", "content": input1}, {"role": "assistant", "content": output1}]

    roles = [turn['role'] for turn in chat]
    if roles[:2] == ['user', 'assistant']:
        prompt_in_llama3_format = tokenizer.apply_chat_template(chat[:1], tokenize=False, add_generation_prompt=True)
        prompt_response_in_llama3_format = tokenizer.apply_chat_template(chat[:2], tokenize=False)
    elif roles[:3] == ['system', 'user', 'assistant']:
        prompt_in_llama3_format = tokenizer.apply_chat_template(chat[:2], tokenize=False, add_generation_prompt=True)
        prompt_response_in_llama3_format = tokenizer.apply_chat_template(chat[:3], tokenize=False)
    else:
        raise ValueError('Unexpected sequence of roles')

    return {
        'prompt': prompt_in_llama3_format,
        'text': prompt_response_in_llama3_format,
        'category': item['category']
    }

def tokenize_text_prompt(example, tokenizer, seq_length=2048):
    text = example["text"]
    prompt = example["prompt"]
    all_tokens = tokenizer(text, return_tensors='pt')
    prompt_tokens = tokenizer(prompt, return_tensors='pt')
    labels = all_tokens['input_ids'].clone()
    labels[:, :prompt_tokens['input_ids'].shape[1]] = -100
    attn_mask = all_tokens['attention_mask']
    
    return {
        'input_ids': all_tokens['input_ids'][:, :seq_length],
        'labels': labels[:, :seq_length],
        'attention_mask': attn_mask[:, :seq_length],
    }

def prepare_batch(example, model, tokenizer):
    item = tokenize_text_prompt(example, tokenizer)
    
    input_ids = item["input_ids"]
    labels = item["labels"]

    attention_mask = item["attention_mask"]

    return {
        "input_ids": input_ids.to(model.device),
        "attention_mask": attention_mask.to(model.device),
        "labels": labels.to(model.device)
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Run speculative decoding benchmark')
    parser.add_argument('--hf_data', type=str, default="togethercomputer/specbench-Llama3.1-405B",
        help='HuggingFace dataset to use for benchmarking')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
        help='HuggingFace model ID or path for target model')
    parser.add_argument('--draft_model', type=str, 
        default='meta-llama/Llama-3.2-1B',
        help='HuggingFace model ID or path for draft model')
    parser.add_argument('--draft_model_type', type=str, choices=["hf", "m2d"], 
        default="hf", 
        help="What is the model type being used")
    parser.add_argument('--temperature', type=float, default=1.0,
        help='Temperature for target model')
    parser.add_argument('--output_key', type=str, default='turn_1_output_temp1',
        help='Key for output in dataset')
    return parser.parse_args()

if __name__ == "__main__":
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0)) 

    args = parse_args()

    model_id = args.target_model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    model.eval()

    draft_model_id = args.draft_model
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_id, device_map="auto")
    draft_model.eval()

    dataset = load_dataset(args.hf_data, split="train")
    all_accept_rates = {}

    pbar = tqdm.tqdm(dataset)
    for example in pbar:
        item = get_prompt_response(example, tokenizer)
        inputs = prepare_batch(item, model, tokenizer)
        draft_inputs = prepare_batch(item, draft_model, tokenizer)
        labels = inputs["labels"]

        with torch.no_grad():
            target_valid_logits_len = (labels != -100).sum(dim=-1)
            if target_valid_logits_len[0] == 0:
                print("skipping this item")
                print(target_valid_logits_len[0])
                continue

            target_output = model.forward(**inputs)
            target_logits = target_output.logits
            target_logits = target_logits[labels != -100]
            # only keep non-ignored tokens
            draft_output = draft_model.forward(**draft_inputs)
            draft_logits = draft_output.logits
            # only keep non-ignored tokens
            draft_valid_logits_len = (labels != -100).sum(dim=-1)
            draft_logits = draft_logits[labels.to(draft_model.device) != -100]

            # we want to make sure the valid logits are the same
            if not torch.all(target_valid_logits_len == draft_valid_logits_len).item():
                raise ValueError("The valid logits lengths are different.")
            
            target_logits = target_logits.unsqueeze(0)
            draft_logits = draft_logits.unsqueeze(0)

            accept_rates = acceptance_rate(target_logits, draft_logits, temp=args.temperature, top_p=1.0)

            # restore 1D shape
            accept_rates = accept_rates.squeeze(0)

            # Use torch.split to divide accept_rates into segments for batch_size
            accept_rates_segments = torch.split(accept_rates, draft_valid_logits_len.tolist())

            # Calculate mean for each segment
            accept_rates = torch.cat([segment.mean().unsqueeze(0) for segment in accept_rates_segments])
            
            if item["category"] not in all_accept_rates:
                all_accept_rates[item["category"]] = []
            all_accept_rates[item["category"]].extend(accept_rates.tolist())

            all_accept_rates_in_category = []
        
            postfix_dict = {}
            for category, accept_rates in all_accept_rates.items():
                running_mean = torch.mean(torch.tensor(accept_rates))    
                postfix_dict[f"running_mean_{category}"] = running_mean.item()
                all_accept_rates_in_category += [running_mean.item()]
            
            all_running_mean = torch.mean(torch.tensor(all_accept_rates_in_category))
            postfix_dict["running_mean_all"] = all_running_mean.item()
            
            pbar.set_postfix(**postfix_dict)

    # Print all acceptance rates with draft model ID
    print(f"Acceptance rates for draft model: {draft_model_id}")
    for category, rates in all_accept_rates.items():
        mean_rate = sum(rates) / len(rates)
        print(f"{category}: {mean_rate:.3f}")
    print(f"Overall mean acceptance rate: {all_running_mean:.3f}")
    print("\n")