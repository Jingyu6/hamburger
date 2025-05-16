import os

import lightning as L
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, TextStreamer,
                          pipeline)

from hamburger.config import GenConfig
from hamburger.model.llama import HAMburgerLlama

L.seed_everything(227)

# create model
hamburger_model: HAMburgerLlama = HAMburgerLlama.load_from_checkpoint(
    "/data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-1B-0506-finish.ckpt", 
    map_location='cpu'
).to('cuda')
hamburger_tokenizer = AutoTokenizer.from_pretrained(hamburger_model.base_model_name)

hf_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
    device_map='cuda', 
)
hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
hf_tokenizer.pad_token = hf_tokenizer.eos_token
base_model = pipeline(task="text-generation", model=hf_model, tokenizer=hf_tokenizer)

MAX_GEN_LEN = 1024

SYS_MSG = "You're a helpful AI assistant, and think carefully before giving your final answer. Wrap your reasoning process in <think> and </think>. "

while True:
    model = input("\033[32mWhat model to use [hamburger/base]?\033[0m ")
    prompt = input("\033[32mInput:\033[0m\n")
    if os.path.exists(prompt):
        with open(prompt, 'r') as f:
            prompt = f.read()

    if model == "hamburger":
        while True:
            reason = input("\033[32mReason mode [yes]/no?\033[0m ")
            if reason in ["", "yes", "no"]:
                break
        
        streamer = TextStreamer(tokenizer=hf_tokenizer, skip_prompt=True)
        streamer.next_tokens_are_prompt = False # remove artifacts

        output = hamburger_model.generate(
            prompt=prompt, 
            config=GenConfig(
                max_gen_len=MAX_GEN_LEN, 
                system_message=SYS_MSG if (reason in ["", "yes"]) else None, 
                remove_think=(reason in ["", "yes"])
            ), 
            streamer=streamer
        )

        print("================================")
        print("Micro Token Output:\n", output["micro_token_output"])
        print("================================")
        print("Speedup:\n", output["speedup"])

    elif model == "base":
        streamer = TextStreamer(tokenizer=hf_tokenizer, skip_prompt=True)
        output = base_model(
            [{"role": "user", "content": prompt}], 
            max_new_tokens=MAX_GEN_LEN, 
            streamer=streamer
        )
    else:
        raise ValueError(f"Unknown model: {model}")
