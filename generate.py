import time

import lightning as L
import torch
from transformers import pipeline

from m2d.config import GenConfig
from m2d.model.llama import M2DLlama

L.seed_everything(227)

# create model
m2d_model: M2DLlama = M2DLlama.load_from_checkpoint(
    "/data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-mha-enhance-finish.ckpt", 
    map_location='cpu'
).to('cuda')

base_model = pipeline(
    task="text-generation", 
    model="meta-llama/Llama-3.2-1B-Instruct", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)

MAX_GEN_LEN = 1024

SYS_MSG = "You're a helpful AI assistant, and think carefully before giving your final answer. Wrap your reasoning process in <think> and </think>. "

while True:
    model = input("\033[32mWhat model to use [m2d/base]?\033[0m ")
    prompt = input("\033[32mInput:\033[0m\n")

    if model == "m2d":
        while True:
            reason = input("\033[32mReason mode [yes]/no?\033[0m ")
            if reason in ["", "yes", "no"]:
                break
        
        gen_start = time.time()
        output = m2d_model.generate(
            prompt=prompt, 
            config=GenConfig(
                max_gen_len=MAX_GEN_LEN, 
                system_message=SYS_MSG if (reason in ["", "yes"]) else None, 
                # repetition_penalty=1.2, 
                remove_think=(reason in ["", "yes"])
            )
        )
        gen_end = time.time()

        print("================================")
        print("Output:\n", output["output"])
        print("================================")
        print("Micro Token Output:\n", output["micro_token_output"])
        print("================================")
        print("Speedup:\n", output["speedup"])

    elif model == "base":
        gen_start = time.time()
        output = base_model(
            [{"role": "user", "content": prompt}], 
            max_new_tokens=MAX_GEN_LEN
        )
        gen_end = time.time()

        print("================================")
        print("Output:\n", output[0]["generated_text"][1]["content"])
    else:
        raise ValueError(f"Unknown model: {model}")

    print(f"Generation latency: {gen_end - gen_start:.5f}s")