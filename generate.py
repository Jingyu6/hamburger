import lightning as L
import torch
from transformers import pipeline

from m2d.config import GenConfig
from m2d.model.llama import M2DLlama

L.seed_everything(227)

# create model
m2d_model: M2DLlama = M2DLlama.load_from_checkpoint(
    "./local/ckpts/m2d-llama-1B-finish.ckpt", 
    map_location='cpu'
).to('cuda')

base_model = pipeline(
    task="text-generation", 
    model="meta-llama/Llama-3.2-1B-Instruct", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)

SYS_MSG = "You're a helpful AI assistant, and think carefully before giving your final answer. Wrap your reasoning process in <think> and </think>. "

while True:
    model = input("\033[32mWhat model to use [m2d/base]?\033[0m ")
    prompt = input("\033[32mInput:\033[0m\n")

    if model == "m2d":
        while True:
            reason = input("\033[32mReason mode [yes]/no?\033[0m ")
            if reason in ["", "yes", "no"]:
                break

        output = m2d_model.generate(
            prompt=prompt, 
            config=GenConfig(
                max_gen_len=1024, 
                system_message=SYS_MSG if (reason in ["", "yes"]) else None, 
                # repetition_penalty=1.2, 
                remove_think=(reason in ["", "yes"])
            )
        )

        print("================================")
        print("Output:\n", output["output"])
        print("================================")
        print("Token Output:\n", output["token_output"])
        print("================================")
        print("Micro Token Output:\n", output["micro_token_output"])
        print("================================")
        print("Speedup:\n", output["speedup"])

    elif model == "base":
        output = base_model(
            [{"role": "user", "content": prompt}], 
            max_new_tokens=256
        )

        print("================================")
        print("Output:\n", output[0]["generated_text"][1]["content"])
    else:
        raise ValueError(f"Unknown model: {model}")
