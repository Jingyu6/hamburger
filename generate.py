import lightning as L

from m2d.model.llama import M2DLlama

L.seed_everything(227)

# create model
model: M2DLlama = M2DLlama.load_from_checkpoint(
    "./local/ckpts/m2d-llama-1B-step=13312.ckpt"
)

prompt = """Could you write me a python program doing quick sort?"""

output = model.generate(
    prompt=prompt, 
)

print("================================")
print("Prompt:\n", prompt)
print("================================")
print("Output:\n", output["output"])
print("================================")
print("Token Output:\n", output["token_output"])
print("================================")
print("Micro Token Output:\n", output["micro_token_output"])
print("================================")
print("Speedup:\n", output["speedup"])
