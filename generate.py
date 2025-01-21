import lightning as L

from m2d.model.llama import M2DLlama

L.seed_everything(227)

# create model
model: M2DLlama = M2DLlama.load_from_checkpoint(
    "./local/ckpts/m2d-llama-1B-step=5120.ckpt"
)

prompt = """Question: Process: - Greenhouse gases are released into the air by human activities - The earth changes energy from the sun into heat - Heat rises from the ground - Greenhouse gas molecules in the atmosphere prevent the heat from going into space - The temperature of the earth increases - The temperature continues to rise and melts the polar ice caps - The temperature rises even faster. Question: suppose animal comes into contact with more sick animals happens, how will it affect More greenhouse gases are produced. How does the supposed perturbation influence the second effect mentioned. Answer by more, less or no effect Answer:"""

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
