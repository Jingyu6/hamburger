import lightning as L

from m2d.model.llama import M2DLlama

L.seed_everything(227)

# create model
model: M2DLlama = M2DLlama.load_from_checkpoint(
    "./local/consolidated.pt"
)

prompt = """Could you write me a json file example?"""

output = model.generate(
    prompt=prompt, 
)

print("Prompt:\n", prompt)
print("Output:\n", output)
