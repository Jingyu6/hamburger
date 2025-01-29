import lightning as L

from m2d.model.llama import M2DLlama

L.seed_everything(227)

# create model
model: M2DLlama = M2DLlama.load_from_checkpoint(
    "./local/ckpts/m2d-llama-1B.ckpt"
)

prompt = """How many 4-letter words with at least one consonant can be constructed from the letters $A$, $B$, $C$, $D$, and $E$? (Note that $B$, $C$, and $D$ are consonants, any word is valid, not just English language words, and letters may be used more than once.)"""

output = model.generate(
    prompt=prompt, 
    max_gen_len=256
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
