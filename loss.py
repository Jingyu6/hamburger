import random
from typing import List

SEQ_LEN = 16
VOCAB_SIZE = 4
STOP = VOCAB_SIZE
SEP = -1
MAX_STEPS = 3

context_len = random.randint(1, SEQ_LEN - 2)
output_len = SEQ_LEN - context_len
input_ids = [random.randint(0, VOCAB_SIZE - 1) for _ in range(SEQ_LEN)]
input_ids[context_len - 1] = SEP

# random split for now
steps = []
total = 0

while True:
    step = random.randint(1, min(MAX_STEPS, output_len - total))
    steps.append(step)
    total += step
    if total == output_len:
        break

print("Input IDs:", input_ids)
print("Steps:", steps)

def emb(
    input_ids: List[int]
):
    # identity for now
    return [str(token) for token in input_ids]

def com(
    embeddings: List[str], 
    steps: List[int], 
    context_len: int
):
    composition = []
    for i in range(context_len):
        composition.append(embeddings[i])
    i = context_len
    for step in steps:
        composition.append("".join(embeddings[i: i + step]))
        i += step
    return composition

def micro_step(
    hidden: List[str]
):
    steps = []
    for _ in hidden:
        steps.extend([random.randint(0, VOCAB_SIZE) for _ in range(MAX_STEPS)])
    return steps

def get_targets(
    input_ids: List[int], 
    steps: List[int], 
    context_len: int
):
    ids = input_ids[context_len:]
    targets = []
    i = 0
    for step in steps:
        t = [STOP for _ in range(MAX_STEPS)]
        t[:step] = ids[i:i + step]
        targets.extend(t)
        i += step
    # EOS tokens
    targets.extend([STOP for _ in range(MAX_STEPS)])
    return targets

def model(
    compositions: List[str]
):
    return compositions

def calc_loss(
    input_ids: List[int], 
    steps: List[int], 
    context_len: int
) -> List[int]:
    # Embedding
    embeddings = emb(input_ids=input_ids)
    # Composition
    compositions = com(embeddings=embeddings, steps=steps, context_len=context_len)
    print("Composition:", compositions)
    # Base model (skip for now)
    hidden = model(compositions)
    # Micro-step
    micro_steps = micro_step(hidden=hidden[context_len - 1:])
    print("Micro steps:", micro_steps)
    # Create targets
    targets = get_targets(input_ids=input_ids, steps=steps, context_len=context_len)
    print("Targets:", targets)
    # Loss
    loss = sum([x != y for x, y in zip(micro_steps, targets) if y != STOP])
    return loss

print("Loss:", calc_loss(
    input_ids=input_ids, 
    steps=steps, 
    context_len=context_len
))
