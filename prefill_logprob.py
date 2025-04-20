import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from m2d.data.strategies import STRATEGIES
from m2d.plot_utils import plot_values

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True)
model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
    device_map="auto", 
)

prompt = """Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?"""
response = """Maila read 12 x 2 = <<12*2=24>>24 pages today. So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday. There are 120 - 36 = <<120-36=84>>84 pages left to be read. Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages. #### 42"""

conversation = [
    {"role": "user", "content": prompt}, 
    {"role": "assistant", "content": response}
]
inputs = tokenizer.apply_chat_template(
    conversation, 
    return_tensors="pt", 
    return_dict=True
)

input_ids = inputs["input_ids"].cuda()
attention_mask = inputs["attention_mask"].cuda()

sliding_window = int(input("Sliding window size: "))
while True:
    mode = input("Visualize mode [entropy, prob]: ")
    if mode in ["entropy", "prob"]:
        break
while True:
    strategy = input("Segmentation strategy: ")
    if strategy in list(STRATEGIES.keys()) + ['none']:
        break

output: CausalLMOutputWithPast = model.forward(
    input_ids=input_ids,
    attention_mask=attention_mask,
    sliding_window=sliding_window if sliding_window > 0 else None, 
    return_dict=True
)

token_str_list = tokenizer.batch_decode(input_ids[0])

logits = output.logits[0]

if mode == "entropy":
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    value_list = (-torch.sum(probs * log_probs, dim=-1)).cpu().tolist()[:-1]
elif mode == "prob":
    # N, V
    probs = torch.nn.functional.softmax(logits[:-1], dim=-1).cpu()
    ids = input_ids[0][1:].cpu()
    value_list = probs[torch.arange(len(ids)), ids].tolist()
else:
    raise ValueError

plot_values(
    token_str_list=token_str_list[1:], 
    value_list=value_list, 
    save_path=f'./local/{mode}_sw={sliding_window}.png', 
    steps=STRATEGIES[strategy](value_list, 4) if strategy != 'none' else None
)
