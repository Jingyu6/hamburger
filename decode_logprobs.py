import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from plot_utils import plot_entropies

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True)
model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct", 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    attn_implementation="eager"
)

content = """
Could you help me generate a json example file? Do not explain. 
"""

conversation = [{
    "role": "user", 
    "content": content
}]
inputs = tokenizer.apply_chat_template(
    conversation, 
    return_tensors="pt", 
    return_dict=True
)

input_ids = inputs["input_ids"].cuda()
attention_mask = inputs["attention_mask"].cuda()

outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,  
    max_new_tokens=64, 
    use_cache=True, 
    output_logits=True, 
    return_dict_in_generate=True, 
    do_sample=False, 
    temperature=None, 
    top_p=None
)

context_len = input_ids.shape[-1]
token_ids = outputs.sequences.view(-1)
token_str_list = tokenizer.batch_decode(token_ids)[context_len:]

logits = torch.concatenate(outputs.logits, dim=0)
probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
token_entropy_list = (-torch.sum(probs * log_probs, dim=-1)).cpu().tolist()

print(token_str_list)
print(token_entropy_list)
plot_entropies(
    token_str_list=token_str_list, 
    token_entropy_list=token_entropy_list
)
