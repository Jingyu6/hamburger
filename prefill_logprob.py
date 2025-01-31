from types import MethodType

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from m2d.plot_utils import plot_entropies

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True)
model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct", 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16, 
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

def prefill(
    self: LlamaForCausalLM,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    num_logits_to_keep=0,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [torch.nn.functional.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return logits

model.prefill = MethodType(prefill, model)

logits = model.prefill(
    input_ids=input_ids,
    attention_mask=attention_mask
)[0]

token_str_list = tokenizer.batch_decode(input_ids[0])

probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
token_entropy_list = (-torch.sum(probs * log_probs, dim=-1)).cpu().tolist()

plot_entropies(
    token_str_list=token_str_list[1:], 
    token_entropy_list=token_entropy_list[:-1]
)
