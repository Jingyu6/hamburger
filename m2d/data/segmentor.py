from types import MethodType
from typing import List

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM


def _prefill(
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


class Segmentor:
    def __init__(
        self, 
        model, 
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model.prefill = MethodType(_prefill, self.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_steps = 4
        self.ratio_threshold = 0.3

    def _calc_steps(
        self, 
        entropy: List[float]
    ):
        steps = []
        last_cnt = 0
        last_max = -1
        for e in entropy:
            if e > self.ratio_threshold * last_max or last_cnt >= self.max_steps:
                # start new segment
                steps.append(last_cnt)
                last_cnt = 1
                last_max = e
            else:
                # keep old segment
                last_cnt += 1
        if last_cnt > 0:
            steps.append(last_cnt)
        steps = steps[1:]
        assert sum(steps) == len(entropy)
        return steps

    @torch.inference_mode
    def segment(
        self, 
        instructions: List[str], 
        responses: List[str], 
    ):
        assert len(instructions) == len(responses)

        inst_lens = []
        for inst in instructions:
            conversation = [{"role": "user", "content": inst}]
            inst_ids = self.tokenizer.apply_chat_template(
                conversation, 
                return_dict=True
            )["input_ids"]

            inst_len = len(inst_ids)
            inst_lens.append(inst_len)
        
        # batch tokenization
        inputs = self.tokenizer.apply_chat_template(
            [
                [{"role": "user", "content": inst}, 
                 {"role": "assistant", "content": resp}]
                for inst, resp in zip(instructions, responses)
            ], 
            return_tensors='pt', 
            return_dict=True, 
            padding=True, 
        )

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # TODO: this part still takes too much mem now
        logits = self.model.prefill(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            use_cache=False
        )

        # TODO: we could optimize this memory later
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_entropy_list = (-torch.sum(probs * log_probs, dim=-1)).cpu().tolist()

        results = {
            "input_ids": [], 
            "steps": [],  
            "inst_lens": inst_lens
        }

        for inst_len, mask, inputs, entropy in zip(
            inst_lens, 
            attention_mask.cpu().tolist(), 
            input_ids.cpu().tolist(), 
            token_entropy_list
        ):
            token_cnt = sum(mask)
            entropy = entropy[inst_len:token_cnt]
            inputs = inputs[:token_cnt]
            steps = self._calc_steps(entropy)
            
            results["input_ids"].append(inputs)
            results["steps"].append(steps)
        
        return results


if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", 
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    segmentor = Segmentor(
        model=model, 
        tokenizer=tokenizer
    )

    print(segmentor.segment(
        instructions=[
            "Could you give me an example of json object?", 
            "Who is Magnus Carlsen?", 
            "What is the capital of China?"
        ], 
        responses=[
            '{\n    "name": "Tony", \n    "age": 16, \n    "married": false, \n    "job": "Student"\n}', 
            "He is the chess world champion.", 
            "Beijing is the capital of China."
        ]
    ))
