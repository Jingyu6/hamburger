from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from m2d.data.strategies import STRATEGIES


class Segmentor:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        strategy: str, 
        sliding_window: Optional[int] = None
    ):
        self.model: AutoModelForCausalLM = model
        self.tokenizer: AutoTokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_steps = 4
        self.step_strategy = STRATEGIES[strategy]
        self.sliding_window = sliding_window

    def _calc_steps(self, entropy: List[float]):
        return self.step_strategy(entropy=entropy, max_steps=self.max_steps)

    @torch.inference_mode
    def segment(
        self, 
        instructions: List[str], 
        responses: List[str], 
        system_message: Optional[str] = None
    ):
        assert len(instructions) == len(responses)

        sm = []
        if system_message is not None:
            sm = [{"role": "system", "content": system_message}]

        inst_lens = []
        for inst in instructions:
            conversation = sm + [{"role": "user", "content": inst}]
            inst_ids = self.tokenizer.apply_chat_template(
                conversation, 
                # this is used to make the model not output the gen prompt
                add_generation_prompt=True, 
                return_dict=True
            )["input_ids"]

            inst_len = len(inst_ids)
            inst_lens.append(inst_len)
        
        # batch tokenization
        inputs = self.tokenizer.apply_chat_template(
            [
                sm + [{"role": "user", "content": inst}, 
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
        logits = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            use_cache=False, 
            sliding_window=self.sliding_window
        ).logits

        results = {
            "input_ids": [], 
            "steps": [],  
            "inst_lens": inst_lens
        }

        for inst_len, mask, inputs, logit in zip(
            inst_lens, 
            attention_mask.cpu().tolist(), 
            input_ids.cpu().tolist(), 
            logits
        ):
            token_cnt = sum(mask)
            # logits_{i - 1} means the prediction of token_{i}
            probs = torch.nn.functional.softmax(logit[inst_len - 1:token_cnt - 1], dim=-1)
            log_probs = torch.nn.functional.log_softmax(logit[inst_len - 1:token_cnt - 1], dim=-1)
            entropy = (-torch.sum(probs * log_probs, dim=-1)).cpu().tolist()
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
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    segmentor = Segmentor(
        model=model, 
        tokenizer=tokenizer, 
        strategy="sliding", 
        sliding_window=4
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
