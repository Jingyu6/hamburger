"""
    This script provide a common OpenAI compatible API for evaluation. 
    We currently support three kinds of models:
        1. Huggingface Models
        2. Byte Latent Transformers
        3. Our M2D Models
"""

import argparse
from typing import Optional

import litserve as ls
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from m2d.config import GenConfig
from m2d.model.llama import M2DLlama


class ModelLitAPI(ls.LitAPI):
    def __init__(
        self, 
        model_name: str, 
        model_type: str, 
        device: str, 
        confidence: Optional[float], 
        **kwargs
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.confidence = confidence
        
    def setup(self, device):
        if self.model_type == "hf":
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2",
                device_map=self.device, 
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            self.model = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
        elif self.model_type == "m2d":
            self.model: M2DLlama = M2DLlama.load_from_checkpoint(self.model_name).to(self.device)
        else:
            raise NotImplemented

    def predict(self, conversation, context):
        max_gen_len = context.get("max_completion_tokens", None)
        if max_gen_len is None:
            max_gen_len = context.get("max_tokens", 128)
        
        # filter non essential fields
        filtered_conversation = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in conversation
        ]

        # generate based on different model type
        if self.model_type == "hf":
            output = self.model(
                filtered_conversation, 
                max_new_tokens=max_gen_len
            )[0]["generated_text"][-1]["content"] # in case of multi-turn
        elif self.model_type == "m2d":
            gen_config = GenConfig(micro_step_confidence=self.confidence)
            gen_config.max_gen_len = max_gen_len
            output = self.model.generate(
                conversation=filtered_conversation, 
                config=gen_config
            )["output"]
            self.model.report.get_speedup()
        else:
            raise NotImplemented
        
        yield output


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--model_type', type=str, default="hf", choices=["hf", "m2d"])
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--confidence', type=float)

    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    api = ModelLitAPI(**vars(args))
    server = ls.LitServer(
        api, 
        devices=1, 
        spec=ls.OpenAISpec()
    )
    server.run(port=args.port)
