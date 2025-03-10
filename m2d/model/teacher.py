from typing import List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class DistillTeacher(nn.Module):
    def __init__(self, teacher_model_name: str):
        super().__init__()
        self.teacher: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            teacher_model_name, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )

        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.teacher.eval()

    def get_logits(
        self, 
        input_ids: torch.LongTensor, 
        seq_lens: List[int], 
        inst_lens: List[int], 
        steps: List[List[int]] 
    ):
        """
            teacher logits:
                t0, t1, t2, t3, t4, t5
            model logits (t' is stop token):
                __, __, t2, t', t3, t4, t', t5, t5'
            align:
                * take off prompt:
                t2, t3, t4, t5
                * add padding
                t2, t', t3, t4, t', t5, t'
        """
        # calculate pos_ids
        position_ids = torch.concat([torch.LongTensor(range(sl)) for sl in seq_lens]).to(input_ids.device)

        # get all logits
        all_logits: CausalLMOutputWithPast = self.teacher.forward(
            input_ids=input_ids[None, ], 
            position_ids=position_ids[None, ], 
            use_cache=False, 
            return_dict=True, 
        ).logits[0] # flattened logits

        # extract only useful logits
        logits = []
        offset = 0
        for seq_len, inst_len, step in zip(seq_lens, inst_lens, steps):
            logits.extend(
                all_logits[offset + inst_len:offset + seq_len].split(
                    split_size=step, 
                    dim=0
                )
            )
            offset += seq_len

        # mask for taking off micro stop
        mask = []

        # add extra pads
        for i in range(len(logits)):
            mask.extend([1] * logits[i].shape[0] + [0])
            logits[i] = torch.concat(
                [logits[i], torch.ones(1, logits[i].shape[-1]).to(logits[i].device)], 
                dim=0
            )

        logits = torch.concat(logits, dim=0)
        mask = torch.BoolTensor(mask).to(logits.device)

        return logits, mask
