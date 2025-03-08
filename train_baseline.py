import os
from typing import Dict, List

import torch
from accelerate import PartialState
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LlamaForCausalLM, Trainer, TrainingArguments)

from m2d.config import M2DConfig
from m2d.data.m2d_data import M2DDataModule


def process_fn(example):
    input_ids = example["input_ids"]
    inst_len = example["inst_lens"]
    # offset will be performed by the trainer itself
    labels = [-100] * len(input_ids)
    labels[inst_len: len(input_ids)] = input_ids[inst_len: len(input_ids)]
    mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids, 
        "labels": labels, 
        "attention_mask": mask
    }


def get_dataset(data_path: str):
    if os.path.exists(data_path):
        train_ds = load_from_disk(data_path)
    else:
        config = M2DConfig.from_path("./local/train.yaml")
        config.print_config()

        data_module = M2DDataModule(
            save_path=config.dataset_names, 
            test_ratio=config.test_ratio, 
            batch_size=config.batch_size, 
        )

        data_module.setup()

        # train dataset
        train_ds = data_module.train_data.map(
            process_fn, 
            remove_columns=data_module.train_data.column_names
        )

        train_ds.save_to_disk(data_path, max_shard_size="1GB")

    return train_ds


class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            # Find the maximum sequence length in the batch for this key
            max_len = max(len(f[key]) for f in features)
            # Pad each sequence to the maximum length
            padded = []
            for f in features:
                seq = f[key]
                if key == "input_ids":
                    padding = [self.pad_token_id] * (max_len - len(seq))
                elif key == "attention_mask":
                    padding = [0] * (max_len - len(seq))
                elif key == "labels":
                    padding = [-100] * (max_len - len(seq))
                padded.append(seq + padding)
            batch[key] = torch.tensor(padded, dtype=torch.long)
        return batch


def main():
    # model
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    device_string = PartialState().process_index

    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        use_cache=False, 
        device_map={"": device_string}
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    config = TrainingArguments(
        output_dir="./local/baseline", 
        run_name="sft_llama_1b", 
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4, 
        learning_rate=1e-5, 
        num_train_epochs=1, 
        logging_steps=32, 
        save_strategy="no", 
        gradient_checkpointing_kwargs={'use_reentrant':False}, 
        bf16=True, 
        fsdp="full_shard", 
        report_to="none", 
        max_steps=16384
    )

    train_dataset = get_dataset("/data/data_persistent1/jingyu/m2d/baseline_data") 

    trainer = Trainer(
        model=model, 
        args=config, 
        train_dataset=train_dataset, 
        data_collator=CustomDataCollator(tokenizer=tokenizer)
    )

    trainer.train()

    # saving
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model("./local/baseline/finish")
    tokenizer.save_pretrained("./local/baseline/finish")

    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__": 
    """
        accelerate launch train_baseline.py
    """
    main()
