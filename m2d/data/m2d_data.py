import os
from typing import Any, Dict, List, Optional

import lightning as L
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from m2d.data.segmentor import Segmentor


class M2DDataModule(L.LightningDataModule):
    def __init__(
        self, 
        save_path: str, 
        test_ratio: float = 0.2, 
        batch_size: int = 8, 
    ):
        super().__init__()
        self.data = load_from_disk(save_path)
        self.data = self.data.train_test_split(
            test_size=int(len(self.data) * test_ratio)
        )
        self.batch_size = batch_size

    @classmethod
    def from_hf_dataset(
        cls, 
        dataset_name: Optional[str] = None, 
        save_path: Optional[str] = None, 
        model: Optional[AutoModelForCausalLM] = None, 
        tokenizer: Optional[AutoTokenizer] = None, 
        inst_name: str = "instruction", 
        resp_name: str = "response", 
        max_num_samples: int = -1, 
        empty_cache_every: int = 1024, 
        max_len: Optional[int] = None, 
        **kwargs
    ):
        assert save_path is not None

        if os.path.exists(save_path):
            try:
                data = cls(save_path, **kwargs)
                return data
            except:
                print("Failed to load existing data. Try creating new data. ")

        if dataset_name is not None:
            assert model is not None
            assert tokenizer is not None

            print(f"Create new {cls.__name__} dataset from {dataset_name}.")

            segmentor = Segmentor(model=model, tokenizer=tokenizer)

            raw_dataset = load_dataset(
                dataset_name, 
                split="train"
            )

            if max_num_samples > 0:
                raw_dataset = raw_dataset.select(range(max_num_samples))

            def process_batch(batch, indices):
                # avoids memory leak
                if any([x % empty_cache_every == 0 for x in indices]):
                    torch.cuda.empty_cache()

                return segmentor.segment(
                    instructions=batch[inst_name], 
                    responses=batch[resp_name]
                )

            processed_data = raw_dataset.map(
                process_batch, 
                batch_size=4, # make sure we dont get OOM
                batched=True, 
                num_proc=1, # need to use 1 since we only have 1 model
                remove_columns=raw_dataset.column_names, 
                with_indices=True
            )
        
            if max_len is not None:
                print(f"Saving the original unfiltered data.")

            processed_data.save_to_disk(
                save_path + "_unfiltered" if max_len is not None else "", 
                max_shard_size="1GB"
            )

            if max_len is not None:
                print(f"Filter data which are longer than {max_len} tokens.")
                original_size = len(processed_data)
                processed_data = processed_data.filter(lambda x: len(x["input_ids"]) <= max_len)
                new_size = len(processed_data)
                print(f"Filtered {original_size - new_size} samples.")
                processed_data.save_to_disk(save_path, max_shard_size="1GB")

        return cls(save_path, **kwargs)

    @staticmethod
    def _collate_fn(
        batch: List[Dict[str, Any]]
    ):
        input_ids = [torch.LongTensor(sample["input_ids"]) for sample in batch]
        seq_lens = [len(ids) for ids in input_ids]
        inst_lens = [sample["inst_lens"] for sample in batch]
        steps = [sample["steps"] for sample in batch]
        
        # flatten ids
        input_ids = torch.concat(input_ids, dim=0)

        return {
            "input_ids": input_ids, 
            "seq_lens": seq_lens, 
            "inst_lens": inst_lens, 
            "steps": steps
        }

    def train_dataloader(self):
        return DataLoader(
            self.data["train"], 
            batch_size=self.batch_size, 
            collate_fn=M2DDataModule._collate_fn, 
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data["test"], 
            batch_size=self.batch_size, 
            collate_fn=M2DDataModule._collate_fn, 
        )

    def get_speedup_estimate(self):
        micro_steps = 0
        macro_steps = 0
        for s in tqdm(self.data['train']):
            steps = s["steps"]
            micro_steps += sum(steps)
            macro_steps += len(steps)
        print(f"Approximate decoding speedup: {(micro_steps / macro_steps) * 100:.2f}%")

if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # first time process the data
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

    data = M2DDataModule.from_hf_dataset(
        dataset_name="imone/OpenOrca_FLAN", 
        save_path="./local/processed_openorca", 
        model=model, 
        tokenizer=tokenizer, 
        max_num_samples=840000
    )

    print(data)
