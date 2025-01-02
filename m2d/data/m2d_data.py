from typing import Optional

import lightning as L
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from m2d.data.segmentor import Segmentor


class M2DDataModule(L.LightningDataModule):
    def __init__(
        self, 
        save_path: str, 
        test_ratio: float = 0.2
    ):
        super().__init__()
        self.data = load_from_disk(save_path)
        self.data = self.data.train_test_split(
            test_size=int(len(self.data) * test_ratio)
        )

    @classmethod
    def from_hf_dataset(
        cls, 
        dataset_name: Optional[str] = None, 
        save_path: Optional[str] = None, 
        model: Optional[AutoModelForCausalLM] = None, 
        tokenizer: Optional[AutoTokenizer] = None, 
        inst_name: str = "instruction", 
        resp_name: str = "response", 
        max_num_samples: int = -1
    ):
        assert dataset_name is not None or save_path is not None

        if dataset_name is not None:
            assert save_path is not None
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

            def process_batch(batch):
                return segmentor.segment(
                    instructions=batch[inst_name], 
                    responses=batch[resp_name]
                )

            processed_data = raw_dataset.map(
                process_batch, 
                batch_size=4, 
                batched=True, 
                num_proc=1, # need to use 1 since we only have 1 model
                remove_columns=raw_dataset.column_names
            )
        
            processed_data.save_to_disk(
                save_path, 
                max_shard_size="1GB"
            )

        return cls(save_path)


if __name__ == "__main__":
    """
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
        max_num_samples=4096
    )
    """

    # afterwards, load the data
    data = M2DDataModule.from_hf_dataset(
        save_path="./local/processed_openorca"
    )

    print(data.data)
