from typing import Optional

import lightning as L
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from m2d.data.segmentor import Segmentor


class M2DDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_hf_dataset(
        cls, 
        dataset_name: Optional[str] = None, 
        save_path: Optional[str] = None, 
        model: Optional[AutoModelForCausalLM] = None, 
        tokenizer: Optional[AutoTokenizer] = None, 
        inst_name: str = "instruction", 
        resp_name: str = "response"
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

            def process_batch(batch):
                return segmentor.segment(
                    instructions=batch[inst_name], 
                    responses=batch[resp_name]
                )

            processed_data = raw_dataset.map(process_batch, batch_size=4, batched=True)
        else:
            print(f"Load {cls.__name__} dataset from {save_path}.")
        

if __name__ == "__main__":
    data = M2DDataModule.from_hf_dataset(
        dataset_name="imone/OpenOrca_FLAN", 
        save_path="./local/processed_data"
    )
