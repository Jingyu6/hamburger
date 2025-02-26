import os
from typing import Any, Callable, Dict, List, Optional

import lightning as L
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from m2d.data.segmentor import Segmentor


class M2DDataModule(L.LightningDataModule):
    def __init__(
        self, 
        save_path: List[str] | str, 
        test_ratio: float = 0.2, 
        batch_size: int = 8, 
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if isinstance(save_path, str):
            self.data = load_from_disk(save_path).shuffle()
        else:
            self.data = concatenate_datasets([load_from_disk(path) for path in save_path]).shuffle()
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
        filter_fn: Optional[Callable] = None, 
        map_fn: Optional[Callable] = None, 
        save_raw: bool = False, 
        subset: Optional[str] = None, 
        batch_size: int = 4, 
        system_message: Optional[str] = None, 
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
                name=subset,  
                split="train"
            ).shuffle() # Randomize length distribution

            if map_fn is not None:
                raw_dataset = raw_dataset.map(map_fn, num_proc=8)

            if filter_fn is not None:
                raw_dataset = raw_dataset.filter(filter_fn)

            if max_num_samples > 0:
                raw_dataset = raw_dataset.select(range(max_num_samples))

            def process_batch(batch, indices):
                # avoids memory leak
                if any([x % empty_cache_every == 0 for x in indices]):
                    torch.cuda.empty_cache()

                return segmentor.segment(
                    instructions=batch[inst_name], 
                    responses=batch[resp_name], 
                    system_message=system_message
                )

            processed_data = raw_dataset.map(
                process_batch, 
                batch_size=batch_size, # make sure we dont get OOM
                batched=True, 
                num_proc=1, # need to use 1 since we only have 1 model
                remove_columns=raw_dataset.column_names, 
                with_indices=True
            )
        
            if max_len is not None:
                print(f"Saving the original unfiltered data.")

            if max_len is None or save_raw:
                processed_data.save_to_disk(
                    save_path + ("_unfiltered" if max_len is not None else ""), 
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

    def get_output_token_count(self):
        token_cnt = 0
        for s in tqdm(self.data['train']):
            token_cnt += (len(s["input_ids"]) - s["inst_lens"])
        print(f"Total number of training tokens: {token_cnt}.")


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
        attn_implementation="flash_attention_2",
        device_map="auto"
    )

    # datasets
    data = M2DDataModule.from_hf_dataset(
        dataset_name="imone/OpenOrca_FLAN", 
        save_path="./local/openorca", 
        model=model, 
        tokenizer=tokenizer, 
        filter_fn=lambda sample: sample["condition"] == "GPT4", 
        max_len=4096
    )

    data = M2DDataModule.from_hf_dataset(
        dataset_name="nampdn-ai/tiny-codes", 
        save_path="./local/tinycode", 
        model=model, 
        tokenizer=tokenizer, 
        inst_name="prompt", 
        resp_name="response", 
        max_len=8192
    )

    data = M2DDataModule.from_hf_dataset(
        dataset_name="teknium/openhermes", 
        save_path="./local/openhermes", 
        model=model, 
        tokenizer=tokenizer, 
        inst_name="instruction", 
        resp_name="output", 
        max_len=8192
    )

    data = M2DDataModule.from_hf_dataset(
        dataset_name="meta-math/MetaMathQA", 
        save_path="./local/metamathqa", 
        model=model, 
        tokenizer=tokenizer, 
        inst_name="query", 
        resp_name="response", 
        max_len=8192
    )

    data = M2DDataModule.from_hf_dataset(
        dataset_name="garage-bAInd/Open-Platypus", 
        save_path="./local/openplatypus", 
        model=model, 
        tokenizer=tokenizer, 
        inst_name="instruction", 
        resp_name="output", 
        max_len=8192
    )
    
    data = M2DDataModule.from_hf_dataset(
        dataset_name="openbmb/UltraInteract_sft", 
        save_path="./local/ultrainteract", 
        model=model, 
        tokenizer=tokenizer, 
        inst_name="instruction", 
        resp_name="response", 
        max_len=8192
    )

    data = M2DDataModule.from_hf_dataset(
        dataset_name="ise-uiuc/Magicoder-Evol-Instruct-110K", 
        save_path="./local/magicoder", 
        model=model, 
        tokenizer=tokenizer, 
        inst_name="instruction", 
        resp_name="response", 
        max_len=8192
    )

    data = M2DDataModule.from_hf_dataset(
        dataset_name="Vezora/Tested-143k-Python-Alpaca", 
        save_path="./local/pythonalpaca", 
        model=model, 
        tokenizer=tokenizer, 
        inst_name="instruction", 
        resp_name="output", 
        max_len=8192
    )

    def _parse_message(example):
        return {"problem": example["reannotated_messages"][0]["content"]}

    data = M2DDataModule.from_hf_dataset(
        dataset_name="ServiceNow-AI/R1-Distill-SFT", 
        subset="v1", 
        save_path="./local/r1distill", 
        model=model, 
        tokenizer=tokenizer, 
        inst_name="problem", 
        resp_name="reannotated_assistant_content", 
        map_fn=_parse_message, 
        filter_fn=lambda x: (len(x["problem"]) + len(x["reannotated_assistant_content"])) <= 4096, 
        system_message="You're a helpful AI assistant, and think carefully before giving your final answer. Wrap your reasoning process in <think> and </think>. ", 
        batch_size=2 # since its longer
    )

    def _parse_message(example):
        return {
            "instruction": example["conversations"][0],
            "response": example["conversations"][1]
        }

    data = M2DDataModule.from_hf_dataset(
        dataset_name="GAIR/lima", 
        save_path="./local/lima", 
        model=model, 
        tokenizer=tokenizer, 
        map_fn=_parse_message, 
        batch_size=2 # since its longer
    )
    
    def _parse_message(example):
        return {
            "instruction": example["messages"][0]["content"],
            "response": example["messages"][1]["content"]
        }

    data = M2DDataModule.from_hf_dataset(
        dataset_name="allenai/tulu-v2-sft-mixture", 
        save_path="./local/tulu", 
        model=model, 
        tokenizer=tokenizer, 
        map_fn=_parse_message
    )

    def _parse_message(example):
        return {
            "instruction": example["conversation"][0]["content"],
            "response": example["conversation"][1]["content"]
        }

    data = M2DDataModule.from_hf_dataset(
        dataset_name="lmsys/lmsys-chat-1m", 
        save_path="./local/lmsys", 
        model=model, 
        tokenizer=tokenizer, 
        map_fn=_parse_message
    )
    
    data = M2DDataModule.from_hf_dataset(
        dataset_name="open-r1/OpenR1-Math-220k", 
        save_path="./local/openr1math", 
        model=model, 
        tokenizer=tokenizer, 
        inst_name="problem", 
        resp_name="solution", 
        max_len=8192, 
    )

    data = M2DDataModule.from_hf_dataset(
        dataset_name="PrimeIntellect/SYNTHETIC-1", 
        save_path="./local/synthetic1", 
        model=model, 
        tokenizer=tokenizer, 
        inst_name="prompt", 
        resp_name="llm_response", 
        system_message="You're a helpful AI assistant, and think carefully before giving your final answer. Wrap your reasoning process in <think> and </think>. ", 
        filter_fn=lambda sample: (
            sample.get("score", None) == 1 and \
            (len(sample["prompt"]) + len(sample["llm_response"])) <= 8192
        ),  
        max_len=8192   
    )

    data = M2DDataModule.from_hf_dataset(
        dataset_name="facebook/natural_reasoning", 
        save_path="./local/naturalreasoning", 
        model=model, 
        tokenizer=tokenizer, 
        inst_name="question", 
        resp_name="output", 
        map_fn=lambda sample: {"output": sample["responses"][0]["response"]}, 
        max_len=8192   
    )
    
    data = M2DDataModule.from_hf_dataset(
        dataset_name="argilla/ifeval-like-data", 
        save_path="./local/ifevallike", 
        model=model, 
        tokenizer=tokenizer, 
        inst_name="prompt", 
        resp_name="response", 
        subset="filtered"
    )

    def _parse_message(example):
        return {
            "instruction": example["conversations"][0]["value"],
            "response": example["conversations"][1]["value"]
        }

    data = M2DDataModule.from_hf_dataset(
        dataset_name="open-thoughts/OpenThoughts-114k", 
        save_path="./local/openthoughts", 
        model=model, 
        tokenizer=tokenizer, 
        map_fn=_parse_message, 
        inst_name="instruction", 
        resp_name="response", 
        batch_size=1 # since its longer
    )
