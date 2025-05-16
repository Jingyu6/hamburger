import argparse
import os
from typing import Any, Callable, Dict, List, Optional

import lightning as L
import torch
from datasets import (Dataset, concatenate_datasets, load_dataset,
                      load_from_disk)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from hamburger.data.segmentor import Segmentor


class HAMburgerDataModule(L.LightningDataModule):
    def __init__(
        self, 
        save_path: List[str] | str, 
        test_ratio: float = 0.2, 
        batch_size: int = 8, 
    ):
        super().__init__()
        self.save_hyperparameters()
        self.save_path = save_path
        self.test_ratio = test_ratio
        self.batch_size = batch_size

        if isinstance(self.save_path, str):
            data = load_from_disk(self.save_path)
            self.data_summary = {
                self.save_path: len(data)
            }
        else:
            assert isinstance(self.save_path, list)
            self.data_summary = {}
            data_list = []
            for p in self.save_path:
                parsed_save_path = p.split(":")
                path = parsed_save_path[0]
                replicate_cnt = 1
                if len(parsed_save_path) == 2:
                    replicate_cnt = float(parsed_save_path[-1])
                ds = load_from_disk(path)
                if replicate_cnt >= 1:
                    data_list.extend([ds] * int(replicate_cnt))
                    self.data_summary[path] = len(ds) * replicate_cnt
                else:
                    new_len = int(len(ds) * replicate_cnt)
                    data_list.append(ds.shuffle().take(new_len))
                    self.data_summary[path] = new_len

            data = concatenate_datasets(data_list)

        data = data.train_test_split(
            test_size=int(len(data) * self.test_ratio)
        )
        self.train_data = data["train"]
        self.test_data = data["test"]

    @classmethod
    def get_distilled_dataset(
        cls, 
        model_name: str, 
        dataset: Dataset, 
        inst_name: str, 
        save_path: str
    ):
        try:
            from vllm import EngineArgs, LLMEngine, SamplingParams
        except:
            raise ImportError("Please install vllm for faster distillation.")

        # greedy sampling
        sampling_params = SamplingParams(
            temperature=0.0, 
            n=1, 
            max_tokens=7168 # TODO: later consider changing it
        )
        # get the model
        engine = LLMEngine.from_engine_args(EngineArgs(
            model=model_name, 
            dtype="bfloat16"
        ))

        # get raw prompt
        prompts = dataset[inst_name]
        total_prompt_cnt = len(prompts)

        # generate
        request_id = 0
        responses = []
        while prompts or engine.has_unfinished_requests():
            if prompts:
                prompt = prompts.pop(0)
                engine.add_request(
                    str(request_id), 
                    prompt, 
                    sampling_params
                )
                request_id += 1

            request_outputs = engine.step()

            for request_output in request_outputs:
                if request_output.finished:
                    responses.append(request_output.outputs[0].text)

            if len(responses) % 200 == 0 and len(responses) > 0:
                print(f"Finished {len(responses)}/{total_prompt_cnt} generations.")

        # store back to disk
        dataset = dataset.remove_columns([n for n in dataset.column_names if n != inst_name])
        dataset = dataset.add_column("distilled_output", responses)
        dataset.save_to_disk(save_path, max_shard_size="1GB")

        return dataset

    @classmethod
    def from_hf_dataset(
        cls, 
        dataset_name: Optional[str] = None, 
        save_path: Optional[str] = None, 
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
        inst_name: str = "instruction", 
        resp_name: str = "response", 
        max_num_samples: int = -1, 
        empty_cache_every: int = 1024, 
        max_len: Optional[int] = None, 
        filter_fn: Optional[Callable] = None, 
        map_fn: Optional[Callable] = None, 
        save_raw: bool = False, 
        subset: Optional[str] = None, 
        split: str = "train", 
        batch_size: int = 4, 
        system_message: Optional[str] = None, 
        strategy: str = "small_group", 
        distill: bool = False, 
        distill_model_name: Optional[str] = None, 
        distill_save_path: Optional[str] = None, 
        data_files: Optional[str] = None, 
        sliding_window: Optional[int] = None, 
        **kwargs
    ):
        assert save_path is not None

        if os.path.exists(save_path):
            try:
                data = cls(save_path, **kwargs)
                return data
            except:
                print("Failed to load existing data. Try creating new data. ")

        assert dataset_name is not None
        print(f"Create new {cls.__name__} dataset from {dataset_name}.")

        if data_files is not None:
            data_files = list(data_files.split(','))
        else:
            data_files = None

        raw_dataset = load_dataset(
            dataset_name,
            name=subset,  
            split=split, 
            data_files=data_files
        ).shuffle() # Randomize length distribution

        if filter_fn is not None:
            raw_dataset = raw_dataset.filter(filter_fn)

        if map_fn is not None:
            raw_dataset = raw_dataset.map(map_fn, num_proc=8)

        if max_num_samples > 0:
            raw_dataset = raw_dataset.select(range(max_num_samples))

        if distill:
            distill_dataset = None
            if os.path.exists(distill_save_path):
                try:
                    print(f"Try reusing old distill dataset.")
                    distill_dataset = cls(distill_save_path, **kwargs)
                except:
                    print(f"Error finding old distill dataset.")
            
            if distill_dataset is None:
                print("Generate distilled dataset first")
                distill_dataset = cls.get_distilled_dataset(
                    model_name=distill_model_name if distill_model_name is not None else model_name, 
                    dataset=raw_dataset, 
                    inst_name=inst_name, 
                    save_path=distill_save_path
                )
            
            raw_dataset = distill_dataset
            resp_name = "distilled_output"

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            device_map="auto"
        )

        segmentor = Segmentor(
            model=model, 
            tokenizer=tokenizer, 
            strategy=strategy, 
            sliding_window=sliding_window
        )

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

        keep_columns = ["input_ids", "steps", "inst_lens"]
        processed_data = processed_data.remove_columns(
            [n for n in processed_data.column_names if n not in keep_columns]
        )

        if max_len is None or save_raw:
            print(f"Saving the original unfiltered data.")
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
            self.train_data, 
            batch_size=self.batch_size, 
            collate_fn=HAMburgerDataModule._collate_fn, 
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            collate_fn=HAMburgerDataModule._collate_fn, 
        )

    def get_speedup_estimate(self):
        micro_steps = 0
        macro_steps = 0
        for s in tqdm(self.train_data):
            steps = s["steps"]
            micro_steps += sum(steps)
            macro_steps += len(steps)
        print(f"Approximate decoding speedup: {(micro_steps / macro_steps) * 100:.2f}%")

    def get_output_token_count(self):
        token_cnt = 0
        for s in tqdm(self.train_data):
            token_cnt += (len(s["input_ids"]) - s["inst_lens"])
        print(f"Total number of training tokens: {token_cnt}.")

    def get_data_summary(self):
        print("Data summary: ")
        total_cnt = len(self.train_data)
        for data_name, data_len in self.data_summary.items():
            print(f"\tData percentage: {(data_len / total_cnt) * 100:.2f}%: {data_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Call the from_hf_dataset class method with command-line arguments."
    )
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name to use")
    parser.add_argument("--inst_name", type=str, default="instruction", help="Key name for instruction")
    parser.add_argument("--resp_name", type=str, default="response", help="Key name for response")
    parser.add_argument("--max_num_samples", type=int, default=-1, help="Maximum number of samples to process")
    parser.add_argument("--empty_cache_every", type=int, default=1024, help="Interval to empty cache")
    parser.add_argument("--max_len", type=int, default=None, help="Maximum length (if applicable)")
    parser.add_argument("--filter_fn", type=str, default=None, help="Python expression for filter function")
    parser.add_argument("--map_fn", type=str, default=None, help="Python expression for map function")
    parser.add_argument("--save_raw", action="store_true", help="Flag to save raw data")
    parser.add_argument("--subset", type=str, default=None, help="Subset of the dataset to use")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (e.g., train, test)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--system_message", type=str, default=None, help="Optional system message")
    parser.add_argument("--strategy", type=str, default="small_group", help="Strategy to use")
    parser.add_argument("--distill", action="store_true", help="Flag to enable distillation")
    parser.add_argument("--distill_model_name", type=str, default=None, help="Model name for distillation")
    parser.add_argument("--distill_save_path", type=str, default=None, help="Path to save the distilled model")
    parser.add_argument("--data_files", type=str, default=None, help="Path to download partial data")
    parser.add_argument("--sliding_window", type=int, default=None, help="Whether to apply sliding window. ")

    args = parser.parse_args()
    filter_fn = eval(args.filter_fn) if args.filter_fn else None
    map_fn = eval(args.map_fn) if args.map_fn else None

    HAMburgerDataModule.from_hf_dataset(
        dataset_name=args.dataset_name,
        save_path=args.save_path,
        model_name=args.model_name,
        inst_name=args.inst_name,
        resp_name=args.resp_name,
        max_num_samples=args.max_num_samples,
        empty_cache_every=args.empty_cache_every,
        max_len=args.max_len,
        filter_fn=filter_fn,
        map_fn=map_fn,
        save_raw=args.save_raw,
        subset=args.subset,
        split=args.split,
        batch_size=args.batch_size,
        system_message=args.system_message,
        strategy=args.strategy,
        distill=args.distill,
        distill_model_name=args.distill_model_name,
        distill_save_path=args.distill_save_path, 
        data_files=args.data_files, 
        sliding_window=args.sliding_window
    )
