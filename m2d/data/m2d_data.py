import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import lightning as L
import torch
from datasets import (Dataset, concatenate_datasets, load_dataset,
                      load_from_disk)
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
                    replicate_cnt = int(parsed_save_path[-1])
                ds = load_from_disk(path)
                data_list.extend([ds] * replicate_cnt)
                self.data_summary[path] = len(ds) * replicate_cnt

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

        raw_dataset = load_dataset(
            dataset_name,
            name=subset,  
            split=split
        ).shuffle() # Randomize length distribution

        if map_fn is not None:
            raw_dataset = raw_dataset.map(map_fn, num_proc=8)

        if filter_fn is not None:
            raw_dataset = raw_dataset.filter(filter_fn)

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
            strategy=strategy
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
            collate_fn=M2DDataModule._collate_fn, 
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            collate_fn=M2DDataModule._collate_fn, 
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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # datasets
    data = M2DDataModule.from_hf_dataset(
        dataset_name="imone/OpenOrca_FLAN", 
        save_path="./local/openorca", 
        filter_fn=lambda sample: sample["condition"] == "GPT4", 
        max_len=4096, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="nampdn-ai/tiny-codes", 
        save_path="./local/tinycodepython", 
        inst_name="prompt", 
        resp_name="response", 
        filter_fn=lambda sample: sample["programming_language"] == "Python", 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="teknium/openhermes", 
        save_path="./local/openhermes", 
        inst_name="instruction", 
        resp_name="output", 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="meta-math/MetaMathQA", 
        save_path="./local/metamathqa", 
        inst_name="query", 
        resp_name="response", 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="garage-bAInd/Open-Platypus", 
        save_path="./local/openplatypus", 
        inst_name="instruction", 
        resp_name="output", 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="openbmb/UltraInteract_sft", 
        save_path="./local/ultrainteract", 
        inst_name="instruction", 
        resp_name="response", 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="ise-uiuc/Magicoder-Evol-Instruct-110K", 
        save_path="./local/magicoder", 
        inst_name="instruction", 
        resp_name="response", 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="Vezora/Tested-143k-Python-Alpaca", 
        save_path="./local/pythonalpaca", 
        inst_name="instruction", 
        resp_name="output", 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    def _parse_message(example):
        return {"problem": example["reannotated_messages"][0]["content"]}
    data = M2DDataModule.from_hf_dataset(
        dataset_name="ServiceNow-AI/R1-Distill-SFT", 
        subset="v1", 
        save_path="./local/r1distill", 
        inst_name="problem", 
        resp_name="reannotated_assistant_content", 
        map_fn=_parse_message, 
        filter_fn=lambda x: (len(x["problem"]) + len(x["reannotated_assistant_content"])) <= 4096, 
        system_message="You're a helpful AI assistant, and think carefully before giving your final answer. Wrap your reasoning process in <think> and </think>. ", 
        batch_size=2, # since its longer
        strategy="decreasing_v2", 
    )
    def _parse_message(example):
        return {
            "instruction": example["conversations"][0],
            "response": example["conversations"][1]
        }
    data = M2DDataModule.from_hf_dataset(
        dataset_name="GAIR/lima", 
        save_path="./local/lima", 
        map_fn=_parse_message, 
        batch_size=2, # since its longer 
        strategy="decreasing_v2", 
    )
    def _parse_message(example):
        return {
            "instruction": example["messages"][0]["content"],
            "response": example["messages"][1]["content"]
        }
    data = M2DDataModule.from_hf_dataset(
        dataset_name="allenai/tulu-v2-sft-mixture", 
        save_path="./local/tulu", 
        map_fn=_parse_message, 
        strategy="decreasing_v2", 
    )
    def _parse_message(example):
        return {
            "instruction": example["conversation"][0]["content"],
            "response": example["conversation"][1]["content"]
        }
    data = M2DDataModule.from_hf_dataset(
        dataset_name="lmsys/lmsys-chat-1m", 
        save_path="./local/lmsys", 
        map_fn=_parse_message, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="open-r1/OpenR1-Math-220k", 
        save_path="./local/openr1math", 
        inst_name="problem", 
        resp_name="solution", 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="PrimeIntellect/SYNTHETIC-1", 
        save_path="./local/synthetic1",  
        inst_name="prompt", 
        resp_name="llm_response", 
        system_message="You're a helpful AI assistant, and think carefully before giving your final answer. Wrap your reasoning process in <think> and </think>. ", 
        filter_fn=lambda sample: (
            sample.get("score", None) == 1 and \
            (len(sample["prompt"]) + len(sample["llm_response"])) <= 8192
        ),  
        max_len=8192, 
        strategy="decreasing_v2",   
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="facebook/natural_reasoning", 
        save_path="./local/naturalreasoning", 
        inst_name="question", 
        resp_name="output", 
        map_fn=lambda sample: {"output": sample["responses"][0]["response"]}, 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="argilla/ifeval-like-data", 
        save_path="./local/ifevallike", 
        inst_name="prompt", 
        resp_name="response", 
        subset="filtered", 
        strategy="decreasing_v2", 
    )
    def _parse_message(example):
        return {
            "instruction": example["conversations"][0]["value"],
            "response": example["conversations"][1]["value"]
        }
    data = M2DDataModule.from_hf_dataset(
        dataset_name="open-r1/OpenThoughts-114k-math", 
        save_path="./local/openthoughts", 
        map_fn=_parse_message, 
        inst_name="problem", 
        resp_name="solution", 
        batch_size=1, # since its longer
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    def _parse_message(example):
        return {
            "prompt": "Write python code to solve the following coding question:\n{question}\n".format(question=example["question"]),
            "response": "```python\n{code}\n```".format(code=example["solutions"][0])
        }
    data = M2DDataModule.from_hf_dataset(
        dataset_name="codeparrot/apps", 
        save_path="./local/apps", 
        inst_name="prompt", 
        resp_name="response", 
        map_fn=_parse_message, 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    def _parse_message(example):
        return {
            "instruction": example["conversations"][0]["value"],
            "response": example["conversations"][1]["value"]
        }
    data = M2DDataModule.from_hf_dataset(
        dataset_name="BAAI/Infinity-Instruct", 
        save_path="./local/infinityinstruct", 
        map_fn=_parse_message, 
        filter_fn=lambda sample: len(sample["instruction"]) + len(sample["response"]) < (8192 * 4), 
        max_len=8192, 
        subset="Gen", 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="TIGER-Lab/MathInstruct", 
        save_path="./local/mathinstruct", 
        inst_name="instruction", 
        resp_name="output", 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="PawanKrd/math-gpt-4o-200k", 
        save_path="./local/mathgpt", 
        inst_name="prompt", 
        resp_name="response", 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="TIGER-Lab/MATH-plus", 
        save_path="./local/mathplus", 
        inst_name="instruction", 
        resp_name="output", 
        max_len=8192, 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="cognitivecomputations/OpenCoder-LLM_opc-sft-stage1-DolphinLabeled", 
        save_path="./local/opencoder", 
        inst_name="instruction", 
        resp_name="output", 
        max_len=8192, 
        subset="filtered_infinity_instruct", 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="OpenCoder-LLM/opc-sft-stage2", 
        save_path="./local/opencoder2", 
        inst_name="instruction", 
        resp_name="output", 
        max_len=8192, 
        subset="educational_instruct", 
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="openai/gsm8k", 
        save_path="./local/gsm8k", 
        inst_name="question", 
        resp_name="answer", 
        subset="main", 
        split="train",  
        strategy="decreasing_v2", 
    )
    data = M2DDataModule.from_hf_dataset(
        dataset_name="AI-MO/NuminaMath-CoT", 
        save_path="./local/mathcot", 
        inst_name="problem", 
        resp_name="solution", 
        strategy="decreasing_v2", 
    )
    def _parse_message(example): 
        raw_answer: str = example["response"]
        raw_question = example["question"]
        return {
            "cot_problem": raw_question + " Reason the question and think step by step. Please end with \"The final answer is [answer]\" where [answer] is your solution. ", 
            "cot_answer": raw_answer.replace("Therefore, the final answer is", "The final answer is")
        }
    data = M2DDataModule.from_hf_dataset(
        dataset_name="ankner/gsm8k-CoT", 
        save_path="./local/gsm8kcot", 
        inst_name="cot_problem", 
        resp_name="cot_answer", 
        map_fn=_parse_message,  
        strategy="decreasing_v2", 
    )
