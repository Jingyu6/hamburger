#!/bin/bash
# Run M2DDataModule.from_hf_dataset for various datasets using the CLI

SAVE_DIR="/data/data_persistent1/jingyu/m2d/3b/"

python -m m2d.data.m2d_data \
  --dataset_name "imone/OpenOrca_FLAN" \
  --save_path $SAVE_DIR + "openorca" \
  --filter_fn 'lambda sample: sample["condition"] == "GPT4"' \
  --max_len 4096 \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "nampdn-ai/tiny-codes" \
  --save_path $SAVE_DIR + "tinycodepython" \
  --inst_name "prompt" \
  --resp_name "response" \
  --filter_fn 'lambda sample: sample["programming_language"] == "Python"' \
  --max_len 8192 \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "teknium/openhermes" \
  --save_path $SAVE_DIR + "openhermes" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len 8192 \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "meta-math/MetaMathQA" \
  --save_path $SAVE_DIR + "metamathqa" \
  --inst_name "query" \
  --resp_name "response" \
  --max_len 8192 \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "garage-bAInd/Open-Platypus" \
  --save_path $SAVE_DIR + "openplatypus" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len 8192 \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "openbmb/UltraInteract_sft" \
  --save_path $SAVE_DIR + "ultrainteract" \
  --inst_name "instruction" \
  --resp_name "response" \
  --max_len 8192 \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "ise-uiuc/Magicoder-Evol-Instruct-110K" \
  --save_path $SAVE_DIR + "magicoder" \
  --inst_name "instruction" \
  --resp_name "response" \
  --max_len 8192 \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "Vezora/Tested-143k-Python-Alpaca" \
  --save_path $SAVE_DIR + "pythonalpaca" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len 8192 \
  --strategy "decreasing_v2"

# ServiceNow-AI/R1-Distill-SFT with map_fn and filter_fn
python -m m2d.data.m2d_data \
  --dataset_name "ServiceNow-AI/R1-Distill-SFT" \
  --subset "v1" \
  --save_path $SAVE_DIR + "r1distill" \
  --inst_name "problem" \
  --resp_name "reannotated_assistant_content" \
  --map_fn 'lambda example: {"problem": example["reannotated_messages"][0]["content"]}' \
  --filter_fn 'lambda x: (len(x["problem"]) + len(x["reannotated_assistant_content"])) <= 4096' \
  --system_message "You're a helpful AI assistant, and think carefully before giving your final answer. Wrap your reasoning process in <think> and </think>. " \
  --batch_size 2 \
  --strategy "decreasing_v2"

# GAIR/lima with map_fn
python -m m2d.data.m2d_data \
  --dataset_name "GAIR/lima" \
  --save_path $SAVE_DIR + "lima" \
  --map_fn 'lambda example: {"instruction": example["conversations"][0], "response": example["conversations"][1]}' \
  --batch_size 2 \
  --strategy "decreasing_v2"

# allenai/tulu-v2-sft-mixture with map_fn
python -m m2d.data.m2d_data \
  --dataset_name "allenai/tulu-v2-sft-mixture" \
  --save_path $SAVE_DIR + "tulu" \
  --map_fn 'lambda example: {"instruction": example["messages"][0]["content"], "response": example["messages"][1]["content"]}' \
  --strategy "decreasing_v2"

# lmsys/lmsys-chat-1m with map_fn
python -m m2d.data.m2d_data \
  --dataset_name "lmsys/lmsys-chat-1m" \
  --save_path $SAVE_DIR + "lmsys" \
  --map_fn 'lambda example: {"instruction": example["conversation"][0]["content"], "response": example["conversation"][1]["content"]}' \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "open-r1/OpenR1-Math-220k" \
  --save_path $SAVE_DIR + "openr1math" \
  --inst_name "problem" \
  --resp_name "solution" \
  --max_len 8192 \
  --strategy "decreasing_v2"

# PrimeIntellect/SYNTHETIC-1 with filter_fn and system_message
python -m m2d.data.m2d_data \
  --dataset_name "PrimeIntellect/SYNTHETIC-1" \
  --save_path $SAVE_DIR + "synthetic1" \
  --inst_name "prompt" \
  --resp_name "llm_response" \
  --system_message "You're a helpful AI assistant, and think carefully before giving your final answer. Wrap your reasoning process in <think> and </think>. " \
  --filter_fn 'lambda sample: (sample.get("score", None) == 1 and (len(sample["prompt"]) + len(sample["llm_response"])) <= 8192)' \
  --max_len 8192 \
  --strategy "decreasing_v2"

# facebook/natural_reasoning with map_fn
python -m m2d.data.m2d_data \
  --dataset_name "facebook/natural_reasoning" \
  --save_path $SAVE_DIR + "naturalreasoning" \
  --inst_name "question" \
  --resp_name "output" \
  --map_fn 'lambda sample: {"output": sample["responses"][0]["response"]}' \
  --max_len 8192 \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "argilla/ifeval-like-data" \
  --save_path $SAVE_DIR + "ifevallike" \
  --inst_name "prompt" \
  --resp_name "response" \
  --subset "filtered" \
  --strategy "decreasing_v2"

# open-r1/OpenThoughts-114k-math with map_fn and custom batch_size
python -m m2d.data.m2d_data \
  --dataset_name "open-r1/OpenThoughts-114k-math" \
  --save_path $SAVE_DIR + "openthoughts" \
  --map_fn 'lambda example: {"instruction": example["conversations"][0]["value"], "response": example["conversations"][1]["value"]}' \
  --inst_name "problem" \
  --resp_name "solution" \
  --batch_size 1 \
  --max_len 8192 \
  --strategy "decreasing_v2"

# codeparrot/apps with map_fn that formats strings
python -m m2d.data.m2d_data \
  --dataset_name "codeparrot/apps" \
  --save_path $SAVE_DIR + "apps" \
  --inst_name "prompt" \
  --resp_name "response" \
  --map_fn 'lambda example: {"prompt": "Write python code to solve the following coding question:\n" + example["question"] + "\n", "response": "```python\n" + example["solutions"][0] + "\n```"}' \
  --max_len 8192 \
  --strategy "decreasing_v2"

# BAAI/Infinity-Instruct with map_fn and filter_fn
python -m m2d.data.m2d_data \
  --dataset_name "BAAI/Infinity-Instruct" \
  --save_path $SAVE_DIR + "infinityinstruct" \
  --map_fn 'lambda example: {"instruction": example["conversations"][0]["value"], "response": example["conversations"][1]["value"]}' \
  --filter_fn 'lambda sample: len(sample["instruction"]) + len(sample["response"]) < (8192 * 4)' \
  --max_len 8192 \
  --subset "Gen" \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "TIGER-Lab/MathInstruct" \
  --save_path $SAVE_DIR + "mathinstruct" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len 8192 \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "PawanKrd/math-gpt-4o-200k" \
  --save_path $SAVE_DIR + "mathgpt" \
  --inst_name "prompt" \
  --resp_name "response" \
  --max_len 8192 \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "TIGER-Lab/MATH-plus" \
  --save_path $SAVE_DIR + "mathplus" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len 8192 \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "cognitivecomputations/OpenCoder-LLM_opc-sft-stage1-DolphinLabeled" \
  --save_path $SAVE_DIR + "opencoder" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len 8192 \
  --subset "filtered_infinity_instruct" \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "OpenCoder-LLM/opc-sft-stage2" \
  --save_path $SAVE_DIR + "opencoder2" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len 8192 \
  --subset "educational_instruct" \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "openai/gsm8k" \
  --save_path $SAVE_DIR + "gsm8k" \
  --inst_name "question" \
  --resp_name "answer" \
  --subset "main" \
  --split "train" \
  --strategy "decreasing_v2"

python -m m2d.data.m2d_data \
  --dataset_name "AI-MO/NuminaMath-CoT" \
  --save_path $SAVE_DIR + "mathcot" \
  --inst_name "problem" \
  --resp_name "solution" \
  --strategy "decreasing_v2"

# ankner/gsm8k-CoT with map_fn that reformats the sample
python -m m2d.data.m2d_data \
  --dataset_name "ankner/gsm8k-CoT" \
  --save_path $SAVE_DIR + "gsm8kcot" \
  --inst_name "cot_problem" \
  --resp_name "cot_answer" \
  --map_fn 'lambda example: {"cot_problem": example["question"] + " Reason the question and think step by step. Please end with \"The final answer is [answer]\" where [answer] is your solution. ", "cot_answer": example["response"].replace("Therefore, the final answer is", "The final answer is")}' \
  --strategy "decreasing_v2"
