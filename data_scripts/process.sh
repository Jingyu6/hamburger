#!/bin/bash
# Run HAMburgerDataModule.from_hf_dataset for various datasets using the CLI

SAVE_DIR="/data/data_persistent1/jingyu/hamburger/decreasing_v2"
STRATEGY="decreasing_v2"
MAX_LEN=8192

python -m hamburger.data.hamburger_data \
  --dataset_name "imone/OpenOrca_FLAN" \
  --save_path "$SAVE_DIR/openorca" \
  --filter_fn 'lambda sample: sample["condition"] == "GPT4"' \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "nampdn-ai/tiny-codes" \
  --save_path "$SAVE_DIR/tinycodepython" \
  --inst_name "prompt" \
  --resp_name "response" \
  --filter_fn 'lambda sample: sample["programming_language"] == "Python"' \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "teknium/openhermes" \
  --save_path "$SAVE_DIR/openhermes" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "meta-math/MetaMathQA" \
  --save_path "$SAVE_DIR/metamathqa" \
  --inst_name "query" \
  --resp_name "response" \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "garage-bAInd/Open-Platypus" \
  --save_path "$SAVE_DIR/openplatypus" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "ise-uiuc/Magicoder-Evol-Instruct-110K" \
  --save_path "$SAVE_DIR/magicoder" \
  --inst_name "instruction" \
  --resp_name "response" \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "Vezora/Tested-143k-Python-Alpaca" \
  --save_path "$SAVE_DIR/pythonalpaca" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "open-r1/OpenR1-Math-220k" \
  --save_path "$SAVE_DIR/openr1math" \
  --inst_name "problem" \
  --resp_name "solution" \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "argilla/ifeval-like-data" \
  --save_path "$SAVE_DIR/ifevallike" \
  --inst_name "prompt" \
  --resp_name "response" \
  --subset "filtered" \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "TIGER-Lab/MathInstruct" \
  --save_path "$SAVE_DIR/mathinstruct" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "PawanKrd/math-gpt-4o-200k" \
  --save_path "$SAVE_DIR/mathgpt" \
  --inst_name "prompt" \
  --resp_name "response" \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "TIGER-Lab/MATH-plus" \
  --save_path "$SAVE_DIR/mathplus" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "cognitivecomputations/OpenCoder-LLM_opc-sft-stage1-DolphinLabeled" \
  --save_path "$SAVE_DIR/opencoder" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len $MAX_LEN \
  --subset "filtered_infinity_instruct" \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "OpenCoder-LLM/opc-sft-stage2" \
  --save_path "$SAVE_DIR/opencoder2" \
  --inst_name "instruction" \
  --resp_name "output" \
  --max_len $MAX_LEN \
  --subset "educational_instruct" \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "openai/gsm8k" \
  --save_path "$SAVE_DIR/gsm8k" \
  --inst_name "question" \
  --resp_name "answer" \
  --subset "main" \
  --split "train" \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "nvidia/OpenMathInstruct-2" \
  --save_path "$SAVE_DIR/openmathinstruct2" \
  --inst_name "problem" \
  --resp_name "generated_solution" \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "KonstantyM/science_qa" \
  --save_path "$SAVE_DIR/scienceqa" \
  --inst_name "question" \
  --resp_name "answer" \
  --max_len $MAX_LEN \
  --max_num_samples 1000000 \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "hotpotqa/hotpot_qa" \
  --save_path "$SAVE_DIR/hotpotqa" \
  --subset "fullwiki" \
  --split "train" \
  --inst_name "prompt" \
  --resp_name "answer" \
  --max_len $MAX_LEN \
  --map_fn 'lambda example: {"prompt": "Answer the question based on the below context (do not explain):\n" + "\n".join([f"Title:\n{t}\nPassage:\n{s}" for t, s in zip(example["context"]["title"], example["context"]["sentences"])]) + "\nQuestion: " + example["question"]}' \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "mrqa-workshop/mrqa" \
  --save_path "$SAVE_DIR/mrqa" \
  --split "train" \
  --inst_name "prompt" \
  --resp_name "response" \
  --max_len $MAX_LEN \
  --map_fn 'lambda example: {"prompt": example["context"] + "\nBased on the above context, answer without explanation: " + example["question"], "response": example["answers"][0]}' \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "deepmind/narrativeqa" \
  --save_path "$SAVE_DIR/narrativeqa" \
  --split "train" \
  --inst_name "prompt" \
  --resp_name "response" \
  --max_len $MAX_LEN \
  --map_fn 'lambda example: {"prompt": example["document"]["summary"]["text"] + "\nBased on the above context, answer without explanation: " + example["question"]["text"], "response": example["answers"][0]["text"]}' \
  --strategy $STRATEGY
