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
  --dataset_name "openbmb/UltraInteract_sft" \
  --save_path "$SAVE_DIR/ultrainteract" \
  --inst_name "instruction" \
  --resp_name "response" \
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

# ServiceNow-AI/R1-Distill-SFT with map_fn and filter_fn
python -m hamburger.data.hamburger_data \
  --dataset_name "ServiceNow-AI/R1-Distill-SFT" \
  --subset "v1" \
  --save_path "$SAVE_DIR/r1distill" \
  --inst_name "problem" \
  --resp_name "reannotated_assistant_content" \
  --map_fn 'lambda example: {"problem": example["reannotated_messages"][0]["content"]}' \
  --filter_fn 'lambda x: (len(x["problem"]) + len(x["reannotated_assistant_content"])) <= $MAX_LEN' \
  --system_message "You're a helpful AI assistant, and think carefully before giving your final answer. Wrap your reasoning process in <think> and </think>. " \
  --batch_size 2 \
  --strategy $STRATEGY

# GAIR/lima with map_fn
python -m hamburger.data.hamburger_data \
  --dataset_name "GAIR/lima" \
  --save_path "$SAVE_DIR/lima" \
  --map_fn 'lambda example: {"instruction": example["conversations"][0], "response": example["conversations"][1]}' \
  --batch_size 2 \
  --strategy $STRATEGY

# allenai/tulu-v2-sft-mixture with map_fn
python -m hamburger.data.hamburger_data \
  --dataset_name "allenai/tulu-v2-sft-mixture" \
  --save_path "$SAVE_DIR/tulu" \
  --map_fn 'lambda example: {"instruction": example["messages"][0]["content"], "response": example["messages"][1]["content"]}' \
  --strategy $STRATEGY

# lmsys/lmsys-chat-1m with map_fn
python -m hamburger.data.hamburger_data \
  --dataset_name "lmsys/lmsys-chat-1m" \
  --save_path "$SAVE_DIR/lmsys" \
  --map_fn 'lambda example: {"instruction": example["conversation"][0]["content"], "response": example["conversation"][1]["content"]}' \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "open-r1/OpenR1-Math-220k" \
  --save_path "$SAVE_DIR/openr1math" \
  --inst_name "problem" \
  --resp_name "solution" \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

# PrimeIntellect/SYNTHETIC-1 with filter_fn and system_message
python -m hamburger.data.hamburger_data \
  --dataset_name "PrimeIntellect/SYNTHETIC-1" \
  --save_path "$SAVE_DIR/synthetic1" \
  --inst_name "prompt" \
  --resp_name "llm_response" \
  --system_message "You're a helpful AI assistant, and think carefully before giving your final answer. Wrap your reasoning process in <think> and </think>. " \
  --filter_fn 'lambda sample: (sample.get("score", None) == 1 and (len(sample["prompt"]) + len(sample["llm_response"])) <= $MAX_LEN)' \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

# facebook/natural_reasoning with map_fn
python -m hamburger.data.hamburger_data \
  --dataset_name "facebook/natural_reasoning" \
  --save_path "$SAVE_DIR/naturalreasoning" \
  --inst_name "question" \
  --resp_name "output" \
  --map_fn 'lambda sample: {"output": sample["responses"][0]["response"]}' \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "argilla/ifeval-like-data" \
  --save_path "$SAVE_DIR/ifevallike" \
  --inst_name "prompt" \
  --resp_name "response" \
  --subset "filtered" \
  --strategy $STRATEGY

# open-r1/OpenThoughts-114k-math with map_fn and custom batch_size
python -m hamburger.data.hamburger_data \
  --dataset_name "open-r1/OpenThoughts-114k-math" \
  --save_path "$SAVE_DIR/openthoughts" \
  --map_fn 'lambda example: {"instruction": example["conversations"][0]["value"], "response": example["conversations"][1]["value"]}' \
  --inst_name "problem" \
  --resp_name "solution" \
  --batch_size 1 \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

# codeparrot/apps with map_fn that formats strings
python -m hamburger.data.hamburger_data \
  --dataset_name "codeparrot/apps" \
  --save_path "$SAVE_DIR/apps" \
  --inst_name "prompt" \
  --resp_name "response" \
  --map_fn 'lambda example: {"prompt": "Write python code to solve the following coding question:\n" + example["question"] + "\n", "response": "```python\n" + example["solutions"][0] + "\n```"}' \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

# BAAI/Infinity-Instruct with map_fn and filter_fn
python -m hamburger.data.hamburger_data \
  --dataset_name "BAAI/Infinity-Instruct" \
  --save_path "$SAVE_DIR/infinityinstruct" \
  --map_fn 'lambda example: {"instruction": example["conversations"][0]["value"], "response": example["conversations"][1]["value"]}' \
  --filter_fn 'lambda sample: len(sample["instruction"]) + len(sample["response"]) < ($MAX_LEN * 4)' \
  --max_len $MAX_LEN \
  --subset "Gen" \
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
  --dataset_name "AI-MO/NuminaMath-CoT" \
  --save_path "$SAVE_DIR/mathcot" \
  --inst_name "problem" \
  --resp_name "solution" \
  --strategy $STRATEGY

# ankner/gsm8k-CoT with map_fn that reformats the sample
python -m hamburger.data.hamburger_data \
  --dataset_name "ankner/gsm8k-CoT" \
  --save_path "$SAVE_DIR/gsm8kcot" \
  --inst_name "cot_problem" \
  --resp_name "cot_answer" \
  --map_fn 'lambda example: {"cot_problem": example["question"] + " Reason the question and think step by step. Please end with \"The final answer is [answer]\" where [answer] is your solution. ", "cot_answer": example["response"].replace("Therefore, the final answer is", "The final answer is")}' \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "nvidia/OpenMathInstruct-2" \
  --save_path "$SAVE_DIR/openmathinstruct2" \
  --inst_name "problem" \
  --resp_name "generated_solution" \
  --max_len $MAX_LEN \
  --strategy $STRATEGY

python -m hamburger.data.hamburger_data \
  --dataset_name "nvidia/Llama-Nemotron-Post-Training-Dataset-v1" \
  --save_path "$SAVE_DIR/nemotroncode" \
  --inst_name "prompt" \
  --resp_name "response" \
  --map_fn 'lambda example: {"prompt": example["input"].replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\ndetailed thinking on<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "").replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", ""), "response": example["output"].replace("<|eot_id|>", "")}' \
  --filter_fn 'lambda example: example["reasoning"] == "on" and (len(example["input"]) + len(example["output"])) <= 8192' \
  --system_message "Think carefully with the reasoning process in <think> and </think>. " \
  --subset "SFT" \
  --data_files "SFT/code/code.jsonl" \
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
