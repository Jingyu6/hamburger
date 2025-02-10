# Long Context
CUDA_VISIBLE_DEVICES=1 lm-eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
    --apply_chat_template \
    --device cuda:0 \
    --batch_size auto \
    --tasks scrolls_narrativeqa_llama_16k,scrolls_govreport_llama_16k \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/baseline/long_context

# General
CUDA_VISIBLE_DEVICES=1 lm-eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
    --apply_chat_template \
    --device cuda:0 \
    --batch_size auto \
    --tasks mmlu_generative,leaderboard_ifeval \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/baseline/general

# Code
CUDA_VISIBLE_DEVICES=1 python -m eval.evalplus \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --dataset humaneval \
    --backend vllm \
    --greedy \
    --enable-chunked-prefill False \
    --root ./local/eval/baseline/code > ./local/eval/baseline/code/humaneval.txt

CUDA_VISIBLE_DEVICES=1 python -m eval.evalplus \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --dataset mbpp \
    --backend vllm \
    --greedy \
    --enable-chunked-prefill False \
    --root ./local/eval/baseline/code > ./local/eval/baseline/code/mbpp.txt

# Math
CUDA_VISIBLE_DEVICES=1 lm-eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
    --tasks gsm8k_cot_llama_3.1_instruct \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --num_fewshot 8 \
    --gen_kwargs temperature=0 \
    --batch_size auto \
    --log_samples \
    --output_path ./local/eval/baseline/math

CUDA_VISIBLE_DEVICES=1 lm-eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
    --apply_chat_template \
    --device cuda:0 \
    --batch_size auto \
    --tasks mathqa \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/baseline/math

# Reasoning
CUDA_VISIBLE_DEVICES=1 lm-eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
    --apply_chat_template \
    --device cuda:0 \
    --batch_size auto \
    --tasks arc_challenge_llama_instruct,gpqa_main_cot_zeroshot \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/baseline/reasoning
