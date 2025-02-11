# Long Context
{
CUDA_VISIBLE_DEVICES=0 lm-eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,attn_implementation="flash_attention_2" \
    --apply_chat_template \
    --device cuda \
    --batch_size auto \
    --tasks scrolls_narrativeqa_llama_16k,scrolls_govreport_llama_16k \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/baseline/long_context
}&

# General
{
CUDA_VISIBLE_DEVICES=1 lm-eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,attn_implementation="flash_attention_2" \
    --apply_chat_template \
    --device cuda \
    --batch_size auto \
    --tasks leaderboard_ifeval \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/baseline/general

CUDA_VISIBLE_DEVICES=1 lm-eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,attn_implementation="flash_attention_2" \
    --apply_chat_template \
    --device cuda \
    --batch_size auto \
    --tasks mmlu_generative \
    --gen_kwargs temperature=0,max_gen_toks=4 \
    --log_samples \
    --output_path ./local/eval/baseline/general
}&

# Math
{
CUDA_VISIBLE_DEVICES=2 lm-eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,attn_implementation="flash_attention_2" \
    --tasks gsm8k_cot_llama \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --num_fewshot 8 \
    --gen_kwargs temperature=0 \
    --batch_size auto \
    --log_samples \
    --output_path ./local/eval/baseline/math
}&

# Reasoning
{
CUDA_VISIBLE_DEVICES=3 lm-eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,attn_implementation="flash_attention_2" \
    --apply_chat_template \
    --device cuda \
    --batch_size auto \
    --tasks arc_challenge_chat,gpqa_main_cot_zeroshot \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/baseline/reasoning
}&

# # Code
# CUDA_VISIBLE_DEVICES=1 evalplus \
#     --model "meta-llama/Llama-3.2-1B-Instruct" \
#     --dataset humaneval \
#     --backend vllm \
#     --greedy \
#     --enable-chunked-prefill False \
#     --root ./local/eval/baseline/code > ./local/eval/baseline/code/humaneval.txt

# CUDA_VISIBLE_DEVICES=1 evalplus \
#     --model "meta-llama/Llama-3.2-1B-Instruct" \
#     --dataset mbpp \
#     --backend vllm \
#     --greedy \
#     --enable-chunked-prefill False \
#     --root ./local/eval/baseline/code > ./local/eval/baseline/code/mbpp.txt