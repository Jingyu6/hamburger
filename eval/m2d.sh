# Long Context
# {
CUDA_VISIBLE_DEVICES=0 lm-eval \
    --model local-chat-completions \
    --model_args model=m2d,base_url=http://127.0.0.1:8000/v1/chat/completions \
    --apply_chat_template \
    --device cuda \
    --batch_size 1 \
    --tasks scrolls_narrativeqa_llama_16k,scrolls_govreport_llama_16k \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/m2d/long_context
# }&

# # General
# {
# CUDA_VISIBLE_DEVICES=1 lm-eval \
#     --model hf \
#     --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,attn_implementation="flash_attention_2" \
#     --apply_chat_template \
#     --device cuda \
#     --batch_size 1 \
#     --tasks leaderboard_ifeval \
#     --gen_kwargs temperature=0 \
#     --log_samples \
#     --output_path ./local/eval/m2d/general

# CUDA_VISIBLE_DEVICES=1 lm-eval \
#     --model hf \
#     --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,attn_implementation="flash_attention_2" \
#     --apply_chat_template \
#     --device cuda \
#     --batch_size 1 \
#     --tasks mmlu_generative \
#     --gen_kwargs temperature=0,max_gen_toks=4 \
#     --log_samples \
#     --output_path ./local/eval/m2d/general
# }&

# # Math
# {
# CUDA_VISIBLE_DEVICES=2 lm-eval \
#     --model hf \
#     --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,attn_implementation="flash_attention_2" \
#     --tasks gsm8k_cot_llama \
#     --apply_chat_template \
#     --fewshot_as_multiturn \
#     --num_fewshot 8 \
#     --gen_kwargs temperature=0 \
#     --batch_size 1 \
#     --log_samples \
#     --output_path ./local/eval/m2d/math
# }&

# # Reasoning
# {
# CUDA_VISIBLE_DEVICES=3 lm-eval \
#     --model hf \
#     --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,attn_implementation="flash_attention_2" \
#     --apply_chat_template \
#     --device cuda \
#     --batch_size 1 \
#     --tasks arc_challenge_chat,gpqa_main_cot_zeroshot \
#     --gen_kwargs temperature=0 \
#     --log_samples \
#     --output_path ./local/eval/m2d/reasoning
# }&
