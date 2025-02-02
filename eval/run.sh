# arc_challenge_llama_instruct,gpqa_main_cot_zeroshot

# {
# CUDA_VISIBLE_DEVICES=0 lm-eval \
#     --model local-chat-completions \
#     --model_args model=m2d,base_url=http://127.0.0.1:8000/v1/chat/completions \
#     --apply_chat_template \
#     --batch_size 1 \
#     --tasks mmlu_generative \
#     --gen_kwargs temperature=0,max_gen_toks=4 \
#     --log_samples \
#     --output_path ./local/eval/m2d
# } &

# {
# CUDA_VISIBLE_DEVICES=1 lm-eval \
#     --model hf \
#     --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
#     --apply_chat_template \
#     --device cuda:1 \
#     --batch_size 8 \
#     --tasks mmlu_generative \
#     --gen_kwargs temperature=0,max_gen_toks=4 \
#     --log_samples \
#     --output_path ./local/eval/baseline 
# } &

{
CUDA_VISIBLE_DEVICES=0 lm-eval \
    --model local-chat-completions \
    --model_args model=m2d,base_url=http://127.0.0.1:8000/v1/chat/completions \
    --apply_chat_template \
    --batch_size 1 \
    --tasks scrolls_narrativeqa_llama_16k,scrolls_govreport_llama_16k \
    --gen_kwargs temperature=0 \
    --log_samples \
    --limit 256 \
    --output_path ./local/eval/m2d_v3
} &

# {
# CUDA_VISIBLE_DEVICES=1 lm-eval \
#     --model hf \
#     --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
#     --apply_chat_template \
#     --device cuda:1 \
#     --batch_size 1 \
#     --tasks scrolls_qasper_llama_16k,scrolls_narrativeqa_llama_16k,scrolls_govreport_llama_16k \
#     --gen_kwargs temperature=0 \
#     --log_samples \
#     --limit 256 \
#     --output_path ./local/eval/baseline 
# } &