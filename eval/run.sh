lm-eval \
    --model local-chat-completions \
    --model_args model=m2d,base_url=http://127.0.0.1:8000/v1/chat/completions \
    --apply_chat_template \
    --batch_size 1 \
    --tasks gsm8k_cot_llama_3.1_instruct