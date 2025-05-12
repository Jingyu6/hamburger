lm-eval \
    --model local-chat-completions \
    --model_args model=m2d,base_url=http://127.0.0.1:8000/v1/chat/completions \
    --apply_chat_template \
    --num_fewshot 1 \
    --tasks gsm8k_cot_llama \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/m2d/math

lm-eval \
    --model local-chat-completions \
    --model_args model=m2d,base_url=http://127.0.0.1:8000/v1/chat/completions \
    --apply_chat_template \
    --batch_size 1 \
    --tasks leaderboard_ifeval \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/m2d/general

lm-eval \
    --model local-chat-completions \
    --model_args model=m2d,base_url=http://127.0.0.1:8000/v1/chat/completions \
    --apply_chat_template \
    --batch_size 1 \
    --tasks arc_challenge_chat \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/m2d/reasoning

lm-eval \
    --model local-chat-completions \
    --model_args model=m2d,base_url=http://127.0.0.1:8000/v1/chat/completions \
    --apply_chat_template \
    --batch_size 1 \
    --tasks mgsm_chat_en \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/m2d/math

evalplus.evaluate \
    --model "m2d" \
    --dataset humaneval \
    --backend openai \
    --greedy \
    --base-url http://127.0.0.1:8000/v1 \
    --root ./local/eval/m2d/code

evalplus.evaluate \
    --model "m2d" \
    --dataset mbpp \
    --backend openai \
    --greedy \
    --base-url http://127.0.0.1:8000/v1 \
    --root ./local/eval/m2d/code
