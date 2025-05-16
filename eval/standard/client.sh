lm-eval \
    --model local-chat-completions \
    --model_args model=hamburger,base_url=http://127.0.0.1:8000/v1/chat/completions \
    --apply_chat_template \
    --num_fewshot 1 \
    --tasks gsm8k_cot_llama \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/hamburger/math

lm-eval \
    --model local-chat-completions \
    --model_args model=hamburger,base_url=http://127.0.0.1:8000/v1/chat/completions \
    --apply_chat_template \
    --batch_size 1 \
    --tasks leaderboard_ifeval \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/hamburger/general

lm-eval \
    --model local-chat-completions \
    --model_args model=hamburger,base_url=http://127.0.0.1:8000/v1/chat/completions \
    --apply_chat_template \
    --batch_size 1 \
    --tasks arc_challenge_chat \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/hamburger/reasoning

lm-eval \
    --model local-chat-completions \
    --model_args model=hamburger,base_url=http://127.0.0.1:8000/v1/chat/completions \
    --apply_chat_template \
    --batch_size 1 \
    --tasks mgsm_chat_en \
    --gen_kwargs temperature=0 \
    --log_samples \
    --output_path ./local/eval/hamburger/math

evalplus.evaluate \
    --model "hamburger" \
    --dataset humaneval \
    --backend openai \
    --greedy \
    --base-url http://127.0.0.1:8000/v1 \
    --root ./local/eval/hamburger/code

evalplus.evaluate \
    --model "hamburger" \
    --dataset mbpp \
    --backend openai \
    --greedy \
    --base-url http://127.0.0.1:8000/v1 \
    --root ./local/eval/hamburger/code
