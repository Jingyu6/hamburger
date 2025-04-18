# GSM8K
python eval_acceptance_rate.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --draft_model_type ar \
    --dataset_name openai/gsm8k \
    --subset main \
    --split test \
    --prompt_key question \
    --max_samples 32

python eval_acceptance_rate.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model /data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-mha-enhance-finish.ckpt \
    --draft_model_type m2d \
    --dataset_name openai/gsm8k \
    --subset main \
    --split test \
    --prompt_key question \
    --max_samples 32

# Humaneval
python eval_acceptance_rate.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --draft_model_type ar \
    --dataset_name openai/openai_humaneval \
    --split test \
    --prompt_key prompt \
    --max_samples 32

python eval_acceptance_rate.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model /data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-mha-enhance-finish.ckpt \
    --draft_model_type m2d \
    --dataset_name openai/openai_humaneval \
    --split test \
    --prompt_key prompt \
    --max_samples 32

# Arc Challenge
python eval_acceptance_rate.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --draft_model_type ar \
    --dataset_name allenai/ai2_arc \
    --subset ARC-Challenge \
    --split test \
    --prompt_key question \
    --max_samples 32

python eval_acceptance_rate.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model /data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-mha-enhance-finish.ckpt \
    --draft_model_type m2d \
    --dataset_name allenai/ai2_arc \
    --subset ARC-Challenge \
    --split test \
    --prompt_key question \
    --max_samples 32

