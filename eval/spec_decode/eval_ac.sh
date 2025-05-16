# GSM8K
python eval_acceptance_rate.py \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --draft_model_type ar \
    --dataset_name openai/gsm8k \
    --subset main \
    --split test \
    --prompt_key question \
    --max_samples 64

python eval_acceptance_rate.py \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --draft_model /data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-1B-mha-enhance-finish.ckpt \
    --draft_model_type hamburger \
    --dataset_name openai/gsm8k \
    --subset main \
    --split test \
    --prompt_key question \
    --max_samples 64 \
    --draft_sep_last

# Humaneval
python eval_acceptance_rate.py \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --draft_model_type ar \
    --dataset_name openai/openai_humaneval \
    --split test \
    --prompt_key prompt \
    --max_samples 64

python eval_acceptance_rate.py \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --draft_model /data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-1B-mha-enhance-finish.ckpt \
    --draft_model_type hamburger \
    --dataset_name openai/openai_humaneval \
    --split test \
    --prompt_key prompt \
    --max_samples 64 \
    --draft_sep_last

# Arc Challenge
python eval_acceptance_rate.py \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --draft_model_type ar \
    --dataset_name allenai/ai2_arc \
    --subset ARC-Challenge \
    --split test \
    --prompt_key question \
    --max_samples 64

python eval_acceptance_rate.py \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --draft_model /data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-1B-mha-enhance-finish.ckpt \
    --draft_model_type hamburger \
    --dataset_name allenai/ai2_arc \
    --subset ARC-Challenge \
    --split test \
    --prompt_key question \
    --max_samples 64 \
    --draft_sep_last

# IFEval
python eval_acceptance_rate.py \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --draft_model_type ar \
    --dataset_name google/IFEval \
    --split train \
    --prompt_key prompt \
    --max_samples 64

python eval_acceptance_rate.py \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --draft_model /data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-1B-mha-enhance-finish.ckpt \
    --draft_model_type hamburger \
    --dataset_name google/IFEval \
    --split train \
    --prompt_key prompt \
    --max_samples 64 \
    --draft_sep_last
