
for task in gsm8k.txt humaneval.txt mgsm.txt multi_news.txt
do
    echo $task
    echo "HAMburger"
    python generate.py \
        --checkpoint_path checkpoints/hamburger/model.pth \
        --is_hamburger \
        --prompt_file prompts/$task \
        --max_new_tokens 512 | grep "Decode TPS"
    
    echo "Llama"
    python generate.py \
        --checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
        --prompt_file prompts/$task \
        --max_new_tokens 512 | grep "Decode TPS"

    echo "Spec-4"
    python generate.py \
        --checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
        --draft_checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model_int8.pth \
        --speculate_k 4 \
        --prompt_file prompts/$task \
        --max_new_tokens 512 | grep "Decode TPS"
    
    echo "Spec-2"
    python generate.py \
        --checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
        --draft_checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model_int8.pth \
        --speculate_k 2 \
        --prompt_file prompts/$task \
        --max_new_tokens 512 | grep "Decode TPS"
done
