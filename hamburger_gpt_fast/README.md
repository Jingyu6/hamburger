## Guide on Inference Performance

### 1. First, we need to convert lightning checkpoint to gpt fast compatible checkpoint. 
```bash
python scripts/convert_hamburger_checkpoint.py \
--lightning_checkpoint_path /data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-1B-0506-finish.ckpt \
--save_checkpoint_path ./checkpoints/hamburger
```

### 2. Generate with Prompts
```bash
python generate.py \
--checkpoint_path checkpoints/hamburger/model.pth \
--is_hamburger \
--prompt "Who is Magnus Carlsen?"

python generate.py \
--checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
--prompt "Who is Magnus Carlsen?"

python generate.py \
--checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
--draft_checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model_int8.pth \
--prompt "Who is Magnus Carlsen?"
```

### 3. Benchmarking with Long Inputs and Outputs
```bash
python generate.py \
--checkpoint_path checkpoints/hamburger/model.pth \
--is_hamburger \
--prompt_file prompts/multi_news.txt \
--max_new_tokens 512

python generate.py \
--checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
--prompt_file prompts/multi_news.txt \
--max_new_tokens 512

python generate.py \
--checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
--draft_checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model_int8.pth \
--speculate_k 4 \
--prompt_file prompts/multi_news.txt \
--max_new_tokens 512

python generate.py \
--checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
--draft_checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model_int8.pth \
--speculate_k 2 \
--prompt_file prompts/multi_news.txt \
--max_new_tokens 512

python generate.py \
--checkpoint_path checkpoints/hamburger/model.pth \
--is_hamburger \
--prompt_file prompts/vcsum.txt \
--max_new_tokens 512

python generate.py \
--checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
--prompt_file prompts/vcsum.txt \
--max_new_tokens 512

python generate.py \
--checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
--draft_checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model_int8.pth \
--speculate_k 4 \
--prompt_file prompts/vcsum.txt \
--max_new_tokens 512

python generate.py \
--checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
--draft_checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model_int8.pth \
--speculate_k 2 \
--prompt_file prompts/vcsum.txt \
--max_new_tokens 512
```
