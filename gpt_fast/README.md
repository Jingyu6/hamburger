```bash
python generate.py \
--checkpoint_path checkpoints/hamburger/model.pth \
--is_hamburger \
--prompt "Who is Magnus Carlsen?"

python generate.py \
--checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
--prompt "Who is Magnus Carlsen?"
```

```bash
python scripts/convert_hamburger_checkpoint.py \
--lightning_checkpoint_path /data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-0506-finish.ckpt \
--save_checkpoint_path ./checkpoints/hamburger
```

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
--checkpoint_path checkpoints/hamburger/model.pth \
--is_hamburger \
--prompt_file prompts/vcsum.txt \
--max_new_tokens 512

python generate.py \
--checkpoint_path checkpoints/meta-llama/Llama-3.2-1B-Instruct/model.pth \
--prompt_file prompts/vcsum.txt \
--max_new_tokens 512
```