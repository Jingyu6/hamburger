```bash
python generate.py \
--checkpoint_path checkpoints/hamburger/model.pth \
--is_hamburger \
--prompt "Hello, my name is"
```

```bash
python scripts/convert_hamburger_checkpoint.py \
--lightning_checkpoint_path /data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-0506-finish.ckpt \
--save_checkpoint_path ./checkpoints/hamburger
```