python pred.py \
    --model "/data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-code-math-skip-finish.ckpt" \
    --model_type "m2d" \
    --exp m2d

python eval.py --exp m2d

python pred.py \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --model_type "hf" \
    --exp baseline

python eval.py --exp baseline