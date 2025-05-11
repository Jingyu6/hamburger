CUDA_VISIBLE_DEVICES=0 python pred.py \
    --model "/data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-0506-finish.ckpt" \
    --model_type "m2d" \
    --confidence 0.9 \
    --exp m2d_p9

python eval.py --exp m2d_p9

CUDA_VISIBLE_DEVICES=1 python pred.py \
    --model "/data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-0506-finish.ckpt" \
    --model_type "m2d" \
    --confidence 0.8 \
    --exp m2d_p8

python eval.py --exp m2d_p8

CUDA_VISIBLE_DEVICES=2 python pred.py \
    --model "/data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-0506-finish.ckpt" \
    --model_type "m2d" \
    --confidence 0.7 \
    --exp m2d_p7

python eval.py --exp m2d_p7

CUDA_VISIBLE_DEVICES=3 python pred.py \
    --model "/data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-0506-finish.ckpt" \
    --model_type "m2d" \
    --confidence 0.6 \
    --exp m2d_p6

python eval.py --exp m2d_p6

CUDA_VISIBLE_DEVICES=4 python pred.py \
    --model "/data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-0506-finish.ckpt" \
    --model_type "m2d" \
    --confidence 0.5 \
    --exp m2d_p5

python eval.py --exp m2d_p5

python pred.py \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --model_type "hf" \
    --exp baseline

python eval.py --exp baseline