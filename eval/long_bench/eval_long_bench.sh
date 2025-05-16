CUDA_VISIBLE_DEVICES=0 python pred.py \
    --model "/data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-1B-0506-finish.ckpt" \
    --model_type "hamburger" \
    --confidence 0.9 \
    --exp hamburger_p9

python eval.py --exp hamburger_p9

CUDA_VISIBLE_DEVICES=1 python pred.py \
    --model "/data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-1B-0506-finish.ckpt" \
    --model_type "hamburger" \
    --confidence 0.8 \
    --exp hamburger_p8

python eval.py --exp hamburger_p8

CUDA_VISIBLE_DEVICES=2 python pred.py \
    --model "/data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-1B-0506-finish.ckpt" \
    --model_type "hamburger" \
    --confidence 0.7 \
    --exp hamburger_p7

python eval.py --exp hamburger_p7

CUDA_VISIBLE_DEVICES=3 python pred.py \
    --model "/data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-1B-0506-finish.ckpt" \
    --model_type "hamburger" \
    --confidence 0.6 \
    --exp hamburger_p6

python eval.py --exp hamburger_p6

CUDA_VISIBLE_DEVICES=4 python pred.py \
    --model "/data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-1B-0506-finish.ckpt" \
    --model_type "hamburger" \
    --confidence 0.5 \
    --exp hamburger_p5

python eval.py --exp hamburger_p5

python pred.py \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --model_type "hf" \
    --exp baseline

python eval.py --exp baseline