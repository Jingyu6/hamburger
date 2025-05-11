
case "$1" in
    "hf")
        python -m eval.server \
            --model_name meta-llama/Llama-3.2-1B-Instruct \
            --model_type hf
        ;;
    "m2d")
        python -m eval.server \
            --model_name /data/data_persistent1/jingyu/m2d/ckpts/m2d-llama-1B-code-math-skip-finish.ckpt \
            --model_type m2d \
            --confidence $2
        ;;
    *)
        echo Invalid choice of model type. 
        exit 1
        ;;
esac
