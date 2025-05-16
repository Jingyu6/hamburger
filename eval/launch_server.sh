
case "$1" in
    "hf")
        python -m eval.server \
            --model_name meta-llama/Llama-3.2-1B-Instruct \
            --model_type hf
        ;;
    "hamburger")
        python -m eval.server \
            --model_name /data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-1B-0506-finish.ckpt \
            --model_type hamburger \
            --confidence $2
        ;;
    *)
        echo Invalid choice of model type. 
        exit 1
        ;;
esac
