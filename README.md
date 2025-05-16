# HAMburger: Accelerating LLM Inference via Token Smashing

## Introduction
HAMburger is a hierachically auto-regressive model that can output multiple tokens per forward. Our approach reduces the growth of KV cache _computation_ from linear to sub-linear w.r.t. the generation length and achieves a TPS speedup proportionally. On both standard tasks and long-context tasks, HAMburger achieves up to 2x TPS boost and 2x KV cache computation (and storage) while maintaining or even surpassing the base model. 

## Architecture
HAMburger stacks a standard LLM (e.g., Llama-3.2-1B-Instruct) with a relative-position-aware compositional embedder before it that smashes multiple tokens into one from the last step, and a micro-step decoder after it that outputs tokens with constant FLOPs. 

## Environment Setup
Use conda as below:
```bash
conda create -yn hamburger python=3.10.15
conda activate hamburger
pip3 install -r requirements.txt
```

## Training
HAMburger is instruction-finetuned with publicly available datasets and we provide both the training code and our trained checkpoints for full reproduction. 

### Data Preparation
We prepared scripts for processing the data automatically and you can easily extend that by adding new datasets:
```bash
bash data_scripts/process.sh
```

### Start Training
We trained our 1B model with 8xH100s. To reproduce our results, we suggest running: 
```bash
python3 -m hamburger.train
```

## Inference
We implement HAMburger on both GPT-Fast and HuggingFace for a balance of simplicity and performance. 

### Generation Demo
To run streaming demo, simply do the following and choose the option based on guidance:
```bash
python generate.py
```

To run GPT-Fast version, please read guidance [here](./hamburger_gpt_fast/README.md). 

### Evaluate Results
All evaluation-related files are stored in `./eval`. To run LongBench, simply run:
```bash
cd ./eval/long_bench
bash eval_long_bench.sh
python summarize_results.py # this is optional
```

To run standard tasks, we rely on `lm_eval` and `evalplus`:

1. First, we need to apply some patches to `lm_eval` by copying (overwritting) `./eval/standard/lm_eval_patch/*` to your conda `lm_eval/tasks`.  
2. Can can setup a server that has common API:
```bash
bash ./eval/launch_server.sh hf # for baseline
bash ./eval/launch_server.sh hamburger 0.8 # for hamburger
```
3. Run any commands found in `./eval/standard/client.sh` for each individual task. 
