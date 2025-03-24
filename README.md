# micro_step

### TODOs
We welcome everyone to try and contribute to the code! Here're some planned TODOs
- [ ] Finish basic evaluation of the new model. 
- [ ] Evaluate code and GSM8k
- [ ] Data ablation
- [ ] vLLM or Sglang implementation & weight conversion script
- [ ] Benchmark output

### Logs
- 2025/02/25: 
    * There seems to be a big performance difference on code using "\n" v.s. \n. 
    * Humaneval needs more output to finish. 

- 2025/03/06:
    * Pretrained models do have higher average conditional entropy than SFT model. 
    * The conditional entropy seem to be transferrable among different sizes. 
    * Need to think about the difference between BLT and M2D
    * Need to factor out the influence of the data by training on the base model. 

- 2025/03/07:
    * Explore BLT local encoder
    * Think about what's the main difference here

- 2025/03/17:
    * Found the best data mix so far
        - Filtered out other languages
        - Added more math data
    * Next step
        - Understand why the model works well for MBPP but not HumanEval
        - Understand where the gap is for GSM8K 8-shot CoT
        - Potentially pre-calculate entropy and do segment strategy ablation
- 2025/03/18:
    * Try zero shot gsm8k + manual answer extraction
        - 8-shot has 23 invalid answers while 0-shot has 430
        - 0-shot is only 2 points lower
        - If 0-shot with manual extraction is higher, that means the model is working bad for few-shot, which might be fixed by adding similar data
- 2025/03/23:
    * Seems like training 3B models results in larger gap.
        - The gap is larger when using the 3D model to segment data
        - Probably due to the fact that we didn't take their confidence difference into account
    * Trying a two-stage training to avoid catastrophic forgetting
        - First freeze all except for newly introduced modules with 1e-4 lr
        - Joint train all weights with 1e-5 lr
        - Consider using LoRA for the base model
- 2025/03/24:
    * For GSM8K, the task provides a prompt prefix.
        - But this would be inconsistent with our training because the any generation should be determined by the model (segmentation), which might cause some ODD problem
        - We can try disabling the prefix and see the result