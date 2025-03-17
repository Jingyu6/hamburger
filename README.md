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
