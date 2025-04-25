

## Prepare BLT Model

| Byte Latent Transformer: [repo link](https://github.com/facebookresearch/blt)

Follow the guidance and download the weights. 
```bash
huggingface-cli download facebook/blt-1b --local-dir=<LOCAL_SAVE_DIR>
```

## Install BLT

Install from source
```bash
git clone https://github.com/facebookresearch/blt.git
cd blt
pip3 install .
```