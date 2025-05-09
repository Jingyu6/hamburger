import os
import re
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs


@torch.inference_mode()
def convert_hamburger_checkpoint(
    lightning_checkpoint_path: Path, 
    save_checkpoint_path: Path
) -> None:
    model_name = "hamburger"

    # Hard coding for now
    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # define weight map
    weight_map = {
        # base model keys
        "model.model.embed_tokens.weight": "model.tok_embeddings.weight",
        "model.model.layers.{}.self_attn.q_proj.weight": "model.layers.{}.attention.wq.weight",
        "model.model.layers.{}.self_attn.k_proj.weight": "model.layers.{}.attention.wk.weight",
        "model.model.layers.{}.self_attn.v_proj.weight": "model.layers.{}.attention.wv.weight",
        "model.model.layers.{}.self_attn.o_proj.weight": "model.layers.{}.attention.wo.weight",
        "model.model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.model.layers.{}.mlp.gate_proj.weight": "model.layers.{}.feed_forward.w1.weight",
        "model.model.layers.{}.mlp.up_proj.weight": "model.layers.{}.feed_forward.w3.weight",
        "model.model.layers.{}.mlp.down_proj.weight": "model.layers.{}.feed_forward.w2.weight",
        "model.model.layers.{}.input_layernorm.weight": "model.layers.{}.attention_norm.weight",
        "model.model.layers.{}.post_attention_layernorm.weight": "model.layers.{}.ffn_norm.weight",
        "model.model.norm.weight": "model.norm.weight",
        "model.lm_head.weight": "model.output.weight",

        # compositional embedder
        "comp_embedder.embedding.weight": None, 
        "comp_embedder.merger.pos": "comp_embedder.merger.pos", 
        "comp_embedder.merger.q_proj.weight": "comp_embedder.merger.q_proj.weight", 
        "comp_embedder.merger.q_proj.bias": "comp_embedder.merger.q_proj.bias", 
        "comp_embedder.merger.k_proj.weight": "comp_embedder.merger.k_proj.weight",  
        "comp_embedder.merger.k_proj.bias": "comp_embedder.merger.k_proj.bias",  
        "comp_embedder.merger.v_proj.weight": "comp_embedder.merger.v_proj.weight", 
        "comp_embedder.merger.v_proj.bias": "comp_embedder.merger.v_proj.bias", 
        "comp_embedder.out_proj.weight": "comp_embedder.out_proj.weight",  
        "comp_embedder.out_proj.bias": "comp_embedder.out_proj.bias", 

        # micro-step decoder
        "micro_step_decoder.decoders.{}.self_attn.q_proj.weight": "micro_step_decoder.decoders.{}.attention.wq.weight",
        "micro_step_decoder.decoders.{}.self_attn.k_proj.weight": "micro_step_decoder.decoders.{}.attention.wk.weight",
        "micro_step_decoder.decoders.{}.self_attn.v_proj.weight": "micro_step_decoder.decoders.{}.attention.wv.weight",
        "micro_step_decoder.decoders.{}.self_attn.o_proj.weight": "micro_step_decoder.decoders.{}.attention.wo.weight",
        "micro_step_decoder.decoders.{}.self_attn.rotary_emb.inv_freq": None,
        "micro_step_decoder.decoders.{}.mlp.gate_proj.weight": "micro_step_decoder.decoders.{}.feed_forward.w1.weight",
        "micro_step_decoder.decoders.{}.mlp.up_proj.weight": "micro_step_decoder.decoders.{}.feed_forward.w3.weight",
        "micro_step_decoder.decoders.{}.mlp.down_proj.weight": "micro_step_decoder.decoders.{}.feed_forward.w2.weight",
        "micro_step_decoder.decoders.{}.input_layernorm.weight": "micro_step_decoder.decoders.{}.attention_norm.weight",
        "micro_step_decoder.decoders.{}.post_attention_layernorm.weight": "micro_step_decoder.decoders.{}.ffn_norm.weight",
    
        "micro_step_decoder.stop_head.0.weight": "micro_step_decoder.stop_head.0.weight", 
        "micro_step_decoder.stop_head.0.bias": "micro_step_decoder.stop_head.0.bias", 
        "micro_step_decoder.stop_head.2.weight": "micro_step_decoder.stop_head.2.weight", 
        "micro_step_decoder.stop_head.2.bias": "micro_step_decoder.stop_head.2.bias", 
        "micro_step_decoder.stop_head.4.weight": "micro_step_decoder.stop_head.4.weight", 
        "micro_step_decoder.stop_head.4.bias": "micro_step_decoder.stop_head.4.bias"
    }

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    # Saving models
    original_result = torch.load(
        str(lightning_checkpoint_path), 
        map_location="cpu", 
        weights_only=True
    )["state_dict"]

    # change keys
    final_result = {}

    # first base model weights
    for key, value in original_result.items():
        if "layers" in key or "decoders" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        if new_key is None:
            continue

        final_result[new_key] = value

    # tied weights or shared weights
    final_result["model.output.weight"] = original_result["model.model.embed_tokens.weight"]
    final_result["comp_embedder.embedding.weight"] = final_result["model.tok_embeddings.weight"]

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = permute(q, config.n_head)
            k = permute(k, config.n_local_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
    
    os.makedirs(save_checkpoint_path, exist_ok=True)

    print(f"Saving checkpoint to {save_checkpoint_path / 'model.pth'}")
    torch.save(final_result, save_checkpoint_path / "model.pth")

    # Saving tokenizers
    print(f"Downloading tokenizer to {save_checkpoint_path / 'tokenizer.model'}")
    tokenizer_path = hf_hub_download(
        repo_id="meta-llama/Llama-3.2-1B-Instruct", 
        filename="original/tokenizer.model", 
        local_dir=save_checkpoint_path, 
    )

    shutil.move(tokenizer_path, save_checkpoint_path)
    shutil.rmtree(save_checkpoint_path / ".cache")
    shutil.rmtree(save_checkpoint_path / "original")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HAMburger checkpoint.')
    parser.add_argument('--lightning_checkpoint_path', type=Path)
    parser.add_argument('--save_checkpoint_path', type=Path)

    args = parser.parse_args()
    convert_hamburger_checkpoint(
        lightning_checkpoint_path=args.lightning_checkpoint_path, 
        save_checkpoint_path=args.save_checkpoint_path
    )
