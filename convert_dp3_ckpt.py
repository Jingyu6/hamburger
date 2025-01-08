from lightning.pytorch.utilities.deepspeed import \
    convert_zero_checkpoint_to_fp32_state_dict

# lightning deepspeed has saved a directory instead of a file
save_path = "./local/ckpts/m2d-llama-1B.ckpt"
output_path = "./local/consolidated.pt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
