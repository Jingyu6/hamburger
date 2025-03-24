import json
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional

import yaml


class _LoadableConfig:
    @classmethod
    def from_path(cls, config_path: Optional[str] = None):
        if config_path is None:
            return cls()

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Get the fields of the dataclass
        field_names = {f.name for f in fields(cls)}
        
        # Check for unused fields in the YAML
        unused_fields = set(data.keys()) - field_names
        if unused_fields:
            raise ValueError(f"Unused fields in YAML: {unused_fields}.")
        
        # Filter out keys that are not fields of the dataclass
        used_data = {k: v for k, v in data.items() if k in field_names}
        
        # Return an instance of the dataclass populated with the filtered data
        return cls(**used_data)

    def print_config(self):
        print("\033[92m{}\033[00m".format(
            f"Using spec config:\n{json.dumps(asdict(self), indent=4)}"))

@dataclass
class M2DConfig(_LoadableConfig): 
    seed: int = 227
    
    base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    pretrained_ckpt_path: Optional[str] = None
    resume_ckpt_path: Optional[str] = None

    strategy: str = "auto"

    dataset_names: List[str] = None
    batch_size: int = 8
    test_ratio: float = 0.005

    accumulate_grad_batches: int = 2

    run_name: str = ""
    
    def __post_init__(self):
        assert self.dataset_names is not None


@dataclass
class GenConfig(_LoadableConfig):
    max_gen_len: int = 256
    system_message: Optional[str] = None
    repetition_penalty: Optional[float] = None
    remove_think: bool = False
    extra_think_steps: int = 512

    @property
    def decode_steps(self):
        return self.max_gen_len + self.extra_think_steps \
            if self.remove_think else self.max_gen_len


_FORMAT_KEYS: List[str] = [
    "prefix_inst", 
    "suffix_inst", 
    "parser_regex", 
    "output_format"
]

@dataclass
class FormatConfig(_LoadableConfig):
    task_configs: Optional[Dict[str, Dict]] = None
    
    def __post_init__(self):
        if self.task_configs is None:
            self.task_configs = {}
        
        for config in self.task_configs.values():
            assert all([(key in _FORMAT_KEYS) for key in config.keys()]), \
                f"Each task requires keys from {_FORMAT_KEYS}"
