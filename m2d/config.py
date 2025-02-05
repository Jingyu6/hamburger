import json
from dataclasses import asdict, dataclass, fields
from typing import List, Optional

import yaml


@dataclass
class M2DConfig: 
    base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    ckpt_path: Optional[str] = None
    dataset_names: List[str] = None
    batch_size: int = 8
    test_ratio: float = 0.005

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
    
    def __post_init__(self):
        assert self.dataset_names is not None

    def print_config(self):
        print("\033[92m{}\033[00m".format(
            f"Using spec config:\n{json.dumps(asdict(self), indent=4)}"))
    