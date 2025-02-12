import os

import litserve as ls

from m2d.config import GenConfig
from m2d.model.llama import M2DLlama

GEN_CONFIG = GenConfig.from_path(
    os.environ.get("GEN_CONFIG_PATH", None)
)

class M2DLitAPI(ls.LitAPI):
    def setup(self, device):
        print(f"Using device={device}")
        self.model: M2DLlama = M2DLlama.load_from_checkpoint(
            "./local/ckpts/m2d-llama-1B-finish.ckpt"
        ).to(device)

    def predict(self, conversation, context):
        max_gen_len = context.get("max_completion_tokens", None)
        if max_gen_len is None:
            max_gen_len = context.get("max_tokens", 128)
        GEN_CONFIG.max_gen_len = max_gen_len

        # filter non essential fields
        formatted_conversation = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in conversation
        ]

        yield self.model.generate(
            conversation=formatted_conversation, 
            config=GEN_CONFIG
        )["output"]


if __name__ == "__main__":
    api = M2DLitAPI()
    server = ls.LitServer(
        api, 
        devices=1, 
        spec=ls.OpenAISpec()
    )
    server.run(port=8000)
