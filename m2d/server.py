import litserve as ls

from m2d.model.llama import M2DLlama


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
        prompt = ""
        for turn in conversation:
            if turn["role"] in ["system", "user"]:
                prompt += turn['content']
        yield self.model.generate(
            prompt=prompt, 
            max_gen_len=max_gen_len,  
        )["output"]


if __name__ == "__main__":
    api = M2DLitAPI()
    server = ls.LitServer(
        api, 
        devices=1, 
        spec=ls.OpenAISpec()
    )
    server.run(port=8000)
