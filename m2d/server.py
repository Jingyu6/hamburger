import litserve as ls

from m2d.model.llama import M2DLlama


class M2DLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model: M2DLlama = M2DLlama.load_from_checkpoint(
            "./local/ckpts/m2d-llama-1B-step=18432.ckpt"
        ).to(device)

    def predict(self, conversation, context):
        max_gen_len = context.get("max_completion_tokens", None)
        if max_gen_len is None:
            max_gen_len = context.get("max_tokens", 128)
        yield self.model.generate(
            prompt=conversation[0]["content"], 
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
