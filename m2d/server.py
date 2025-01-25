import litserve as ls

from m2d.model.llama import M2DLlama


class M2DLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model: M2DLlama = M2DLlama.load_from_checkpoint(
            "./local/ckpts/m2d-llama-1B-step=18432.ckpt"
        ).to(device)

    def predict(self, conversation, context):
        yield self.model.generate(
            prompt=conversation[0]["content"], 
            max_gen_len=context.get("max_completion_tokens", context.get("max_tokens", 128))
        )["output"][len("<|start_header_id|>assistant<|end_header_id|>\n\n"):]


if __name__ == "__main__":
    api = M2DLitAPI()
    server = ls.LitServer(
        api, 
        devices=1, 
        stream=False, 
        spec=ls.OpenAISpec()
    )
    server.run(port=8000)
