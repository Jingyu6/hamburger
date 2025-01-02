import lightning as L
from transformers import AutoTokenizer, LlamaModel

from m2d.model.m2d_modules import CompositionalEmbedder, MicroStepDecoder


class M2DLlama(L.LightningModule):
    def __init__(
        self, 
        base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
        max_steps: int = 4
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model: LlamaModel = LlamaModel.from_pretrained(base_model_name)
        self.comp_embedder = CompositionalEmbedder(
            self.base_model.embed_tokens, 
            max_steps
        )
        self.micro_step_decoder = MicroStepDecoder()

    def forward(self, **inputs):
        pass

    def training_step(self, batch, batch_idx):
        print(batch)
        exit()

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass


if __name__ == "__main__":
    model = M2DLlama()
