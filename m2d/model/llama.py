import lightning as L
from transformers import LlamaModel


class M2DLlama(L.LightningModule):
    def __init__(
        self, 
        base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.base_model = LlamaModel.from_pretrained(base_model_name)

    def forward(self, **inputs):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass


if __name__ == "__main__":
    model = M2DLlama()
