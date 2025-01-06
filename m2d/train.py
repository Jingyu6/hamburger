import lightning as L

from m2d.data.m2d_data import M2DDataModule
from m2d.model.llama import M2DLlama

# create model
model = M2DLlama()

# prepare data
data_module = M2DDataModule.from_hf_dataset(
    save_path="./local/processed_openorca", 
    test_ratio=0.1, 
    batch_size=4, 
)

# create trainer
trainer = L.Trainer(
    max_epochs=1, 
    gradient_clip_val=1.0, 
    accumulate_grad_batches=2
)

# start training
trainer.fit(
    model=model, 
    datamodule=data_module
)
