import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from m2d.data.m2d_data import M2DDataModule
from m2d.model.llama import M2DLlama

# create model
model = M2DLlama()

# prepare data
data_module = M2DDataModule.from_hf_dataset(
    save_path="./local/processed_openorca", 
    test_ratio=0.1, 
    batch_size=2, 
)

# checkpoint callbacks
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="step",
    mode="max",
    every_n_train_steps=1024, 
    dirpath="./local/ckpts",
    filename="m2d-llama-1B-{global_step}",
)

# lr_monitor_callback = LearningRateMonitor(
#     logging_interval="step"
# )

# logger
wandb_logger = WandbLogger(
    project="m2d", 
    log_model="all"
)

# create trainer
trainer = L.Trainer(
    strategy="deepspeed", 
    max_epochs=1, 
    gradient_clip_val=1.0, 
    accumulate_grad_batches=4, 
    callbacks=[
        checkpoint_callback, 
        # lr_monitor_callback
    ], 
    logger=wandb_logger
)

# start training
trainer.fit(
    model=model, 
    datamodule=data_module
)
