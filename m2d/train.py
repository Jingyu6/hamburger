import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from m2d.config import M2DConfig
from m2d.data.m2d_data import M2DDataModule
from m2d.model.llama import M2DLlama

L.seed_everything(227)

config = M2DConfig.from_path("./local/config.yaml")
config.print_config()

# create model
if config.pretrained_ckpt_path is not None:
    print("Starting from pretrained ckpt...")
    model = M2DLlama.load_from_checkpoint(config.pretrained_ckpt_path, map_location="cpu")
else:
    print("Starting from scratch...")
    model = M2DLlama(base_model_name=config.base_model_name)

# prepare data
data_module = M2DDataModule(
    save_path=config.dataset_names, 
    test_ratio=config.test_ratio, 
    batch_size=config.batch_size, 
)

# checkpoint callbacks
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="step",
    mode="max",
    every_n_train_steps=1024, 
    dirpath="./local/ckpts",
    filename="m2d-llama-1B-{step}",
)

# logger
wandb_logger = WandbLogger(
    project="m2d", 
    log_model="all"
)

# This can take lots of disk space
# wandb_logger.watch(model, log="all")

# create trainer
trainer = L.Trainer(
    strategy=config.strategy, 
    max_epochs=1, 
    gradient_clip_val=1.0, 
    accumulate_grad_batches=config.accumulate_grad_batches, 
    num_sanity_val_steps=0, 
    precision="bf16", 
    callbacks=[checkpoint_callback], 
    val_check_interval=512, 
    logger=wandb_logger
)

# start training
trainer.fit(
    model=model, 
    datamodule=data_module, 
    ckpt_path=config.resume_ckpt_path
)

trainer.save_checkpoint("./local/ckpts/m2d-llama-1B-finish.ckpt")
