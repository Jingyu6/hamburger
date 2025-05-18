import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from hamburger.config import HAMburgerConfig
from hamburger.data.hamburger_data import HAMburgerDataModule
from hamburger.model.llama import HAMburgerLlama

# make sure the huge cache file is not in /home
os.environ["WANDB_CACHE_DIR"] = "/data/data_persistent1/jingyu/wandb_cache"

config = HAMburgerConfig.from_path("./train.yaml")
config.print_config()

L.seed_everything(config.seed)

# create model
if config.pretrained_ckpt_path is not None:
    print("Starting from pretrained ckpt...")
    model = HAMburgerLlama.load_from_checkpoint(
        config.pretrained_ckpt_path, map_location="cpu")
else:
    print("Starting from scratch...")
    model = HAMburgerLlama(base_model_name=config.base_model_name)

# prepare data
data_module = HAMburgerDataModule(
    save_path=config.dataset_names, 
    test_ratio=config.test_ratio, 
    batch_size=config.batch_size, 
)
data_module.get_data_summary()

# checkpoint callbacks
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="step",
    mode="max",
    every_n_train_steps=2048, 
    dirpath="/data/data_persistent1/jingyu/hamburger/ckpts",
    filename="hamburger-llama" + f"-{config.run_name}-" + "{step}",
)

# logger
wandb_logger = WandbLogger(
    name=config.run_name if config.run_name != "" else None, 
    project="hamburger", 
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

trainer.save_checkpoint(
    f"/data/data_persistent1/jingyu/hamburger/ckpts/hamburger-llama-{config.run_name}-finish.ckpt")
