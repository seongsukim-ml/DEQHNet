#!/usr/bin/env python3
import os
import sys
import torch
import hydra
import logging
import shutil
import subprocess
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch_geometric.loader import DataLoader

# from ori_dataset_traj import MD17_DFT_trajectory, random_split, get_mask
from qh9_datasets_split import QH9Stable, QH9Dynamic

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Insert the parent directory at the beginning of sys.path
sys.path.insert(0, parent_dir)

from models import get_model, get_pl_model


# import copy
from pathlib import Path
import warnings
import omegaconf

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config_qh9", config_name="config")
def main(conf):
    # Copy the auxiliary basis file to the output directory.
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    shutil.copy(os.path.join(os.path.dirname(__file__), "auxiliary.gbs"), output_dir)
    logger.info("Copied auxiliary basis file")
    cmd = subprocess.Popen("set basis AUXILIARY", shell=True)
    cmd.wait()
    logger.info("Set auxiliary basis")

    # Set the default tensor type and seed.
    default_type = torch.float64 if conf.data_type == "float64" else torch.float32
    torch.set_default_dtype(default_type)

    pl.seed_everything(0)

    # Load the dataset.
    # root_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-5]))
    root_path = "/home/seongsukim/dft/DEQHNet/src/QHNet_flow"

    logger.info(f"Root path: {root_path}")
    logger.info(f"Loading {conf.dataset.dataset_name} dataset...")
    # dataset = MD17_DFT_trajectory(
    #     os.path.join(root_path, "dataset"),
    #     name=conf.dataset.dataset_name,
    #     transform=get_mask,
    #     prefix=conf.dataset.prefix,
    # )
    if conf.dataset.dataset_name == "QH9Stable":
        dataset = QH9Stable(
            os.path.join(root_path, "dataset"),
            split=conf.dataset.split,
        )
    elif conf.dataset.dataset_name == "QH9Dynamic":
        dataset = QH9Dynamic(
            os.path.join(root_path, "dataset"),
            split=conf.dataset.split,
            version=conf.dataset.version,
        )
    train_dataset = dataset[dataset.train_mask]
    valid_dataset = dataset[dataset.val_mask]
    test_dataset = dataset[dataset.test_mask]

    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.dataset.train_batch_size,
        shuffle=True,
        num_workers=conf.dataset.num_workers,
        pin_memory=conf.dataset.pin_memory,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=conf.dataset.train_batch_size,
        shuffle=False,
        num_workers=conf.dataset.num_workers,
        pin_memory=conf.dataset.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=conf.dataset.test_batch_size,
        shuffle=False,
        num_workers=conf.dataset.num_workers,
        pin_memory=conf.dataset.pin_memory,
    )

    # Initialize the wandb logger.
    os.makedirs(output_dir / "wandb", exist_ok=True)

    wandb_logger = WandbLogger(
        project=conf.wandb.project,
        name=conf.wandb.run_name,
        save_dir=output_dir,
        mode=getattr(conf.wandb, "mode", "online"),
        id=conf.wandb.run_id,
        tags=getattr(conf.wandb, "tags", None),
    )
    wandb_config = omegaconf.OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )
    wandb_logger.log_hyperparams(wandb_config)

    # Set up model checkpointing.
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best-{epoch:02d}",
        )
    )
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Initialize the LightningModule.
    pl_model_cls = get_pl_model(conf)
    logger.info(f"Using model: {pl_model_cls}")
    lit_model = pl_model_cls(conf)

    wandb_logger.watch(
        model=lit_model,
        log_freq=500,
    )

    # Create the PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        max_steps=conf.num_training_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=64 if conf.data_type == "float64" else 32,
        log_every_n_steps=conf.dataset.train_batch_interval,
        accelerator="auto",
        devices=1,
        # val_check_interval=conf.dataset.validation_interval,
        check_val_every_n_epoch=conf.check_val_every_n_epoch,
        enable_progress_bar=True,
        gradient_clip_val=5.0,
    )
    warnings.filterwarnings("ignore")
    # Start training.
    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=conf.continune_ckpt,
    )
    logger.info("Testing...")
    trainer.test(lit_model, test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
