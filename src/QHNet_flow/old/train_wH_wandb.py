#!/usr/bin/env python3
import os
import torch
import hydra
import logging
import shutil
import subprocess
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models import get_model
from torch_geometric.loader import DataLoader
from ori_dataset import MD17_DFT, random_split, get_mask
from torch_ema import ExponentialMovingAverage
from transformers import get_polynomial_decay_schedule_with_warmup

from pathlib import Path
import warnings


logger = logging.getLogger(__name__)


def criterion(outputs, target, loss_weights):
    error_dict = {}
    for key in loss_weights.keys():
        diff = outputs[key] - target[key]
        # diff = outputs[key] - target[key] + target.init_ham
        mse = torch.mean(diff**2)
        mae = torch.mean(torch.abs(diff))
        error_dict[key + "_mae"] = mae
        error_dict[key + "_rmse"] = torch.sqrt(mse)
        loss = mse + mae
        error_dict[key] = loss
        if "loss" in error_dict:
            error_dict["loss"] += loss_weights[key] * loss
        else:
            error_dict["loss"] = loss_weights[key] * loss
    return error_dict


def post_processing(batch, default_type):
    if "hamiltonian" in batch.keys:
        batch.hamiltonian = batch.hamiltonian.view(
            batch.hamiltonian.shape[0] // batch.hamiltonian.shape[1],
            batch.hamiltonian.shape[1],
            batch.hamiltonian.shape[1],
        )
    if "overlap" in batch.keys:
        batch.overlap = batch.overlap.view(
            batch.overlap.shape[0] // batch.overlap.shape[1],
            batch.overlap.shape[1],
            batch.overlap.shape[1],
        )
    for key in batch.keys:
        if torch.is_floating_point(batch[key]):
            batch[key] = batch[key].type(default_type)
    return batch


class LitModel(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.model = get_model(conf.model)
        # Set up the model on the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.set(device)
        self.loss_weights = {"hamiltonian": 1.0}
        self.default_type = (
            torch.float64 if conf.data_type == "float64" else torch.float32
        )
        torch.set_default_dtype(self.default_type)

        # Optional: set up EMA if enabled
        self.ema = None
        self.use_init_hamiltonian = conf.use_init_hamiltonian
        self.use_init_hamiltonian_residue = conf.use_init_hamiltonian_residue
        if conf.ema_start_epoch > -1:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.99)

    def forward(self, batch):
        if self.use_init_hamiltonian:
            if self.use_init_hamiltonian_residue:
                output = self.model(batch, batch.init_ham)
                output["hamiltonian"] = output["hamiltonian"] + batch.init_ham
                return output
            else:
                return self.model(batch, batch.init_ham)
        else:
            return self.model(batch)

    def training_step(self, batch, batch_idx):
        batch = post_processing(batch, self.default_type)
        outputs = self(batch)
        errors = criterion(outputs, batch, loss_weights=self.loss_weights)
        loss = errors["loss"]
        for key in errors.keys():
            self.log(
                f"train/{key}",
                errors[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        batch = post_processing(batch, self.default_type)
        with self.ema.average_parameters():
            ema_outputs = self(batch)
            ema_errors = criterion(ema_outputs, batch, loss_weights=self.loss_weights)
            ema_loss = ema_errors["loss"]
            for key in ema_errors.keys():
                self.log(
                    f"val_ema/{key}",
                    ema_errors[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
        outputs = self(batch)
        errors = criterion(outputs, batch, loss_weights=self.loss_weights)
        loss = errors["loss"]
        for key in errors.keys():
            self.log(
                f"val/{key}",
                errors[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return errors

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.conf.dataset.learning_rate,
            betas=(0.99, 0.999),
            amsgrad=False,
        )
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.conf.warmup_step,
            num_training_steps=self.conf.num_training_steps,
            lr_end=self.conf.end_lr,
            power=1.0,
            last_epoch=-1,
        )
        # The scheduler will be updated every training step
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA after each training batch if enabled and past the EMA start epoch.
        if self.ema is not None and self.current_epoch > self.conf.ema_start_epoch:
            self.ema.update()


@hydra.main(config_path="config", config_name="config")
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
    root_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-3]))
    logger.info(f"Loading {conf.dataset.dataset_name} dataset...")
    dataset = MD17_DFT(
        os.path.join(root_path, "dataset"),
        name=conf.dataset.dataset_name,
        transform=get_mask,
    )
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        [
            conf.dataset.num_train,
            conf.dataset.num_valid,
            len(dataset) - (conf.dataset.num_train + conf.dataset.num_valid),
        ],
        seed=conf.split_seed,
    )

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

    # Initialize the wandb logger.
    os.makedirs(output_dir / "wandb", exist_ok=True)
    wandb_logger = WandbLogger(
        project=conf.wandb_project,
        name=conf.wandb_run_name,
        save_dir=output_dir,
    )

    # Set up model checkpointing.
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{val_hamiltonian_mae:.8f}",
    )
    callbacks.append(checkpoint_callback)
    lr_callback = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_callback)

    # Initialize the LightningModule.
    lit_model = LitModel(conf)

    # Create the PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        max_steps=conf.num_training_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=conf.dataset.train_batch_interval,
        accelerator="auto",
        devices=1,
        enable_model_summary=True,
        enable_progress_bar=True,
        gradient_clip_val=(
            conf.dataset.clip_norm if conf.dataset.use_gradient_clipping else None
        ),
    )
    warnings.filterwarnings("ignore")
    # Start training.
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
