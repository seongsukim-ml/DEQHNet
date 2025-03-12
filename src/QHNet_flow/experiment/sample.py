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

from models import get_model, get_pl_model
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from ori_dataset import MD17_DFT, random_split, get_mask
from torch_ema import ExponentialMovingAverage
from transformers import get_polynomial_decay_schedule_with_warmup

# import copy
from pathlib import Path
import warnings
from tqdm import tqdm
import time


logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="config")
def main(conf):
    # Copy the auxiliary basis file to the output directory.
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    shutil.copy(os.path.join(os.path.dirname(__file__), "auxiliary.gbs"), output_dir)
    logger.info(f"Output directory: {output_dir}")
    logger.info("Copied auxiliary basis file")
    root_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-5]))
    logger.info(f"Root path: {root_path}")

    cmd = subprocess.Popen("set basis AUXILIARY", shell=True)
    cmd.wait()
    logger.info("Set auxiliary basis")

    # Set the default tensor type and seed.
    default_type = torch.float64 if conf.data_type == "float64" else torch.float32
    torch.set_default_dtype(default_type)
    pl.seed_everything(0)

    # Load the dataset.
    logger.info(f"Loading {conf.dataset.dataset_name} dataset...")
    dataset = MD17_DFT(
        os.path.join(root_path, "dataset"),
        name=conf.dataset.dataset_name,
        transform=get_mask,
    )
    logger.info(f"dataset loaded..")
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        [
            conf.dataset.num_train,
            conf.dataset.num_valid,
            len(dataset) - (conf.dataset.num_train + conf.dataset.num_valid),
        ],
        seed=conf.split_seed,
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=conf.dataset.test_batch_size,
        shuffle=False,
        num_workers=conf.dataset.num_workers,
        pin_memory=conf.dataset.pin_memory,
    )
    logger.info(f"split dataset")
    # Initialize the wandb logger.
    output_dir_name = f"output"
    os.makedirs(output_dir / output_dir_name, exist_ok=True)
    logger.info(f"Output directory: {output_dir / output_dir_name}")

    # Initialize the LightningModule.
    if conf.model_path is None or conf.model_path == "":
        logger.info("Model path is not provided")
        raise ValueError
    if not os.path.exists(conf.model_path):
        logger.info("Model path does not exist")
        raise ValueError
    logger.info(f"Loading model from {conf.model_path}")

    pl_model_cls = get_pl_model(conf)
    lit_model = pl_model_cls.load_from_checkpoint(conf.model_path, conf=conf)
    logger.info("Model loaded")
    # Create the PyTorch Lightning Trainer.
    warnings.filterwarnings("ignore")
    logger.info("Testing...")
    errors, h_output = lit_model.test_over_dataset(test_data_loader, default_type)
    msg = f"dataset {conf.dataset.dataset_name}: {errors.get('total_items')} :"

    for key in errors.keys():
        if key == "hamiltonian" or key == "orbital_energies":
            msg += f"{key}: {errors[key]*1e6:.3f}(10^-6), "
        elif key == "orbital_coefficients":
            msg += f"{key}: {errors[key]*1e2:.4f}(10^-2)"
        else:
            msg += f"{key}: {errors[key]:.8f}, "
    logger.info(msg)
    with open(output_dir / output_dir_name / "results.txt", "w") as f:
        f.write(msg)
    torch.save(h_output, output_dir / output_dir_name / "h_output.pt")


if __name__ == "__main__":
    main()
