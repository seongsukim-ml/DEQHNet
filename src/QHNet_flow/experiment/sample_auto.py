#!/usr/bin/env python3
import os
import hydra.conf
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
import sys

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# config_path = [cp for cp in sys.argv if "job_path" in cp][0][len("job_path=") :]
# overrides: list[str] = list(OmegaConf.load(Path(config_path) / "overrides.yaml"))
# logger.info(f"Overrides: {overrides}")
# sys.argv += overrides
# sys.argv += [
#     "hydra.run.dir: outputs/${dataset.dataset_name}/${model.version}${prefix}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
# ]


# config_path = [
#     path["path"] for path in cfg.runtime.config_sources if path["schema"] == "file"
# ][0]
@hydra.main(config_path="../config_v3", config_name="config")
def main(conf):
    # Copy the auxiliary basis file to the output directory.
    cfg = HydraConfig.get()
    config_name = cfg.job.config_name

    job_path = conf.job_path
    logger.info(f"Config name: {config_name}")
    logger.info(f"Config path: {job_path}")
    conf = hydra.compose(
        config_name=config_name,  # same config_name as used by @hydra.main
        overrides=OmegaConf.load(Path(job_path) / ".hydra" / "overrides.yaml"),
    )
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    shutil.copy(os.path.join(os.path.dirname(__file__), "auxiliary.gbs"), output_dir)
    logger.info(f"Output directory: {output_dir}")
    logger.info("Copied auxiliary basis file")
    # root_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-5]))
    root_path = "/home/seongsukim/dft/DEQHNet/src/QHNet_flow"
    logger.info(f"Root path: {root_path}")

    logger.info(f"Config: {conf}")

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
    if getattr(conf, "model_path", None) is None or conf.model_path == "":
        logger.info("Model path is not provided")
        ckpt_list = list(Path(job_path).parent.rglob("*.ckpt"))
        if len(ckpt_list) == 0:
            logger.info("No checkpoint found")
            raise ValueError
        else:
            if getattr(conf, "use_last", False):
                ckpt_list = [ckpt for ckpt in ckpt_list if "last" in ckpt.name]
                if len(ckpt_list) == 0:
                    logger.info("No last checkpoint found")
                    raise ValueError
                else:
                    model_path = ckpt_list[0]
            else:
                best_ckpt_list = [ckpt for ckpt in ckpt_list if "best" in ckpt.name]
                if len(best_ckpt_list) == 0:
                    logger.info("No best checkpoint found")
                    model_path = ckpt_list[0]
                else:
                    best_ckpt_list.sort(
                        key=lambda x: int(x.name.split("=")[-1].split(".")[0])
                    )
                    model_path = best_ckpt_list[-1]
    elif not os.path.exists(conf.model_path):
        logger.info("Model path does not exist")
        raise ValueError
    else:
        model_path = conf.model_path
    logger.info(f"Loading model from {model_path}")

    pl_model_cls = get_pl_model(conf)
    lit_model = pl_model_cls.load_from_checkpoint(model_path, conf=conf)
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
