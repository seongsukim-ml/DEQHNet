import torch
import pytorch_lightning as pl
from models import get_model

# from src.QHNet_flow.utils import ExponentialMovingAverage, self.self.post_processing

from torch_ema import ExponentialMovingAverage
from transformers import get_polynomial_decay_schedule_with_warmup
import logging
import time
from tqdm import tqdm
from torch_scatter import scatter_sum


logger = logging.getLogger(__name__)


class LitModel(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        # Set up the model on the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_type = (
            torch.float64 if conf.data_type == "float64" else torch.float32
        )
        torch.set_default_dtype(self.default_type)

        self.loss_weights = conf.loss_weights
        # self.loss_weights = {"hamiltonian": 1.0}
        self.model = get_model(conf.model)
        self.model.set(device)

        # Optional: set up EMA if enabled
        self.ema = None
        self.use_init_hamiltonian = getattr(conf, "use_init_hamiltonian", False)
        self.use_init_hamiltonian_residue = getattr(
            conf, "use_init_hamiltonian_residue", False
        )
        self.ema_start_epoch = getattr(conf, "ema_start_epoch", -1)
        if self.ema_start_epoch > -1:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.99)

    @staticmethod
    def post_processing(batch, default_type):
        if "hamiltonian" in batch.keys:
            if batch.hamiltonian.dim() == 2:
                batch.hamiltonian = batch.hamiltonian.view(
                    batch.hamiltonian.shape[0] // batch.hamiltonian.shape[1],
                    batch.hamiltonian.shape[1],
                    batch.hamiltonian.shape[1],
                )
        if "overlap" in batch.keys:
            if batch.overlap.dim() == 2:
                batch.overlap = batch.overlap.view(
                    batch.overlap.shape[0] // batch.overlap.shape[1],
                    batch.overlap.shape[1],
                    batch.overlap.shape[1],
                )
        for key in batch.keys:
            if torch.is_floating_point(batch[key]):
                batch[key] = batch[key].type(default_type)
        return batch

    @staticmethod
    def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
        eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
        eps = 1e-8 * torch.ones_like(eigvals)
        eigvals = torch.where(eigvals > 1e-8, eigvals, eps)
        frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

        Fs = torch.bmm(
            torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap
        )
        orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
        orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
        return orbital_energies, orbital_coefficients

    # @staticmethod
    # def criterion(outputs, target, loss_weights):
    #     error_dict = {}
    #     if "waloss" in loss_weights.keys():
    #         energy, orb = LitModel.cal_orbital_and_energies(
    #             target.overlap, target.hamiltonian
    #         )
    #         target.orbital_energies = torch.diag_embed(energy).to(
    #             target.hamiltonian.device
    #         )
    #         target.orbital_coefficients = orb.to(target.hamiltonian.device)
    #     for key in loss_weights.keys():
    #         if key == "hamiltonian":
    #             diff = outputs[key] - target[key]
    #         elif key == "waloss":
    #             diff = outputs["hamiltonian"].bmm(target.orbital_coefficients)
    #             diff = torch.bmm(target.orbital_coefficients.transpose(-1, -2), diff)
    #             diff = diff - target.orbital_energies

    #         mse = torch.mean(diff**2)
    #         mae = torch.mean(torch.abs(diff))
    #         error_dict[key + "_mae"] = mae
    #         error_dict[key + "_rmse"] = torch.sqrt(mse)
    #         # loss = mse + mae
    #         loss = mse + mae
    #         error_dict[key] = loss
    #         if "loss" in error_dict:
    #             error_dict["loss"] += loss_weights[key] * loss
    #         else:
    #             error_dict["loss"] = loss_weights[key] * loss

    #     return error_dict

    @staticmethod
    def criterion(outputs, target, loss_weights):
        error_dict = {}
        keys = loss_weights.keys()
        try:
            for key in keys:
                row = target.edge_index[0]
                edge_batch = target.batch[row]
                diff_diagonal = (
                    outputs[f"{key}_diagonal_blocks"] - target[f"diagonal_{key}"]
                )
                mse_diagonal = torch.sum(
                    diff_diagonal**2 * target[f"diagonal_{key}_mask"], dim=[1, 2]
                )
                mae_diagonal = torch.sum(
                    torch.abs(diff_diagonal) * target[f"diagonal_{key}_mask"],
                    dim=[1, 2],
                )
                count_sum_diagonal = torch.sum(
                    target[f"diagonal_{key}_mask"], dim=[1, 2]
                )
                mse_diagonal = scatter_sum(mse_diagonal, target.batch)
                mae_diagonal = scatter_sum(mae_diagonal, target.batch)
                count_sum_diagonal = scatter_sum(count_sum_diagonal, target.batch)

                diff_non_diagonal = (
                    outputs[f"{key}_non_diagonal_blocks"]
                    - target[f"non_diagonal_{key}"]
                )
                mse_non_diagonal = torch.sum(
                    diff_non_diagonal**2 * target[f"non_diagonal_{key}_mask"],
                    dim=[1, 2],
                )
                mae_non_diagonal = torch.sum(
                    torch.abs(diff_non_diagonal) * target[f"non_diagonal_{key}_mask"],
                    dim=[1, 2],
                )
                count_sum_non_diagonal = torch.sum(
                    target[f"non_diagonal_{key}_mask"], dim=[1, 2]
                )
                mse_non_diagonal = scatter_sum(mse_non_diagonal, edge_batch)
                mae_non_diagonal = scatter_sum(mae_non_diagonal, edge_batch)
                count_sum_non_diagonal = scatter_sum(count_sum_non_diagonal, edge_batch)

                mae = (
                    (mae_diagonal + mae_non_diagonal)
                    / (count_sum_diagonal + count_sum_non_diagonal)
                ).mean()
                mse = (
                    (mse_diagonal + mse_non_diagonal)
                    / (count_sum_diagonal + count_sum_non_diagonal)
                ).mean()

                error_dict[key + "_mae"] = mae
                error_dict[key + "_rmse"] = torch.sqrt(mse)
                error_dict[key + "_diagonal_mae"] = (
                    mae_diagonal / count_sum_diagonal
                ).mean()
                error_dict[key + "_non_diagonal_mae"] = (
                    mae_non_diagonal / count_sum_non_diagonal
                ).mean()
                loss = mse + mae
                error_dict[key] = loss
                if "loss" in error_dict.keys():
                    error_dict["loss"] = error_dict["loss"] + loss_weights[key] * loss
                else:
                    error_dict["loss"] = loss_weights[key] * loss
        except Exception as exc:
            raise exc
        return error_dict

    def forward(self, batch, H=None, keep_blocks=True):
        if self.use_init_hamiltonian:
            output = self.model(batch, batch.init_ham, keep_blocks=keep_blocks)
        else:
            output = self.model(batch, keep_blocks=keep_blocks)
        if self.use_init_hamiltonian_residue:
            output["hamiltonian"] = output["hamiltonian"] + batch.init_ham
        return output

    def training_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        outputs = self(batch)
        errors = self.criterion(outputs, batch, loss_weights=self.loss_weights)
        loss = errors["loss"]
        for key in errors.keys():
            self.log(
                f"train/{key}",
                errors[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True if key == "loss" else False,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        with self.ema.average_parameters():
            ema_outputs = self(batch)
            ema_errors = self.criterion(
                ema_outputs, batch, loss_weights=self.loss_weights
            )
            ema_loss = ema_errors["loss"]
            for key in ema_errors.keys():
                self.log(
                    f"val_ema/{key}",
                    ema_errors[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True if key == "loss" else False,
                    sync_dist=True,
                )
            ema_orb_and_eng_error = self._orb_and_eng_error(ema_outputs, batch)
            for key in ema_orb_and_eng_error.keys():
                self.log(
                    f"val_ema/{key}",
                    ema_orb_and_eng_error[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True if key == "loss" else False,
                    sync_dist=True,
                )
        outputs = self(batch)
        errors = self.criterion(outputs, batch, loss_weights=self.loss_weights)
        loss = errors["loss"]
        for key in errors.keys():
            self.log(
                f"val/{key}",
                errors[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True if key == "loss" else False,
                sync_dist=True,
            )
        orb_and_eng_error = self._orb_and_eng_error(outputs, batch)
        for key in orb_and_eng_error.keys():
            self.log(
                f"val/{key}",
                orb_and_eng_error[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True if key == "loss" else False,
                sync_dist=True,
            )
        return errors

    def test_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        outputs = self(batch)
        errors = self.criterion(outputs, batch, loss_weights=self.loss_weights)
        loss = errors["loss"]
        for key in errors.keys():
            self.log(
                f"test/{key}",
                errors[key],
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        orb_and_eng_error = self._orb_and_eng_error(outputs, batch)
        for key in orb_and_eng_error.keys():
            self.log(
                f"test/{key}",
                orb_and_eng_error[key],
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return errors

    def _orb_and_eng_error(self, _outputs, _target):
        total_error_dict = {"total_items": 0}
        loss_weights = {
            "hamiltonian": 1.0,
            "orbital_energies": 1.0,
            "orbital_coefficients": 1.0,
        }
        outputs = _outputs
        target = _target.clone()

        for key in outputs.keys():
            if isinstance(outputs[key], torch.Tensor):
                outputs[key] = outputs[key].to("cpu")

        target = target.to("cpu")

        outputs["orbital_energies"], outputs["orbital_coefficients"] = (
            self.cal_orbital_and_energies(target["overlap"], outputs["hamiltonian"])
        )
        target.orbital_energies, target.orbital_coefficients = (
            self.cal_orbital_and_energies(target["overlap"], target["hamiltonian"])
        )

        num_orb = int(target.atoms[target.ptr[0] : target.ptr[1]].sum() / 2)
        (
            outputs["orbital_energies"],
            outputs["orbital_coefficients"],
            target.orbital_energies,
            target.orbital_coefficients,
        ) = (
            outputs["orbital_energies"][:, :num_orb],
            outputs["orbital_coefficients"][:, :, :num_orb],
            target.orbital_energies[:, :num_orb],
            target.orbital_coefficients[:, :, :num_orb],
        )
        error_dict = self.criterion_test(outputs, target, loss_weights)
        return {
            "orbital_energies": error_dict["orbital_energies"],
            "orbital_coefficients": error_dict["orbital_coefficients"],
            "sample_hamiltonian": error_dict["hamiltonian"],
        }

    def configure_optimizers(self):
        torch.set_default_dtype(self.default_type)
        if self.conf.optimizer.lower() == "AdamW".lower():
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=getattr(self.conf, "dataset", {}).get("learning_rate", 5e-4),
                betas=(0.99, 0.999),
                amsgrad=False,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.conf.optimizer} is not implemented."
            )
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=getattr(self.conf, "warmup_step", 1000),
            num_training_steps=getattr(self.conf, "num_training_steps", 200000),
            lr_end=getattr(self.conf, "end_lr", 1e-8),
            power=getattr(self.conf, "scheduler_power", 1.0),
            last_epoch=-1,
        )
        # The scheduler will be updated every training step
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA after each training batch if enabled and past the EMA start epoch.
        if self.ema is not None and self.current_epoch > self.ema_start_epoch:
            self.ema.update()

    @staticmethod
    def criterion_test(outputs, target, names):
        error_dict = {}
        for key in names:
            if key == "orbital_coefficients":
                error_dict[key] = (
                    torch.cosine_similarity(outputs[key], target[key], dim=1)
                    .abs()
                    .mean()
                )
            else:
                diff = outputs[key] - target[key]
                mae = torch.mean(torch.abs(diff))
                error_dict[key] = mae
        return error_dict

    @torch.no_grad()
    def test_over_dataset(self, test_data_loader, default_type):
        self.eval()
        total_error_dict = {"total_items": 0}
        loss_weights = {
            "hamiltonian": 1.0,
            "orbital_energies": 1.0,
            "orbital_coefficients": 1.0,
        }
        total_time = 0
        total_graph = 0
        # total_traj = []
        last_traj = []
        logger.info("num of test data: {}".format(len(test_data_loader)))
        for idx, batch in tqdm(enumerate(test_data_loader)):
            batch = self.post_processing(batch, default_type)
            batch = batch.to(self.model.device)
            tic = time.time()
            # ham = batch.hamiltonian.cpu()
            # outputs, traj, _ = self(batch)
            outputs = self(batch, batch.init_ham)

            last_traj.append(outputs["hamiltonian"])

            duration = time.time() - tic
            total_graph = total_graph + batch.ptr.shape[0] - 1
            total_time = duration + total_time
            for key in outputs.keys():
                if isinstance(outputs[key], torch.Tensor):
                    outputs[key] = outputs[key].to("cpu")

            batch = batch.to("cpu")
            outputs["orbital_energies"], outputs["orbital_coefficients"] = (
                self.cal_orbital_and_energies(batch["overlap"], outputs["hamiltonian"])
            )
            batch.orbital_energies, batch.orbital_coefficients = (
                self.cal_orbital_and_energies(batch["overlap"], batch["hamiltonian"])
            )
            num_orb = int(batch.atoms[batch.ptr[0] : batch.ptr[1]].sum() / 2)
            (
                outputs["orbital_energies"],
                outputs["orbital_coefficients"],
                batch.orbital_energies,
                batch.orbital_coefficients,
            ) = (
                outputs["orbital_energies"][:, :num_orb],
                outputs["orbital_coefficients"][:, :, :num_orb],
                batch.orbital_energies[:, :num_orb],
                batch.orbital_coefficients[:, :, :num_orb],
            )
            error_dict = self.criterion_test(outputs, batch, loss_weights)
            secs = duration / batch.hamiltonian.shape[0]
            msg = f"batch {idx} / {secs*100:.2f}(10^-2)s : "
            for key in error_dict.keys():
                if key == "hamiltonian" or key == "orbital_energies":
                    msg += f"{key}: {error_dict[key]*1e6:.3f}(10^-6), "
                elif key == "orbital_coefficients":
                    msg += f"{key}: {error_dict[key]*1e2:.4f}(10^-2)"
                else:
                    msg += f"{key}: {error_dict[key]:.8f}, "

                if key in total_error_dict.keys():
                    total_error_dict[key] += (
                        error_dict[key].item() * batch.hamiltonian.shape[0]
                    )
                else:
                    total_error_dict[key] = (
                        error_dict[key].item() * batch.hamiltonian.shape[0]
                    )
            logger.info(msg)
            total_error_dict["total_items"] += batch.hamiltonian.shape[0]
        for key in total_error_dict.keys():
            if key != "total_items":
                total_error_dict[key] = (
                    total_error_dict[key] / total_error_dict["total_items"]
                )
        last_traj = torch.cat(last_traj, dim=0)
        return total_error_dict, last_traj
