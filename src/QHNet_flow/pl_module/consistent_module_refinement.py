import torch

# import pytorch_lightning as pl
# from models import get_model

# from src.QHNet_flow.utils import ExponentialMovingAverage, self.post_processing

# from torch_ema import ExponentialMovingAverage
# from transformers import get_polynomial_decay_schedule_with_warmup
from pl_module.base_module import LitModel
from torch_geometric.data import Batch
from utils import AOData
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

default_loss_weights = {
    "consistent_mae": 3.0,
    "consistent_mse": 3.0,
    "hamiltonian_mae": 1.0,
    "hamiltonian_mse": 1.0,
}

# loss_type: 1 -> consistent loss divide by step size
# loss_type: 2 -> consistent loss without divide by step size


class LitModel_consistent_refinement(LitModel):
    def __init__(self, conf, inf_model_pl):
        super().__init__(conf=conf)
        self.max_T = conf.consistent.get("max_T", 50)
        self.loss_weights = conf.get("loss_weights", default_loss_weights)
        self.error_threshold = conf.consistent.get("error_threshold", 1)
        self.batch_mul = conf.consistent.get("batch_mul", 1)
        # self.num_ode_steps_inf = conf.consistent.get("num_ode_steps_inf", self.max_T)
        self.num_ode_steps_inf = conf.consistent.get("num_ode_steps_inf", 1)
        self.loss_type = conf.consistent.get("loss_type", 1)
        self.inf_model_pl = inf_model_pl
        for name, param in self.inf_model_pl.named_parameters():
            param.requires_grad = False
        self.train_noise_scale = conf.refinement.get("train_noise_scale", 0)
        self.start_scf_idx = conf.refinement.get("start_scf_idx", 0)
        self.low_t = conf.refinement.get("low_t", 0.5)

    @staticmethod
    def batch_repeat(batch, mul=1, default_type=torch.float64, repeat_style="repeat"):
        assert repeat_style in ["append", "repeat"]
        if mul == 1:
            return batch
        batch_list = []
        for idx in range(batch.num_graphs):
            bb = batch.batch
            pos = batch.pos[bb == idx]
            atoms = batch.atoms[bb == idx]
            forces = batch.force[bb == idx]
            energy = batch.energy[idx]
            overlap = batch.overlap[idx].unsqueeze(0)
            hamiltonian = batch.hamiltonian[idx].unsqueeze(0)
            init_ham = batch.init_ham[idx].unsqueeze(0)
            mask_row = batch.mask_row[bb == idx]

            len_orb = batch.hamiltonian.shape[-1]
            AO_index = batch.AO_index[:, idx * len_orb : (idx + 1) * len_orb]
            AO_index[0] -= batch.ptr[idx]
            AO_index[2] -= idx
            Q = batch.Q[idx * len_orb : (idx + 1) * len_orb]

            hamiltonian_sch = batch.hamiltonian_sch[idx].unsqueeze(0)
            hamiltonian_traj = batch.hamiltonian_traj[batch.cycle_batch == idx]
            cycle = batch.cycle[idx]

            data = AOData(
                pos=pos,
                atoms=atoms,
                force=forces,
                energy=energy,
                overlap=overlap,
                hamiltonian=hamiltonian,
                init_ham=init_ham,
                AO_index=AO_index,
                Q=Q,
                mask_row=mask_row,
                hamiltonian_sch=hamiltonian_sch,
                hamiltonian_traj=hamiltonian_traj,
                cycle=cycle,
            )
            if repeat_style == "repeat":
                for _ in range(mul):
                    batch_list.append(data.clone())
            else:
                batch_list.append(data.clone())

        if repeat_style == "append":
            new_batch_list = []
            for _ in range(mul):
                new_batch_list += batch_list
            batch_list = new_batch_list

        return LitModel_consistent_refinement.post_processing(
            Batch.from_data_list(batch_list), default_type
        )

    @staticmethod
    def post_processing(batch, default_type, start_scf_idx=0):
        device = batch.hamiltonian.device
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
        if "cycle" in batch.keys:
            batch.cycle_batch = torch.repeat_interleave(
                torch.arange(batch.hamiltonian.shape[0]).to(device), batch.cycle
            )
            batch.cycle_ptr = torch.cat(
                [torch.tensor([0]).to(device), torch.cumsum(batch.cycle, dim=0)], dim=0
            )
            assert len(batch.cycle_ptr) == batch.hamiltonian.shape[0] + 1
            
        if start_scf_idx > 0:
            batch.init_ham = batch.hamiltonian_traj[batch.cycle_ptr[:-1]+start_scf_idx]
        # for key in batch.keys:
        #     if torch.is_floating_point(batch[key]):
        #         batch[key] = batch[key].type(default_type)
        return batch

    @staticmethod
    def sample_t(num_batch, device, low=0, high=1):
        t_batch = torch.rand(num_batch).to(device)
        t_batch = t_batch * (high - low) + low
        return t_batch

    def consistent_sample(self, batch, mul=1, low=0.25, high=1):
        batch = self.batch_repeat(batch, mul)
        device = batch.hamiltonian.device
        t_batch = self.sample_t(batch.hamiltonian.shape[0], device, low, high)
        t_int = (t_batch * batch.cycle).long() # t_int in [0, max_T - 1], not max_T
        
        batch.t_int = t_int  # t_int in [0, max_T - 1], not max_T
        batch.t = t_batch.double()  # t in [0, 1)
        batch.hamiltonian_t = batch.hamiltonian_traj[t_int]

        return batch
    
    @staticmethod
    def criterion_refine(outputs_dict, target, loss_weights, loss_type=1):
        error_dict = {"loss": 0}
        diff = {}
        dict_keys = {
            "hamiltonian": "K",
            "refinement": "R",
        }

        if "K" in outputs_dict.keys():
            ham_K = outputs_dict["K"]["hamiltonian"]
            diff["hamiltonian"] = ham_K - (
                target["hamiltonian"] - target["hamiltonian_t"]
            )

        if "R" in outputs_dict.keys():
            ham_R = outputs_dict["R"]["hamiltonian"]
            diff["refinement"] = ham_R - (target["hamiltonian"] - target["inf"])

        for key in diff.keys():
            if dict_keys[key] in outputs_dict.keys():
                mse = torch.mean(diff[key] ** 2)
                mae = torch.mean(torch.abs(diff[key]))
                error_dict[key + "_mae"] = mae
                error_dict[key + "_rmse"] = torch.sqrt(mse)

                cur_loss = 0
                if key + "_mae" in loss_weights.keys():
                    cur_loss = cur_loss + loss_weights[key + "_mae"] * mae
                if key + "_mse" in loss_weights.keys():
                    cur_loss = cur_loss + loss_weights[key + "_mse"] * mse
                error_dict[key] = cur_loss
                error_dict["loss"] += cur_loss

        for key in diff.keys():
            if dict_keys[key] in outputs_dict.keys():
                for _bin in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
                    s, e = _bin
                    if key == "hamiltonian":
                        mask = (target.t >= s) & (target.t < e)
                    cur_diff = diff[key][mask]
                    if cur_diff.numel() == 0:
                        cur_diff = torch.tensor([0.0]).to(target.hamiltonian.device)
                    mse = torch.mean(cur_diff**2)
                    mae = torch.mean(torch.abs(cur_diff))
                    error_dict[key + f"_mae@{s:.2f}_{e:.2f}"] = mae
                    error_dict[key + f"_rmse@{s:.2f}_{e:.2f}"] = torch.sqrt(mse)

        return error_dict

    def forward(self, batch, cur_H):
        output = self.model(batch, cur_H)
        return output

    def training_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type, self.start_scf_idx)
        batch = self.consistent_sample(batch, self.batch_mul, self.low_t)
        outputs = self.forward(batch, batch.hamiltonian_t)
        with torch.no_grad():
            batch.inf = self.inf_model_pl(batch)["hamiltonian"]
        if self.train_noise_scale > 0:
            noise = torch.randn_like(batch.inf) * self.train_noise_scale
            batch.inf = batch.inf + noise
        outputs_R = self.forward(batch, batch.inf)

        errors = self.criterion_refine(
            {"K": outputs, "R": outputs_R},
            batch,
            loss_weights=self.loss_weights,
            loss_type=self.loss_type,
        )
        loss = errors["loss"]
        self._log_error(errors, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type, self.start_scf_idx)
        batch_one = batch.clone()
        batch = self.consistent_sample(batch, self.batch_mul, self.low_t)
        outputs = self.forward(batch, batch.hamiltonian_t)
        batch.inf = self.inf_model_pl(batch)["hamiltonian"]
        outputs_R = self.forward(batch, batch.inf)

        errors = self.criterion_refine(
            {"K": outputs, "R": outputs_R},
            batch,
            loss_weights=self.loss_weights,
            loss_type=self.loss_type,
        )
        loss = errors["loss"]
        self._log_error(errors, "val")

        # fmt: off
        if loss < self.error_threshold:
            # Short trajectory
            self._log_sample_error(
                batch_one,
                prefix="val",
            )
            self._log_sample_error(
                batch_one,
                prefix="val",
                sample_type="full",
                post_fix="full",
            )

        # fmt: on
        return errors

    def test_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type, self.start_scf_idx)
        batch_one = batch.clone()
        batch = self.consistent_sample(batch, self.batch_mul)
        outputs = self.forward(batch, batch.hamiltonian_t)
        batch.inf = self.inf_model_pl(batch)["hamiltonian"]
        outputs_R = self.forward(batch, batch.inf)

        errors = self.criterion_refine(
            {"K": outputs, "R": outputs_R},
            batch,
            loss_weights=self.loss_weights,
            loss_type=self.loss_type,
        )
        loss = errors["loss"]
        self._log_error(errors, "test")
        # fmt: off
        if loss < self.error_threshold:
            self._log_sample_error(
                batch_one,
                prefix="test",
            )
            self._log_sample_error(
                batch_one,
                prefix="test",
                sample_type="full",
                post_fix="full",
            )
        # fmt: on
        return errors

    def sample(self, batch, type="delta"):
        assert type in ["delta", "full"]
        if type == "delta":
            output_1 = self.inf_model_pl(batch)
            output_2 = self.model(batch, output_1["hamiltonian"])
            res_outputs = {
                "hamiltonian": output_2["hamiltonian"] + output_1["hamiltonian"]
            }

        if type == "full":
            output = self.model(batch, batch.init_ham)
            res_outputs = output

        return res_outputs

    def _log_error(self, errors, prefix):
        for key in errors.keys():
            if "@" in key:
                _key, _time_bin = key.split("@")[0], key.split("@")[1]
                self.log(
                    f"{prefix}_{_time_bin}/{_key}_{_time_bin}",
                    errors[key],
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
            else:
                self.log(
                    f"{prefix}/{key}",
                    errors[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True if key == "loss" else False,
                    sync_dist=True,
                )

    def _log_sample_error(self, batch_one, prefix, post_fix="", sample_type="delta"):
        try:
            sample = self.sample(batch_one, sample_type)
            orb_and_eng_error = self._orb_and_eng_error(sample, batch_one)
            if post_fix != "" and post_fix is not None:
                post_fix = f"_{post_fix}"
            else:
                post_fix = ""
            for key in orb_and_eng_error.keys():
                self.log(
                    f"{prefix}/{key}{post_fix}",
                    orb_and_eng_error[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True if key == "loss" else False,
                    sync_dist=True,
                )
        except Exception as e:
            logger.error(f"Error in logging sample error: {e}")