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


class LitModel_consistent_t(LitModel):
    def __init__(self, conf):
        super().__init__(conf=conf)
        self.max_T = conf.consistent.get("max_T", 50)
        self.loss_weights = conf.get("loss_weights", default_loss_weights)
        self.error_threshold = conf.consistent.get("error_threshold", 1)
        self.batch_mul = conf.consistent.get("batch_mul", 1)
        # self.num_ode_steps_inf = conf.consistent.get("num_ode_steps_inf", self.max_T)
        self.num_ode_steps_inf = conf.consistent.get("num_ode_steps_inf", 1)
        self.loss_type = conf.consistent.get("loss_type", 1)

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

        return LitModel_consistent_t.post_processing(
            Batch.from_data_list(batch_list), default_type
        )

    @staticmethod
    def post_processing(batch, default_type):
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
        # for key in batch.keys:
        #     if torch.is_floating_point(batch[key]):
        #         batch[key] = batch[key].type(default_type)
        return batch

    @staticmethod
    def sample_t_and_h(num_batch, max_T, device):
        # Sample t in [0, max_T)
        t_batch = torch.randint(low=0, high=(max_T), size=(num_batch,)).to(device)
        max_h = max_T - t_batch
        # Sample u in [0,1) for each element
        u = torch.rand(num_batch).to(device)
        # Sample h in [1, max_h]
        h_batch = (u * (max_h)).floor().long() + 1

        assert t_batch.max() < max_T
        assert (t_batch + h_batch).max() <= max_T
        assert (t_batch + h_batch).min() >= 1
        assert h_batch.max() <= max_h.max()
        assert h_batch.min() >= 1

        return t_batch, h_batch

    def consistent_sample(self, batch, mul=1):
        batch = self.batch_repeat(batch, mul)
        device = batch.hamiltonian.device
        t_batch, h_batch = self.sample_t_and_h(
            batch.hamiltonian.shape[0], self.max_T, device
        )

        batch.t_int = t_batch  # t_int in [0, max_T - 1], not max_T
        batch.h_int = h_batch  # h_int in [1, max_T - t], t_int + h_int <= max_T
        batch.t = t_batch.double() / (self.max_T)  # t in [0, 1)
        batch.h = h_batch.double() / (self.max_T)  # h in [0, 1]
        batch.t_0 = torch.zeros_like(batch.t).to(device).double()
        batch.K = torch.ones_like(batch.h).to(device).double()

        t_clip = t_batch.clone()
        t_h_clip = (t_batch + h_batch).clone()

        # Clip the t and h to the t in [0, max_T - 1] and h in [1, max_T - t]
        t_clip[t_batch >= batch.cycle] = batch.cycle[t_batch >= batch.cycle] - 1
        t_h_clip[t_h_clip >= batch.cycle] = batch.cycle[t_h_clip >= batch.cycle] - 1

        t_clip = t_clip + batch.cycle_ptr[:-1]
        t_h_clip = t_h_clip + batch.cycle_ptr[:-1]
        assert t_clip.max() < batch.hamiltonian_traj.shape[0]
        assert t_h_clip.max() < batch.hamiltonian_traj.shape[0]

        batch.hamiltonian_t = batch.hamiltonian_traj[t_clip]
        batch.hamiltonian_t_h = batch.hamiltonian_traj[t_h_clip]

        return batch

    @staticmethod
    def criterion_consistent(outputs_dict, target, loss_weights, loss_type=1):
        error_dict = {"loss": 0}
        diff = {}
        dict_keys = {
            "hamiltonian": "K",
            "consistent": "H",
        }

        if "H" in outputs_dict.keys():
            ham_h = outputs_dict["H"]["hamiltonian"]
            h = target["h"].reshape(-1, 1, 1)
            if loss_type == 1:
                diff["consistent"] = (
                    ham_h - (target["hamiltonian_t_h"] - target["hamiltonian_t"]) / h
                )
            elif loss_type == 2:
                diff["consistent"] = ham_h - (
                    target["hamiltonian_t_h"] - target["hamiltonian_t"]
                )
        if "K" in outputs_dict.keys():
            ham_K = outputs_dict["K"]["hamiltonian"]
            K = target["K"].reshape(-1, 1, 1)
            diff["hamiltonian"] = (
                ham_K - (target["hamiltonian"] - target["init_ham"]) / K
            )

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
                    if key == "consistent":
                        mask = (target.h >= s) & (target.h < e)
                    cur_diff = diff[key][mask]
                    if cur_diff.numel() == 0:
                        cur_diff = torch.tensor([0.0]).to(target.hamiltonian.device)
                    mse = torch.mean(cur_diff**2)
                    mae = torch.mean(torch.abs(cur_diff))
                    error_dict[key + f"_mae@{s:.2f}_{e:.2f}"] = mae
                    error_dict[key + f"_rmse@{s:.2f}_{e:.2f}"] = torch.sqrt(mse)

        return error_dict

    def forward(self, batch, cur_H, cur_t, step):
        output = self.model(batch, cur_H, cur_t, step)
        return output

    def _forward_H_and_K(self, batch):
        batch_mul = self.batch_repeat(batch, mul=2, repeat_style="append")
        device = batch.hamiltonian.device

        outputs = self(
            batch_mul,
            torch.cat([batch.hamiltonian_t, batch.init_ham]).to(device),
            torch.cat([batch.t, batch.t_0]).to(device),
            torch.cat([batch.h, batch.K]).to(device),
        )
        # fmt: off
        outputs_h = {"hamiltonian": outputs["hamiltonian"][: batch.hamiltonian_t.shape[0]]}
        outputs_K = {"hamiltonian": outputs["hamiltonian"][batch.hamiltonian_t.shape[0] :]}
        # fmt: on
        return outputs_h, outputs_K

    def training_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        batch = self.consistent_sample(batch, self.batch_mul)
        outputs, outputs_K = self._forward_H_and_K(batch)

        errors = self.criterion_consistent(
            {"H": outputs, "K": outputs_K},
            batch,
            loss_weights=self.loss_weights,
            loss_type=self.loss_type,
        )
        loss = errors["loss"]
        self._log_error(errors, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        batch_one = batch.clone()
        batch = self.consistent_sample(batch, self.batch_mul)
        outputs, outputs_K = self._forward_H_and_K(batch)

        errors = self.criterion_consistent(
            {"H": outputs, "K": outputs_K},
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
                num_timesteps=self.num_ode_steps_inf
            )
            # Long trajectory
            self._log_sample_error(
                batch_one,
                prefix="val",
                num_timesteps=self.max_T,
                post_fix=f"_{self.max_T}",
            )
        # fmt: on
        return errors

    def test_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        batch_one = batch.clone()
        batch = self.consistent_sample(batch, mul=self.batch_mul)
        outputs, outputs_K = self._forward_H_and_K(batch)

        errors = self.criterion_consistent(
            {"H": outputs, "K": outputs_K},
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
                num_timesteps=self.num_ode_steps_inf
            )
            self._log_sample_error(
                batch_one,
                prefix="test",
                num_timesteps=self.max_T,
                post_fix=f"_{self.max_T}",
            )
        # fmt: on
        return errors

    def sample(
        self,
        batch,
        num_timesteps=1,
    ):
        device = self.model.device
        lin_t = torch.linspace(0, 1.0, num_timesteps + 1).to(device)
        cur_t = lin_t[0]
        batch.hamiltonian_t = batch.init_ham
        # batch.init_ham_t = torch.randn_like(batch.hamiltonian)
        hamiltonian_traj = [batch.hamiltonian_t.cpu()]
        predictions = []

        for idx, next_t in enumerate(lin_t[1:]):
            batch.t = cur_t.repeat(batch.init_ham.shape[0])
            dt = next_t - cur_t
            assert dt > 0
            assert dt == 1.0 if num_timesteps == 1 else dt < 1.0

            dt = dt.repeat(batch.init_ham.shape[0])
            outputs = self(batch, batch.hamiltonian_t, batch.t, dt)
            if self.loss_type == 1:
                dt = dt.reshape(-1, 1, 1)
                ham_t = batch.hamiltonian_t + outputs["hamiltonian"] * dt
            else:
                ham_t = batch.hamiltonian_t + outputs["hamiltonian"]
            hamiltonian_traj.append(ham_t.cpu())
            predictions.append(outputs["hamiltonian"].cpu())

            # Update the previous timestep and the current Hamiltonian
            cur_t = next_t
            batch.hamiltonian_t = ham_t

        res_outputs = {"hamiltonian": batch.hamiltonian_t}

        return res_outputs, hamiltonian_traj, predictions

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

    def _log_sample_error(self, batch_one, prefix, num_timesteps=1, post_fix=""):
        try:
            sample, traj, pred = self.sample(batch_one, num_timesteps=num_timesteps)
            orb_and_eng_error = self._orb_and_eng_error(sample, batch_one)
            if post_fix != "":
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
            outputs, traj, _ = self.sample(batch, num_timesteps=self.num_ode_steps_inf)
            # outputs = self(batch, batch.init_ham)
            last_traj.append(traj[-1])

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
