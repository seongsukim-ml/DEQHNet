import torch

# import pytorch_lightning as pl
# from models import get_model

# from src.QHNet_flow.utils import ExponentialMovingAverage, self.post_processing

# from torch_ema import ExponentialMovingAverage
# from transformers import get_polynomial_decay_schedule_with_warmup
from pl_module.base_module import LitModel
from torch_geometric.data import Batch
from utils import AOData


class LitModel_flow(LitModel):
    def __init__(self, conf):
        super().__init__(conf=conf)
        self.batch_mul = conf.flow.get("batch_mul", 10)
        self.use_t_scale = conf.flow.get("use_t_scale", True)
        self.num_ode_steps = conf.flow.get("num_ode_steps", 10)
        self.init_gauss = conf.flow.get("init_gauss", False)
        self.error_threshold = conf.flow.get("error_threshold", 1e-5)

    @staticmethod
    def batch_repeat(batch, mul=1):
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
            )
            for _ in range(mul):
                batch_list.append(data.clone())
        return Batch.from_data_list(batch_list)

    def corrupt(self, batch, mul=1):
        batch = self.batch_repeat(batch, mul)
        # batch = Batch.from_data_list(batch_list)
        batch_t = self.sample_t(batch.hamiltonian.shape[0], batch.hamiltonian.device)
        batch.t = batch_t

        if self.init_gauss:
            random_ham = torch.randn_like(batch.hamiltonian)
        else:
            random_ham = batch.init_ham

        if self.use_init_hamiltonian_residue:
            target_ham = batch.hamiltonian - batch.init_ham
        else:
            target_ham = batch.hamiltonian

        batch_t_reshape = batch_t.reshape(-1, 1, 1)
        batch.init_ham_t = (
            random_ham * (1 - batch_t_reshape) + target_ham * batch_t_reshape
        )
        return batch

    @staticmethod
    def sample_t(num_batch, device, min_t=0.01):
        t = torch.rand(num_batch, device=device)
        return t * (1 - 2 * min_t) + min_t  # [min_t, 1-min_t]

    @staticmethod
    def criterion(outputs, target, loss_weights, use_t_scale=False):
        error_dict = {}
        if "waloss" in loss_weights.keys():
            energy, orb = LitModel.cal_orbital_and_energies(
                target.overlap, target.hamiltonian
            )
            target.orbital_energies = torch.diag_embed(energy).to(
                target.hamiltonian.device
            )
            target.orbital_coefficients = orb.to(target.hamiltonian.device)
        for key in loss_weights.keys():
            scale = 1
            if key == "hamiltonian":
                diff = outputs[key] - target[key]
                if use_t_scale:
                    scale = 1 - torch.min(target.t, torch.tensor(0.9))

            elif key == "waloss":
                diff = outputs["hamiltonian"].bmm(target.orbital_coefficients)
                diff = torch.bmm(target.orbital_coefficients.transpose(-1, -2), diff)
                diff = diff - target.orbital_energies
                if use_t_scale:
                    scale = 1 - torch.min(target.t, torch.tensor(0.9))

            mse = torch.mean(diff**2)
            mae = torch.mean(torch.abs(diff))
            error_dict[key + "_mae"] = mae
            error_dict[key + "_rmse"] = torch.sqrt(mse)

            if key == "hamiltonian":
                loss = mse + mae
            if key == "waloss":
                loss = mse
            loss = loss * scale
            loss = torch.mean(loss)
            error_dict[key] = loss
            if "loss" in error_dict:
                error_dict["loss"] += loss_weights[key] * loss
            else:
                error_dict["loss"] = loss_weights[key] * loss

        for key in loss_weights.keys():
            if key == "waloss":
                continue
            for _bin in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
                s, e = _bin
                mask = (target.t >= s) & (target.t < e)
                diff = outputs[key][mask] - target[key][mask]
                mse = torch.mean(diff**2)
                mae = torch.mean(torch.abs(diff))
                error_dict[key + f"_mae@{s:.2f}_{e:.2f}"] = mae
                error_dict[key + f"_rmse@{s:.2f}_{e:.2f}"] = torch.sqrt(mse)

        return error_dict

    def forward(self, batch, H):
        if self.use_init_hamiltonian:
            output = self.model(batch, H)
            if self.use_init_hamiltonian_residue:
                output["hamiltonian"] = output["hamiltonian"] + batch.init_ham
            return output
        else:
            return self.model(batch)

    def training_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        batch = self.corrupt(batch, mul=self.batch_mul)
        outputs = self(batch, batch.init_ham_t)
        errors = self.criterion(
            outputs, batch, loss_weights=self.loss_weights, use_t_scale=self.use_t_scale
        )
        loss = errors["loss"]
        for key in errors.keys():
            if "@" in key:
                _key, _time_bin = key.split("@")[0], key.split("@")[1]
                self.log(
                    f"train_{_time_bin}/{_key}_{_time_bin}",
                    errors[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True if key == "loss" else False,
                    sync_dist=True,
                )
            else:
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
        batch_one = batch.clone()
        batch = self.corrupt(batch, mul=self.batch_mul)
        with self.ema.average_parameters():
            ema_outputs = self(batch, batch.init_ham_t)
            ema_errors = self.criterion(
                ema_outputs,
                batch,
                loss_weights=self.loss_weights,
                use_t_scale=self.use_t_scale,
            )
            ema_loss = ema_errors["loss"]
            for key in ema_errors.keys():
                if "@" in key:
                    _key, _time_bin = key.split("@")[0], key.split("@")[1]
                    self.log(
                        f"val_{_time_bin}/{_key}_{_time_bin}",
                        ema_errors[key],
                        on_step=True,
                        on_epoch=True,
                        sync_dist=True,
                    )
                else:
                    self.log(
                        f"val_ema/{key}",
                        ema_errors[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True if key == "loss" else False,
                        sync_dist=True,
                    )
            if ema_loss < self.error_threshold:
                try:
                    ema_sample, traj, pred = self.sample(
                        batch_one, num_timesteps=self.num_ode_steps
                    )
                    ema_orb_and_eng_error = self._orb_and_eng_error(
                        ema_sample, batch_one
                    )
                    for key in ema_orb_and_eng_error.keys():
                        self.log(
                            f"val_ema/{key}",
                            ema_orb_and_eng_error[key],
                            on_step=True,
                            on_epoch=True,
                            prog_bar=True if key == "loss" else False,
                            sync_dist=True,
                        )
                except Exception as e:
                    print(e)
                    pass
        outputs = self(batch, batch.init_ham_t)
        errors = self.criterion(
            outputs, batch, loss_weights=self.loss_weights, use_t_scale=self.use_t_scale
        )
        loss = errors["loss"]
        for key in errors.keys():
            # if key has @
            if "@" in key:
                _key, _time_bin = key.split("@")[0], key.split("@")[1]
                self.log(
                    f"val_{_time_bin}/{_key}_{_time_bin}",
                    errors[key],
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
            else:
                self.log(
                    f"val/{key}",
                    errors[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True if key == "loss" else False,
                    sync_dist=True,
                )
        if loss < self.error_threshold:
            try:
                sample, traj, pred = self.sample(
                    batch_one, num_timesteps=self.num_ode_steps
                )
                orb_and_eng_error = self._orb_and_eng_error(sample, batch_one)
                for key in orb_and_eng_error.keys():
                    self.log(
                        f"val/{key}",
                        orb_and_eng_error[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True if key == "loss" else False,
                        sync_dist=True,
                    )
            except Exception as e:
                print(e)
                pass
        return errors

    def sample(
        self,
        batch,
        num_timesteps=100,
        min_t=0.01,
    ):
        device = self.model.device
        lin_t = torch.linspace(min_t, 1.0, num_timesteps).to(device)
        cur_t = lin_t[0]
        # import pdb
        # pdb.set_trace()
        batch.init_ham_t = batch.init_ham
        # batch.init_ham_t = torch.randn_like(batch.hamiltonian)
        hamiltonian_traj = [batch.init_ham_t.cpu()]
        predictions = [None]

        for idx, next_t in enumerate(lin_t[1:]):
            batch.t = cur_t.repeat(batch.init_ham.shape[0])
            outputs = self(batch, batch.init_ham_t)
            dt = next_t - cur_t
            assert dt > 0
            # vector_field = outputs["hamiltonian"] / (1 - cur_t)
            vector_field = (outputs["hamiltonian"] - batch.init_ham_t) / (1 - cur_t)
            ham_t = batch.init_ham_t + vector_field * dt.reshape(-1, 1, 1)
            hamiltonian_traj.append(ham_t.cpu())
            predictions.append(outputs["hamiltonian"].cpu())

            # Update the previous timestep and the current Hamiltonian
            cur_t = next_t
            batch.init_ham_t = ham_t

        res_outputs = {"hamiltonian": ham_t}

        return res_outputs, hamiltonian_traj, predictions
