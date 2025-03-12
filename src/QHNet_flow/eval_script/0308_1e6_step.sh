# mal
# python -m experiment.train_v2 --config-name=config_flow_eval \
# dataset=malondialdehyde \
# model=QHNet_flow \
# prefix=_MAE_BM1_T1e5 \
# model.version=QHNet_flow_v5 \
# data_type=float64 \
# use_init_hamiltonian_residue=True \
# use_mse_and_mae=True \
# flow.use_res_target=True \
# flow.init_gauss=True \
# flow.init_gauss_center=False \
# model.hidden_size=128 \
# check_val_every_n_epoch=1 \
# scheduler_power=3.5 \
# num_training_steps=100000 \
# dataset.batch_size=20 \
# end_lr=1e-9 \
# +mode=eval \
# +flow.sample_random=True \
# "model_ckpt='/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/malondialdehyde/QHNet_flow_v5_MAE_BM1_T1e5/2025-03-08/12-26-26/DFT-hamiltonian_V2/31h4tdlh/checkpoints/best-epoch=19.ckpt'"

python -m experiment.train_v2 --config-name=config_flow_eval \
dataset=malondialdehyde \
model=QHNet_flow \
prefix=_MAE_BM1_T2e5 \
model.version=QHNet_flow_v5 \
data_type=float64 \
use_init_hamiltonian_residue=True \
use_mse_and_mae=True \
flow.use_res_target=True \
flow.init_gauss=True \
flow.init_gauss_center=False \
model.hidden_size=128 \
check_val_every_n_epoch=1 \
scheduler_power=3.5 \
num_training_steps=200000 \
dataset.batch_size=20 \
end_lr=1e-9 \
+mode=eval \
+flow.sample_random=True \
"model_ckpt='/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/completed/malondialdehyde/QHNet_flow_v5_MAE_BM1_T2e5/DFT-hamiltonian_V2/4h4p62h1/checkpoints/best-epoch=39.ckpt'"



# ethanol
python -m experiment.train_v2 --config-name=config_flow_eval \
dataset=ethanol \
model=QHNet_flow \
prefix=_MAE_BM1_T1e5 \
model.version=QHNet_flow_v5 \
data_type=float64 \
use_init_hamiltonian_residue=True \
use_mse_and_mae=True \
flow.use_res_target=True \
flow.init_gauss=True \
flow.init_gauss_center=False \
model.hidden_size=128 \
check_val_every_n_epoch=1 \
scheduler_power=3.5 \
num_training_steps=100000 \
dataset.batch_size=20 \
end_lr=1e-9 \
+mode=eval \
"model_ckpt='/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/completed/ethanol/QHNet_flow_v5_MAE_BM1_T1e5/DFT-hamiltonian_V2/ihl6uq1w/checkpoints/best-epoch=19.ckpt'"

python -m experiment.train_v2 --config-name=config_flow_eval \
dataset=ethanol \
model=QHNet_flow \
prefix=_MAE_BM1_T2e5 \
model.version=QHNet_flow_v5 \
data_type=float64 \
use_init_hamiltonian_residue=True \
use_mse_and_mae=True \
flow.use_res_target=True \
flow.init_gauss=True \
flow.init_gauss_center=False \
model.hidden_size=128 \
check_val_every_n_epoch=1 \
scheduler_power=3.5 \
num_training_steps=200000 \
dataset.batch_size=20 \
end_lr=1e-9 \
+mode=eval \
"model_ckpt='/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/completed/ethanol/QHNet_flow_v5_MAE_BM1_T2e5/DFT-hamiltonian_V2/allzamwv/checkpoints/best-epoch=39.ckpt'"


# water
python -m experiment.train_v2 --config-name=config_flow_eval \
dataset=water \
model=QHNet_flow \
prefix=_MAE_BM1_T1e5 \
model.version=QHNet_flow_v5 \
data_type=float64 \
use_init_hamiltonian_residue=True \
use_mse_and_mae=True \
flow.use_res_target=True \
flow.init_gauss=True \
flow.init_gauss_center=False \
model.hidden_size=128 \
check_val_every_n_epoch=1 \
scheduler_power=3.5 \
num_training_steps=100000 \
dataset.batch_size=20 \
end_lr=1e-9 \
+mode=eval \
+flow.sample_random=True \
"model_ckpt='/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/completed/water/QHNet_flow_v5_MAE_BM1_T1e5/DFT-hamiltonian_V2/mvq1wlii/checkpoints/best-epoch=1929.ckpt'"

# under train
python -m experiment.train_v2 --config-name=config_flow_eval \
dataset=water \
model=QHNet_flow \
prefix=_MAE_BM1_T2e5 \
model.version=QHNet_flow_v5 \
data_type=float64 \
use_init_hamiltonian_residue=True \
use_mse_and_mae=True \
flow.use_res_target=True \
flow.init_gauss=True \
flow.init_gauss_center=False \
model.hidden_size=128 \
check_val_every_n_epoch=1 \
scheduler_power=3.5 \
num_training_steps=200000 \
dataset.batch_size=20 \
end_lr=1e-9 \
+mode=eval \
+flow.sample_random=True \
"model_ckpt='/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/water/QHNet_flow_v5_MAE_BM1_T2e5/2025-03-08/02-29-13/DFT-hamiltonian_V2/1zm1colm/checkpoints/last.ckpt'"


# uracil
python -m experiment.train_v2 --config-name=config_flow_eval \
dataset=uracil \
model=QHNet_flow \
prefix=_MAE_BM1_T1e5 \
model.version=QHNet_flow_v5 \
data_type=float64 \
use_init_hamiltonian_residue=True \
use_mse_and_mae=True \
flow.use_res_target=True \
flow.init_gauss=True \
flow.init_gauss_center=False \
model.hidden_size=128 \
check_val_every_n_epoch=1 \
scheduler_power=3.5 \
num_training_steps=100000 \
dataset.batch_size=5 \
end_lr=1e-9 \
+mode=eval \
+flow.sample_random=True \
+flow.num_ode_steps_inf=5 \
"model_ckpt='/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/completed/uracil/QHNet_flow_v5_MAE_BM1_T1e5/DFT-hamiltonian_V2/sgptoscw/checkpoints/best-epoch=19.ckpt'"


# uracil

# under train
python -m experiment.train_v2 --config-name=config_flow_eval \
dataset=uracil \
model=QHNet_flow \
prefix=_MAE_BM1_T2e5 \
model.version=QHNet_flow_v5 \
data_type=float64 \
use_init_hamiltonian_residue=True \
use_mse_and_mae=True \
flow.use_res_target=True \
flow.init_gauss=True \
flow.init_gauss_center=False \
model.hidden_size=128 \
check_val_every_n_epoch=1 \
scheduler_power=3.5 \
num_training_steps=200000 \
dataset.batch_size=40 \
end_lr=1e-9 \
+mode=eval \
+flow.sample_random=True \
+flow.num_ode_steps_inf=20 \
"model_ckpt='/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/completed/uracil/QHNet_flow_v5_MAE_BM1_T2e5/DFT-hamiltonian_V2/fisf326b/checkpoints/best-epoch=37.ckpt'"
