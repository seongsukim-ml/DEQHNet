#!/bin/bash

# module purge
# module load cuda/12.4 intel/2023 mpi cmake

# python -m experiment.sample \
#     --config-name=config_eval2 \
#     model.version=Real_QHNet \
#     use_init_hamiltonian=False\
#     use_init_hamiltonian_residue=False
# flow.num_ode_steps=10

# python -m experiment.sample \
#     --config-name=config_eval2 \
#     model.version=QHNet \
#     use_init_hamiltonian=True \
#     use_init_hamiltonian_residue=False \
#     'model_path="/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/water/QHNet/2025-02-26/04-28-11/DFT-hamiltonian_V2/tejlaoyw/checkpoints/last.ckpt"'


# python -m experiment.sample \
#     --config-name=config_eval2 \
#     model.version=QHNet \
#     use_init_hamiltonian=False\
#     use_init_hamiltonian_residue=False\
    #  'model_path="/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/water/QHNet/2025-02-26/04-28-11/DFT-hamiltonian_V2/tejlaoyw/checkpoints/last.ckpt"'

python -m experiment.sample \
    --config-name=config_flow_eval \
    model.version=QHNet_flow_v2 \
    data_type=float64 \
    use_init_hamiltonian=False \
    use_init_hamiltonian_residue=False \
    flow.init_gauss=True \
    dataset.batch_size=100 \
    flow.num_ode_steps=30 \
    'model_path="/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/water/QHNet_flow_v2/2025-02-26/04-34-56/DFT-hamiltonian_V2/t1kpfrod/checkpoints/best-epoch=3289.ckpt"'

python -m experiment.sample \
    --config-name=config_flow_eval \
    model.version=QHNet_flow_v3 \
    data_type=float64 \
    use_init_hamiltonian=False \
    use_init_hamiltonian_residue=False \
    flow.init_gauss=True \
    dataset.batch_size=100 \
    flow.num_ode_steps=30 \
    'model_path="/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/water/QHNet_flow_v3/2025-02-26/04-38-10/DFT-hamiltonian_V2/6lenaifl/checkpoints/best-epoch=2209.ckpt"'

python -m experiment.sample_auto \
    --config-name=config_flow5 \
    +job_path="/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/water/QHNet_flow_v3_v4_5/2025-02-26/22-41-01"