#!/bin/bash

# module purge
# module load cuda/12.4 intel/2023 mpi cmake

python -m experiment.train


python -m experiment.train_qh9-cont --config-name=config-cont dataset=QH9Dynamic dataset.split=geometry dataset.version=300k dataset.batch_size=32 dataset.learning_rate=1e-3 +SLURM_JOB_ID=580318 +hpgpu=hpgpu model=QHNet model.version=Real_QHNet prefix=_P3_5 data_type=float32 model.hidden_size=128 use_init_hamiltonian_residue=True check_val_every_n_epoch=1 scheduler_power=3.5 mode=eval