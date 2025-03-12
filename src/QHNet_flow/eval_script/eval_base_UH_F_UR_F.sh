#! /bin/bash
# MODEL_PATH="/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/2025-02-21/22-30-25/DFT-hamiltonian/yvp82qw4/checkpoints/best-epoch=31976-val_hamiltonian_mae=0.00000000.ckpt"
# MODEL_PATH='model_path="/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/2025-02-21/22-30-25/DFT-hamiltonian/yvp82qw4/checkpoints/best-epoch=31976-val_hamiltonian_mae=0.00000000.ckpt"'

python test_wH_fixed.py --config-name=config_eval model.version=QHNet \
    'model_path="/home/seongsukim/dft/DEQHNet/src/QHNet_flow/outputs/2025-02-21/22-30-25/DFT-hamiltonian/yvp82qw4/checkpoints/best-epoch=31976-val_hamiltonian_mae=0.00000000.ckpt"' \
    use_init_hamiltonian=False \
    use_init_hamiltonian_residue=False