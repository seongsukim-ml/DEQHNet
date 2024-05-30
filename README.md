<div align="center">

# Infusing Self-Consistency into Quantum Hamiltonian Prediction via Deep Equilibrium Models


[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pyg](https://img.shields.io/badge/-pyg_2.3.0-34e1e9)](https://pytorch-geometric.readthedocs.io/en/latest/#)



</div>

## 📌  Introduction

Infusing Self-Consistency into Quantum Hamiltonian Prediction via Deep Equilibrium Models


## 🚀  Quickstart

Install dependencies

```bash
# clone project
git clone https://github.com/Zun-Wang/DEQHNet.git
cd DEQHNet

# [OPTIONAL] create conda environment
[Optional] conda create -n DEQHNet python=3.10
[Optional] conda activate DEQHNet

# Recommed to install part of dependencies in advance
# Take `cuda121` version as an example
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch_geometric==2.3.0

pip install pytorch-lightning==1.8.3

pip install pyscf==2.2.1
conda install psi4 python=3.9 -c conda-forge

pip install requirements.txt

pip install -e .
```

Train DEQHNet, e.g., 
```bash
cp auxiliary.gbs src/QHNet/
cd src/QHNet/
set basis AUXILIARY
python src/QHNet/train_wH.py dataset=uracil model=QHNet model.version=DEQHNet
```


## Citation
TBD


## Acknowledgements
This project is based on the repo [AIRS](https://github.com/divelab/AIRS.git).