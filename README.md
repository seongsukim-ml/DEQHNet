<div align="center">

# Infusing Self-Consistency into Density Functional Theory Hamiltonian Prediction via Deep Equilibrium Models


[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pyg](https://img.shields.io/badge/-pyg_2.3.0-34e1e9)](https://pytorch-geometric.readthedocs.io/en/latest/#)



</div>

## 📌  Introduction

Infusing Self-Consistency into Density Functional Theory Hamiltonian Prediction via Deep Equilibrium Models.


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
conda install -c conda-forge gxx_linux-64==11.1.0
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch_geometric==2.3.0

pip install pytorch-lightning==1.8.5

# pip install pyscf==2.2.1
pip install hydra-core
# conda install psi4 python=3.9 -c conda-forge

pip install -r requirements.txt
pip install scipy==1.10
pip install pydantic==1.10.21
pip install numpy==1.23
pip install -e .
```

``` bash
conda create -n p4_cu124_2 python=3.9 psi4 pyscf pytorch==2.5.0 torchvision==0.20.1 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia  -c pyscf -c pyg


# conda install -c pyscf -c conda-forge pyscf
# conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install pytorch-lightning
pip install hydra-core
pip install ase
pip install torch_ema tqdm wandb PyYAML
pip install e3nn gdown transformers tensorboard torchdeq lmdb
```

<!-- conda install pytorch3d -c pytorch3d
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 -->


``` bash
conda create -n p4_QH4 python=3.9 pytorch==2.1.2 pytorch-cuda=12.1  psi4 pyscf=2.2.1 pytorch3d pytorch-lightning==1.8.5 -c pytorch -c nvidia -c pyscf -c pytorch3d 
conda activate p4_QH4
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
pip install pytorch-lightning==1.8.5
# pip install hydra-core
# pip install ase==3.22.1
# pip install torch_ema tqdm wandb PyYAML
# pip install e3nn gdown transformers tensorboard torchdeq lmdb
pip install -r requirements.txt
pip install scipy==1.10
pip install pydantic==1.10.21
```



Train DEQHNet, e.g., 
```bash
cp auxiliary.gbs src/QHNet/
cd src/QHNet/
set basis AUXILIARY
python src/QHNet/train_wH.py dataset=uracil model=QHNet model.version=DEQHNet
```


## Citation
```
@inproceedings{wang2024infusing,
  title={Infusing Self-Consistency into Density Functional Theory Hamiltonian Prediction via Deep Equilibrium Models},
  author={Wang, Zun and Liu, Chang and Zou, Nianlong and Zhang, He and Wei, Xinran and Huang, Lin and Wu, Lijun and Shao, Bin},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=PSVkinBs4u}
}
```


## Acknowledgements
This project is based on the repo [AIRS](https://github.com/divelab/AIRS.git).


## Final
```bash
conda create -n p4_QH2  python=3.9 psi4=1.7 pytorch3d  -c pytorch3d -c psi4 
conda activate p4_QH2
# conda install -c conda-forge gxx_linux-64==11.1.0
# conda install -c conda-forge gcc=12.1.0
pip install pyscf==2.2.1
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch_geometric==2.3.0

pip install pytorch-lightning==1.8.5

# pip install pyscf==2.2.1
pip install hydra-core
# conda install psi4 python=3.9 -c conda-forge

pip install -r requirements.txt
pip install scipy==1.10
pip install pydantic==1.10.21
pip install numpy==1.23
pip install pillow==11.0.0
```