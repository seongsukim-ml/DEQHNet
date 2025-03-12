import os
import torch
from torch import Tensor
import tarfile
import numpy as np
import os.path as osp
from tqdm import tqdm
from ase.db import connect
from argparse import Namespace
from torch_geometric.data import InMemoryDataset, download_url

import logging
from utils import AOData, Onsite_3idx_Overlap_Integral, build_molecule, build_AO_index

from pyscf import gto, scf, dft
import concurrent.futures
from pathlib import Path

logger = logging.getLogger()


convention_dict = {
    "pyscf_631G": Namespace(
        atom_to_orbitals_map={1: "ss", 6: "ssspp", 7: "ssspp", 8: "ssspp", 9: "ssspp"},
        orbital_idx_map={"s": [0], "p": [2, 0, 1], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1],
            6: [0, 1, 2, 3, 4],
            7: [0, 1, 2, 3, 4],
            8: [0, 1, 2, 3, 4],
            9: [0, 1, 2, 3, 4],
        },
    ),
    "pyscf_def2svp": Namespace(
        atom_to_orbitals_map={
            1: "ssp",
            6: "sssppd",
            7: "sssppd",
            8: "sssppd",
            9: "sssppd",
        },
        orbital_idx_map={"s": [0], "p": [1, 2, 0], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],
            6: [0, 1, 2, 3, 4, 5],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
        },
    ),
    "back2pyscf": Namespace(
        atom_to_orbitals_map={
            1: "ssp",
            6: "sssppd",
            7: "sssppd",
            8: "sssppd",
            9: "sssppd",
        },
        orbital_idx_map={"s": [0], "p": [2, 0, 1], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],
            6: [0, 1, 2, 3, 4, 5],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
        },
    ),
}


def matrix_transform(matrices, atoms, convention="pyscf_631G"):
    conv = convention_dict[convention]
    orbitals = ""
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    matrices_new = matrices[..., transform_indices, :]
    matrices_new = matrices_new[..., :, transform_indices]
    matrices_new = matrices_new * transform_signs[:, None]
    matrices_new = matrices_new * transform_signs[None, :]
    return matrices_new


def random_split(dataset, lengths, seed=None):
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = np.random.RandomState(seed=seed).permutation(sum(lengths))
    return [
        torch.utils.data.Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(torch._utils._accumulate(lengths), lengths)
    ]


def get_mask(data):
    mask_period_group_1 = torch.tensor([1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    mask_period_group_2 = torch.ones(14)

    mask_row = []
    for atom in data.atoms:
        mask_row.append(mask_period_group_1 if atom < 2 else mask_period_group_2)

    data.mask_row = torch.stack(mask_row, dim=0)
    return data


def hamiltonian_transform(hamiltonian, atoms):
    conv = Namespace(
        atom_to_orbitals_map={"H": "ssp", "O": "sssppd", "C": "sssppd", "N": "sssppd"},
        orbital_idx_map={"s": [0], "p": [2, 0, 1], "d": [4, 2, 0, 1, 3]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            "H": [0, 1, 2],
            "O": [0, 1, 2, 3, 4, 5],
            "C": [0, 1, 2, 3, 4, 5],
            "N": [0, 1, 2, 3, 4, 5],
        },
    )

    orbitals = ""
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int64)
    transform_signs = np.concatenate(transform_signs)

    hamiltonian_new = hamiltonian[..., transform_indices, :]
    hamiltonian_new = hamiltonian_new[..., :, transform_indices]
    hamiltonian_new = hamiltonian_new * transform_signs[:, None]
    hamiltonian_new = hamiltonian_new * transform_signs[None, :]
    return hamiltonian_new


db_split_dict = {
    "water": 2500,
    "ethanol": 5000,
    "malondialdehyde": 5000,
    "uracil": 2500,
}


class MD17_DFT_trajectory(InMemoryDataset):
    def __init__(
        self,
        root="dataset/",
        name="water",
        prefix="_trajectory",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        db_idx=None,
        db_split=None,
    ):
        # water, ethanol, malondialdehyde, uracil
        self.db_idx = db_idx
        if db_split is None:
            db_split = db_split_dict[name]
            print(
                f"db_split is not provided. Using default value of {name}: {db_split}"
            )
        self.db_split = db_split  # how many molecules are included in one sub database
        self.name = name
        if prefix is None:
            prefix = "_trajectory"
        self.folder = osp.join(root, self.name + prefix)
        self.url = "http://quantum-machine.org/data/schnorb_hamiltonian"
        self.chemical_symbols = ["n", "H", "He", "Li", "Be", "B", "C", "N", "O"]
        self.atom_types = None
        self.init_ham = {}

        orbitals_ref = {}
        orbitals_ref[1] = np.array([0, 0, 1])  # H: 2s 1p
        orbitals_ref[6] = np.array([0, 0, 0, 1, 1, 2])  # C: 3s 2p 1d
        orbitals_ref[7] = np.array([0, 0, 0, 1, 1, 2])  # N: 3s 2p 1d
        orbitals_ref[8] = np.array([0, 0, 0, 1, 1, 2])  # O: 3s 2p 1d
        self.orbitals_ref = orbitals_ref

        orbitals = []
        assert name in ["water", "ethanol", "malondialdehyde", "uracil", "aspirin"]
        if name == "water":
            atoms = [8, 1, 1]
            atom_list = ["O", "H"]
            H_size = 24
        elif name == "ethanol":
            atoms = [6, 6, 8, 1, 1, 1, 1, 1, 1]
            atom_list = ["C", "O", "H"]
            H_size = 72
        elif name == "malondialdehyde":
            atoms = [6, 6, 6, 8, 8, 1, 1, 1, 1]
            atom_list = ["C", "O", "H"]
            H_size = 90
        elif name == "uracil":
            atoms = [6, 6, 7, 6, 7, 6, 8, 8, 1, 1, 1, 1]
            atom_list = ["C", "N", "O", "H"]
            H_size = 132
        elif name == "aspirin":
            atoms = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]
            atom_list = ["C", "O", "H"]
            raise NotImplementedError

        self.Q_dict = Onsite_3idx_Overlap_Integral(
            atom_list=atom_list, basis="def2-svp"
        ).Q_table()
        self.Q = (
            torch.stack(
                [
                    torch.block_diag(*[self.Q_dict[z][l] for z in atoms])
                    for l in range(60)
                ]
            )
            .double()
            .numpy()
        )
        self.Q = (
            torch.from_numpy(
                matrix_transform(self.Q, atoms, convention="pyscf_def2svp")
            )
            .double()
            .permute(1, 2, 0)
        )
        self.Q[:, :, 16:40] = (
            self.Q[:, :, 16:40]
            .reshape(H_size, H_size, -1, 3)[:, :, :, [1, 2, 0]]
            .reshape(H_size, H_size, 24)
        )

        for Z in atoms:
            orbitals.append(tuple((int(Z), int(l)) for l in self.orbitals_ref[Z]))

        self.orbitals = tuple(orbitals)
        self.print_once = False

        super(MD17_DFT_trajectory, self).__init__(
            self.folder, transform, pre_transform, pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])
        if not self.atom_types:
            self.atom_types = "".join([self.chemical_symbols[i] for i in self[0].atoms])

    @property
    def raw_file_names(self):
        if self.name == "ethanol":
            return [
                f"schnorb_hamiltonian_{self.name}_dft.tgz",
                f"schnorb_hamiltonian_{self.name}_dft.db",
            ]
        elif self.name == "aspirin":
            return [
                f"schnorb_hamiltonian_{self.name}_quambo.db",
                f"schnorb_hamiltonian_{self.name}_quambo.db",
            ]
        else:
            return [
                f"schnorb_hamiltonian_{self.name}.tgz",
                f"schnorb_hamiltonian_{self.name}.db",
            ]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        if self.name == "ethanol":
            url = f"{self.url}/schnorb_hamiltonian_{self.name}" + "_dft.tgz"
        else:
            url = f"{self.url}/schnorb_hamiltonian_{self.name}" + ".tgz"
        download_url(url, self.raw_dir)
        extract_path = self.raw_dir
        tar = tarfile.open(os.path.join(self.raw_dir, self.raw_file_names[0]), "r")
        for item in tar:
            tar.extract(item, extract_path)

    def process(self):
        db = connect(osp.join(self.raw_dir, self.raw_file_names[1]))
        data_list = []
        if not getattr(self, "atom_types"):
            self.atom_types = "".join(
                [self.chemical_symbols[i] for i in next(db.select(1))["numbers"]]
            )

        def merge_db():
            db_paths = Path(self.processed_paths[0]).parent.glob("**/data_*.pt")
            db_paths = [db_path for db_path in db_paths]
            fin_data_list = []
            if len(db_paths) == max_splits:
                for db_path in db_paths:
                    data = torch.load(db_path)
                    fin_data_list.append(data)
                fin_data_list = [item for sublist in fin_data_list for item in sublist]
                data, slices = self.collate(fin_data_list)
                torch.save((data, slices), self.processed_paths[0])
                return
            else:
                print(
                    f"Failed to merge db, only {len(db_paths)}/ {max_splits} files are saved"
                )
        
        max_splits = len(db) // self.db_split
        print(len(db),self.db_split,len(db) % self.db_split)
        if len(db) % self.db_split > 0:
            max_splits += 1
        print(f"Max splits: {max_splits}")

        merge_db()
        if self.db_idx is not None:
            print("Processing...")
            idx_start = self.db_split * self.db_idx
            idx_end = self.db_split * (self.db_idx + 1)
            rows = list(db.select())[idx_start:idx_end]
            print(f"Lenght of db: {len(db)}")
            # print(f"Max splits: {max_splits}")
            print(f"Processing {len(rows)} molecules")
            if self.db_idx > max_splits:
                print("db_idx is larger than max_splits")
                raise ValueError
            print(f"Current split: {self.db_idx + 1} / {max_splits}")

            ## parrallelize this part
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                data_list = list(
                    tqdm(executor.map(self.get_mol, rows), total=len(rows))
                )

            # for row in tqdm(db.select()):
            #     data_list.append(self.get_mol(row))

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            # data, slices = self.collate(data_list)

            print("Saving...")
            save_path = Path(self.processed_paths[0]).parent / f"data_{self.db_idx}.pt"
            torch.save((data_list), save_path)

            merge_db()

    def calc_traj(self, atoms, pos, init="minao"):
        def get_infos(envs):
            setattr(envs["mf"], "total_cycle", envs["cycle"])
            # info = {'fock'    : envs['fock'],
            #         'dm'      : envs['dm'],
            #         'mo_coeff': envs['mo_coeff'],
            #         'mo_energy':envs['mo_energy'],
            #         'e_tot'   : envs['e_tot']}
            fock = envs["fock"]
            getattr(envs["mf"], "focks").append(fock)

        mol = gto.Mole()
        t = [[atoms[atom_idx], pos[atom_idx]] for atom_idx in range(len(atoms))]
        mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")
        # mf = scf.RHF(mol)
        mf = dft.RKS(mol)
        mf.callback = get_infos
        mf.focks = []
        if init == "minao":
            init_dm = mf.init_guess_by_minao()
        elif init == "1e":
            init_dm = mf.init_guess_by_1e()
        else:
            print(f"init {init} not implemented")
            raise NotImplementedError
        init_ham = scf.hf.get_fock(mf, dm=init_dm)
        mf.kernel()
        Ham_fianl = mf.get_fock()
        Ham_traj = np.stack(mf.focks, axis=0)
        # Ham_final != Ham_traj[-1]
        return Ham_fianl, init_ham, Ham_traj

    def calc_traj_fix(self, atoms, pos, init="minao"):
        def get_infos(envs):
            setattr(envs["mf"], "total_cycle", envs["cycle"])
            cur_mf = envs["mf"]
            fock = envs["fock"]  # Focks without DIIS
            # S, C, E = cur_mf.get_ovlp(), envs["mo_coeff"], envs["mo_energy"]
            # SCE = np.einsum("ij, jk, k -> ik", S, C, E)
            # fock2 = SCE.dot(C.T).dot(S)  # Focks with DIIS
            S = cur_mf.get_ovlp()
            E2, C2 = cur_mf.eig(fock,S)
            setattr(cur_mf, "_dm_last", envs["dm"])
            getattr(cur_mf, "focks").append(fock)
            # getattr(cur_mf, "focks2").append(fock2)
            getattr(cur_mf, "S").append(S)
            getattr(cur_mf, "C").append(C2)
            getattr(cur_mf, "E").append(E2)

        mol = gto.Mole()
        t = [[atoms[atom_idx], pos[atom_idx]] for atom_idx in range(len(atoms))]
        mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")
        # mf = scf.RHF(mol)
        mf = dft.RKS(mol)
        mf.xc = "pbe,pbe"
        mf.basis = "def2svp"
        mf.callback = get_infos
        mf.focks = []
        mf.S, mf.C , mf.E = ([], [], [])
        if init == "minao":
            init_dm = mf.init_guess_by_minao()
        elif init == "1e":
            init_dm = mf.init_guess_by_1e()
        else:
            print(f"init {init} not implemented")
            raise NotImplementedError
        init_ham = scf.hf.get_fock(mf, dm=init_dm)
        mf.kernel()
        Ham_fianl = mf.get_fock(dm=mf._dm_last)
        Ham_traj = np.stack(mf.focks, axis=0)
        E_traj = np.stack(mf.E, axis=0)
        C_traj = np.stack(mf.C, axis=0)
        # Ham_final != Ham_traj[-1]
        return Ham_fianl, init_ham, Ham_traj, E_traj, C_traj

    def get_mol(self, row):
        # from angstrom to bohr
        # make sure the original data type is float or double
        pos = torch.tensor(row["positions"] * 1.8897261258369282, dtype=torch.float64)
        atoms = torch.tensor(row["numbers"], dtype=torch.int64).view(-1, 1)
        energy = torch.tensor(row.data["energy"], dtype=torch.float64)
        force = torch.tensor(row.data["forces"], dtype=torch.float64)
        hamiltonian_sch = torch.tensor(
            hamiltonian_transform(row.data["hamiltonian"], self.atom_types),
            dtype=torch.float64,
        ).unsqueeze(0)
        overlap = torch.tensor(
            hamiltonian_transform(row.data["overlap"], self.atom_types),
            dtype=torch.float64,
        )

        atom_string = build_molecule(row["numbers"], row["positions"])
        AO_index = build_AO_index(atom_string, "def2-svp")

        (hamiltonian_final, init_ham, hamiltonian_traj, E_traj, C_traj)= self.calc_traj_fix(
            row["numbers"], row["positions"]
        )
        hamiltonian = torch.tensor(
            matrix_transform(
                hamiltonian_final, row["numbers"], convention="pyscf_def2svp"
            ),
            dtype=torch.float64,
        )
        init_ham = torch.tensor(
            matrix_transform(init_ham, row["numbers"], convention="pyscf_def2svp"),
            dtype=torch.float64,
        ).unsqueeze(0)
        hamiltonian_traj = torch.tensor(
            matrix_transform(
                hamiltonian_traj, row["numbers"], convention="pyscf_def2svp"
            ),
            dtype=torch.float64,
        )  # (Cycle, H, H) Cycle is not fixed
        E_traj = torch.tensor(E_traj, dtype=torch.float64)
        C_traj = torch.tensor(C_traj, dtype=torch.float64) # Do we have to transform this?
        
        cycle = hamiltonian_traj.shape[0] # mf.cycle + 1 / final value is target hamiltonian
        assert len(hamiltonian_traj.shape) == 3

        data = AOData(
            pos=pos,
            atoms=atoms,
            energy=energy,
            force=force,
            AO_index=AO_index,
            init_ham=init_ham,
            hamiltonian_sch=hamiltonian_sch,
            hamiltonian=hamiltonian,
            hamiltonian_traj=hamiltonian_traj,
            cycle=cycle,
            overlap=overlap,
        )

        if not self.print_once:
            print(data)
            self.print_once = True

        return data

    @staticmethod
    def post_process_data(data):
        ret_data = data
        # ret_data.hamiltonian_traj = torch.cat(
        #     [data.init_ham, data.hamiltonian_traj, data.hamiltonian.unsqueeze(0)],
        #     dim=0,
        # )
        # ret_data.cycle = data.cycle + 2
        ret_data.hamiltonian = data.hamiltonian.unsqueeze(0)
        ret_data.overlap = data.overlap.unsqueeze(0)
        return ret_data

    def __getitem__(self, idx):
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            data = self.get(self.indices()[idx])
            data = self.post_process_data(data)
            data = data if self.transform is None else self.transform(data)
            data.Q = self.Q
            return data

        else:
            return self.post_process_data(self.index_select(idx))


if __name__ == "__main__":
    # get arguments
    import argparse

    parser = argparse.ArgumentParser(description="Generation")
    parser.add_argument(
        "--root",
        type=str,
        default="/home/seongsukim/dft/DEQHNet/src/QHNet_flow/dataset",
    )
    parser.add_argument("--name", type=str, default="water")
    parser.add_argument("--db_idx", type=int, default=None)
    parser.add_argument("--db_split", type=int, default=None)
    parser.add_argument("--pdb", action="store_true")

    args = parser.parse_args()
    dataset = MD17_DFT_trajectory(
        root=args.root,
        name=args.name,
        prefix="_trajectory_v2",
        db_idx=args.db_idx,
        db_split=args.db_split,
    )
    if args.pdb:
        import pdb

        pdb.set_trace()
    print("Finished")
