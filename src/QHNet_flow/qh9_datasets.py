import os
import lmdb
import random
import torch
import numpy as np
import os.path as osp
from argparse import Namespace
import pickle
import gdown
import pyscf
from pyscf import gto
from tqdm import tqdm
from apsw import Connection
from torch_geometric.data import InMemoryDataset, Data
from pyscf import gto, scf, dft
import concurrent.futures

from utils import AOData, Onsite_3idx_Overlap_Integral, build_molecule, build_AO_index

BOHR2ANG = 1.8897259886

GoogleDriveLink = (
    "https://drive.google.com/drive/u/0/folders/1LXTC8uaOQzmb76FsuGfwSocAbK5Hshfj"
)

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
}

atomrefs = {
    6: [0.0, 0.0, 0.0, 0.0, 0.0],
    7: [-13.61312172, -1029.86312267, -1485.30251237, -2042.61123593, -2713.48485589],
    8: [-13.5745904, -1029.82456413, -1485.26398105, -2042.5727046, -2713.44632457],
    9: [-13.54887564, -1029.79887659, -1485.2382935, -2042.54701705, -2713.42063702],
    10: [-13.90303183, -1030.25891228, -1485.71166277, -2043.01812778, -2713.88796536],
    11: [0.0, 0.0, 0.0, 0.0, 0.0],
}

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414
conversion = torch.tensor(
    [
        1.0,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        KCALMOL2EV,
        KCALMOL2EV,
        KCALMOL2EV,
        KCALMOL2EV,
        1.0,
        1.0,
        1.0,
    ]
)

atomrefs_tensor = torch.zeros(5, 19)
atomrefs_tensor[:, 7] = torch.tensor(atomrefs[7])
atomrefs_tensor[:, 8] = torch.tensor(atomrefs[8])
atomrefs_tensor[:, 9] = torch.tensor(atomrefs[9])
atomrefs_tensor[:, 10] = torch.tensor(atomrefs[10])


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


class QH9Stable(InMemoryDataset):
    url = "https://drive.google.com/file/d/1LcEJGhB8VUGkuyb0oQ_9ANJdSkky9xMS/view?usp=sharing"

    def __init__(
        self,
        root="datasets/",
        split="random",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.folder = osp.join(root, "QH9Stable")
        self.split = split
        self.db_dir = os.path.join(self.folder, "processed")
        self.full_orbitals = 14
        self.orbital_mask = {}
        idx_1s_2s_2p = torch.tensor([0, 1, 3, 4, 5])
        orbital_mask_line1 = idx_1s_2s_2p
        orbital_mask_line2 = torch.arange(self.full_orbitals)
        for i in range(1, 11):
            self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

        atom_list = ["H", "C", "N", "O", "F"]

        self.Q_dict = Onsite_3idx_Overlap_Integral(
            atom_list=atom_list, basis="def2-svp"
        ).Q_table()

        super(QH9Stable, self).__init__(
            self.folder, transform, pre_transform, pre_filter
        )
        self.train_mask, self.val_mask, self.test_mask = torch.load(
            self.processed_paths[0]
        )
        self.slices = {
            "id": torch.arange(
                self.train_mask.shape[0]
                + self.val_mask.shape[0]
                + self.test_mask.shape[0]
                + 1
            )
        }

    def download(self):
        try:
            print(f"Downloading the QH9Stable dataset to through {self.url}")
            gdown.download(self.url, output=self.raw_paths[0], fuzzy=True)
        except:
            print(
                f"Downloading failed! Please download the QH9Stable dataset to {self.raw_paths[0]} through {self.url}"
            )
            print(f"Or you can try to download the zip file through {GoogleDriveLink}")
            raise FileNotFoundError(f"QH9Stable needs to be downloaded.")

    @property
    def raw_file_names(self):
        return [f"QH9Stable.db"]

    @property
    def processed_file_names(self):
        if self.split == "random":
            return ["processed_QH9Stable_random_12.pt", "QH9Stable.lmdb/data.mdb"]
        elif self.split == "size_ood":
            return ["processed_QH9Stable_size_ood.pt", "QH9Stable.lmdb/data.mdb"]

    def process_row(self, row):
        # Process a single row: extract data, compute values and prepare the dictionary.
        atoms = np.frombuffer(row[3], np.int32)
        pos = np.frombuffer(row[4], np.float64)
        ovlp, init_ham = self.calc_ovlp_and_ham_init(atoms, pos)
        ori_data_dict = {
            "id": row[0],
            "geo_id": row[1],
            "num_nodes": row[2],
            "atoms": row[2],
            "pos": row[3],  # angstrom
            "Ham": row[4],
            "ovlp": ovlp.tobytes(),
            "init_ham": init_ham.tobytes(),
        }
        key = ori_data_dict["id"].to_bytes(length=4, byteorder="big")
        data_dict = pickle.dumps(ori_data_dict)
        return key, data_dict

    def process(self):
        assert pyscf.__version__ == "2.2.1"  # pyscf (2.2.1) (QH9-Stable)

        if self.split == "size_ood":
            num_nodes_list = []

        for raw_file_name in self.raw_file_names:
            connection = Connection(os.path.join(self.root, "raw", raw_file_name))
            cursor = connection.cursor()
            data = cursor.execute("select * from data").fetchall()

            lmdb_path = os.path.join(self.processed_dir, "QH9Stable.lmdb")
            if not os.path.isdir(lmdb_path):
                # dataloader with lmdb
                db_env = lmdb.open(lmdb_path, map_size=1048576000000)
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                    results = list(
                        tqdm(executor.map(self.process_row, data), total=len(data))
                    )

                with db_env.begin(write=True) as txn:
                    for key, data_dict in results:
                        txn.put(key, data_dict)
                db_env.close()
                print("Saving lmdb database...")
            else:
                print("lmdb database exists. Jump the lmdb database creation step.")

        if self.split == "random":
            print("Random splitting...")
            data_ratio = [0.8, 0.1, 0.1]
            data_split = [
                int(len(data) * data_ratio[0]),
                int(len(data) * data_ratio[1]),
            ]
            data_split.append(len(data) - sum(data_split))
            indices = np.random.RandomState(seed=43).permutation(len(data))
            train_mask = indices[: data_split[0]]
            val_mask = indices[data_split[0] : data_split[0] + data_split[1]]
            test_mask = indices[data_split[0] + data_split[1] :]

        elif self.split == "size_ood":
            print("Size OOD splitting...")
            num_nodes_list = [row[1] for row in data]
            num_nodes_array = np.array(num_nodes_list)
            train_indices = np.where(num_nodes_array <= 20)
            val_condition = np.logical_and(num_nodes_array >= 21, num_nodes_array <= 22)
            val_indices = np.where(val_condition)
            test_indices = np.where(num_nodes_array >= 23)
            train_mask = train_indices[0].astype(np.int64)
            val_mask = val_indices[0].astype(np.int64)
            test_mask = test_indices[0].astype(np.int64)

        torch.save((train_mask, val_mask, test_mask), self.processed_paths[0])

    def process_old(self):
        assert pyscf.__version__ == "2.2.1"  # pyscf (2.2.1) (QH9-Stable)

        if self.split == "size_ood":
            num_nodes_list = []

        for raw_file_name in self.raw_file_names:
            connection = Connection(os.path.join(self.root, "raw", raw_file_name))
            cursor = connection.cursor()
            data = cursor.execute("select * from data").fetchall()
            if not os.path.isdir(os.path.join(self.processed_dir, "QH9Stable.lmdb")):
                # dataloader with lmdb
                db_env = lmdb.open(
                    os.path.join(self.processed_dir, "QH9Stable.lmdb"),
                    map_size=1048576000000,
                )
                for row in tqdm(data):
                    atoms = np.frombuffer(row[2], np.int32)
                    pos = np.frombuffer(row[3], np.float64)
                    ovlp, init_ham = self.calc_ovlp_and_ham_init(atoms, pos)
                    with db_env.begin(write=True) as txn:
                        ori_data_dict = {
                            "id": row[0],
                            "num_nodes": row[1],
                            "atoms": row[2],
                            "pos": row[3],  # ang
                            "Ham": row[4],
                            "ovlp": ovlp.tobytes(),
                            "init_ham": init_ham.tobytes(),
                        }
                        data_dict = pickle.dumps(ori_data_dict)
                        txn.put(
                            ori_data_dict["id"].to_bytes(length=4, byteorder="big"),
                            data_dict,
                        )
                db_env.close()
                print("Saving lmdb database...")
            else:
                print("lmdb database exists. Jump the lmdb database creation step.")

        if self.split == "random":
            print("Random splitting...")
            data_ratio = [0.8, 0.1, 0.1]
            data_split = [
                int(len(data) * data_ratio[0]),
                int(len(data) * data_ratio[1]),
            ]
            data_split.append(len(data) - sum(data_split))
            indices = np.random.RandomState(seed=43).permutation(len(data))
            train_mask = indices[: data_split[0]]
            val_mask = indices[data_split[0] : data_split[0] + data_split[1]]
            test_mask = indices[data_split[0] + data_split[1] :]

        elif self.split == "size_ood":
            print("Size OOD splitting...")
            num_nodes_list = [row[1] for row in data]
            num_nodes_array = np.array(num_nodes_list)
            train_indices = np.where(num_nodes_array <= 20)
            val_condition = np.logical_and(num_nodes_array >= 21, num_nodes_array <= 22)
            val_indices = np.where(val_condition)
            test_indices = np.where(num_nodes_array >= 23)
            train_mask = train_indices[0].astype(np.int64)
            val_mask = val_indices[0].astype(np.int64)
            test_mask = test_indices[0].astype(np.int64)

        torch.save((train_mask, val_mask, test_mask), self.processed_paths[0])

    def cut_matrix(self, matrix, atoms):
        all_diagonal_matrix_blocks = []
        all_non_diagonal_matrix_blocks = []
        all_diagonal_matrix_block_masks = []
        all_non_diagonal_matrix_block_masks = []
        col_idx = 0
        edge_index_full = []
        for idx_i, atom_i in enumerate(atoms):  # (src)
            row_idx = 0
            atom_i = atom_i.item()
            mask_i = self.orbital_mask[atom_i]
            for idx_j, atom_j in enumerate(atoms):  # (dst)
                if idx_i != idx_j:
                    edge_index_full.append([idx_j, idx_i])
                atom_j = atom_j.item()
                mask_j = self.orbital_mask[atom_j]

                if len(matrix.size()) == 2:
                    matrix_block = torch.zeros(
                        self.full_orbitals, self.full_orbitals
                    ).type(torch.float64)
                    matrix_block_mask = torch.zeros(
                        self.full_orbitals, self.full_orbitals
                    ).type(torch.float64)
                else:
                    len_ = matrix.size(-1)
                    matrix_block = torch.zeros(
                        self.full_orbitals, self.full_orbitals, len_
                    ).type(torch.float64)
                    matrix_block_mask = torch.zeros(
                        self.full_orbitals, self.full_orbitals, len_
                    ).type(torch.float64)

                extracted_matrix = matrix[
                    row_idx : row_idx + len(mask_j), col_idx : col_idx + len(mask_i)
                ]

                # for matrix_block
                tmp = matrix_block[mask_j]
                tmp[:, mask_i] = extracted_matrix
                matrix_block[mask_j] = tmp

                tmp = matrix_block_mask[mask_j]
                tmp[:, mask_i] = 1
                matrix_block_mask[mask_j] = tmp

                if idx_i == idx_j:
                    all_diagonal_matrix_blocks.append(matrix_block)
                    all_diagonal_matrix_block_masks.append(matrix_block_mask)
                else:
                    all_non_diagonal_matrix_blocks.append(matrix_block)
                    all_non_diagonal_matrix_block_masks.append(matrix_block_mask)
                row_idx = row_idx + len(mask_j)
            col_idx = col_idx + len(mask_i)
        return (
            torch.stack(all_diagonal_matrix_blocks, dim=0),
            torch.stack(all_non_diagonal_matrix_blocks, dim=0),
            torch.stack(all_diagonal_matrix_block_masks, dim=0),
            torch.stack(all_non_diagonal_matrix_block_masks, dim=0),
            torch.tensor(edge_index_full).transpose(-1, -2),
        )

    def get_mol(self, atoms, pos, Ham, ovlp, int_ham):
        hamiltonian = torch.tensor(
            matrix_transform(Ham, atoms, convention="pyscf_def2svp"),
            dtype=torch.float64,
        )
        (
            diagonal_hamiltonian,
            non_diagonal_hamiltonian,
            diagonal_hamiltonian_mask,
            non_diagonal_hamiltonian_mask,
            edge_index_full,
        ) = self.cut_matrix(hamiltonian, atoms)

        init_ham = torch.tensor(
            matrix_transform(init_ham, atoms, convention="pyscf_def2svp"),
            dtype=torch.float64,
        )

        (
            diagonal_init_ham,
            non_diagonal_init_ham,
            diagonal_init_ham_mask,
            non_diagonal_init_ham_mask,
            _,
        ) = self.cut_matrix(init_ham, atoms)

        overlap = torch.tensor(
            matrix_transform(ovlp, atoms, convention="pyscf_def2svp"),
            dtype=torch.float64,
        )
        diagonal_overlap, non_diagonal_overlap, _, _, _ = self.cut_matrix(
            overlap, atoms
        )

        Q = (
            torch.stack(
                [
                    torch.block_diag(*[self.Q_dict[z][l] for z in atoms])
                    for l in range(60)
                ]
            )
            .double()
            .numpy()
        )
        Q = (
            torch.from_numpy(matrix_transform(Q, atoms, convention="pyscf_def2svp"))
            .double()
            .permute(1, 2, 0)
        )
        H_size = hamiltonian.size(-1)
        Q[:, :, 16:40] = (
            Q[:, :, 16:40]
            .reshape(H_size, H_size, -1, 3)[:, :, :, [1, 2, 0]]
            .reshape(H_size, H_size, 24)
        )
        diagonal_Q, non_diagonal_Q, _, _, _ = self.cut_matrix(Q, atoms)

        data = Data(
            pos=torch.tensor(pos, dtype=torch.float64),
            atoms=torch.tensor(atoms, dtype=torch.int64).view(-1, 1),
            diagonal_hamiltonian=diagonal_hamiltonian,
            non_diagonal_hamiltonian=non_diagonal_hamiltonian,
            diagonal_hamiltonian_mask=diagonal_hamiltonian_mask,
            non_diagonal_hamiltonian_mask=non_diagonal_hamiltonian_mask,
            diagonal_init_ham=diagonal_init_ham,
            non_diagonal_init_ham=non_diagonal_init_ham,
            diagonal_init_ham_mask=diagonal_init_ham_mask,
            non_diagonal_init_ham_mask=non_diagonal_init_ham_mask,
            diagonal_overlap=diagonal_overlap,
            non_diagonal_overlap=non_diagonal_overlap,
            diagonal_Q=diagonal_Q,
            non_diagonal_Q=non_diagonal_Q,
            edge_index_full=edge_index_full,
        )
        return data

    def get(self, idx):
        db_env = lmdb.open(
            os.path.join(self.processed_dir, "QH9Stable.lmdb"),
            readonly=True,
            lock=False,
        )
        with db_env.begin() as txn:
            data_dict = txn.get(int(idx).to_bytes(length=4, byteorder="big"))
            data_dict = pickle.loads(data_dict)
            _, num_nodes, atoms, pos, Ham, ovlp, init_ham = (
                data_dict["id"],
                data_dict["num_nodes"],
                np.frombuffer(data_dict["atoms"], np.int32),
                np.frombuffer(data_dict["pos"], np.float64),
                np.frombuffer(data_dict["Ham"], np.float64),
                np.frombuffer(data_dict["ovlp"], np.float64),
                np.frombuffer(data_dict["init_ham"], np.float64),
            )
            pos = pos.reshape(num_nodes, 3)
            num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
            Ham = Ham.reshape(num_orbitals, num_orbitals)
            init_ham = init_ham.reshape(num_orbitals, num_orbitals)
            ovlp = ovlp.reshape(num_orbitals, num_orbitals)
            data = self.get_mol(atoms, pos, Ham, ovlp, init_ham)
        db_env.close()
        return data

    def calc_ovlp(self, atoms, pos):
        mol = gto.Mole()
        t = [[atoms[atom_idx], pos[atom_idx]] for atom_idx in range(len(atoms))]
        mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")

        ovlp = mol.intor("int1e_ovlp")

        return ovlp

    def calc_ovlp_and_ham_init(self, atoms, pos, init="minao"):
        mol = gto.Mole()
        t = [[atoms[atom_idx], pos[atom_idx]] for atom_idx in range(len(atoms))]
        mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")

        ovlp = mol.intor("int1e_ovlp")
        mf_hf = dft.RKS(mol)
        mf_hf.xc = "b3lyp"
        mf_hf.basis = "def2svp"
        if init == "minao":
            init_dm = mf_hf.init_guess_by_minao()
        elif init == "1e":
            init_dm = mf_hf.init_guess_by_1e()
        # init_ham = scf.hf.get_fock(mf_hf, dm=init_dm)
        init_ham = mf_hf.get_fock(dm=init_dm)

        return ovlp, init_ham


class QH9Dynamic(InMemoryDataset):
    url = {
        "100k": "https://drive.google.com/file/d/1SNWk0GD6Nt96qNAJJU2uedwWDQ4bbB1w/view?usp=sharing",
        "300k": "https://drive.google.com/file/d/1sbf-sFhh3ZmhXgTcN2ke_la39MaG0Yho/view?usp=sharing",
    }

    def __init__(
        self,
        root="datasets/",
        task="",
        split="geometry",
        version="300k",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        assert task in [""]
        self.version = version
        if self.version == "300k":
            self.folder = osp.join(root, "QH9Dynamic_300k_2")
        elif self.version == "100k":
            self.folder = osp.join(root, "QH9Dynamic_100k")
        else:
            print(
                f"Current version parameter is {version}, which is not included in [100k, 300k]."
            )
            print(f"Using 300k version instead...")
            self.folder = osp.join(root, "QH9Dynamic_300k")
            self.version = "300k"

        self.split = split
        self.db_dir = os.path.join(self.folder, "processed")
        self.full_orbitals = 14
        self.orbital_mask = {}
        idx_1s_2s_2p = torch.tensor([0, 1, 3, 4, 5])
        orbital_mask_line1 = idx_1s_2s_2p
        orbital_mask_line2 = torch.arange(self.full_orbitals)
        for i in range(1, 11):
            self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

        atom_list = ["H", "C", "N", "O", "F"]

        self.Q_dict = Onsite_3idx_Overlap_Integral(
            atom_list=atom_list, basis="def2-svp"
        ).Q_table()

        super(QH9Dynamic, self).__init__(
            self.folder, transform, pre_transform, pre_filter
        )
        self.train_mask, self.val_mask, self.test_mask = torch.load(
            self.processed_paths[0]
        )
        self.slices = {
            "id": torch.arange(
                self.train_mask.shape[0]
                + self.val_mask.shape[0]
                + self.test_mask.shape[0]
                + 1
            )
        }

    def download(self):
        try:
            print(
                f"Downloading the QH9Dynamic_{self.version} dataset through {self.url}"
            )
            gdown.download(self.url[self.version], output=self.raw_paths[0], fuzzy=True)
        except:
            print(
                f"Downloading failed! Please download the QH9Dynamic_{self.version} dataset to {self.raw_paths[0]} "
                f"through {self.url[self.version]}"
            )
            print(f"Or you can try to download the zip file through {GoogleDriveLink}")
            raise FileNotFoundError(
                f"QH9Dynamic_{self.version} needs to be downloaded."
            )

    @property
    def raw_file_names(self):
        if self.version == "300k":
            return [f"QH9Dynamic_300k.db"]
        elif self.version == "100k":
            return [f"QH9Dynamic_100k.db"]

    @property
    def processed_file_names(self):
        if self.split == "geometry":
            return ["processed_QH9Dynamic_geometry.pt", "QH9Dynamic.lmdb/data.mdb"]
        elif self.split == "mol":
            return ["processed_QH9Dynamic_mol.pt", "QH9Dynamic.lmdb/data.mdb"]

    @staticmethod
    def process_row(row, calc_func):
        # Process a single row: extract data, compute values and prepare the dictionary.
        atoms = np.frombuffer(row[2], np.int32)
        pos = np.frombuffer(row[3], np.float64)
        ovlp, init_ham = calc_func(atoms, pos)
        ori_data_dict = {
            "id": row[0],
            "num_nodes": row[1],
            "atoms": row[2],
            "pos": row[3],  # angstrom
            "Ham": row[4],
            "ovlp": ovlp.tobytes(),
            "init_ham": init_ham.tobytes(),
        }
        data_dict = pickle.dumps(ori_data_dict)
        key = ori_data_dict["id"].to_bytes(length=4, byteorder="big")
        return key, data_dict

    def process(
        self,
        num_geometry_per_mol=100,
        num_train_geometry_per_mol=80,
        num_val_geometry_per_mol=10,
        num_mol=2998,
    ):
        if self.version == "300k":
            # assert pyscf.__version__ == "2.2.1"  # pyscf (2.2.1) (QH9-Dynamic-300k)

            num_mol = 2998
        elif self.version == "100k":
            # assert pyscf.__version__ == "2.3.0"  # pyscf (2.3.0) (QH9-Dynamic-100k)

            num_mol = 999

        if not os.path.isdir(os.path.join(self.processed_dir, "QH9Dynamic.lmdb")):
            SAVE_TO_LMDB = True
            print("lmdb database is created.")
        else:
            SAVE_TO_LMDB = False
            print("lmdb database exists. Jump the lmdb database creation step.")

        db_env = lmdb.open(
            os.path.join(self.processed_dir, "QH9Dynamic.lmdb"), map_size=1048576000000
        )
        if self.split == "geometry":
            print("Geometry-wise splitting...")
            train_mask = np.array([], dtype=np.int64)
            val_mask = np.array([], dtype=np.int64)
            test_mask = np.array([], dtype=np.int64)
            cur_index = 0
            for raw_file_name in self.raw_file_names:
                connection = Connection(os.path.join(self.root, "raw", raw_file_name))
                cursor = connection.cursor()
                data = cursor.execute("select * from data").fetchall()
                import pdb

                pdb.set_trace()
                for ind, row in enumerate(tqdm(data)):
                    atoms = np.frombuffer(row[3], np.int32)
                    pos = np.frombuffer(row[4], np.float64) / BOHR2ANG
                    ovlp, init_ham = self.calc_ovlp_and_ham_init(atoms, pos)

                    # dataloader with lmdb
                    if SAVE_TO_LMDB:
                        with db_env.begin(write=True) as txn:
                            ori_data_dict = {
                                "id": int(np.frombuffer(row[0], dtype=np.int64)),
                                "geo_id": row[1],
                                "num_nodes": row[2],
                                "atoms": row[3],
                                "pos": row[4],  # Bhor, and then transfer to ang
                                "Ham": row[9],
                                "ovlp": ovlp.tobytes(),
                                "init_ham": init_ham.tobytes(),
                            }
                            data_dict = pickle.dumps(ori_data_dict)
                            txn.put(ind.to_bytes(length=4, byteorder="big"), data_dict)

                    if (
                        ind + 1
                    ) % num_geometry_per_mol == 0:  # Finish traversing one molecule
                        indices = np.random.RandomState(seed=ind).permutation(
                            num_geometry_per_mol
                        )  # Different random split for different molecules
                        train_mask_cur_mol = (
                            cur_index + indices[:num_train_geometry_per_mol]
                        )
                        val_mask_cur_mol = (
                            cur_index
                            + indices[
                                num_train_geometry_per_mol : num_train_geometry_per_mol
                                + num_val_geometry_per_mol
                            ]
                        )
                        test_mask_cur_mol = (
                            cur_index
                            + indices[
                                num_train_geometry_per_mol + num_val_geometry_per_mol :
                            ]
                        )
                        train_mask = np.concatenate((train_mask, train_mask_cur_mol))
                        val_mask = np.concatenate((val_mask, val_mask_cur_mol))
                        test_mask = np.concatenate((test_mask, test_mask_cur_mol))
                        cur_index += num_geometry_per_mol

        elif self.split == "mol":
            print("Molecule-wise splitting...")
            for raw_file_name in self.raw_file_names:
                connection = Connection(os.path.join(self.root, "raw", raw_file_name))
                cursor = connection.cursor()
                data = cursor.execute("select * from data").fetchall()
                if SAVE_TO_LMDB:
                    for ind, row in enumerate(tqdm(data)):
                        atoms = np.frombuffer(row[3], np.int32)
                        pos = np.frombuffer(row[4], np.float64) / BOHR2ANG
                        ovlp, init_ham = self.calc_ovlp_and_ham_init(atoms, pos)

                        # dataloader with lmdb
                        with db_env.begin(write=True) as txn:
                            ori_data_dict = {
                                "id": int(np.frombuffer(row[0], dtype=np.int64)),
                                "geo_id": row[1],
                                "num_nodes": row[2],
                                "atoms": row[3],
                                "pos": row[4],  # Bhor, and then transfer to ang
                                "Ham": row[9],
                                "ovlp": ovlp.tobytes(),
                                "init_ham": init_ham.tobytes(),
                            }
                            data_dict = pickle.dumps(ori_data_dict)
                            txn.put(ind.to_bytes(length=4, byteorder="big"), data_dict)

            if SAVE_TO_LMDB:
                db_env.close()

            mol_id_list = [
                i for i in range(num_mol) for _ in range(num_geometry_per_mol)
            ]
            data_ratio = [0.8, 0.1, 0.1]
            index_list = [i for i in range(max(mol_id_list) + 1)]
            random.seed(43)
            random.shuffle(index_list)
            train_mol_ids = np.array(index_list[: int(len(index_list) * data_ratio[0])])
            val_mol_ids = np.array(
                index_list[
                    int(len(index_list) * data_ratio[0]) : int(
                        len(index_list) * data_ratio[0]
                    )
                    + int(len(index_list) * data_ratio[1])
                ]
            )
            test_mol_ids = np.array(
                index_list[
                    int(len(index_list) * data_ratio[0])
                    + int(len(index_list) * data_ratio[1]) :
                ]
            )
            mol_id_array = np.array(mol_id_list)
            train_mask = np.where(np.isin(mol_id_array, train_mol_ids))[0].astype(
                np.int64
            )
            val_mask = np.where(np.isin(mol_id_array, val_mol_ids))[0].astype(np.int64)
            test_mask = np.where(np.isin(mol_id_array, test_mol_ids))[0].astype(
                np.int64
            )
        print("Saving...")

        torch.save((train_mask, val_mask, test_mask), self.processed_paths[0])

    def cut_matrix(self, matrix, atoms):
        all_diagonal_matrix_blocks = []
        all_non_diagonal_matrix_blocks = []
        all_diagonal_matrix_block_masks = []
        all_non_diagonal_matrix_block_masks = []
        col_idx = 0
        edge_index_full = []
        for idx_i, atom_i in enumerate(atoms):  # (src)
            row_idx = 0
            atom_i = atom_i.item()
            mask_i = self.orbital_mask[atom_i]
            for idx_j, atom_j in enumerate(atoms):  # (dst)
                atom_j = atom_j.item()
                if idx_i != idx_j:
                    edge_index_full.append([idx_j, idx_i])
                mask_j = self.orbital_mask[atom_j]

                if len(matrix.size()) == 2:
                    matrix_block = torch.zeros(
                        self.full_orbitals, self.full_orbitals
                    ).type(torch.float64)
                    matrix_block_mask = torch.zeros(
                        self.full_orbitals, self.full_orbitals
                    ).type(torch.float64)
                else:
                    len_ = matrix.size(-1)
                    matrix_block = torch.zeros(
                        self.full_orbitals, self.full_orbitals, len_
                    ).type(torch.float64)
                    matrix_block_mask = torch.zeros(
                        self.full_orbitals, self.full_orbitals, len_
                    ).type(torch.float64)

                extracted_matrix = matrix[
                    row_idx : row_idx + len(mask_j), col_idx : col_idx + len(mask_i)
                ]

                # for matrix_block
                tmp = matrix_block[mask_j]
                tmp[:, mask_i] = extracted_matrix
                matrix_block[mask_j] = tmp

                tmp = matrix_block_mask[mask_j]
                tmp[:, mask_i] = 1
                matrix_block_mask[mask_j] = tmp

                if idx_i == idx_j:
                    all_diagonal_matrix_blocks.append(matrix_block)
                    all_diagonal_matrix_block_masks.append(matrix_block_mask)
                else:
                    all_non_diagonal_matrix_blocks.append(matrix_block)
                    all_non_diagonal_matrix_block_masks.append(matrix_block_mask)
                row_idx = row_idx + len(mask_j)
            col_idx = col_idx + len(mask_i)
        return (
            torch.stack(all_diagonal_matrix_blocks, dim=0),
            torch.stack(all_non_diagonal_matrix_blocks, dim=0),
            torch.stack(all_diagonal_matrix_block_masks, dim=0),
            torch.stack(all_non_diagonal_matrix_block_masks, dim=0),
            torch.tensor(edge_index_full).transpose(-1, -2),
        )

    def get_mol(self, atoms, pos, Ham, ovlp, init_ham):
        hamiltonian = torch.tensor(
            matrix_transform(Ham, atoms, convention="pyscf_def2svp"),
            dtype=torch.float64,
        )
        (
            diagonal_hamiltonian,
            non_diagonal_hamiltonian,
            diagonal_hamiltonian_mask,
            non_diagonal_hamiltonian_mask,
            edge_index_full,
        ) = self.cut_matrix(hamiltonian, atoms)

        init_ham = torch.tensor(
            matrix_transform(init_ham, atoms, convention="pyscf_def2svp"),
            dtype=torch.float64,
        )

        (
            diagonal_init_ham,
            non_diagonal_init_ham,
            diagonal_init_ham_mask,
            non_diagonal_init_ham_mask,
            _,
        ) = self.cut_matrix(init_ham, atoms)

        overlap = torch.tensor(
            matrix_transform(ovlp, atoms, convention="pyscf_def2svp"),
            dtype=torch.float64,
        )
        diagonal_overlap, non_diagonal_overlap, _, _, _ = self.cut_matrix(
            overlap, atoms
        )

        Q = (
            torch.stack(
                [
                    torch.block_diag(*[self.Q_dict[z][l] for z in atoms])
                    for l in range(60)
                ]
            )
            .double()
            .numpy()
        )
        Q = (
            torch.from_numpy(matrix_transform(Q, atoms, convention="pyscf_def2svp"))
            .double()
            .permute(1, 2, 0)
        )
        H_size = hamiltonian.size(-1)
        Q[:, :, 16:40] = (
            Q[:, :, 16:40]
            .reshape(H_size, H_size, -1, 3)[:, :, :, [1, 2, 0]]
            .reshape(H_size, H_size, 24)
        )
        diagonal_Q, non_diagonal_Q, _, _, _ = self.cut_matrix(Q, atoms)

        data = Data(
            pos=torch.tensor(pos, dtype=torch.float64),
            atoms=torch.tensor(atoms, dtype=torch.int64).view(-1, 1),
            diagonal_hamiltonian=diagonal_hamiltonian,
            non_diagonal_hamiltonian=non_diagonal_hamiltonian,
            diagonal_hamiltonian_mask=diagonal_hamiltonian_mask,
            non_diagonal_hamiltonian_mask=non_diagonal_hamiltonian_mask,
            diagonal_init_ham=diagonal_init_ham,
            non_diagonal_init_ham=non_diagonal_init_ham,
            diagonal_init_ham_mask=diagonal_init_ham_mask,
            non_diagonal_init_ham_mask=non_diagonal_init_ham_mask,
            diagonal_overlap=diagonal_overlap,
            non_diagonal_overlap=non_diagonal_overlap,
            diagonal_Q=diagonal_Q,
            non_diagonal_Q=non_diagonal_Q,
            edge_index_full=edge_index_full,
        )
        return data

    def get(self, idx):
        db_env = lmdb.open(
            os.path.join(self.processed_dir, "QH9Dynamic.lmdb"),
            readonly=True,
            lock=False,
        )
        with db_env.begin() as txn:
            data_dict = txn.get(int(idx).to_bytes(length=4, byteorder="big"))
            data_dict = pickle.loads(data_dict)
            _, num_nodes, atoms, pos, Ham, ovlp, init_ham = (
                data_dict["id"],
                data_dict["num_nodes"],
                np.frombuffer(data_dict["atoms"], np.int32),
                np.frombuffer(data_dict["pos"], np.float64),
                np.frombuffer(data_dict["Ham"], np.float64),
                np.frombuffer(data_dict["ovlp"], np.float64),
                np.frombuffer(data_dict["init_ham"], np.float64),
            )
            pos = pos.reshape(num_nodes, 3)
            pos = pos / BOHR2ANG  # transfer the unit back to ang
            # pos = pos
            num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
            Ham = Ham.reshape(num_orbitals, num_orbitals)
            init_ham = init_ham.reshape(num_orbitals, num_orbitals)
            ovlp = ovlp.reshape(num_orbitals, num_orbitals)
            data = self.get_mol(atoms, pos, Ham, ovlp, init_ham)
        db_env.close()
        return data

    def calc_ovlp(self, atoms, pos):
        mol = gto.Mole()
        t = [[atoms[atom_idx], pos[atom_idx]] for atom_idx in range(len(atoms))]
        mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")

        ovlp = mol.intor("int1e_ovlp")
        return ovlp

    def calc_ovlp_and_ham_init(self, atoms, pos, init="minao"):
        mol = gto.Mole()
        t = [[atoms[atom_idx], pos[atom_idx]] for atom_idx in range(len(atoms))]
        mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")

        ovlp = mol.intor("int1e_ovlp")
        mf_hf = dft.RKS(mol)
        mf_hf.xc = "b3lyp"
        if init == "minao":
            init_dm = mf_hf.init_guess_by_minao()
        elif init == "1e":
            init_dm = mf_hf.init_guess_by_1e()
        init_ham = scf.hf.get_fock(mf_hf, dm=init_dm)

        return ovlp, init_ham


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generation")
    parser.add_argument(
        "--root",
        type=str,
        default="/home/seongsukim/dft/DEQHNet/src/QHNet_flow/dataset",
    )
    parser.add_argument("--name", type=str, default="QH9Stable")
    parser.add_argument("--version", type=str, default="300k")

    args = parser.parse_args()
    assert args.name in ["QH9Stable", "QH9Dynamic"]
    if args.name == "QH9Stable":
        dataset = QH9Stable(root=args.root)
    elif args.name == "QH9Dynamic":
        assert args.version in ["100k", "300k"]
        dataset = QH9Dynamic(root=args.root, version=args.version)
    print(dataset[0])
