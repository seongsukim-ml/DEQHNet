import warnings
import torch
from torch import Tensor
from torch_geometric.data import Data
from pyscf import gto
from typing import Optional, List
from tqdm import tqdm
from copy import deepcopy
from ase.data import chemical_symbols, atomic_numbers
import psi4


def build_molecule(Z, pos):
    res = ""
    for i in range(len(Z)):
        res += f"{chemical_symbols[Z[i]]} {pos[i][0]} {pos[i][1]} {pos[i][2]};"

    return res[:-1]


def build_AO_index(atom, basis):
    r"""`AO_index` is a (2, |AO|)-shape tensor, which means AO per atom.
    For example, the AO index of H2 molecule is
    AO_index = [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    where [0, 0, 0, 1, 1, 1] means each H atom uses three AOs respectively, and [0, 0, 0, 0, 0, 0] means all these AOs belong to a molecule.
    """
    ao_map = {"s": 0, "p": 1, "d": 2, "f": 3}
    mol = gto.M(atom=atom, basis=basis)
    AO_index = torch.tensor(
        [
            [int(i.split()[0]) for i in mol.ao_labels()],
            [ao_map[i.split()[-1][1]] for i in mol.ao_labels()],
            [0 for _ in range(mol.nao)],
        ]
    ).long()
    return AO_index


def AO2Irreps(AO: List[int]):
    ao_map1 = {0: 1, 1: 3, 2: 5, 3: 7}
    ao_map2 = {0: "e", 1: "o", 2: "e", 3: "o"}

    irreps = ""
    count = 1
    for i in range(1, len(AO)):
        if AO[i] == AO[i - 1]:
            count += 1
        else:
            irreps += f"{count // ao_map1[AO[i-1]]}x{AO[i-1]}{ao_map2[AO[i-1]]}+"
            count = 1

    irreps += f"{count // ao_map1[AO[i-1]]}x{AO[i-1]}{ao_map2[AO[i-1]]}"
    return irreps


class AOData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "AO_index":
            return torch.tensor([[self.num_nodes], [0], [1]])
        else:
            return super(AOData, self).__inc__(key, value, *args, **kwargs)


class Onsite_3idx_Overlap_Integral:
    r"""The on-site three-index overlap integral :math:`\tilde{Q}` from the
    `"Informing geometric deep learning with electronic interactionsto accelerate quantum chemistry"
    <https://www.pnas.org/doi/epdf/10.1073/pnas.2205221119>`_ paper

    .. math::
        \tilde{Q}^{n,l,m}^{\mu, \nu} = \int_{r\in\mathbb{R}^3} (\Phi_A^{n_1, l_1, m_1}(r))^*\Phi_A^{n_2, l_2, m_3}(r)
        \tilde{\Phi}_A^{n, l, m} (r) dr
    """

    def __init__(self, atom_list: Optional[List[str]] = None, basis: str = "def2-svp"):
        self.atom_list = (
            chemical_symbols[1:58] if atom_list is None else atom_list
        )  # `def2-svp` only support from H to La

        if basis != "def2-svp":
            warnings.warn(
                f"The class is only tested for `def2-svp` rather than {basis}. Be careful!!!"
            )

        self.basis = basis

    def calc_Q(self, atom: str):
        psi4.core.be_quiet()

        # NOTE: prevent Psi4 from moving the molecule in space.
        # Refer to https://forum.psicode.org/t/how-to-align-the-atomic-orbitals-between-pyscf-and-psi4/3025/2
        mol = psi4.geometry(
            f"""
            {atom} 0 0 0
            nocom
            noreorient
        """
        )

        # Basis Set
        psi4.set_options({"basis": self.basis})

        # Build new wavefunction
        wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("basis"))

        # Initialize MintsHelper with wavefunction's basis set
        mints = psi4.core.MintsHelper(wfn.basisset())

        # Refer to https://psicode.org/psi4manual/4.0b5/quickaddbasis.html
        psi4.set_options({"basis": "auxiliary"})

        # Build new wavefunction
        wfn_aux = psi4.core.Wavefunction.build(
            mol, psi4.core.get_global_option("basis")
        )

        Q = mints.ao_3coverlap(wfn.basisset(), wfn.basisset(), wfn_aux.basisset())
        return torch.from_numpy(Q.np.T).double()

    def Q_table(self):
        Q_dict = {}
        for atom in tqdm(
            self.atom_list, desc="Building on-site three-index overlap integral table"
        ):
            Q = self.calc_Q(atom)
            Q_dict[atomic_numbers[atom]] = self.transform_Q(Q, atom)

        return Q_dict

    @property
    def AO_transform_row(self):  # psi4 -> pyscf
        """The on-site three-index overlap integral is calculated with `psi4`, while Hamiltonian obtained from `psycf`,
        which represents Hamiltonian under AOs with different ordering from that in `psi4`.

        Refer to https://psicode.org/psi4manual/master/prog_blas.html#how-to-name-orbital-bases-e-g-ao-so:
        If Spherical Harmonics are used, :math: `2L + 1` real combinations of the spherical harmonics are built
        from the :math: `(L+1)(L+2)/2` Cartesian Gaussians. Unlike Cartesian functions, these functions are all strictly normalized.
        Note that in PSI4, the real combinations of spherical harmonic functions are ordered as: :math: `0, 1+, 1-, 2+, 2-, \cdots`.

        Refer to https://github.com/pyscf/pyscf/blob/master/pyscf/lib/parameters.py#L68-L76:
        In `pyscf`,
            REAL_SPHERIC = (
                ('',),
                ('x', 'y', 'z'),
                ('xy', 'yz', 'z^2', 'xz', 'x2-y2',),
                ('-3', '-2', '-1', '+0', '+1', '+2', '+3'),
                ('-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3', '+4'),
                ('-5', '-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3', '+4', '+5'),
                ('-6', '-5', '-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3', '+4', '+5', '+6'),
            )
        Refer to https://github.com/pyscf/pyscf/issues/2123#issuecomment-1985951880:
        `p` functions are special in pyscf, as they don't follow the same order as functions and higher.
        Refer to https://github.com/pyscf/pyscf/blob/master/pyscf/symm/Dmatrix.py#L29:
        ```python
        if reorder_p and l == 1:
            D = D[[2,0,1]][:,[2,0,1]]
        ```
        """
        return {
            "s": torch.tensor([[1.0]]).double(),
            "p": torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
            ).double(),
            "d": torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            ).double(),
        }

    def parse_basis(self, atom: str):
        AOs, AO_slices = [], []
        tmp = 0
        for info in gto.format_basis({atom: self.basis})[atom]:
            AO_slices.append(AO_slices[-1] + tmp if len(AO_slices) != 0 else 0)

            if info[0] == 0:
                AOs.append("s")
                tmp = 1
            elif info[0] == 1:
                AOs.append("p")
                tmp = 3
            elif info[0] == 2:
                AOs.append("d")
                tmp = 5
            else:
                raise NotImplementedError

        AO_slices.append(AO_slices[-1] + tmp if len(AO_slices) != 0 else 0)
        return AOs, AO_slices

    def transform_Q(self, Q: Tensor, atom: str):
        AOs, AO_slices = self.parse_basis(atom)
        Q_aligned_orb = deepcopy(Q)
        for i, mu in enumerate(AOs):
            for j, nu in enumerate(AOs):
                Q_aligned_orb[
                    :, AO_slices[i] : AO_slices[i + 1], AO_slices[j] : AO_slices[j + 1]
                ] = torch.einsum(
                    "hik, kl -> hil",
                    torch.einsum(
                        "ij, hjk -> hik",
                        self.AO_transform_row[mu],
                        Q[
                            :,
                            AO_slices[i] : AO_slices[i + 1],
                            AO_slices[j] : AO_slices[j + 1],
                        ],
                    ),
                    self.AO_transform_row[nu].T,  # transpose for column
                )

        Q_aligned_aux = deepcopy(Q_aligned_orb)

        for i in range(8):
            Q_aligned_aux[16 + i * 3 : 16 + (i + 1) * 3] = torch.einsum(
                "ij, jkl -> ikl",
                self.AO_transform_row["p"],
                Q_aligned_orb[16 + i * 3 : 16 + (i + 1) * 3],
            )

        for i in range(4):
            Q_aligned_aux[40 + i * 5 : 40 + (i + 1) * 5] = torch.einsum(
                "ij, jkl -> ikl",
                self.AO_transform_row["d"],
                Q_aligned_orb[40 + i * 5 : 40 + (i + 1) * 5],
            )

        return Q_aligned_aux


"""
Generalized QR Decomposition in PyTorch
============

Author:
-------
Yuchao Lin

"""
import torch


@torch.jit.script
def find_independent_vectors_cuda(P):
    """Find rank(P) linearly independent vectors from P"""
    n = P.size(0)
    r = int(torch.linalg.matrix_rank(P.to(torch.float32)))

    indices = torch.arange(r)
    done = False
    while not done:
        subset = P[indices, :]
        if torch.linalg.matrix_rank(subset.to(torch.float32)) == r:
            return subset
        done = True
        for i in range(r - 1, -1, -1):
            if indices[i] != i + n - r:
                indices[i] += 1
                for j in range(i + 1, r):
                    indices[j] = indices[j - 1] + 1
                done = False
                break

    return None


@torch.jit.script
def find_independent_vectors_complex_cuda(P):
    """Find rank(P) linearly independent vectors from P"""
    n = P.size(0)
    r = int(torch.linalg.matrix_rank(P.to(torch.complex64)))

    indices = torch.arange(r)
    done = False
    while not done:
        subset = P[indices, :]
        if torch.linalg.matrix_rank(subset.to(torch.complex64)) == r:
            return subset
        done = True
        for i in range(r - 1, -1, -1):
            if indices[i] != i + n - r:
                indices[i] += 1
                for j in range(i + 1, r):
                    indices[j] = indices[j - 1] + 1
                done = False
                break

    return None


@torch.jit.script
def qr_decomposition(coords):
    """QR decomposition on induced set"""
    vecs = find_independent_vectors_cuda(coords)
    assert vecs is not None
    Q, R = torch.linalg.qr(vecs.transpose(0, 1), mode="complete")
    for j in range(R.size(1)):
        if R[j, j] < 0:
            R[j, :] *= -1
            Q[:, j] *= -1
    return Q, R


@torch.jit.script
def qr_decomposition_complex(coords):
    """Complex QR decomposition on induced set"""
    vecs = find_independent_vectors_complex_cuda(coords)
    assert vecs is not None
    Q, R = torch.linalg.qr(vecs.H, mode="complete")
    for j in range(R.size(1)):
        if R[j, j].real < 0:
            R[j, :] *= -1
            Q[:, j] *= -1
    return Q, R


@torch.jit.script
def inner_product(u, v, eta):
    return torch.dot(u, torch.mv(eta, v))


@torch.jit.script
def project(u, v, eta):
    norm_sq = inner_product(u, u, eta)
    assert norm_sq != 0.0
    coeff = inner_product(u, v, eta) / norm_sq
    return coeff * u


@torch.jit.script
def gram_schmidt(A, eta):
    """Generalized Gram-Schmidt orthogonalization"""
    m, n = A.size()
    metric_length = eta.size(0)
    Q = torch.zeros((m, metric_length), dtype=A.dtype).to(A.device)
    R = torch.zeros((metric_length, n), dtype=A.dtype).to(A.device)
    eta_c = torch.zeros(metric_length, dtype=A.dtype).to(A.device)

    for j in range(n):
        v = A[:, j]

        for i in range(j):
            v -= project(Q[:, i], A[:, j], eta)
        norm_sq = inner_product(v, v, eta)
        norm_sq = torch.sqrt(torch.abs(norm_sq))
        assert norm_sq != 0.0
        Q[:, j] = v / norm_sq
        Rjj = inner_product(Q[:, j], A[:, j], eta)
        if Rjj < 0:
            Q[:, j] = -Q[:, j]
            Rjj = -Rjj
        R[j, j] = Rjj
        eta_c[j] = torch.sign(inner_product(Q[:, j], Q[:, j], eta))
        for i in range(j):
            R[i, j] = inner_product(Q[:, i], A[:, j], eta)

    return Q, eta_c, R


@torch.jit.script
def generate_permutation(eta_c, eta, vecs):
    """Algorithm 2"""
    n, d = vecs.size()
    S = torch.eye(d).to(vecs.dtype)
    a = eta_c
    b = torch.diag(eta)

    for i in range(d):
        if a[i] != 0 and a[i] != b[i]:
            for j in range(i + 1, d):
                if a[j] != b[j] and a[j] == b[i]:
                    S_prime = torch.eye(d).to(vecs.dtype)
                    S_prime[i, i] = 0
                    S_prime[j, j] = 0
                    S_prime[i, j] = 1
                    S_prime[j, i] = 1
                    a = torch.matmul(a, S_prime)
                    S = torch.matmul(S, S_prime)
                    break
    return S.T


@torch.jit.script
def generate_minkowski_permutation_matrix(Q, eta):
    """Algorithm 2 but only for O(1,d-1)/SO(1,d-1)"""
    diag_elements = torch.diag(torch.matmul(Q.T, torch.matmul(eta, Q)))
    swap_index = int(torch.argmax(diag_elements).item())
    P = torch.eye(len(diag_elements)).to(Q.dtype).to(Q.device)
    P[0, 0] = 0.0
    P[swap_index, swap_index] = 0.0
    P[0, swap_index] = 1.0
    P[swap_index, 0] = 1.0
    return P


@torch.jit.script
def generalized_qr_decomposition(coords, eta):
    """Generalized QR decomposition"""
    vecs = find_independent_vectors_cuda(coords)
    assert vecs is not None
    Q, eta_c, R = gram_schmidt(vecs.transpose(0, 1), eta)
    # P = generate_minkowski_permutation_matrix(Q, eta)
    P = generate_permutation(eta_c, eta, vecs)
    Q = Q @ P
    R = P.T @ torch.diag(eta_c) @ R
    return Q, R
