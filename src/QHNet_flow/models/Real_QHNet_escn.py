import time

import torch.nn.functional as F
from .modules import *

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from e3nn import o3
from torch_scatter import scatter
from e3nn.nn import FullyConnectedNet, Gate, Activation
from e3nn.o3 import Linear, TensorProduct, FullyConnectedTensorProduct
from torch.nn.init import zeros_

# from .scn import *
from .scn.escn import MessageBlock
from .equiformer_v2.transformer_block import *
from .equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)
from .equiformer_v2.input_block import EdgeDegreeEmbedding
from .equiformer_v2.smearing import GaussianSmearing

import math
import logging
import torch

### Methods for sample points on a sphere


class ModuleListInfo(torch.nn.ModuleList):
    def __init__(self, info_str, modules=None) -> None:
        super().__init__(modules)
        self.info_str = str(info_str)

    def __repr__(self) -> str:
        return self.info_str


def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)


def softplus_inverse(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))


def get_nonlinear(nonlinear: str):
    if nonlinear.lower() == "ssp":
        return ShiftedSoftPlus
    elif nonlinear.lower() == "silu":
        return F.silu
    elif nonlinear.lower() == "tanh":
        return F.tanh
    elif nonlinear.lower() == "abs":
        return torch.abs
    else:
        raise NotImplementedError


def get_feasible_irrep(irrep_in1, irrep_in2, cutoff_irrep_out, tp_mode="uvu"):
    irrep_mid = []
    instructions = []

    for i, (_, ir_in) in enumerate(irrep_in1):
        for j, (_, ir_edge) in enumerate(irrep_in2):
            for ir_out in ir_in * ir_edge:
                if ir_out in cutoff_irrep_out:
                    if (cutoff_irrep_out.count(ir_out), ir_out) not in irrep_mid:
                        k = len(irrep_mid)
                        irrep_mid.append((cutoff_irrep_out.count(ir_out), ir_out))
                    else:
                        k = irrep_mid.index((cutoff_irrep_out.count(ir_out), ir_out))
                    instructions.append((i, j, k, tp_mode, True))

    irrep_mid = o3.Irreps(irrep_mid)
    normalization_coefficients = []
    for ins in instructions:
        ins_dict = {
            "uvw": (irrep_in1[ins[0]].mul * irrep_in2[ins[1]].mul),
            "uvu": irrep_in2[ins[1]].mul,
            "uvv": irrep_in1[ins[0]].mul,
            "uuw": irrep_in1[ins[0]].mul,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": irrep_in1[ins[0]].mul * (irrep_in2[ins[1]].mul - 1) // 2,
        }
        alpha = irrep_mid[ins[2]].ir.dim
        x = sum([ins_dict[ins[3]] for ins in instructions])
        if x > 0.0:
            alpha /= x
        normalization_coefficients += [math.sqrt(alpha)]

    irrep_mid, p, _ = irrep_mid.sort()
    instructions = [
        (i_in1, i_in2, p[i_out], mode, train, alpha)
        for (i_in1, i_in2, i_out, mode, train), alpha in zip(
            instructions, normalization_coefficients
        )
    ]
    return irrep_mid, instructions


class NormGate(torch.nn.Module):
    def __init__(self, irrep):
        super(NormGate, self).__init__()
        self.irrep = irrep
        self.norm = o3.Norm(self.irrep)

        num_mul, num_mul_wo_0 = 0, 0
        for mul, ir in self.irrep:
            num_mul += mul
            if ir.l != 0:
                num_mul_wo_0 += mul

        self.mul = o3.ElementwiseTensorProduct(
            self.irrep[1:], o3.Irreps(f"{num_mul_wo_0}x0e")
        )
        self.fc = nn.Sequential(
            nn.Linear(num_mul, num_mul), nn.SiLU(), nn.Linear(num_mul, num_mul)
        )

        self.num_mul = num_mul
        self.num_mul_wo_0 = num_mul_wo_0

    def forward(self, x):
        norm_x = self.norm(x)[:, self.irrep.slices()[0].stop :]
        f0 = torch.cat([x[:, self.irrep.slices()[0]], norm_x], dim=-1)
        gates = self.fc(f0)
        gated = self.mul(
            x[:, self.irrep.slices()[0].stop :], gates[:, self.irrep.slices()[0].stop :]
        )
        x = torch.cat([gates[:, self.irrep.slices()[0]], gated], dim=-1)
        return x


class SO2ConvNetLayer(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        sphere_channels_all: int,
        hidden_channels: int,
        edge_attr_dim: int,
        lmax_list: List[int],
        mmax_list: List[int],
        SO3_grid: SO3_Grid,
        act,
        distance_expansion,
        max_num_elements=10,
        resnet: bool = True,
        layer_idx: int = 0,
    ):
        super(SO2ConvNetLayer, self).__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.sphere_channels_all = sphere_channels_all
        self.edge_attr_dim = edge_attr_dim
        self.resnet = resnet
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.SO3_grid = SO3_grid
        self.act = act
        self.layer_idx = layer_idx

        self.message_block = MessageBlock(
            self.layer_idx,
            self.sphere_channels,
            self.hidden_channels,
            self.edge_attr_dim,
            self.lmax_list,
            self.mmax_list,
            distance_expansion,
            max_num_elements,
            self.SO3_grid,
            self.act,
        )
        # Non-linear point-wise comvolution for the aggregated messages
        self.fc1_sphere = nn.Linear(
            2 * self.sphere_channels_all, self.sphere_channels_all, bias=False
        )
        self.fc2_sphere = nn.Linear(
            self.sphere_channels_all, self.sphere_channels_all, bias=False
        )
        self.fc3_sphere = nn.Linear(
            self.sphere_channels_all, self.sphere_channels_all, bias=False
        )
        # self.irrep_hidden = o3.Irreps(
        #     [(self.sphere_channels_all, (l, 1)) for l in range(self.lmax_list[0] + 1)]
        # )
        self.irrep_hidden = o3.Irreps(
            [
                (self.sphere_channels_all, (l, 1 if l % 2 == 0 else -1))
                for l in range(self.lmax_list[0] + 1)
            ]
        )
        self.max_lmax = max(self.lmax_list)

        self.norm_1 = get_normalization_layer(
            "rms_norm_sh", lmax=self.max_lmax, num_channels=sphere_channels
        )
        self.norm_2 = get_normalization_layer(
            "rms_norm_sh", lmax=self.max_lmax, num_channels=sphere_channels
        )

    def forward(self, data, x_emb, SO3_edge_rot, mappingReduced):
        atomic_numbers = data.atoms.squeeze()
        edge_distance = data.edge_distance
        edge_index = data.edge_index

        x_res = x_emb.embedding
        x_emb.embedding = self.norm_1(x_res)

        x_message = self.message_block(
            x_emb,
            atomic_numbers,
            edge_distance,
            edge_index,
            SO3_edge_rot,
            mappingReduced,
        )
        # Compute point-wise spherical non-linearity on aggregated messages

        # Project to grid
        x_grid_message = x_message.to_grid(self.SO3_grid, lmax=self.max_lmax)
        x_grid = x_emb.to_grid(self.SO3_grid, lmax=self.max_lmax)
        x_grid = torch.cat([x_grid, x_grid_message], dim=3)

        # Perform point-wise convolution
        x_grid = self.act(self.fc1_sphere(x_grid))
        x_grid = self.act(self.fc2_sphere(x_grid))
        x_grid = self.fc3_sphere(x_grid)

        # Project back to spherical harmonic coefficients
        x_message._from_grid(x_grid, self.SO3_grid, lmax=self.max_lmax)

        x_message.embedding = x_message.embedding + x_res

        return x_message


class InnerProduct(torch.nn.Module):
    def __init__(self, irrep_in):
        super(InnerProduct, self).__init__()
        self.irrep_in = o3.Irreps(irrep_in).simplify()
        irrep_out = o3.Irreps([(mul, "0e") for mul, _ in self.irrep_in])
        instr = [
            (i, i, i, "uuu", False, 1 / ir.dim)
            for i, (mul, ir) in enumerate(self.irrep_in)
        ]
        self.tp = o3.TensorProduct(
            self.irrep_in,
            self.irrep_in,
            irrep_out,
            instr,
            irrep_normalization="component",
        )
        self.irrep_out = irrep_out.simplify()

    def forward(self, features_1, features_2):
        out = self.tp(features_1, features_2)
        return out


class PairNetLayer(torch.nn.Module):
    def __init__(
        self,
        irrep_in_node,
        irrep_bottle_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        resnet: bool = True,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinear="ssp",
    ):
        super(PairNetLayer, self).__init__()
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.irrep_in_node = (
            irrep_in_node
            if isinstance(irrep_in_node, o3.Irreps)
            else o3.Irreps(irrep_in_node)
        )
        self.irrep_bottle_hidden = (
            irrep_bottle_hidden
            if isinstance(irrep_bottle_hidden, o3.Irreps)
            else o3.Irreps(irrep_bottle_hidden)
        )
        self.irrep_out = (
            irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        )
        self.sh_irrep = (
            sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)
        )

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(
            self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden
        )
        self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_bottle_hidden,
            tp_mode="uuu",
        )

        self.irrep_tp_out_node_pair_msg, instruction_node_pair_msg = get_feasible_irrep(
            self.irrep_tp_in_node,
            self.sh_irrep,
            self.irrep_bottle_hidden,
            tp_mode="uvu",
        )

        self.linear_node_pair = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_tp_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        self.linear_node_pair_n = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )
        self.linear_node_pair_inner = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        self.tp_node_pair = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node_pair,
            instruction_node_pair,
            shared_weights=False,
            internal_weights=False,
        )

        self.fc_node_pair = FullyConnectedNet(
            [self.edge_attr_dim]
            + invariant_layers * [invariant_neurons]
            + [self.tp_node_pair.weight_numel],
            self.nonlinear_layer,
        )

        if self.irrep_in_node == self.irrep_out and resnet:
            self.resnet = True
        else:
            self.resnet = False

        self.linear_node_pair = Linear(
            irreps_in=self.irrep_tp_out_node_pair,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )
        self.norm_gate = NormGate(self.irrep_tp_out_node_pair)
        self.inner_product = InnerProduct(self.irrep_in_node)
        self.norm = o3.Norm(self.irrep_in_node)
        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul = num_mul + mul

        self.norm_gate_pre = NormGate(self.irrep_tp_out_node_pair)
        self.fc = nn.Sequential(
            nn.Linear(self.irrep_in_node[0][0] + num_mul, self.irrep_in_node[0][0]),
            nn.SiLU(),
            nn.Linear(self.irrep_in_node[0][0], self.tp_node_pair.weight_numel),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, node_attr, node_pair_attr=None):
        dst, src = data.full_edge_index
        node_attr_0 = self.linear_node_pair_inner(node_attr)
        s0 = self.inner_product(node_attr_0[dst], node_attr_0[src])[
            :, self.irrep_in_node.slices()[0].stop :
        ]
        s0 = torch.cat(
            [
                node_attr_0[dst][:, self.irrep_in_node.slices()[0]],
                node_attr_0[src][:, self.irrep_in_node.slices()[0]],
                s0,
            ],
            dim=-1,
        )

        node_attr = self.norm_gate_pre(node_attr)
        node_attr = self.linear_node_pair_n(node_attr)

        node_pair = self.tp_node_pair(
            node_attr[src],
            node_attr[dst],
            self.fc_node_pair(data.full_edge_attr) * self.fc(s0),
        )

        node_pair = self.norm_gate(node_pair)
        node_pair = self.linear_node_pair(node_pair)

        if self.resnet and node_pair_attr is not None:
            node_pair = node_pair + node_pair_attr
        return node_pair


class SelfNetLayer(torch.nn.Module):
    def __init__(
        self,
        irrep_in_node,
        irrep_bottle_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        resnet: bool = True,
        nonlinear="ssp",
    ):
        super(SelfNetLayer, self).__init__()
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.sh_irrep = sh_irrep
        self.irrep_in_node = (
            irrep_in_node
            if isinstance(irrep_in_node, o3.Irreps)
            else o3.Irreps(irrep_in_node)
        )
        self.irrep_bottle_hidden = (
            irrep_bottle_hidden
            if isinstance(irrep_bottle_hidden, o3.Irreps)
            else o3.Irreps(irrep_bottle_hidden)
        )
        self.irrep_out = (
            irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        )

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.resnet = resnet
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(
            self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden
        )
        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_bottle_hidden,
            tp_mode="uuu",
        )

        # - Build modules -
        self.linear_node_1 = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        self.linear_node_2 = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )
        self.tp = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node,
            instruction_node,
            shared_weights=True,
            internal_weights=True,
        )
        self.norm_gate = NormGate(self.irrep_out)
        self.norm_gate_1 = NormGate(self.irrep_in_node)
        self.norm_gate_2 = NormGate(self.irrep_in_node)
        self.linear_node_3 = Linear(
            irreps_in=self.irrep_tp_out_node,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

    def forward(self, data, x, old_fii):
        old_x = x
        xl = self.norm_gate_1(x)
        xl = self.linear_node_1(xl)
        xr = self.norm_gate_2(x)
        xr = self.linear_node_2(xr)
        x = self.tp(xl, xr)
        if self.resnet:
            x = x + old_x
        x = self.norm_gate(x)
        x = self.linear_node_3(x)
        if self.resnet and old_fii is not None:
            x = old_fii + x
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class Expansion(nn.Module):
    def __init__(self, irrep_in, irrep_out_1, irrep_out_2):
        super(Expansion, self).__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2
        self.instructions = self.get_expansion_path(irrep_in, irrep_out_1, irrep_out_2)
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum(
            [prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0]
        )
        if self.num_path_weight > 0:
            self.weights = nn.Parameter(
                torch.rand(self.num_path_weight + self.num_bias)
            )
        self.num_weights = self.num_path_weight + self.num_bias

    def forward(self, x_in, weights=None, bias_weights=None):
        batch_num = x_in.shape[0]
        if len(self.irrep_in) == 1:
            x_in_s = [
                x_in.reshape(batch_num, self.irrep_in[0].mul, self.irrep_in[0].ir.dim)
            ]
        else:
            x_in_s = [
                x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
                for i, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)
            ]

        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        for ins in self.instructions:
            mul_ir_in = self.irrep_in[ins[0]]
            mul_ir_out1 = self.irrep_out_1[ins[1]]
            mul_ir_out2 = self.irrep_out_2[ins[2]]
            x1 = x_in_s[ins[0]]
            x1 = x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
            w3j_matrix = (
                o3.wigner_3j(ins[1], ins[2], ins[0]).to(self.device).type(x1.type())
            )
            if ins[3] is True or weights is not None:
                if weights is None:
                    weight = self.weights[
                        flat_weight_index : flat_weight_index + prod(ins[-1])
                    ].reshape(ins[-1])
                    result = (
                        torch.einsum(f"wuv, ijk, bwk-> buivj", weight, w3j_matrix, x1)
                        / mul_ir_in.mul
                    )
                else:
                    weight = weights[
                        :, flat_weight_index : flat_weight_index + prod(ins[-1])
                    ].reshape([-1] + ins[-1])
                    result = torch.einsum(f"bwuv, bwk-> buvk", weight, x1)
                    if ins[0] == 0 and bias_weights is not None:
                        bias_weight = bias_weights[
                            :, bias_weight_index : bias_weight_index + prod(ins[-1][1:])
                        ].reshape([-1] + ins[-1][1:])
                        bias_weight_index += prod(ins[-1][1:])
                        result = result + bias_weight.unsqueeze(-1)
                    result = (
                        torch.einsum(f"ijk, buvk->buivj", w3j_matrix, result)
                        / mul_ir_in.mul
                    )
                flat_weight_index += prod(ins[-1])
            else:
                result = torch.einsum(
                    f"uvw, ijk, bwk-> buivj",
                    torch.ones(ins[-1]).type(x1.type()).to(self.device),
                    w3j_matrix,
                    x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim),
                )

            result = result.reshape(batch_num, mul_ir_out1.dim, mul_ir_out2.dim)
            key = (ins[1], ins[2])
            if key in outputs.keys():
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result

        rows = []
        for i in range(len(self.irrep_out_1)):
            blocks = []
            for j in range(len(self.irrep_out_2)):
                if (i, j) not in outputs.keys():
                    blocks += [
                        torch.zeros(
                            (
                                x_in.shape[0],
                                self.irrep_out_1[i].dim,
                                self.irrep_out_2[j].dim,
                            ),
                            device=x_in.device,
                        ).type(x_in.type())
                    ]
                else:
                    blocks += [outputs[(i, j)]]
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2)
        return output

    def get_expansion_path(self, irrep_in, irrep_out_1, irrep_out_2):
        instructions = []
        for i, (num_in, ir_in) in enumerate(irrep_in):
            for j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append(
                            [i, j, k, True, 1.0, [num_in, num_out1, num_out2]]
                        )
        return instructions

    @property
    def device(self):
        return next(self.parameters()).device

    def __repr__(self):
        return (
            f"{self.irrep_in} -> {self.irrep_out_1}x{self.irrep_out_1} and bias {self.num_bias}"
            f"with parameters {self.num_path_weight}"
        )


class QHNet(nn.Module):
    def __init__(
        self,
        in_node_features=1,
        sh_lmax=4,
        hidden_size=128,
        bottle_hidden_size=32,
        num_gnn_layers=5,
        max_radius=12,
        num_nodes=10,
        radius_embed_dim=32,
        **kwargs,
    ):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super(QHNet, self).__init__()
        self.order = sh_lmax

        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = hidden_size
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = self.hs
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.num_nodes = num_nodes
        self.node_embedding = nn.Embedding(self.num_nodes, self.hs)
        self.hidden_irrep = o3.Irreps(
            f"{self.hs}x0e + {self.hs}x1o + {self.hs}x2e + {self.hs}x3o + {self.hs}x4e"
        )
        self.hidden_bottle_irrep = o3.Irreps(
            f"{self.hbs}x0e + {self.hbs}x1o + {self.hbs}x2e + {self.hbs}x3o + {self.hbs}x4e"
        )
        # self.hidden_irrep_base = o3.Irreps(
        #     f"{self.hs}x0e + {self.hs}x1e + {self.hs}x2e + {self.hs}x3e + {self.hs}x4e"
        # )
        # self.hidden_bottle_irrep_base = o3.Irreps(
        #     f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e"
        # )
        self.final_out_irrep = o3.Irreps(
            f"{self.hs * 3}x0e + {self.hs * 2}x1o + {self.hs}x2e"
        ).simplify()
        self.input_irrep = o3.Irreps(f"{self.hs}x0e")
        # self.distance_expansion = ExponentialBernsteinRadialBasisFunctions(
        #     self.radius_embed_dim, self.max_radius
        # )
        self.distance_expansion = GaussianSmearing(
            0.0,
            self.max_radius,
            self.radius_embed_dim,
            2.0,
        )
        # self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        # self.nonlinear_gates = {1: "ssp", -1: "abs"}
        # self.num_fc_layer = 1

        self.e3_gnn_layer = nn.ModuleList()
        # self.e3_gnn_node_pair_layer = nn.ModuleList()
        # self.e3_gnn_node_layer = nn.ModuleList()
        # self.start_layer = 2

        self.lmax_list = [4]
        self.mmax_list = [2]
        self.num_resolutions: int = len(self.lmax_list)
        self.sphere_channels = hidden_size
        self.sphere_channels_all: int = self.num_resolutions * self.sphere_channels
        self.SO3_grid = ModuleListInfo(
            f"({max(self.lmax_list)}, {max(self.lmax_list)})"
        )
        self.grid_resolution = None
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=self.grid_resolution,
                        normalization="component",
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        self.act = nn.SiLU()
        self.edge_channels = self.sphere_channels_all
        self.distance_expansion_SO2 = GaussianSmearing(
            0.0,
            self.max_radius,
            self.hs,
            2.0,
        )
        self.edge_channels_list = [int(self.distance_expansion_SO2.num_output)] + [
            self.edge_channels
        ] * 2

        self.use_atom_edge_embedding = True
        self.share_atom_edge_embedding = False
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding

        self.attn_hidden_channels = self.hs
        self.num_heads = 8
        self.attn_alpha_channels = 32
        self.attn_value_channels = 16
        self.ffn_hidden_channels = self.hs * 2
        self.max_num_elements = self.num_nodes
        self.use_m_share_rad = False
        self.attn_activation = "scaled_silu"
        self.use_s2_act_attn = False
        self.use_attn_renorm = True
        self.ffn_activation = "scaled_silu"
        self.use_gate_act = False
        self.use_grid_mlp = False
        self.use_sep_s2_act = True
        self.norm_type = "rms_norm_sh"
        self.alpha_drop = 0.1
        self.drop_path_rate = 0.05
        self.proj_drop = 0.0

        self.sphere_embedding = nn.Embedding(self.num_nodes, self.sphere_channels_all)

        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=1.0,
        )
        for i in range(self.num_gnn_layers):
            self.e3_gnn_layer.append(
                TransBlockV2(
                    self.sphere_channels,
                    self.attn_hidden_channels,
                    self.num_heads,
                    self.attn_alpha_channels,
                    self.attn_value_channels,
                    self.ffn_hidden_channels,
                    self.sphere_channels,
                    self.lmax_list,
                    self.mmax_list,
                    self.SO3_rotation,
                    self.mappingReduced,
                    self.SO3_grid,
                    self.max_num_elements,
                    self.edge_channels_list,
                    self.block_use_atom_edge_embedding,
                    self.use_m_share_rad,
                    self.attn_activation,
                    self.use_s2_act_attn,
                    self.use_attn_renorm,
                    self.ffn_activation,
                    self.use_gate_act,
                    self.use_grid_mlp,
                    self.use_sep_s2_act,
                    self.norm_type,
                    self.alpha_drop,
                    self.drop_path_rate,
                    self.proj_drop,
                )
            )

        self.e3_gnn_node_layer = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            # self.sphere_channels,
            self.hbs,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
        )

        # self.e3_gnn_node_pair_layer = SO2EquivariantGraphAttentionEdge(
        #     self.sphere_channels,
        #     self.attn_hidden_channels,
        #     self.num_heads,
        #     self.attn_alpha_channels,
        #     self.attn_value_channels,
        #     self.ffn_hidden_channels,
        #     self.sphere_channels,
        #     self.lmax_list,
        #     self.mmax_list,
        #     self.SO3_rotation,
        #     self.mappingReduced,
        #     self.SO3_grid,
        #     self.max_num_elements,
        #     self.edge_channels_list,
        #     self.block_use_atom_edge_embedding,
        #     self.use_m_share_rad,
        #     self.attn_activation,
        #     self.use_s2_act_attn,
        #     self.use_attn_renorm,
        #     self.ffn_activation,
        #     self.use_gate_act,
        #     self.use_grid_mlp,
        #     self.use_sep_s2_act,
        #     self.norm_type,
        #     self.alpha_drop,
        #     self.drop_path_rate,
        #     self.proj_drop,
        # )
        self.e3_gnn_node_pair_layer = SO2EquivariantGraphAttentionEdge(
            self.sphere_channels,
            self.attn_hidden_channels,
            self.num_heads,
            self.attn_alpha_channels,
            self.attn_value_channels,
            # self.sphere_channels,
            self.hbs,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.SO3_grid,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            self.use_m_share_rad,
            self.attn_activation,
            self.use_s2_act_attn,
            self.use_attn_renorm,
            self.use_gate_act,
            self.use_sep_s2_act,
            alpha_drop=0.0,
        )
        self.irrep_so2_in_node = o3.Irreps(
            [(self.hbs, (l, 1)) for l in range(self.lmax_list[0] + 1)]
        )
        print(self.irrep_so2_in_node)

        self.nonlinear_layer = get_nonlinear("ssp")
        (
            self.expand_ii,
            self.expand_ij,
            self.fc_ii,
            self.fc_ij,
            self.fc_ii_bias,
            self.fc_ij_bias,
        ) = (
            nn.ModuleDict(),
            nn.ModuleDict(),
            nn.ModuleDict(),
            nn.ModuleDict(),
            nn.ModuleDict(),
            nn.ModuleDict(),
        )
        for name in {"hamiltonian"}:
            input_expand_ii = self.hidden_bottle_irrep
            input_expand_ij = self.hidden_bottle_irrep
            # input_expand_ii = self.irrep_so2_in_node
            # input_expand_ij = self.irrep_so2_in_node

            self.expand_ii[name] = Expansion(
                input_expand_ii,
                o3.Irreps("3x0e + 2x1e + 1x2e"),
                o3.Irreps("3x0e + 2x1e + 1x2e"),
            )
            self.fc_ii[name] = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ii[name].num_path_weight),
            )
            self.fc_ii_bias[name] = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ii[name].num_bias),
            )

            self.expand_ij[name] = Expansion(
                input_expand_ij,
                o3.Irreps("3x0e + 2x1e + 1x2e"),
                o3.Irreps("3x0e + 2x1e + 1x2e"),
            )

            self.fc_ij[name] = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight),
            )

            self.fc_ij_bias[name] = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias),
            )
        # self.output_ii_tp = o3.FullyConnectedTensorProduct(
        #     self.irrep_so2_in_node,
        #     self.irrep_so2_in_node,
        #     self.hidden_bottle_irrep,
        # )
        # self.output_ij_tp = o3.FullyConnectedTensorProduct(
        #     self.irrep_so2_in_node,
        #     self.irrep_so2_in_node,
        #     self.hidden_bottle_irrep,
        # )

        # self.output_ii = Linear(self.irrep_so2_in_node, self.hidden_bottle_irrep)
        # self.output_ij = Linear(self.irrep_so2_in_node, self.hidden_bottle_irrep)

        self.concat_idx = [
            torch.arange(0, 1 * self.hs),
            torch.arange(1 * self.hs, 4 * self.hs),
            torch.arange(4 * self.hs, 9 * self.hs),
            torch.arange(9 * self.hs, 16 * self.hs),
            torch.arange(16 * self.hs, 25 * self.hs),
        ]
        self.concat_idx = []
        for j in range(max(self.lmax_list) + 1):
            for i in range(self.hbs):
                offset = i * (self.lmax_list[0] + 1) ** 2
                self.concat_idx.append(
                    torch.arange(j**2 + offset, j**2 + (2 * j + 1) + offset)
                )
        self.concat_idx = torch.cat(self.concat_idx, dim=0)

        self.norm1 = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.sphere_channels,
        )
        self.norm2 = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.hbs,
        )

    def get_number_of_parameters(self):
        num = 0
        for param in self.parameters():
            if param.requires_grad:
                num += param.numel()
        return num

    def set(self, device):
        self = self.to(device)
        self.orbital_mask = self.get_orbital_mask()
        for key in self.orbital_mask.keys():
            self.orbital_mask[key] = self.orbital_mask[key].to(self.device)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, data, H=None, keep_blocks=False):
        node_attr, edge_index, rbf_new, edge_sh, _ = self.build_graph(
            data, self.max_radius
        )
        node_attr = self.node_embedding(node_attr)
        data.node_attr, data.edge_index, data.edge_attr, data.edge_sh = (
            node_attr,
            edge_index,
            rbf_new,
            edge_sh,
        )

        _, full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index = (
            self.build_graph(data, 10000)
        )
        data.full_edge_index, data.full_edge_attr, data.full_edge_sh = (
            full_edge_index,
            full_edge_attr,
            full_edge_sh,
        )
        dst, src = edge_index
        full_dst, full_src = data.full_edge_index

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(data, edge_index, data.edge_vec)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################
        atomic_numbers = data.atoms.squeeze()
        num_atoms = atomic_numbers.shape[0]
        # Init per node representations using an atomic number based embedding
        offset = 0

        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )
        x.embedding[:, 0, :] = node_attr

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion_SO2(data.edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers, edge_distance, edge_index
        )
        x.embedding = x.embedding + edge_degree.embedding

        for layer_idx, layer in enumerate(self.e3_gnn_layer):
            x = layer(
                x,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch,  # for GraphDropPath
            )
        x.embedding = self.norm1(x.embedding)
        fii_so2 = self.e3_gnn_node_layer(x)
        fij_so2 = self.e3_gnn_node_pair_layer(
            x,  # SO3_Embedding
            atomic_numbers,
            edge_distance,
            edge_index,
        )
        # fii_so2.embedding = self.norm(fii_so2.embedding)
        # fij_so2.embedding = self.norm(fij_so2.embedding)
        # self.concat_idx = self.concat_idx.to(self.device)

        fii = fii_so2.embedding.flatten(1, 2)
        fij = fij_so2.embedding.flatten(1, 2)
        # fii = fii.index_select(-1, self.concat_idx).contiguous()
        # fij = fij.index_select(-1, self.concat_idx).contiguous()

        # fii = self.output_ii(fii)
        # fij = self.output_ij(fij)
        # fii = self.output_ii_tp(fii, fii)
        # fij = self.output_ij_tp(fij, fij)
        hamiltonian_diagonal_matrix = self.expand_ii["hamiltonian"](
            fii,
            self.fc_ii["hamiltonian"](data.node_attr),
            self.fc_ii_bias["hamiltonian"](data.node_attr),
        )
        node_pair_embedding = torch.cat(
            [data.node_attr[dst], data.node_attr[src]], dim=-1
        )
        hamiltonian_non_diagonal_matrix = self.expand_ij["hamiltonian"](
            fij,
            self.fc_ij["hamiltonian"](node_pair_embedding),
            self.fc_ij_bias["hamiltonian"](node_pair_embedding),
        )

        if keep_blocks is False:
            hamiltonian_matrix = self.build_final_matrix(
                data, hamiltonian_diagonal_matrix, hamiltonian_non_diagonal_matrix
            )
            hamiltonian_matrix = hamiltonian_matrix + hamiltonian_matrix.transpose(
                -1, -2
            )
            results = {}
            results["hamiltonian"] = hamiltonian_matrix
        else:
            ret_hamiltonian_diagonal_matrix = (
                hamiltonian_diagonal_matrix
                + hamiltonian_diagonal_matrix.transpose(-1, -2)
            )

            # the transpose should considers the i, j
            ret_hamiltonian_non_diagonal_matrix = (
                hamiltonian_non_diagonal_matrix
                + hamiltonian_non_diagonal_matrix[transpose_edge_index].transpose(
                    -1, -2
                )
            )

            results = {}
            results["hamiltonian_diagonal_blocks"] = ret_hamiltonian_diagonal_matrix
            results["hamiltonian_non_diagonal_blocks"] = (
                ret_hamiltonian_non_diagonal_matrix
            )
        return results

    def build_graph(self, data, max_radius):
        node_attr = data.atoms.squeeze()
        # data.pos = data.pos * 3
        radius_edges = radius_graph(data.pos, max_radius, data.batch)

        dst, src = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        data.edge_vec = edge_vec
        data.edge_distance = edge_vec.norm(dim=-1, keepdim=True)
        rbf = (
            self.distance_expansion(edge_vec.norm(dim=-1).unsqueeze(-1))
            .squeeze()
            .type(data.pos.type())
        )

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep,
            edge_vec[:, [1, 2, 0]],
            normalize=True,
            normalization="component",
        ).type(data.pos.type())

        start_edge_index = 0
        all_transpose_index = []
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx + 1] - data.ptr[graph_idx]
            graph_edge_index = radius_edges[
                :, start_edge_index : start_edge_index + num_nodes * (num_nodes - 1)
            ]
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = (
                sub_graph_edge_index[0] * (num_nodes - 1)
                + sub_graph_edge_index[1]
                - bias
            )
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            start_edge_index = start_edge_index + num_nodes * (num_nodes - 1)

        return (
            node_attr,
            radius_edges,
            rbf,
            edge_sh,
            torch.cat(all_transpose_index, dim=-1),
        )

    def build_final_matrix(self, data, diagonal_matrix, non_diagonal_matrix):
        # concate the blocks together and then select once.
        final_matrix = []
        dst, src = data.full_edge_index
        for graph_idx in range(data.ptr.shape[0] - 1):
            matrix_block_col = []
            for src_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                matrix_col = []
                for dst_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                    if src_idx == dst_idx:
                        matrix_col.append(
                            diagonal_matrix[src_idx]
                            .index_select(
                                -2, self.orbital_mask[data.atoms[dst_idx].item()]
                            )
                            .index_select(
                                -1, self.orbital_mask[data.atoms[src_idx].item()]
                            )
                        )
                    else:
                        mask1 = src == src_idx
                        mask2 = dst == dst_idx
                        index = torch.where(mask1 & mask2)[0].item()

                        matrix_col.append(
                            non_diagonal_matrix[index]
                            .index_select(
                                -2, self.orbital_mask[data.atoms[dst_idx].item()]
                            )
                            .index_select(
                                -1, self.orbital_mask[data.atoms[src_idx].item()]
                            )
                        )
                matrix_block_col.append(torch.cat(matrix_col, dim=-2))
            final_matrix.append(torch.cat(matrix_block_col, dim=-1))
        final_matrix = torch.stack(final_matrix, dim=0)
        return final_matrix

    def get_orbital_mask(self):
        idx_1s_2s = torch.tensor([0, 1])
        idx_2p = torch.tensor([3, 4, 5])
        orbital_mask_line1 = torch.cat([idx_1s_2s, idx_2p])
        orbital_mask_line2 = torch.arange(14)
        orbital_mask = {}
        for i in range(1, 11):
            orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2
        return orbital_mask

    def split_matrix(self, data):
        diagonal_matrix, non_diagonal_matrix = torch.zeros(
            data.atoms.shape[0], 14, 14
        ).type(data.pos.type()).to(self.device), torch.zeros(
            data.edge_index.shape[1], 14, 14
        ).type(
            data.pos.type()
        ).to(
            self.device
        )

        data.matrix = data.matrix.reshape(
            len(data.ptr) - 1, data.matrix.shape[-1], data.matrix.shape[-1]
        )

        num_atoms = 0
        num_edges = 0
        for graph_idx in range(data.ptr.shape[0] - 1):
            slices = [0]
            for atom_idx in data.atoms[
                range(data.ptr[graph_idx], data.ptr[graph_idx + 1])
            ]:
                slices.append(slices[-1] + len(self.orbital_mask[atom_idx.item()]))

            for node_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                node_idx = node_idx - num_atoms
                orb_mask = self.orbital_mask[data.atoms[node_idx].item()]
                diagonal_matrix[node_idx][orb_mask][:, orb_mask] = data.matrix[
                    graph_idx
                ][
                    slices[node_idx] : slices[node_idx + 1],
                    slices[node_idx] : slices[node_idx + 1],
                ]

            for edge_index_idx in range(num_edges, data.edge_index.shape[1]):
                dst, src = data.edge_index[:, edge_index_idx]
                if dst > data.ptr[graph_idx + 1] or src > data.ptr[graph_idx + 1]:
                    break
                num_edges = num_edges + 1
                orb_mask_dst = self.orbital_mask[data.atoms[dst].item()]
                orb_mask_src = self.orbital_mask[data.atoms[src].item()]
                graph_dst, graph_src = dst - num_atoms, src - num_atoms
                non_diagonal_matrix[edge_index_idx][orb_mask_dst][
                    :, orb_mask_src
                ] = data.matrix[graph_idx][
                    slices[graph_dst] : slices[graph_dst + 1],
                    slices[graph_src] : slices[graph_src + 1],
                ]

            num_atoms = num_atoms + data.ptr[graph_idx + 1] - data.ptr[graph_idx]
        return diagonal_matrix, non_diagonal_matrix

    def _init_edge_rot_mat(self, coords, edge_index, edge_distance_vec):
        edge_vec_0 = edge_distance_vec
        edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

        # Make sure the atoms are far enough apart
        if torch.min(edge_vec_0_distance) < 0.0001:
            logging.error(
                "Error edge_vec_0_distance: {}".format(torch.min(edge_vec_0_distance))
            )
            (minval, minidx) = torch.min(edge_vec_0_distance, 0)
            logging.error(
                "Error edge_vec_0_distance: {} {} {} {} {}".format(
                    minidx,
                    edge_index[0, minidx],
                    edge_index[1, minidx],
                    coords[edge_index[0, minidx]],
                    coords[edge_index[1, minidx]],
                )
            )

        norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

        edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
        edge_vec_2 = edge_vec_2 / (
            torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
        )
        # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
        # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
        edge_vec_2b = edge_vec_2.clone()
        edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
        edge_vec_2b[:, 1] = edge_vec_2[:, 0]
        edge_vec_2c = edge_vec_2.clone()
        edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
        edge_vec_2c[:, 2] = edge_vec_2[:, 1]
        vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
        vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
        # Check the vectors aren't aligned
        assert torch.max(vec_dot) < 0.99

        norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
        norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
        norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
        norm_y = torch.cross(norm_x, norm_z, dim=1)
        norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

        # Construct the 3D rotation matrix
        norm_x = norm_x.view(-1, 3, 1)
        norm_y = -norm_y.view(-1, 3, 1)
        norm_z = norm_z.view(-1, 3, 1)

        edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
        edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

        return edge_rot_mat.detach()
