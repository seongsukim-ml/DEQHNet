import time
import collections
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from e3nn import o3
from torch_scatter import scatter
import torch_geometric
from torch_geometric.nn import global_mean_pool, global_max_pool
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Linear, TensorProduct
from e3nn.o3._norm import Norm
from e3nn.math import normalize2mom, perm
from e3nn.util.jit import compile_mode
from torch.nn import init
from torchdeq import get_deq

from .modules import *
from .scn.escn import *
from .QHNet import *


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class SO2ConvNetLayer(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        sphere_channels_all: int,
        hidden_channels: int,
        # irrep_out,
        # sh_irrep,
        edge_attr_dim: int,
        lmax_list: List[int],
        mmax_list: List[int],
        SO3_grid: SO3_Grid,
        act,
        distance_expansion,
        max_num_elements=10,
        resnet: bool = True,
    ):
        super(SO2ConvNetLayer, self).__init__()
        # self.irrep_out = (
        #     irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        # )
        # self.irreps_out = irrep_out
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.sphere_channels_all = sphere_channels_all
        self.edge_attr_dim = edge_attr_dim
        self.resnet = resnet
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.SO3_grid = SO3_grid
        self.act = act

        self.message_block = MessageBlock(
            0,
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

        self.irrep_hidden = o3.Irreps(
            [
                (self.sphere_channels_all, (l, 1 if l % 2 == 0 else -1))
                for l in range(self.lmax_list[0] + 1)
            ]
        )

    def forward(self, data, x_emb, SO3_edge_rot, mappingReduced, x_irrep_out_old=None):
        atomic_numbers = data.atoms.squeeze()
        edge_distance = data.edge_distance
        edge_index = data.edge_index
        x_message = self.message_block(
            x_emb,
            atomic_numbers,
            edge_distance,
            edge_index,
            SO3_edge_rot,
            mappingReduced,
        )
        # Compute point-wise spherical non-linearity on aggregated messages
        max_lmax = max(self.lmax_list)

        # Project to grid
        x_grid_message = x_message.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid = x_emb.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid = torch.cat([x_grid, x_grid_message], dim=3)

        # Perform point-wise convolution
        x_grid = self.act(self.fc1_sphere(x_grid))
        x_grid = self.act(self.fc2_sphere(x_grid))
        x_grid = self.fc3_sphere(x_grid)

        # Project back to spherical harmonic coefficients
        x_message._from_grid(x_grid, self.SO3_grid, lmax=max_lmax)

        x_irrep_out = x_message.embedding.flatten(1, 2)
        if self.resnet and x_irrep_out_old is not None:
            x_irrep_out = x_irrep_out + x_irrep_out_old

        return x_message, x_irrep_out


class QHNet_flow(nn.Module):
    def __init__(
        self,
        in_node_features=1,
        sh_lmax=4,
        hidden_size=128,
        bottle_hidden_size=32,
        num_gnn_layers=5,
        max_radius=12,
        num_nodes=10,
        radius_embed_dim=32,  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        use_block_S=False,
        use_block_H=False,
        **deq_kwargs,
    ):
        super(QHNet_flow, self).__init__()
        self.order = sh_lmax

        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = hidden_size
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.num_nodes = num_nodes
        self.node_embedding = nn.Embedding(num_nodes, self.hs)
        self.hidden_irrep = o3.Irreps(
            f"{self.hs}x0e + {self.hs}x1o + {self.hs}x2e + {self.hs}x3o + {self.hs}x4e"
        )
        self.hidden_bottle_irrep = o3.Irreps(
            f"{self.hbs}x0e + {self.hbs}x1o + {self.hbs}x2e + {self.hbs}x3o + {self.hbs}x4e"
        )
        self.hidden_irrep_base = o3.Irreps(
            f"{self.hs}x0e + {self.hs}x1e + {self.hs}x2e + {self.hs}x3e + {self.hs}x4e"
        )
        self.hidden_bottle_irrep_base = o3.Irreps(
            f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e"
        )
        self.final_out_irrep = o3.Irreps(
            f"{self.hs * 3}x0e + {self.hs * 2}x1o + {self.hs}x2e"
        ).simplify()
        # self.input_irrep = o3.Irreps(f'{self.hs}x0e')
        self.input_irrep = o3.Irreps(f"{self.hs}x0e")
        self.distance_expansion = ExponentialBernsteinRadialBasisFunctions(
            self.radius_embed_dim, self.max_radius
        )
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.num_fc_layer = 1

        self.onebody_reduction = OneBody_Reduction()
        self.norm_layer = "layer"
        self.irreps_node_attr = "1x0e"
        fc_neurons = [64, 64]
        self.fc_neurons = [self.radius_embed_dim] + fc_neurons
        self.irreps_head = o3.Irreps("32x0e+16x1o+8x2e")
        self.num_heads = 4
        self.irreps_pre_attn = None
        self.rescale_degree = False
        self.nonlinear_message = False
        self.alpha_drop = 0.0
        self.proj_drop = 0.0
        self.out_drop = 0.0
        self.drop_path_rate = 0.0
        self.irreps_mlp_mid = "128x0e+64x1e+32x2e"

        self.e3_gnn_layer = nn.ModuleList()
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        self.use_block_S = use_block_S
        if self.use_block_S:
            self.blocks_S = torch.nn.ModuleList()
        self.use_block_H = use_block_H
        if self.use_block_H:
            self.blocks_H_init = torch.nn.ModuleList()
        self.blocks_H_cur = torch.nn.ModuleList()
        self.blocks_Linear = torch.nn.ModuleList()
        self.start_layer = 2

        # self.node_embedding_t = nn.Linear(self.hs * 2, self.hs, bias=False)
        self.sigma_embedding = nn.Sequential(
            nn.Linear(2 * self.hs, self.hs), nn.SiLU(), nn.Linear(self.hs, self.hs)
        )

        self.lmax_list = [4]
        self.mmax_list = [2]
        self.num_resolutions: int = len(self.lmax_list)
        self.sphere_channels = hidden_size
        self.sphere_channels_all: int = self.num_resolutions * self.sphere_channels
        self.SO3_grid = nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(SO3_Grid(lval, m))

            self.SO3_grid.append(SO3_m_grid)
        self.act = nn.SiLU()
        self.edge_channels = self.sphere_channels_all

        self.distance_expansion_SO2 = GaussianSmearing(
            0.0,
            self.max_radius,
            self.edge_channels,
            1.0,
        )

        self.irreps_node_embedding_0 = o3.Irreps("16x0e+8x1o+4x2e")
        self.num_concat = 2
        if self.use_block_S:
            self.num_concat += 1
        if self.use_block_H:
            self.num_concat += 1

        self.hidden_irrep_concat = o3.Irreps(
            f"{self.hs * self.num_concat}x0e + \
              {self.hs * self.num_concat}x1o + \
              {self.hs * self.num_concat}x2e + \
              {self.hs * self.num_concat}x3o + \
              {self.hs * self.num_concat}x4e"
        )
        self.concat_idx = [
            torch.arange(0, 1 * self.hs),
            torch.arange(1 * self.hs, 4 * self.hs),
            torch.arange(4 * self.hs, 9 * self.hs),
            torch.arange(9 * self.hs, 16 * self.hs),
            torch.arange(16 * self.hs, 25 * self.hs),
        ]
        self.hidden_irrep_concat_idx = []
        for group in self.concat_idx:
            for i in range(self.num_concat):
                self.hidden_irrep_concat_idx.append(group + i * 25 * self.hs)

        self.hidden_irrep_concat_idx = torch.concat(self.hidden_irrep_concat_idx)

        for i in range(self.num_gnn_layers):
            # input_irrep = self.input_irrep if i == 0 else self.hidden_irrep
            self.e3_gnn_layer.append(
                SO2ConvNetLayer(
                    self.sphere_channels,
                    self.sphere_channels_all,
                    self.hs,
                    edge_attr_dim=self.edge_channels,
                    max_num_elements=self.num_nodes,
                    lmax_list=self.lmax_list,
                    mmax_list=self.mmax_list,
                    SO3_grid=self.SO3_grid,
                    distance_expansion=self.distance_expansion_SO2,
                    act=self.act,
                    resnet=False if i == 0 else True,
                )
            )

            self.irrep_so2_in_node = self.e3_gnn_layer[-1].irrep_hidden

            self.irreps_node_embedding = (
                o3.Irreps("16x0e+8x1o+4x2e") if i == 0 else self.irrep_so2_in_node
            )

            self.blocks_H_cur.append(
                TransBlock(
                    irreps_node_input=self.irreps_node_embedding,
                    irreps_node_attr=self.irreps_node_attr,
                    irreps_edge_attr=self.sh_irrep,
                    irreps_node_output=self.irrep_so2_in_node,
                    fc_neurons=self.fc_neurons,
                    irreps_head=self.irreps_head,
                    num_heads=self.num_heads,
                    irreps_pre_attn=self.irreps_pre_attn,
                    rescale_degree=self.rescale_degree,
                    nonlinear_message=self.nonlinear_message,
                    alpha_drop=self.alpha_drop,
                    proj_drop=self.proj_drop,
                    drop_path_rate=self.drop_path_rate,
                    irreps_mlp_mid=self.irreps_mlp_mid,
                    norm_layer=self.norm_layer,
                )
            )
            if self.use_block_H:
                self.blocks_H_init.append(
                    TransBlock(
                        irreps_node_input=self.irreps_node_embedding,
                        irreps_node_attr=self.irreps_node_attr,
                        irreps_edge_attr=self.sh_irrep,
                        irreps_node_output=self.irrep_so2_in_node,
                        fc_neurons=self.fc_neurons,
                        irreps_head=self.irreps_head,
                        num_heads=self.num_heads,
                        irreps_pre_attn=self.irreps_pre_attn,
                        rescale_degree=self.rescale_degree,
                        nonlinear_message=self.nonlinear_message,
                        alpha_drop=self.alpha_drop,
                        proj_drop=self.proj_drop,
                        drop_path_rate=self.drop_path_rate,
                        irreps_mlp_mid=self.irreps_mlp_mid,
                        norm_layer=self.norm_layer,
                    )
                )
            if self.use_block_S:
                self.blocks_S.append(
                    TransBlock(
                        irreps_node_input=self.irreps_node_embedding,
                        irreps_node_attr=self.irreps_node_attr,
                        irreps_edge_attr=self.sh_irrep,
                        irreps_node_output=self.irrep_so2_in_node,
                        fc_neurons=self.fc_neurons,
                        irreps_head=self.irreps_head,
                        num_heads=self.num_heads,
                        irreps_pre_attn=self.irreps_pre_attn,
                        rescale_degree=self.rescale_degree,
                        nonlinear_message=self.nonlinear_message,
                        alpha_drop=self.alpha_drop,
                        proj_drop=self.proj_drop,
                        drop_path_rate=self.drop_path_rate,
                        irreps_mlp_mid=self.irreps_mlp_mid,
                        norm_layer=self.norm_layer,
                    )
                )
            self.blocks_Linear.append(
                Linear(self.hidden_irrep_concat, self.hidden_irrep)
            )

            self.norm = get_norm_layer(self.norm_layer)(self.hidden_irrep)

            if i > self.start_layer:
                self.e3_gnn_node_layer.append(
                    SelfNetLayer(
                        # irrep_in_node=self.hidden_irrep_base,
                        irrep_in_node=self.irrep_so2_in_node,
                        irrep_bottle_hidden=self.irrep_so2_in_node,
                        irrep_out=self.irrep_so2_in_node,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        resnet=True,
                    )
                )

                self.e3_gnn_node_pair_layer.append(
                    PairNetLayer(
                        irrep_in_node=self.irrep_so2_in_node,
                        irrep_bottle_hidden=self.irrep_so2_in_node,
                        irrep_out=self.irrep_so2_in_node,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        invariant_layers=self.num_fc_layer,
                        invariant_neurons=self.hs,
                        resnet=True,
                    )
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
            # input_expand_ii = self.hidden_bottle_irrep
            # input_expand_ij = self.hidden_bottle_irrep
            input_expand_ii = self.irrep_so2_in_node
            input_expand_ij = self.irrep_so2_in_node

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

        # self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        # self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

        # self.deq = get_deq(**deq_kwargs)

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

    def injection(self, data):
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
        return data, node_attr, edge_sh, rbf_new, transpose_edge_index

    def filter(
        self,
        H,
        data,
        node_attr,
        edge_sh,
        rbf_new,
        transpose_edge_index,
        keep_blocks=False,
    ):
        node_attr_R = node_attr
        # embeded_t = self.sigma_embedding(get_time_embedding(data.t, self.hs))
        # node_attr_R = torch.cat([node_attr_R, embeded_t], dim=-1)
        # node_attr_R = self.node_embedding_t(node_attr_R)
        # node_attr_R = node_attr_R + embeded_t[data.batch]
        embedded_t = get_time_embedding(data.t, self.hs)
        node_attr_R = torch.cat([node_attr_R, embedded_t[data.batch]], dim=-1)
        node_attr_R = self.sigma_embedding(node_attr_R)

        edge_dst, edge_src = data.edge_index
        node_feats_H = self.onebody_reduction(data, H, keep_blocks)
        if self.use_block_H:
            node_feats_H_init = self.onebody_reduction(data, data.init_ham, keep_blocks)
        if self.use_block_S:
            node_feats_S = self.onebody_reduction(data, data.overlap, keep_blocks)
        node_attr_R_init = node_attr_R
        # node_attr_R_init = data.node_attr

        full_dst, full_src = data.full_edge_index

        ### ESCN
        edge_distance_vec, data.edge_distance = get_edge_vectors_and_lengths(
            positions=data.pos,
            edge_index=data.edge_index,
            shifts=torch.zeros([data.edge_index.shape[1], 3]).to(data.pos.device),
        )
        edge_rot_mat = self._init_edge_rot_mat(
            data.pos, data.edge_index, edge_distance_vec
        )
        self.SO3_edge_rot = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_edge_rot.append(SO3_Rotation(edge_rot_mat, self.lmax_list[i]))

        x = SO3_Embedding(
            data.pos.shape[0],
            self.lmax_list,
            self.sphere_channels,
            self.device,
            torch.float64,
        )

        offset_res = 0
        offset = 0
        # Initialize the l=0,m=0 coefficients for each resolution
        for i in range(self.num_resolutions):
            x.embedding[:, offset_res, :] = data.node_attr[
                :, offset : offset + self.sphere_channels
            ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # This can be expensive to compute (not implemented efficiently), so only do it once and pass it along to each layer
        mappingReduced = CoefficientMapping(self.lmax_list, self.mmax_list, self.device)

        # tic = time.time()
        fii = None
        fij = None
        for layer_idx, layer in enumerate(self.e3_gnn_layer):
            x, node_attr_R = layer(
                data,
                x,
                SO3_edge_rot=self.SO3_edge_rot,
                mappingReduced=mappingReduced,
                x_irrep_out_old=node_attr_R,
            )
            node_attr = torch.ones_like(node_feats_H.narrow(1, 0, 1))
            node_concat = [node_attr_R]
            node_feats_H = self.blocks_H_cur[layer_idx](
                node_input=node_feats_H,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=rbf_new,
                batch=data.batch,
            )
            node_concat.append(node_feats_H)
            if self.use_block_H:
                node_feats_H_init = self.blocks_H_init[layer_idx](
                    node_input=node_feats_H_init,
                    node_attr=node_attr,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    edge_attr=edge_sh,
                    edge_scalars=rbf_new,
                    batch=data.batch,
                )
                node_concat.append(node_feats_H_init)

            if self.use_block_S:
                node_feats_S = self.blocks_S[layer_idx](
                    node_input=node_feats_S,
                    node_attr=node_attr,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    edge_attr=edge_sh,
                    edge_scalars=rbf_new,
                    batch=data.batch,
                )
                node_concat.append(node_feats_S)

            node_concat = (
                torch.cat(node_concat, dim=-1)
                .index_select(-1, self.hidden_irrep_concat_idx.to(node_attr_R.device))
                .contiguous()
            )
            node_attr_R = self.blocks_Linear[layer_idx](node_concat)
            node_attr_R = self.norm(node_attr_R, batch=data.batch)

            if layer_idx > self.start_layer:
                fii = self.e3_gnn_node_layer[layer_idx - self.start_layer - 1](
                    data, node_attr_R, fii
                )
                fij = self.e3_gnn_node_pair_layer[layer_idx - self.start_layer - 1](
                    data, node_attr_R, fij
                )

        # fii = self.output_ii(fii)
        # fij = self.output_ij(fij)
        hamiltonian_diagonal_matrix = self.expand_ii["hamiltonian"](
            fii,
            self.fc_ii["hamiltonian"](node_attr_R_init),
            self.fc_ii_bias["hamiltonian"](node_attr_R_init),
        )
        node_pair_embedding = torch.cat(
            [node_attr_R_init[full_dst], node_attr_R_init[full_src]], dim=-1
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

            return hamiltonian_matrix
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
            return ret_hamiltonian_diagonal_matrix, ret_hamiltonian_non_diagonal_matrix

    def forward(self, data, H, keep_blocks=False):
        data, node_attr, edge_sh, rbf_new, transpose_edge_index = self.injection(data)
        H_pred = self.filter(
            H, data, node_attr, edge_sh, rbf_new, transpose_edge_index, keep_blocks
        )
        results = {}
        results["hamiltonian"] = H_pred
        return results

    def build_graph(self, data, max_radius):
        node_attr = data.atoms.squeeze()
        radius_edges = radius_graph(data.pos, max_radius, data.batch)

        dst, src = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
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
