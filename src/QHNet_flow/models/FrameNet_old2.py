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
from .QHNet import *

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import GraphNorm


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


class EdgeEmbdding(nn.Module):
    def __init__(self, hidden_size, radius_embed_dim):
        super(EdgeEmbdding, self).__init__()
        self.hidden_size = hidden_size
        self.linear_pos = nn.Linear(3, hidden_size // 2)
        self.linear_rbf = nn.Linear(radius_embed_dim, hidden_size - hidden_size // 2)
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, rel_pos, rbf):
        _rel_pos = self.linear_pos(rel_pos)
        _rbf = self.linear_rbf(rbf)
        e = torch.cat([_rel_pos, _rbf], dim=-1)
        e = self.act(e)
        e = self.linear(e)

        return e


@torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        # x: node features, edge_index: edge indices
        x = self.W(x)
        row, col = edge_index
        alpha = self.leakyrelu(self.a(torch.cat([x[row], x[col]], dim=1)))
        alpha = softmax_dropout(alpha, self.dropout, self.training)
        out = scatter(alpha * x[col], row, dim=0, dim_size=x.size(0), reduce="sum")
        return out


class NodeMessagePassingAttention(MessagePassing):
    def __init__(self, hidden_size, dropout=0.1, **kwargs):
        super(NodeMessagePassingAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.linear_edge = nn.Linear(3 * hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.attention_layer = GraphAttentionLayer(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x, edge_index, edge_attr):
        # x_i: node features of the source nodes, edge_index: edge indices
        row, col = edge_index
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.attention_layer(out, edge_index)
        out = self.linear_out(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        # x_j: node features of the source nodes, edge_attr: edge features
        out = self.linear_edge(torch.cat([x_i, x_j, edge_attr], dim=-1))
        out = self.act(out)
        return out


class NodeMessagePassing(MessagePassing):
    def __init__(self, hidden_size, **kwargs):
        super(NodeMessagePassing, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.linear_edge = nn.Linear(3 * hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x, edge_index, edge_attr):
        # x_i: node features of the source nodes, edge_index: edge indices
        row, col = edge_index
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.linear_out(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        # x_j: node features of the source nodes, edge_attr: edge features
        out = self.linear_edge(torch.cat([x_i, x_j, edge_attr], dim=-1))
        out = self.act(out)
        return out


class SelfInteractionBlock(MessagePassing):
    def __init__(self, hidden_size, ham_hidden, **kwargs):
        super(SelfInteractionBlock, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.linear_message = nn.Linear(3 * hidden_size + ham_hidden, hidden_size)
        self.linear_out = nn.Linear(hidden_size, ham_hidden)
        self.act = nn.SiLU()

    def forward(self, fii, x, edge_index, edge_attr):
        # x: node features, edge_index: edge indices, edge_attr: edge features
        row, col = edge_index
        out = self.propagate(edge_index, x=x, H=fii, edge_attr=edge_attr)
        out = self.linear_out(out)
        return out

    def message(self, x_i, x_j, H_i, H_j, edge_attr):
        # x_j: node features of the source nodes, edge_attr: edge features
        out = self.linear_message(torch.cat([x_i, x_j, H_i, edge_attr], dim=-1))
        out = self.act(out)
        return out


class SelfInteractionBlockAttention(MessagePassing):
    def __init__(self, hidden_size, ham_hidden, **kwargs):
        super(SelfInteractionBlockAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.linear_message = nn.Linear(3 * hidden_size + ham_hidden, hidden_size)
        self.linear_out = nn.Linear(hidden_size, ham_hidden)
        self.attention_layer = GraphAttentionLayer(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, fii, x, edge_index, edge_attr):
        # x: node features, edge_index: edge indices, edge_attr: edge features
        row, col = edge_index
        out = self.propagate(edge_index, x=x, H=fii, edge_attr=edge_attr)
        out = self.attention_layer(out, edge_index)
        out = self.linear_out(out)
        return out

    def message(self, x_i, x_j, H_i, H_j, edge_attr):
        # x_j: node features of the source nodes, edge_attr: edge features
        out = self.linear_message(torch.cat([x_i, x_j, H_i, edge_attr], dim=-1))
        out = self.act(out)
        return out


class PairInteractionBlock(nn.Module):
    def __init__(self, hidden_size, ham_hidden, **kwargs):
        super(PairInteractionBlock, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(2 * hidden_size + ham_hidden, hidden_size)
        self.act = nn.SiLU()
        self.linear_out2 = nn.Linear(hidden_size, ham_hidden)

    def forward(self, fij, x_i, x_j):
        out = torch.cat([fij, x_i, x_j], dim=-1)
        out = self.linear(out)
        out = self.act(out)
        out = self.linear_out2(out)
        return out


class FrameNet(nn.Module):
    def __init__(
        self,
        in_node_features=1,
        hidden_size=128,
        bottle_hidden_size=32,
        num_gnn_layers=5,
        max_radius=12,
        num_nodes=10,
        radius_embed_dim=32,  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        ham_dim=24,
        ham_hidden=512,
        **deq_kwargs,
    ):
        super(FrameNet, self).__init__()
        self.hs = hidden_size
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.distance_expansion = ExponentialBernsteinRadialBasisFunctions(
            self.radius_embed_dim, self.max_radius
        )
        self.num_gnn_layers = num_gnn_layers
        self.ham_dim = ham_dim
        self.ham_hidden = ham_hidden
        self.ham_dim_square = ham_dim * ham_dim
        self.node_embedding = nn.Embedding(num_nodes, self.hs)
        self.edge_embedding = EdgeEmbdding(self.hs, self.radius_embed_dim)
        self.ham_embedding = nn.Sequential(
            nn.Linear(self.ham_dim_square, ham_hidden),
            nn.SiLU(),
            nn.Linear(ham_hidden, ham_hidden),
        )
        self.sigma_embedding = nn.Sequential(
            nn.Linear(2 * self.hs, self.hs), nn.SiLU(), nn.Linear(self.hs, self.hs)
        )

        self.node_layers = nn.ModuleList()
        for i in range(self.num_gnn_layers):
            self.node_layers.append(NodeMessagePassingAttention(self.hs, aggr="add"))

        self.self_interaction_blocks = nn.ModuleList()
        self.pair_interaction_blocks = nn.ModuleList()
        for i in range(self.num_gnn_layers):
            self.self_interaction_blocks.append(
                SelfInteractionBlockAttention(self.hs, self.ham_hidden, aggr="add")
            )
            self.pair_interaction_blocks.append(
                PairInteractionBlock(self.hs, self.ham_hidden)
            )

        self.hidden_irrep_dim = self.hbs * 25
        self.ham_decoder_self = nn.Sequential(
            nn.Linear(self.ham_hidden, self.ham_hidden),
            nn.SiLU(),
            nn.Linear(self.ham_hidden, self.hidden_irrep_dim),
        )
        self.ham_decoder_pair = nn.Sequential(
            nn.Linear(self.ham_hidden, self.ham_hidden),
            nn.SiLU(),
            nn.Linear(self.ham_hidden, self.hidden_irrep_dim),
        )
        # self.deq = get_deq(**deq_kwargs)

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
            input_expand_ii = o3.Irreps(
                f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e"
            ) # self.hbs * (1 + 3 + 5 + 7 + 9) = 25 * self.hbs

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
                o3.Irreps(
                    f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e"
                ),
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

    def injection(self, data, frame=True):
        node_attr, edge_index, edge_vec, rbf_new, _ = self.build_graph(
            data, self.max_radius, frame
        )
        node_attr = self.node_embedding(node_attr)
        data.node_attr, data.edge_index, data.edge_attr = (
            node_attr,
            edge_index,
            rbf_new,
        )

        _, full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index = (
            self.build_graph(data, 10000, frame)
        )
        data.full_edge_index, data.full_edge_attr, data.full_edge_sh = (
            full_edge_index,
            full_edge_attr,
            full_edge_sh,
        )
        return data, node_attr, rbf_new, edge_vec, transpose_edge_index

    def filter(
        self,
        H,
        data,
        node_attr,
        edge_vec,
        rbf_new,
        transpose_edge_index,
        keep_blocks=False,
    ):
        node_attr_R = node_attr
        embedded_t = get_time_embedding(data.t, self.hs)

        edge_attr = self.edge_embedding(edge_vec, rbf_new)
        node_attr_R = torch.cat([node_attr_R, embedded_t[data.batch]], dim=-1)
        node_attr_R = self.sigma_embedding(node_attr_R)
        node_attr_H = self.ham_embedding(H.reshape(-1, self.ham_dim_square))

        node_attr_R2 = node_attr_R

        for i in range(len(self.node_layers)):
            node_attr_R2 = node_attr_R2 + self.node_layers[i](
                node_attr_R2, data.edge_index, edge_attr
            )

        edge_dst, edge_src = data.edge_index
        full_dst, full_src = data.full_edge_index

        # tic = time.time()
        fii = node_attr_H[data.batch]
        # fij = torch.cat([node_attr_H[full_dst], node_attr_H[full_src]], dim=-1)
        fij = node_attr_H[data.batch][full_dst]

        for i in range(self.num_gnn_layers):
            # print(fii.shape, fij.shape)
            fii = fii + self.self_interaction_blocks[i](
                fii, node_attr_R2, data.edge_index, edge_attr
            )
            fij = fij + self.pair_interaction_blocks[i](
                fij, node_attr_R2[full_dst], node_attr_R2[full_src]
            )

        hamiltonian_diagonal_matrix = self.ham_decoder_self(fii)
        hamiltonian_non_diagonal_matrix = self.ham_decoder_pair(fij)
        
        hamiltonian_diagonal_matrix = torch.einsum("bi,bj->bij",hamiltonian_diagonal_matrix,hamiltonian_diagonal_matrix)
        hamiltonian_non_diagonal_matrix =torch.einsum("bi,bj->bij",hamiltonian_non_diagonal_matrix,hamiltonian_non_diagonal_matrix)
        
    
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

    def forward(self, data, H, keep_blocks=False, frame=True):
        data, node_attr, edge_vec, rbf_new, transpose_edge_index = self.injection(
            data, frame
        )
        if frame:
            WD = data.WD_block_diag.detach()
            H = torch.bmm(WD, torch.bmm(H, WD.transpose(-1, -2)))

        H_pred = self.filter(
            H, data, node_attr, edge_vec, rbf_new, transpose_edge_index, keep_blocks
        )

        if frame:
            H_pred = torch.bmm(WD.transpose(-1, -2), torch.bmm(H_pred, WD))
            # H_pred = torch.bmm(WD, torch.bmm(H_pred, WD.transpose(-1, -2)))

        results = {}
        results["hamiltonian"] = H_pred
        return results

    def build_graph(self, data, max_radius, frame=True):
        if frame:
            pos = data.pos_rot
        node_attr = data.atoms.squeeze()
        radius_edges = radius_graph(pos, max_radius, data.batch)

        dst, src = radius_edges
        edge_vec = pos[dst.long()] - pos[src.long()]
        rbf = (
            self.distance_expansion(edge_vec.norm(dim=-1).unsqueeze(-1))
            .squeeze()
            .type(pos.type())
        )

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
            edge_vec,
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
