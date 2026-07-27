#!/usr/bin/env python3
"""
PlantRACER reaction / protein encoders and CLIP-style joint model.
"""

from typing import List, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Set2Set


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MoleculeGraphEncoder(nn.Module):
    """
    MPNN + attention pooling graph encoder.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # message MLP: concat(x_j, e_ij) -> hidden, 使用 nn.Sequential 实现
        self.message_mlps = nn.ModuleList([nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
                for _ in range(num_layers)
            ]
        )

        self.update_mlps = nn.ModuleList([nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
                for _ in range(num_layers)
            ]
        )
        # second-stage update: concat(x_i, delta_i) -> hidden
        self.update_out_mlps = nn.ModuleList([nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Set2Set pooling (graph-level)
        # 输出维度为 2 * hidden_dim
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        # 将 2H 映射回 H，保持与下游接口一致
        self.set2set_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def _message_passing(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        layer_idx: int,
    ) -> Tensor:
        """
        Message passing.
        """
        row, col = edge_index  # row: dst, col: src
        src = x[col]  # [E, H]
        e = edge_attr  # [E, H]

        m_ij = torch.cat([src, e], dim=-1)  # [E, 2H]
        m_ij = self.message_mlps[layer_idx](m_ij)  # [E, H]

        m_i = torch.zeros_like(x)
        m_i.index_add_(0, row, m_ij)
        deg = torch.bincount(row, minlength=x.size(0)).clamp(min=1).float().unsqueeze(-1)
        m_i = m_i / deg  # [N, H]

        upd_in = torch.cat([x, m_i], dim=-1)  # [N, 2H]
        delta = self.update_mlps[layer_idx](upd_in)  # [N, H]

        out_in = torch.cat([x, delta], dim=-1)  # [N, 2H]
        x_new = self.update_out_mlps[layer_idx](out_in)  # [N, H]
        return x_new

    def forward(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """
        输入: PyG Batch, 包含 x, edge_index, edge_attr, batch (节点->分子 id)
        输出:
        - mol_repr: [num_molecules, hidden_dim] 分子级表示
        - node_repr: [num_nodes, hidden_dim] 节点级表示
        """
        x = self.node_proj(batch.x)  # [N, H]
        e = self.edge_proj(batch.edge_attr)  # [E, H]

        for l in range(self.num_layers):
            x = self._message_passing(x, batch.edge_index, e, layer_idx=l)
            x = F.relu(x)
            x = self.dropout(x)

        # Set2Set pooling 到分子级别
        # batch.batch: [N]，表示每个节点所属的图索引
        mol_repr_2h = self.set2set(x, batch.batch)  # [num_graphs, 2H]
        mol_repr = self.set2set_proj(mol_repr_2h)   # [num_graphs, H]

        return mol_repr, x


class ReactionEncoder(nn.Module):
    """
    对一个反应里的所有分子级表示做 self-attention，得到反应级表示。

    输入:
    - mol_repr: [num_molecules, hidden_dim]
    - reaction_idx: [num_molecules] (分子 -> 反应 id)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        #self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def _self_attention(self, x: Tensor) -> Tensor:
        """
        x: [M, H]
        """
        if x.size(0) == 1:
            return x.squeeze(0)
        Q = self.q_proj(x)  # [M,H]
        K = self.k_proj(x)
        scale = self.hidden_dim ** -0.5
        attn = torch.matmul(Q, K.t()) * scale  # [M,M]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        alpha = attn.mean(dim=0)   # [M]
        out = (alpha.unsqueeze(-1) * x).sum(dim=0)
        
        return out

    def forward(self, mol_repr: Tensor, reaction_idx: Tensor) -> Tensor:
        """
        返回 reaction_repr: [num_reactions, hidden_dim]
        """
        device = mol_repr.device
        if mol_repr.numel() == 0:
            return torch.empty(0, self.hidden_dim, device=device)

        num_reactions = int(reaction_idx.max().item()) + 1
        reaction_repr = []
        for r in range(num_reactions):
            mask = reaction_idx == r
            x_r = mol_repr[mask]
            if x_r.size(0) == 0:
                reaction_repr.append(torch.zeros(self.hidden_dim, device=device))
            else:
                reaction_repr.append(self._self_attention(x_r))
        return torch.stack(reaction_repr, dim=0)  # [R,H]


class ProteinEncoder(nn.Module):
    """
    Protein encoder.
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = MLP(in_dim, [hidden_dim,hidden_dim,hidden_dim], hidden_dim, dropout=dropout)

    def forward(self, emb: Tensor) -> Tensor:
        """
        emb: [B, in_dim] or [in_dim]
        """
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return F.normalize(self.mlp(emb), p=2, dim=-1)


class EnzymeReactionCLIPModel(nn.Module):
    """
    Enzyme-reaction matching model.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        protein_emb_dim: int,
        hidden_dim: int = 256,
        mpnn_layers: int = 3,
        dropout: float = 0.1,
        side: Literal["substrate", "product", "both"] = "substrate",
    ) -> None:
        super().__init__()
        assert side in ("substrate", "product", "both")
        self.side = side
        self.hidden_dim = hidden_dim

        self.substrate_encoder = MoleculeGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=mpnn_layers,
            dropout=dropout,
        )
        self.product_encoder = MoleculeGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=mpnn_layers,
            dropout=dropout,
        )
        self.reaction_encoder = ReactionEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.protein_encoder = ProteinEncoder(
            in_dim=protein_emb_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def encode_reactions(
        self, batch_data: List[Data], side: Literal["substrate", "product", "both"] = None
    ) -> Tensor:
        """
        Encode reactions.
        """
        side = side or self.side

        # 收集所有 substrates / products 及其所在的 reaction 索引
        all_sub_mols: List[Data] = []
        sub_rxn_idx: List[int] = []
        all_prod_mols: List[Data] = []
        prod_rxn_idx: List[int] = []

        for i, d in enumerate(batch_data):
            subs = list(getattr(d, "substrates", []) or [])
            prods = list(getattr(d, "products", []) or [])

            if subs:
                all_sub_mols.extend(subs)
                sub_rxn_idx.extend([i] * len(subs))
            if prods:
                all_prod_mols.extend(prods)
                prod_rxn_idx.extend([i] * len(prods))

        device = batch_data[0].substrates[0].x.device if batch_data and getattr(batch_data[0], "substrates", None) else torch.device("cpu")

        def _encode_side(
            mols: List[Data],
            rxn_idx: List[int],
            encoder: MoleculeGraphEncoder,
        ) -> Tuple[Tensor, Tensor]:
            if not mols:
                return (
                    torch.empty(0, self.hidden_dim, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                )
            batch = Batch.from_data_list(mols)
            mol_repr, _ = encoder(batch)
            rxn_idx_tensor = torch.tensor(rxn_idx, dtype=torch.long, device=mol_repr.device)
            return mol_repr, rxn_idx_tensor

        # 根据 side 选择编码逻辑
        if side == "substrate":
            mol_repr_sub, rxn_idx_sub = _encode_side(
                all_sub_mols, sub_rxn_idx, self.substrate_encoder
            )
            if mol_repr_sub.numel() == 0:
                return torch.zeros(0, self.hidden_dim, device=device)
            reaction_repr = self.reaction_encoder(mol_repr_sub, rxn_idx_sub)
        elif side == "product":
            mol_repr_prod, rxn_idx_prod = _encode_side(
                all_prod_mols, prod_rxn_idx, self.product_encoder
            )
            if mol_repr_prod.numel() == 0:
                return torch.zeros(0, self.hidden_dim, device=device)
            reaction_repr = self.reaction_encoder(mol_repr_prod, rxn_idx_prod)
        else:  # both: 两侧分别编码，再在 reaction 级别合并
            mol_repr_sub, rxn_idx_sub = _encode_side(
                all_sub_mols, sub_rxn_idx, self.substrate_encoder
            )
            mol_repr_prod, rxn_idx_prod = _encode_side(
                all_prod_mols, prod_rxn_idx, self.product_encoder
            )
            if mol_repr_sub.numel() == 0 and mol_repr_prod.numel() == 0:
                return torch.zeros(0, self.hidden_dim, device=device)
            if mol_repr_sub.numel() == 0:
                reaction_repr = self.reaction_encoder(mol_repr_prod, rxn_idx_prod)
            elif mol_repr_prod.numel() == 0:
                reaction_repr = self.reaction_encoder(mol_repr_sub, rxn_idx_sub)
            else:
                mol_repr_cat = torch.cat([mol_repr_sub, mol_repr_prod], dim=0)
                rxn_idx_cat = torch.cat([rxn_idx_sub, rxn_idx_prod], dim=0)
                reaction_repr = self.reaction_encoder(mol_repr_cat, rxn_idx_cat)

        return F.normalize(reaction_repr, p=2, dim=-1)

    def encode_proteins(self, protein_emb: Tensor) -> Tensor:
        """
        protein_emb: [B, Dp] or [Dp]
        """
        return self.protein_encoder(protein_emb)

    def forward(
        self, batch_data: List[Data], protein_emb: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        返回:
        - reaction_repr: [B,H]
        - protein_repr: [B,H]
        """
        reaction_repr = self.encode_reactions(batch_data, side=self.side)
        protein_repr = self.encode_proteins(protein_emb)
        return reaction_repr, protein_repr


__all__ = [
    "MoleculeGraphEncoder",
    "ReactionEncoder",
    "ProteinEncoder",
    "EnzymeReactionCLIPModel",
]

