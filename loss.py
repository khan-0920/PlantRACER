#!/usr/bin/env python3
"""
Loss functions for PlantRACER CLIP training.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def clip_loss(
    reaction_repr: Tensor,
    protein_repr: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """
    reaction_repr, protein_repr: [B, H]
    """
    assert reaction_repr.shape == protein_repr.shape
    logits = reaction_repr @ protein_repr.t()  # [B,B]
    logits = logits / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_r2p = F.cross_entropy(logits, targets)
    loss_p2r = F.cross_entropy(logits.t(), targets)
    return (loss_r2p + loss_p2r) * 0.5


def clip_loss_with_negatives(
    reaction_repr: Tensor,
    protein_repr: Tensor,
    neg_repr_batch: List[Optional[Tensor]],
    temperature: float = 0.07,
) -> Tensor:
    
    assert reaction_repr.shape == protein_repr.shape
    B, H = reaction_repr.shape
    device = reaction_repr.device

    logits = reaction_repr @ protein_repr.t()  # [B,B]
    logits = logits / temperature
    targets = torch.arange(B, device=device)

    # 反应->酶 方向: 标准 CLIP
    loss_r2p = F.cross_entropy(logits, targets)

    # 酶->反应 方向: 分母中加入负反应
    logits_p2r = logits.t()  # [B,B]
    losses_p2r: List[Tensor] = []

    for i in range(B):
        row = logits_p2r[i]  # [B]
        neg_repr = neg_repr_batch[i]  # [N_neg_i, H] 或 None

        # 若该样本没有有效副反应（None 或空），退化为标准 CE
        if neg_repr is None or neg_repr.numel() == 0:
            losses_p2r.append(
                F.cross_entropy(row.unsqueeze(0), targets[i : i + 1])
            )
            continue

        # 该酶与负反应的相似度
        p_vec = protein_repr[i].unsqueeze(0)  # [1,H]
        neg_logits = (neg_repr @ p_vec.t()).squeeze(-1)  # [N_neg_i]
        neg_logits = neg_logits / temperature

        # 数值稳定的 softmax 分母: batch 内所有正反应 + 副反应
        max_val = torch.maximum(row.max(), neg_logits.max())
        exp_row = torch.exp(row - max_val)      # [B]
        exp_neg = torch.exp(neg_logits - max_val)  # [N_neg]

        numerator = torch.exp(row[i] - max_val)
        denom = exp_row.sum() + exp_neg.sum()
        loss_i = -torch.log(numerator / denom)
        losses_p2r.append(loss_i)

    loss_p2r = torch.stack(losses_p2r).mean()
    return 0.5 * (loss_r2p + loss_p2r)


__all__ = ["clip_loss", "clip_loss_with_negatives"]

