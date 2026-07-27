#!/usr/bin/env python3
import os
import argparse
import pickle
import random
from typing import List, Optional, Tuple, Dict, Any, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.data import Data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import EnzymeReactionCLIPModel
from loss import clip_loss, clip_loss_with_negatives
from process_dataset import atom_featurizer, bond_featurizer
import pandas as pd

def _bytes_to_sample(b: bytes) -> Data:
    return pickle.loads(b)


class InMemoryPairDataset(Dataset):
    """
    直接基于 reaction.pt / protein.pt 和 pair CSV 的内存数据集。

    - reaction_dict: {RHEA_ID: {"substrates": [Data...], "products": [Data...]}}
    - protein_dict: {UniprotID: Tensor[Dp]}
    - pair_csv: 包含 RHEA_ID, UniprotID 的 CSV（如 train_pair_filtered.csv）
    """

    def __init__(
        self,
        reaction_dict: Dict[int, Dict[str, Any]],
        protein_dict: Dict[str, torch.Tensor],
        pair_csv: str,
        neg_reaction_dict: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> None:
        super().__init__()
        self.reaction_dict = reaction_dict
        self.protein_dict = protein_dict
        self.neg_reaction_dict = neg_reaction_dict or {}

        df = pd.read_csv(pair_csv)
        if "RHEA_ID" not in df.columns or "UniprotID" not in df.columns:
            raise ValueError(
                f"pair CSV {pair_csv} must contain 'RHEA_ID' and 'UniprotID' columns"
            )

        pairs: List[Tuple[int, str]] = []
        for _, row in df.iterrows():
            try:
                rhea_id = str(row["RHEA_ID"])
                uid = str(row["UniprotID"]).strip()
            except Exception:
                continue
            if not uid:
                continue
            if rhea_id not in self.reaction_dict:
                continue
            if uid not in self.protein_dict:
                continue
            pairs.append((rhea_id, uid))

        if not pairs:
            raise RuntimeError(f"No valid pairs found in {pair_csv}")

        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Data:
        rhea_id, uid = self.pairs[idx]
        rxn = self.reaction_dict[rhea_id]
        prot_emb = self.protein_dict[uid]

        # 正反应分子图
        substrates = rxn["substrates"]
        products = rxn["products"]

        # 该正反应对应的所有副反应 (已在 preprocess 阶段转为图结构字典)
        neg_items = self.neg_reaction_dict.get(rhea_id, [])
        neg_reactions: List[Data] = []
        for item in neg_items:
            subs = item.get("substrates") or []
            prods = item.get("products") or []
            if not subs and not prods:
                continue
            neg_reactions.append(
                Data(substrates=subs, products=prods)
            )

        return Data(
            substrates=substrates,
            products=products,
            embedding=prot_emb,
            reaction_id=rhea_id,
            neg_reactions=neg_reactions,
        )


def collate_clip(
    batch: List[Data],
) -> Tuple[List[Data], torch.Tensor, List[List[Data]]]:
    """
    collate 函数:
    - 保持反应级 Data 列表原样，用于 `EnzymeReactionCLIPModel` 的 encode_reactions
    - 将 protein embedding 堆叠成 [B, Dp]
    """
    if not batch:
        return [], torch.empty(0, 0), torch.empty(0, dtype=torch.long), []

    protein_emb_list = []
    neg_reactions_batch: List[List[Data]] = []

    for b_idx, d in enumerate(batch):
        if not hasattr(d, "embedding"):
            raise ValueError(
                "Data object missing 'embedding' field required for CLIP training"
            )
        if not hasattr(d, "reaction_id"):
            raise ValueError(
                "Data object missing 'reaction_id' field required for negative sampling"
            )
        if not hasattr(d, "neg_reactions"):
            raise ValueError(
                "Data object missing 'neg_reactions' field required for negative sampling"
            )
        protein_emb_list.append(d.embedding.unsqueeze(0))
        neg_reactions_batch.append(list(d.neg_reactions or []))

        # 为 substrates / products 中的每个分子添加:
        # - mol_idx: 在该样本内部的分子索引 (atom-level 所属分子索引)
        # - batch_idx: 该分子所属的反应在当前 batch 中的索引
        subs = list(getattr(d, "substrates", []) or [])
        prods = list(getattr(d, "products", []) or [])
        for m_idx, mol in enumerate(subs):
            setattr(mol, "mol_idx", m_idx)
            setattr(mol, "batch_idx", b_idx)
        for m_idx, mol in enumerate(prods):
            setattr(mol, "mol_idx", m_idx)
            setattr(mol, "batch_idx", b_idx)

    protein_emb = torch.cat(protein_emb_list, dim=0)  # [B, Dp]
    return batch, protein_emb, neg_reactions_batch


class UniquePairBatchSampler(Sampler[List[int]]):
    """
    自定义 BatchSampler:
    - 每个 batch 中的 RHEA_ID 和 UniprotID 都互不相同
    - 每个 epoch 中，每个 (RHEA_ID, UniprotID) pair 最多被使用一次
    - 如果剩余样本无法再组成满足条件的完整 batch，则本 epoch 直接结束（不产生残缺 batch）
    """

    def __init__(
        self,
        pairs: List[Tuple[int, str]],
        batch_size: int,
    ) -> None:
        self.pairs = pairs
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[int]]:
        # 每个 epoch 打乱一次索引
        indices = list(range(len(self.pairs)))
        random.shuffle(indices)

        remaining = indices

        while True:
            batch: List[int] = []
            used_rhea: set[int] = set()
            used_prot: set[str] = set()
            selected: List[int] = []

            # 贪心地从 remaining 中抽样，构造一个满足“反应/蛋白都唯一”的 batch
            for idx in remaining:
                rhea_id, uid = self.pairs[idx]
                if rhea_id in used_rhea or uid in used_prot:
                    continue
                batch.append(idx)
                selected.append(idx)
                used_rhea.add(rhea_id)
                used_prot.add(uid)
                if len(batch) >= self.batch_size:
                    break

            # 无法再组成完整 batch，则本 epoch 结束
            if len(batch) < self.batch_size:
                break

            yield batch

            # 从 remaining 中移除本 batch 用过的样本，保证一个 epoch 中每个 pair 只用一次
            used_set = set(selected)
            remaining = [i for i in remaining if i not in used_set]
            if not remaining:
                break

    def __len__(self) -> int:
        # 近似长度（真实生成的 batch 数可能略小于此值）
        return len(self.pairs) // self.batch_size


class PlantRACERDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for CLIP training on PlantRACER.

    - 期望 data_dir 下存在:
        - reaction.pt: {RHEA_ID: {"substrates": [Data...], "products": [Data...]}}
        - protein.pt: {UniprotID: Tensor[Dp]}
        - train_pair_filtered.csv / test_pair_filtered.csv
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[InMemoryPairDataset] = None
        self.reaction_dict: Optional[Dict[int, Dict[str, Any]]] = None
        self.protein_dict: Optional[Dict[str, torch.Tensor]] = None
        self.neg_reaction_dict: Optional[Dict[int, List[Dict[str, Any]]]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # 一次性读入 reaction.pt / protein.pt / neg_reaction.pt
        reaction_pt = os.path.join(self.data_dir, "reaction.pt")
        protein_pt = os.path.join(self.data_dir, "protein.pt")
        neg_reaction_pt = os.path.join(self.data_dir, "neg_reaction.pt")
        if not os.path.isfile(reaction_pt):
            raise FileNotFoundError(f"reaction.pt not found at {reaction_pt}")
        if not os.path.isfile(protein_pt):
            raise FileNotFoundError(f"protein.pt not found at {protein_pt}")

        self.reaction_dict = torch.load(reaction_pt)
        self.protein_dict = torch.load(protein_pt)
        if os.path.isfile(neg_reaction_pt):
            self.neg_reaction_dict = torch.load(neg_reaction_pt)
        else:
            self.neg_reaction_dict = {}

        train_pairs_csv = os.path.join(self.data_dir, "train_pair_filtered.csv")
        if not os.path.isfile(train_pairs_csv):
            raise FileNotFoundError(f"train_pair_filtered.csv not found at {train_pairs_csv}")

        self.train_ds = InMemoryPairDataset(
            reaction_dict=self.reaction_dict,
            protein_dict=self.protein_dict,
            pair_csv=train_pairs_csv,
            neg_reaction_dict=self.neg_reaction_dict,
        )


    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        batch_sampler = UniquePairBatchSampler(
            pairs=self.train_ds.pairs,  # type: ignore[attr-defined]
            batch_size=self.batch_size,
        )
        return DataLoader(
            self.train_ds,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=collate_clip,
        )



class PlantRACERCLIPLightningModule(pl.LightningModule):
    """
    Lightning wrapper for CLIP-style PlantRACER model.
    """

    def __init__(
        self,
        model: EnzymeReactionCLIPModel,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.save_hyperparameters(ignore=["model"])

    def forward(
        self,
        reactions: List[Data],
        protein_emb: torch.Tensor,
        neg_reactions_batch: List[List[Data]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        前向:
        - 编码正样本反应和蛋白
        - 同时为每个样本编码其所有副反应，返回每个样本对应的 neg_repr 张量
        """
        reaction_repr, protein_repr = self.model(reactions, protein_emb)

        neg_repr_batch: List[Optional[torch.Tensor]] = []
        for neg_list in neg_reactions_batch:
            if not neg_list:
                # 无副反应时直接使用 None，由 loss 函数处理为“退化为标准 CLIP”
                neg_repr_batch.append(None)
                continue
            neg_repr = self.model.encode_reactions(
                neg_list, side=self.model.side
            )  # [N_neg_i, H]
            if neg_repr is None:
                neg_repr_batch.append(None)
            else:
                neg_repr_batch.append(neg_repr)

        return reaction_repr, protein_repr, neg_repr_batch

    def _step(self, batch, prefix: str) -> torch.Tensor:
        reactions, protein_emb, neg_reactions_batch = batch
        reaction_repr, protein_repr, neg_repr_batch = self(
            reactions, protein_emb, neg_reactions_batch
        )
        # 始终使用带副反应的 loss；当某个样本的 neg_repr 为空时，
        # clip_loss_with_negatives 会对该样本自动退化为标准 CLIP 行为
        loss = clip_loss_with_negatives(
            reaction_repr=reaction_repr,
            protein_repr=protein_repr,
            neg_repr_batch=neg_repr_batch,
            temperature=self.temperature,
        )
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=protein_emb.size(0),
        )
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


def main():
    parser = argparse.ArgumentParser(
        description="PlantRACER CLIP-style training with PyTorch Lightning"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing reaction.pt / protein.pt / *_pair_filtered.csv",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of reaction/protein encoders",
    )
    parser.add_argument(
        "--mpnn_layers",
        type=int,
        default=5,
        help="Number of graph message passing layers",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["substrate", "product", "both"],
        default="both",
        help="Use substrates / products / both to encode reactions",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Softmax temperature for CLIP InfoNCE loss",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ckpt",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Lightning strategy, e.g. 'ddp' for multi-GPU",
    )
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    task_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(task_dir, "data")
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 维度信息: node_dim / edge_dim / protein_emb_dim
    #node_dim = atom_featurizer.dim + 3  # 多了 reaction-center 三维 one-hot
    node_dim = 179
    #edge_dim = bond_featurizer.dim
    edge_dim = 14
    # protein_emb_dim: 来自 ESM2 per-residue embedding 的维度，假设与 BindDB 一致 (1280)
    protein_emb_dim = 1280

    model = EnzymeReactionCLIPModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        protein_emb_dim=protein_emb_dim,
        hidden_dim=args.hidden_dim,
        mpnn_layers=args.mpnn_layers,
        dropout=0.2,
        side=args.side,
    )

    lightning_module = PlantRACERCLIPLightningModule(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
    )

    datamodule = PlantRACERDataModule(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=0,
        save_last=True,
    )

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
    else:
        accelerator = "cpu"
        devices = 1

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=args.strategy,
        callbacks=[checkpoint_callback],
        default_root_dir=save_dir,
    )
    trainer.fit(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()

