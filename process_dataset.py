#!/usr/bin/env python3

import os
import argparse
import pickle
import hashlib
import re
from typing import Iterable, List, Optional, Tuple, Dict, Set

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from torch_geometric.data import Data


class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            self.features_mapping[k] = dict(
                zip(s, range(self.dim, len(s) + self.dim))
            )
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,), dtype=np.float32)
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output

    def misc_vector(self):
        vec = np.zeros((self.dim,), dtype=np.float32)
        for name_feature, feature_mapping in self.features_mapping.items():
            if "misc" in feature_mapping:
                vec[feature_mapping["misc"]] = 1.0
        return vec


class AtomFeaturizer(Featurizer):
    def possible_atomic_num_list(self, atom):
        return atom.GetAtomicNum()

    def possible_chirality_list(self, atom):
        return atom.GetChiralTag()

    def possible_degree_list(self, atom):
        return atom.GetTotalDegree()

    def possible_formal_charge_list(self, atom):
        return atom.GetFormalCharge()

    def possible_numH_list(self, atom):
        return atom.GetTotalNumHs()

    def possible_number_radical_e_list(self, atom):
        return atom.GetNumRadicalElectrons()

    def possible_hybridization_list(self, atom):
        return atom.GetHybridization().name.lower()

    def possible_is_aromatic_list(self, atom):
        return atom.GetIsAromatic()

    def possible_is_in_ring_list(self, atom):
        return atom.IsInRing()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1  # extra dim for "no bond"

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def possible_bond_type_list(self, bond):
        return str(bond.GetBondType())

    def possible_bond_stereo_list(self, bond):
        return str(bond.GetStereo())

    def possible_is_conjugated_list(self, bond):
        return bond.GetIsConjugated()


atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
        "possible_chirality_list": [
            "CHI_UNSPECIFIED",
            "CHI_TETRAHEDRAL_CW",
            "CHI_TETRAHEDRAL_CCW",
            "CHI_TRIGONALBIPYRAMIDAL",
            "CHI_OCTAHEDRAL",
            "CHI_SQUAREPLANAR",
            "CHI_OTHER",
        ],
        "possible_degree_list": list(range(0, 11)) + ["misc"],
        "possible_formal_charge_list": list(range(-5, 6)) + ["misc"],
        "possible_numH_list": list(range(0, 9)) + ["misc"],
        "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
        "possible_hybridization_list": ["sp", "sp2", "sp3", "sp3d", "sp3d2", "misc"],
        "possible_is_aromatic_list": [False, True],
        "possible_is_in_ring_list": [False, True],
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
        "possible_bond_stereo_list": [
            "STEREONONE",
            "STEREOZ",
            "STEREOE",
            "STEREOCIS",
            "STEREOTRANS",
            "STEREOANY",
        ],
        "possible_is_conjugated_list": [False, True],
    }
)


_HF_MODEL = None
_HF_TOKENIZER = None
_HF_DEVICE = None
_HF_LOADED_PATH = None


def compute_esm2_embedding(
    seq: str,
    device: torch.device,
    esm_ckpt: str,
) -> np.ndarray:
    """
    使用本地 ESM2 checkpoint 计算单条蛋白序列的 per-residue embedding。

    使用 Hugging Face Transformers 从本地 checkpoint 计算单条蛋白序列的 per-residue embedding。

    假定 esm_ckpt 是一个包含 config/model 权重等的本地目录，兼容 AutoModel / AutoTokenizer。

    返回:
        np.ndarray, 形状 [L, D]，不做 pooling。
    """
    global _HF_MODEL, _HF_TOKENIZER, _HF_DEVICE, _HF_LOADED_PATH

    if _HF_MODEL is None or _HF_LOADED_PATH != esm_ckpt or _HF_DEVICE != str(device):
        _HF_TOKENIZER = AutoTokenizer.from_pretrained(esm_ckpt)
        _HF_MODEL = AutoModel.from_pretrained(esm_ckpt)
        _HF_MODEL.eval()
        _HF_MODEL.to(device)
        _HF_DEVICE = str(device)
        _HF_LOADED_PATH = esm_ckpt

    seq = seq.strip()
    if not seq:
        hidden_size = _HF_MODEL.config.hidden_size
        return np.zeros((0, hidden_size), dtype=np.float32)

    inputs = _HF_TOKENIZER(
        seq,
        return_tensors="pt",
        add_special_tokens=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _HF_MODEL(**inputs)
    # outputs.last_hidden_state: [1, L_tokens, D]
    token_reprs = outputs.last_hidden_state[0]  # [L_tokens, D]

    # 通常第一个/最后一个是特殊 token，这里简单去掉首尾各一个
    if token_reprs.size(0) > 2:
        token_reprs = token_reprs[1:-1]

    return token_reprs.cpu().numpy().astype(np.float32)


def augment_atom_features(
    x: np.ndarray,
    center_atom_indices: Optional[Iterable[int]] = None,
    neighbor_atom_indices: Optional[Iterable[int]] = None,
    second_shell_indices: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """
    在原子级特征上增加三个维度:
    - [1,0,0]: 原子直接参与键的断裂与形成
    - [0,1,0]: 与上述原子直接连接 / 以及“反应周围一圈”的原子
    - [0,0,1]: 其他原子

    当前若未提供反应中心标注，则全部设为 [0,0,1]。
    当提供反应中心时:
    - center_atom_indices: 参与键断裂/形成的原子在分子中的索引集合
    - neighbor_atom_indices: 与上述原子直接连接的原子索引集合
    - second_shell_indices: 额外指定的“反应周围一圈”原子索引集合
    """
    n_atoms = x.shape[0]
    extra = np.zeros((n_atoms, 3), dtype=np.float32)
    if (
        center_atom_indices is None
        and neighbor_atom_indices is None
        and second_shell_indices is None
    ):
        # 默认所有原子视作“其他原子”
        extra[:, 2] = 1.0
    else:
        center_mask = np.zeros((n_atoms,), dtype=bool)
        neighbor_mask = np.zeros((n_atoms,), dtype=bool)

        if center_atom_indices is not None:
            for idx in center_atom_indices:
                if 0 <= idx < n_atoms:
                    center_mask[int(idx)] = True

        if neighbor_atom_indices is not None:
            for idx in neighbor_atom_indices:
                if 0 <= idx < n_atoms:
                    neighbor_mask[int(idx)] = True

        # second_shell_indices 也直接计入第二列（neighbor_mask）
        if second_shell_indices is not None:
            for idx in second_shell_indices:
                if 0 <= idx < n_atoms:
                    neighbor_mask[int(idx)] = True

        # 确保中心原子优先级最高: 若某原子既在中心又在周围一圈，则只标记为中心
        neighbor_mask[center_mask] = False

        other_mask = ~(center_mask | neighbor_mask)

        extra[center_mask, 0] = 1.0
        extra[neighbor_mask, 1] = 1.0
        extra[other_mask, 2] = 1.0

    return np.concatenate([x, extra], axis=-1)


def smiles_to_graph(
    smiles: str,
) -> Optional[Data]:
    """
    Convert SMILES string to graph (x, edge_index, edge_attr).
    Returns None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        #mol = Chem.AddHs(mol)

        misc_atom_vec = atom_featurizer.misc_vector()
        x_list = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # dummy atom (*)
                x_list.append(misc_atom_vec)
            else:
                x_list.append(atom_featurizer.encode(atom))
        x = np.stack(x_list, axis=0).astype(np.float32)
        x = augment_atom_features(x)

        edge_index: List[List[int]] = [[], []]
        edge_attr_list: List[np.ndarray] = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0] += [start, end]
            edge_index[1] += [end, start]
            bond_feat = bond_featurizer.encode(bond)
            edge_attr_list += [bond_feat, bond_feat]

        if len(edge_index[0]) == 0:
            edge_index_arr = np.zeros((2, 0), dtype=np.int64)
            edge_attr_arr = np.zeros((0, bond_featurizer.dim), dtype=np.float32)
        else:
            edge_index_arr = np.asarray(edge_index, dtype=np.int64)
            edge_attr_arr = np.stack(edge_attr_list, axis=0).astype(np.float32)

        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index_arr, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr_arr, dtype=torch.float32),
        )
    except Exception:
        return None


def parse_cano_rxn(cano_rxn_str: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """
    Parse CANO_RXN string: "sub1.sub2>>prod1.prod2", SMILES separated by '.'.
    """
    if pd.isna(cano_rxn_str) or not str(cano_rxn_str).strip():
        return None, None
    s = str(cano_rxn_str).strip()
    if ">>" not in s:
        return None, None
    left, right = s.split(">>", 1)
    subs = [x.strip() for x in left.split(".") if x.strip()]
    prods = [x.strip() for x in right.split(".") if x.strip()]
    if not subs or not prods:
        return None, None
    return subs, prods


def smiles_list_to_graphs(smiles_list: Iterable[str]) -> Optional[List[Data]]:
    graphs: List[Data] = []
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g is None:
            return None
        graphs.append(g)
    return graphs


def smiles_to_graph_with_reaction_center(
    smiles: str,
    reaction_center_map_ids: Optional[Set[int]] = None,
    second_shell_map_ids: Optional[Set[int]] = None,
) -> Optional[Data]:
    """
    与 smiles_to_graph 类似，但会使用 SMILES 中的 atom map id 以及
    reaction_center_map_ids / second_shell_map_ids 信息，将原子划分到三类并调用 augment_atom_features:
        - [1,0,0]: 原子 map id 在 reaction_center_map_ids 中
        - [0,1,0]: 与上述原子直接连接，或 map id 在 second_shell_map_ids 中
        - [0,0,1]: 其他原子
    """
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    #mol = Chem.AddHs(mol)

    misc_atom_vec = atom_featurizer.misc_vector()
    x_list = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # dummy atom (*)
            x_list.append(misc_atom_vec)
        else:
            x_list.append(atom_featurizer.encode(atom))
    x = np.stack(x_list, axis=0).astype(np.float32)

        # 根据 atom map id 确定反应中心原子及其一阶邻居
    center_indices: Set[int] = set()
    second_shell_indices: Set[int] = set()
    if reaction_center_map_ids:
        for atom in mol.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap != 0 and amap in reaction_center_map_ids:
                center_indices.add(atom.GetIdx())

    if second_shell_map_ids:
        for atom in mol.GetAtoms():
            amap = atom.GetAtomMapNum()
            if (
                amap != 0
                and amap in second_shell_map_ids
                and atom.GetIdx() not in center_indices
            ):
                second_shell_indices.add(atom.GetIdx())

    neighbor_indices: Set[int] = set()
    if center_indices:
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            if u in center_indices and v not in center_indices:
                neighbor_indices.add(v)
            if v in center_indices and u not in center_indices:
                neighbor_indices.add(u)

    x = augment_atom_features(
        x,
        center_atom_indices=center_indices,
        neighbor_atom_indices=neighbor_indices,
        second_shell_indices=second_shell_indices,
    )

    edge_index: List[List[int]] = [[], []]
    edge_attr_list: List[np.ndarray] = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0] += [start, end]
        edge_index[1] += [end, start]
        bond_feat = bond_featurizer.encode(bond)
        edge_attr_list += [bond_feat, bond_feat]

    if len(edge_index[0]) == 0:
        edge_index_arr = np.zeros((2, 0), dtype=np.int64)
        edge_attr_arr = np.zeros((0, bond_featurizer.dim), dtype=np.float32)
    else:
        edge_index_arr = np.asarray(edge_index, dtype=np.int64)
        edge_attr_arr = np.stack(edge_attr_list, axis=0).astype(np.float32)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index_arr, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr_arr, dtype=torch.float32),
    )



def smiles_list_to_graphs_with_reaction_center(
    smiles_list: Iterable[str],
    reaction_center_map_ids: Optional[Set[int]] = None,
    second_shell_map_ids: Optional[Set[int]] = None,
) -> Optional[List[Data]]:
    graphs: List[Data] = []
    for smi in smiles_list:
        g = smiles_to_graph_with_reaction_center(
            smi,
            reaction_center_map_ids=reaction_center_map_ids,
            second_shell_map_ids=second_shell_map_ids,
        )
        if g is None:
            return None
        graphs.append(g)
    return graphs




def _parse_reaction_center_field(val: object) -> Set[int]:
    """
    将 reaction_mapped.csv 中 reaction_center 列解析为一组 atom map id (从 1 开始)。
    支持格式示例:
        - "1,2,3"
        - "1 2 3"
        - "1;2;3"
        - "[1, 2, 3]"
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    s = str(val).strip()
    if not s:
        return set()
    # 去掉可能的括号
    s = s.strip("[]()")
    if not s:
        return set()
    tokens = re.split(r"[,\s;]+", s)
    ids: Set[int] = set()
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        try:
            ids.add(int(t))
        except ValueError:
            continue
    return ids


def build_reaction_dict_from_csv(
    reaction_csv: str,
) -> Dict[int, Dict[str, List[Data]]]:
    """
    从 data/reaction_mapped_with_atoms.csv 或类似格式构建:
        { RHEA_ID: {"substrates": [Data...], "products": [Data...]} }
    其中:
        - CANO_RXN_SMILES / atom_map: 反应 SMILES
        - First: 反应中心原子的 atom map id 列表 (从 1 开始)
        - Second: 反应中心周围一圈原子的 atom map id 列表 (从 1 开始)
    """
    if not os.path.exists(reaction_csv):
        raise FileNotFoundError(f"reaction CSV not found: {reaction_csv}")

    df = pd.read_csv(reaction_csv)
    #required_cols = {"RHEA_ID", "CANO_RXN_SMILES", "atom_map"}
    required_cols = {"reaction_id","atom_map"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"reaction CSV missing columns: {missing}")

    use_first = "First" in df.columns
    use_second = "Second" in df.columns

    reaction_dict: Dict[str, Dict[str, List[Data]]] = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="build_reaction_dict"):
        try:
            rhea_id = str(row["reaction_id"])
        except Exception:
            continue

        mapped_rxn = row.get("atom_map")
        rxn_str = None
        if isinstance(mapped_rxn, str) and mapped_rxn.strip():
            rxn_str = mapped_rxn.strip()
        else:
            cano = row.get("CANO_RXN_SMILES")
            if isinstance(cano, str) and cano.strip():
                rxn_str = cano.strip()
        if rxn_str is None:
            continue

        subs_smi, prods_smi = parse_cano_rxn(rxn_str)
        if subs_smi is None or prods_smi is None:
            continue

        center_ids: Set[int] = set()
        if use_first:
            center_ids = _parse_reaction_center_field(row.get("First"))

        second_ids: Set[int] = set()
        if use_second:
            second_ids = _parse_reaction_center_field(row.get("Second"))

        # 如果存在 First/Second，就把 First 当反应中心，Second 当“周围一圈”，
        # 传到带反应中心的构图函数中（在那里会映射到 augment_atom_features 的第二列）
        if center_ids or second_ids:
            substrates = smiles_list_to_graphs_with_reaction_center(
                subs_smi,
                reaction_center_map_ids=center_ids,
                second_shell_map_ids=second_ids,
            )
            products = smiles_list_to_graphs_with_reaction_center(
                prods_smi,
                reaction_center_map_ids=center_ids,
                second_shell_map_ids=second_ids,
            )
        else:
            substrates = smiles_list_to_graphs(subs_smi)
            products = smiles_list_to_graphs(prods_smi)

        if substrates is None or products is None:
            continue

        reaction_dict[rhea_id] = {
            "substrates": substrates,
            "products": products,
        }

    return reaction_dict


def _read_all_uniprot_from_fasta(fasta_path: str) -> Dict[str, str]:
    """
    从 protein.fa 中读取所有蛋白:
        header 的第一个 token 视作 UniprotID，后续多行拼接为序列。
    """
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"protein FASTA not found: {fasta_path}")

    uniprot_to_seq: Dict[str, str] = {}
    current_id: Optional[str] = None
    seq_chunks: List[str] = []

    with open(fasta_path, "r") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None and seq_chunks:
                    uniprot_to_seq[current_id] = "".join(seq_chunks)
                header = line[1:].strip()
                token = header.split()[0]
                current_id = token
                seq_chunks = []
            else:
                if current_id is not None:
                    seq_chunks.append(line.strip())

    if current_id is not None and seq_chunks:
        uniprot_to_seq[current_id] = "".join(seq_chunks)

    return uniprot_to_seq


def build_protein_dict_from_esm2(
    protein_fasta: str,
    esm_ckpt: str,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    基于 protein.fa 中的所有条目，直接调用 ESM2 计算蛋白 embedding:
        { UniprotID: Tensor[Dp] } 的蛋白级 embedding 字典。

    注意:
    - 需要 compute_esm2_embedding 可用（来自 BindDB 任务的工具）。
    - 每条序列得到 per-residue embedding [L, Dp]，然后在长度维度做 mean pooling。
    """
    if compute_esm2_embedding is None:
        raise ImportError(
            "compute_esm2_embedding not available. 请确认 BindDB ESM2 工具已安装，"
            "并且 old/build_protein_emb_dict.py 中的依赖可用。"
        )

    # 直接从 FASTA 读取所有 ID 和序列
    uniprot_to_seq = _read_all_uniprot_from_fasta(protein_fasta)
    if not uniprot_to_seq:
        return {}

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    protein_dict: Dict[str, torch.Tensor] = {}
    for uid, seq in tqdm(sorted(uniprot_to_seq.items()), desc="build_protein_dict_esm2"):
        # compute_esm2_embedding 约定返回 per-residue embedding [L, Dp]
        arr = compute_esm2_embedding(seq, torch_device, esm_ckpt)  # type: ignore
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        mean_emb = arr.mean(axis=0)
        protein_dict[uid] = torch.from_numpy(mean_emb).float()

    return protein_dict


def run(
    training_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Tuple[int, int]:
    """
    主入口（新的保存逻辑）:
    - 从 data/reaction_mapped.csv 构建反应的分子图表示:
        {RHEA_ID: {"substrates": [Data...], "products": [Data...]}} -> reaction.pt
    - 从 data/train_pair_filtered.csv / data/test_pair_filtered.csv、
      data/protein.fa 和 protein_emb.lmdb 构建蛋白 embedding 字典:
        {UniprotID: Tensor[Dp]} -> protein.pt
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = data_dir or os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    reaction_csv = os.path.join(data_dir, "reaction_mapped_with_atoms.csv")
    protein_fasta_default = os.path.join(data_dir, "protein.fa")
    protein_fasta_fixed = os.path.join(data_dir, "protein_fixed.fa")
    protein_fasta = (
        protein_fasta_fixed
        if os.path.isfile(protein_fasta_fixed)
        else protein_fasta_default
    )

    # 1) 反应图表示
    reaction_dict = build_reaction_dict_from_csv(reaction_csv)
    reaction_pt_path = os.path.join(data_dir, "reaction.pt")
    torch.save(reaction_dict, reaction_pt_path)

    # 2) 蛋白 embedding 字典（直接通过 ESM2 计算，不再依赖 LMDB）
    esm_ckpt_path = "/lustre/home/1800011712/GEOM/ckpt/esm2_650m_local"
    protein_dict = build_protein_dict_from_esm2(
        protein_fasta=protein_fasta,
        esm_ckpt=esm_ckpt_path,
    )
    protein_pt_path = os.path.join(data_dir, "protein.pt")
    torch.save(protein_dict, protein_pt_path)

    print(
        f"Done. Saved {len(reaction_dict)} reactions to {reaction_pt_path}, "
        f"{len(protein_dict)} proteins to {protein_pt_path}"
    )
    return len(reaction_dict), len(protein_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSV -> LMDB with substrates, products, protein embedding, label"
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        default=None,
        help="Dir containing train.csv, test.csv ",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Output dir for train.lmdb, test.lmdb",
    )
    args = parser.parse_args()
    run(
        training_dir=args.training_dir,
        data_dir=args.data_dir,
    )

