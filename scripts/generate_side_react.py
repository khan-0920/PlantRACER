#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdkit import Chem
from rdkit.Chem import rdchem
from collections import defaultdict


# ============================================================
# User input: atom-mapped reaction SMILES
# Can be replaced with your own reaction
# ============================================================
rxn_smi = (
    "[CH3:1][C:2](=[O:3])[S:52][CH2:51][CH2:50][NH:49][C:47](=[O:48])[CH2:46][CH2:45][NH:44][C:42](=[O:43])[C@H:40]([OH:41])[C:6]([CH3:5])([CH3:7])[CH2:8][O:9][P:10](=[O:11])([O-:12])[O:13][P:14](=[O:15])([O-:16])[O:17][CH2:18][C@H:19]1[O:20][C@@H:21]([N:22]2[CH:23]=[N:24][C:25]3=[C:26]2[N:27]=[CH:28][N:29]=[C:30]3[NH2:31])[C@H:32]([OH:33])[C@@H:34]1[O:35][P:36](=[O:37])([O-:38])[O-:39].[O:54]=[C:53]([O-:4])/[CH:55]=[CH:56]/[CH2:57][C:58](=[O:59])[O-:60]>>"
    "[CH3:1][C:2](=[O:3])[O-:4]."
    "[CH3:5][C:6]([CH3:7])([CH2:8][O:9][P:10](=[O:11])([O-:12])[O:13][P:14](=[O:15])([O-:16])[O:17][CH2:18][C@H:19]1[O:20][C@@H:21]([N:22]2[CH:23]=[N:24][C:25]3=[C:26]2[N:27]=[CH:28][N:29]=[C:30]3[NH2:31])[C@H:32]([OH:33])[C@@H:34]1[O:35][P:36](=[O:37])([O-:38])[O-:39])[C@@H:40]([OH:41])[C:42](=[O:43])[NH:44][CH2:45][CH2:46][C:47](=[O:48])[NH:49][CH2:50][CH2:51][S:52][C:53](=[O:54])/[CH:55]=[CH:56]/[CH2:57][C:58](=[O:59])[O-:60]"
)

rxn_smi = "[*:1][NH:2][C@@H:3]([CH2:4][CH2:5][C:6](=[O:7])[NH2:23])[C:20]([*:21])=[O:22].[NH2:8][CH2:9][C@H:10]([OH:11])[c:12]1[cH:13][cH:14][c:15]([OH:16])[c:17]([OH:18])[cH:19]1>>[*:1][NH:2][C@@H:3]([CH2:4][CH2:5][C:6](=[O:7])[NH:8][CH2:9][C@H:10]([OH:11])[c:12]1[cH:13][cH:14][c:15]([OH:16])[c:17]([OH:18])[cH:19]1)[C:20]([*:21])=[O:22].[NH3:23]"


# ============================================================
# 基础函数
# ============================================================

def mol_from_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"SMILES 解析失败: {smi}")
    return mol


def split_reaction_smiles(rxn_smi: str):
    left, right = rxn_smi.split(">>")
    reactant_smis = [x for x in left.split(".") if x]
    product_smis = [x for x in right.split(".") if x]
    reactants = [mol_from_smiles(s) for s in reactant_smis]
    products = [mol_from_smiles(s) for s in product_smis]
    return reactants, products, reactant_smis, product_smis


def get_map_to_atominfo(mols):
    """
    Return:
        map_to_atom: map_num -> atom object
        map_to_mol_idx: map_num -> molecule index
        map_to_atom_idx: map_num -> atom idx in that molecule
    """
    map_to_atom = {}
    map_to_mol_idx = {}
    map_to_atom_idx = {}

    for mi, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap > 0:
                map_to_atom[amap] = atom
                map_to_mol_idx[amap] = mi
                map_to_atom_idx[amap] = atom.GetIdx()
    return map_to_atom, map_to_mol_idx, map_to_atom_idx


# ============================================================
# 键级转换
# ============================================================

def bondtype_to_order(bt):
    if bt == rdchem.BondType.SINGLE:
        return 1.0
    elif bt == rdchem.BondType.DOUBLE:
        return 2.0
    elif bt == rdchem.BondType.TRIPLE:
        return 3.0
    elif bt == rdchem.BondType.AROMATIC:
        return 1.5
    else:
        return None


def order_to_bondtype(order):
    if order == 1.0:
        return rdchem.BondType.SINGLE
    elif order == 2.0:
        return rdchem.BondType.DOUBLE
    elif order == 3.0:
        return rdchem.BondType.TRIPLE
    elif order == 1.5:
        return rdchem.BondType.AROMATIC
    else:
        raise ValueError(f"不支持的键级: {order}")


def get_bond_order_dict(mols):
    """
    返回:
        bond_dict[(min_map, max_map)] = bond_order (float)
    """
    bond_dict = {}
    for mol in mols:
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetAtomMapNum()
            a2 = bond.GetEndAtom().GetAtomMapNum()
            if a1 > 0 and a2 > 0:
                key = tuple(sorted((a1, a2)))
                bond_dict[key] = bondtype_to_order(bond.GetBondType())
    return bond_dict


# ============================================================
# 统一提取键变化
# ============================================================

def extract_bond_changes(reactants, products):
    """
    Extract all bond changes:
        (a,b,before_order,after_order)

    Examples:
        Single bond broken: (a,b,1.0,0.0)
        Single bond formed: (a,b,0.0,1.0)
        Double bond changed to single: (a,b,2.0,1.0)
        Double bond changed to single: (a,b,2.0,1.0)
    """
    rbonds = get_bond_order_dict(reactants)
    pbonds = get_bond_order_dict(products)

    all_keys = sorted(set(rbonds.keys()) | set(pbonds.keys()))
    bond_changes = []

    for key in all_keys:
        before_order = rbonds.get(key, 0.0)
        after_order = pbonds.get(key, 0.0)
        if before_order != after_order:
            a, b = key
            bond_changes.append((a, b, before_order, after_order))

    return bond_changes, rbonds, pbonds


def get_reactant_present_changes(bond_changes):
    """
    before>0 bond change, i.e. the bond change exists in the reactants
    Includes:
        1->0, 2->1, 2->0, 1.5->1 ...
    """
    return [x for x in bond_changes if x[2] > 0.0]


def get_product_present_changes(bond_changes):
    """
    after>0 bond change, i.e. the bond change exists in the products
    包括:
        0->1, 1->2, 0->2, 1->1.5 ...
    """
    return [x for x in bond_changes if x[3] > 0.0]


# ============================================================
# 官能团定义
# ============================================================

FG_SMARTS = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "carboxylate": "[CX3](=O)[O-]",
    "amide": "[NX3][CX3](=O)[#6]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "alcohol": "[OX2H][#6]",
    "water_oxygen": "[OH2]",
    "thiolester": "[CX3](=O)[SX2]",
    "thioester_sulfur": "[SX2][CX3](=O)",
    "carbonyl_oxygen": "[OX1]=[CX3]",
    "aryl": "a",
    "aryl_carbon": "[c]",
    "aryl_CH": "[cH]",
    "sp3_carbon": "[CX4]",
}


def get_atom_functional_groups(mol):
    atom_fg = defaultdict(set)

    for label, smarts in FG_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        for match in matches:
            for idx in match:
                atom_fg[idx].add(label)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if len(atom_fg[idx]) == 0:
            atom_fg[idx].add(f"elem_{atom.GetAtomicNum()}")
            atom_fg[idx].add("aromatic" if atom.GetIsAromatic() else "non_aromatic")

    return atom_fg


def same_functional_group(atom_fg1, atom_fg2):
    return len(set(atom_fg1) & set(atom_fg2)) > 0


# ============================================================
# Relaxed atom signature
# ============================================================

def atom_signature(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    neigh_nums = sorted([n.GetAtomicNum() for n in atom.GetNeighbors()])
    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic(),
        str(atom.GetHybridization()),
        tuple(neigh_nums),
    )


# ============================================================
# Explicit H change extraction
# ============================================================

def get_explicit_h_map(mols):
    amap_to_h = {}
    for mol in mols:
        for atom in mol.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap > 0:
                amap_to_h[amap] = atom.GetNumExplicitHs()
    return amap_to_h


def extract_atom_h_changes(reactants, products):
    """
    Extract explicit H changes in the main reaction mapped atoms
    Return:
        h_changes[map] = {"from": h_r, "to": h_p, "delta": h_p - h_r}
    """
    rh = get_explicit_h_map(reactants)
    ph = get_explicit_h_map(products)

    all_maps = sorted(set(rh.keys()) & set(ph.keys()))
    h_changes = {}

    for amap in all_maps:
        hr = rh[amap]
        hp = ph[amap]
        if hr != hp:
            h_changes[amap] = {
                "from": hr,
                "to": hp,
                "delta": hp - hr
            }
    return h_changes


# ============================================================
# Stage A: Search for bond change replacement candidates
# When replacing, the same bond change before->after must occur
# ============================================================

def find_bond_change_replacements(
    reactants,
    map_to_mol_idx,
    map_to_atom_idx,
    atom_fg_by_mol,
    bond_change,
):
    """
    For a bond change (a,b,before_order,after_order),
    find candidates (u,v) in the same reactant molecule, requiring:
      - u/v match the atom environment of a/b
      - functional group matching
      - candidate bond current bond level == before_order
      - same bond change before_order -> after_order after replacement
    """
    a, b, before_order, after_order = bond_change

    # Only search for replacements in the reactants that originally existed
    if before_order <= 0.0:
        return []

    orig_mol_idx = map_to_mol_idx[a]
    if map_to_mol_idx[b] != orig_mol_idx:
        return []

    mol = reactants[orig_mol_idx]
    atom_fg = atom_fg_by_mol[orig_mol_idx]

    idx_a = map_to_atom_idx[a]
    idx_b = map_to_atom_idx[b]

    sig_a = atom_signature(mol, idx_a)
    sig_b = atom_signature(mol, idx_b)

    ref_bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
    if ref_bond is None:
        return []

    ref_before_order = bondtype_to_order(ref_bond.GetBondType())
    if ref_before_order != before_order:
        return []

    fg_a = atom_fg[idx_a]
    fg_b = atom_fg[idx_b]

    candidates = []

    for bond in mol.GetBonds():
        u_idx = bond.GetBeginAtomIdx()
        v_idx = bond.GetEndAtomIdx()

        u_map = mol.GetAtomWithIdx(u_idx).GetAtomMapNum()
        v_map = mol.GetAtomWithIdx(v_idx).GetAtomMapNum()

        if u_map <= 0 or v_map <= 0:
            continue
        if set((u_map, v_map)) == set((a, b)):
            continue

        cand_before_order = bondtype_to_order(bond.GetBondType())
        if cand_before_order != before_order:
            continue

        sig_u = atom_signature(mol, u_idx)
        sig_v = atom_signature(mol, v_idx)

        fg_u = atom_fg[u_idx]
        fg_v = atom_fg[v_idx]

        # Direct matching
        if sig_u == sig_a and sig_v == sig_b:
            if same_functional_group(fg_u, fg_a) and same_functional_group(fg_v, fg_b):
                candidates.append((u_map, v_map, before_order, after_order))

        # Reverse matching, rearrange to a->u, b->v
        elif sig_u == sig_b and sig_v == sig_a:
            if same_functional_group(fg_u, fg_b) and same_functional_group(fg_v, fg_a):
                candidates.append((v_map, u_map, before_order, after_order))

    candidates = sorted(set(candidates))
    return candidates


# ============================================================
# Stage B: Search for atom replacements in formed-like changes that are not "non reactant-present-change endpoint"
# ============================================================

def find_atom_replacements(
    reactants,
    map_to_mol_idx,
    map_to_atom_idx,
    atom_fg_by_mol,
    target_map,
    forbidden_maps=None,
):
    if forbidden_maps is None:
        forbidden_maps = set()

    orig_mol_idx = map_to_mol_idx[target_map]
    mol = reactants[orig_mol_idx]
    atom_fg = atom_fg_by_mol[orig_mol_idx]

    target_idx = map_to_atom_idx[target_map]
    target_sig = atom_signature(mol, target_idx)
    target_fg = atom_fg[target_idx]

    candidates = []
    for atom in mol.GetAtoms():
        amap = atom.GetAtomMapNum()
        idx = atom.GetIdx()

        if amap <= 0:
            continue
        if amap == target_map:
            continue
        if amap in forbidden_maps:
            continue

        sig = atom_signature(mol, idx)
        fg = atom_fg[idx]

        if sig == target_sig and same_functional_group(fg, target_fg):
            candidates.append(amap)

    candidates = sorted(set(candidates))
    return candidates


# ============================================================
# Atom property change remapping (explicit H)
# ============================================================

def remap_atom_property_changes(base_h_changes, replace_map):
    mapped = {}
    for amap, info in base_h_changes.items():
        new_amap = replace_map.get(amap, amap)
        mapped[new_amap] = dict(info)
    return mapped


# ============================================================
# Global bond change replacement logic
# ============================================================

def apply_bond_change_replacement_to_global_changes(all_bond_changes, orig_change, new_change):
    """
    Replace one bond change operation:
        orig_change = (a,b,before,after)
        new_change  = (u,v,before,after)

    Other bond changes that are not replaced are still all retained.
    Also replace all other bond change endpoints that reference a/b with u/v.
    """
    a, b, before_order, after_order = orig_change
    u, v, nbefore, nafter = new_change

    if before_order != nbefore or after_order != nafter:
        raise ValueError("Before and after must be the same bond change type")

    replace_map = {a: u, b: v}

    new_all_changes = []
    for x, y, bo, ao in all_bond_changes:
        if (set((x, y)) == set((a, b))) and bo == before_order and ao == after_order:
            new_all_changes.append((u, v, bo, ao))
        else:
            nx = replace_map.get(x, x)
            ny = replace_map.get(y, y)
            new_all_changes.append((nx, ny, bo, ao))

    return new_all_changes, replace_map


def apply_formed_atom_replacement_to_global_changes(all_bond_changes, orig_atom, new_atom):
    """
    When replacing the atom orig_atom -> new_atom in a bond_change:
      - All bond change operations are still retained
      - Only replace all references to orig_atom with new_atom
    """
    replace_map = {orig_atom: new_atom}

    new_all_changes = []
    for x, y, bo, ao in all_bond_changes:
        nx = replace_map.get(x, x)
        ny = replace_map.get(y, y)
        new_all_changes.append((nx, ny, bo, ao))

    return new_all_changes, replace_map


# ============================================================
# Combine molecules / sanitize
# ============================================================

def combine_molecules_with_map_tracking(mols):
    combo = None
    map_to_combo_idx = {}
    offset = 0

    for mol in mols:
        if combo is None:
            combo = Chem.Mol(mol)
            for atom in mol.GetAtoms():
                amap = atom.GetAtomMapNum()
                if amap > 0:
                    map_to_combo_idx[amap] = atom.GetIdx()
            offset = mol.GetNumAtoms()
        else:
            new_combo = Chem.CombineMols(combo, mol)
            for atom in mol.GetAtoms():
                amap = atom.GetAtomMapNum()
                if amap > 0:
                    map_to_combo_idx[amap] = atom.GetIdx() + offset
            offset += mol.GetNumAtoms()
            combo = new_combo

    return combo, map_to_combo_idx


def sanitize_mol_safe(mol):
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


# ============================================================
# Apply unified bond change set + explicit H change
# ============================================================

def apply_bond_changes_to_reactants(
    reactants,
    bond_changes,
    mapped_h_changes=None,
    verbose=False,
):
    """
    Apply the unified bond change set to the reactants:
        (a,b,before_order,after_order)

    Supports:
      - Break bond: before>0, after=0
      - Form bond: before=0, after>0
      - Change bond level: before>0, after>0, before!=after
    """
    if mapped_h_changes is None:
        mapped_h_changes = {}

    combo, map_to_combo_idx = combine_molecules_with_map_tracking(reactants)
    rw = Chem.RWMol(combo)

    for a, b, before_order, after_order in bond_changes:
        if a not in map_to_combo_idx or b not in map_to_combo_idx:
            if verbose:
                print(f"[DEBUG] missing atom in bond change: {(a,b,before_order,after_order)}")
            return None

        ia = map_to_combo_idx[a]
        ib = map_to_combo_idx[b]
        bond = rw.GetBondBetweenAtoms(ia, ib)

        current_order = 0.0 if bond is None else bondtype_to_order(bond.GetBondType())

        # 严格检查：当前键级应等于 before_order
        if current_order != before_order:
            if verbose:
                print(f"[DEBUG] current order mismatch for {(a,b)}: current={current_order}, expected={before_order}")
            return None

        # 应用 before -> after
        if before_order > 0.0 and after_order == 0.0:
            rw.RemoveBond(ia, ib)

        elif before_order == 0.0 and after_order > 0.0:
            rw.AddBond(ia, ib, order_to_bondtype(after_order))

        elif before_order > 0.0 and after_order > 0.0:
            bond = rw.GetBondBetweenAtoms(ia, ib)
            if bond is None:
                if verbose:
                    print(f"[DEBUG] bond missing for order change: {(a,b)}")
                return None
            bond.SetBondType(order_to_bondtype(after_order))

        else:
            # 0->0 无意义
            pass

    # 显式H修改
    for amap, info in mapped_h_changes.items():
        if amap not in map_to_combo_idx:
            continue
        idx = map_to_combo_idx[amap]
        atom = rw.GetAtomWithIdx(idx)
        atom.SetNumExplicitHs(int(info["to"]))
        atom.SetNoImplicit(True)

    mol = rw.GetMol()

    if not sanitize_mol_safe(mol):
        if verbose:
            print("[DEBUG] sanitize failed")
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                print("[DEBUG] sanitize error:", e)
        return None

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    frag_smis = [Chem.MolToSmiles(f, canonical=True) for f in frags]
    frag_smis = sorted(frag_smis)

    return mol, frag_smis


# ============================================================
# 去重 key
# ============================================================

def canonical_bond_change_reaction_key(bond_changes, mapped_h_changes=None):
    k1 = tuple(sorted(
        (min(a, b), max(a, b), bo, ao) for a, b, bo, ao in bond_changes
    ))
    if mapped_h_changes is None:
        k2 = ()
    else:
        k2 = tuple(sorted((amap, info["to"]) for amap, info in mapped_h_changes.items()))
    return (k1, k2)


# ============================================================
# 阶段A：全局键变化替换副反应生成
# ============================================================

def generate_global_bond_change_replacement_side_reactions(
    reactants,
    bond_changes,
    bond_change_replacements,
    base_h_changes,
):
    all_side_reactions = []
    seen = set()

    for orig_change, cand_list in bond_change_replacements.items():
        for new_change in cand_list:
            new_all_changes, replace_map = apply_bond_change_replacement_to_global_changes(
                all_bond_changes=bond_changes,
                orig_change=orig_change,
                new_change=new_change,
            )

            mapped_h_changes = remap_atom_property_changes(base_h_changes, replace_map)

            key = canonical_bond_change_reaction_key(new_all_changes, mapped_h_changes)
            if key in seen:
                continue

            applied = apply_bond_changes_to_reactants(
                reactants=reactants,
                bond_changes=new_all_changes,
                mapped_h_changes=mapped_h_changes,
                verbose=False,
            )
            if applied is None:
                continue

            _, prod_smis = applied
            seen.add(key)

            all_side_reactions.append({
                "type": "stageA_global_bond_change_replacement",
                "replaced_operation": {
                    "from": orig_change,
                    "to": new_change,
                },
                "replace_map": replace_map,
                "bond_changes": new_all_changes,
                "mapped_h_changes": mapped_h_changes,
                "products": prod_smis,
            })

    return all_side_reactions


# ============================================================
# Stage B: Global formed-like atom replacement side reaction generation
# ============================================================

def generate_global_formed_replacement_side_reactions(
    reactants,
    bond_changes,
    atom_replacements,
    base_h_changes,
):
    all_side_reactions = []
    seen = set()

    for orig_atom, cand_list in atom_replacements.items():
        for new_atom in cand_list:
            new_all_changes, replace_map = apply_formed_atom_replacement_to_global_changes(
                all_bond_changes=bond_changes,
                orig_atom=orig_atom,
                new_atom=new_atom,
            )

            mapped_h_changes = remap_atom_property_changes(base_h_changes, replace_map)

            key = canonical_bond_change_reaction_key(new_all_changes, mapped_h_changes)
            if key in seen:
                continue

            applied = apply_bond_changes_to_reactants(
                reactants=reactants,
                bond_changes=new_all_changes,
                mapped_h_changes=mapped_h_changes,
                verbose=False,
            )
            if applied is None:
                continue

            _, prod_smis = applied
            seen.add(key)

            all_side_reactions.append({
                "type": "stageB_global_formed_replacement",
                "replaced_atom": {
                    "from": orig_atom,
                    "to": new_atom,
                },
                "replace_map": replace_map,
                "bond_changes": new_all_changes,
                "mapped_h_changes": mapped_h_changes,
                "products": prod_smis,
            })

    return all_side_reactions


# ============================================================
# Main logic
# ============================================================

def generate_side_reactions(rxn_smi):
    reactants, products, reactant_smis, product_smis = split_reaction_smiles(rxn_smi)

    map_to_atom_r, map_to_mol_idx_r, map_to_atom_idx_r = get_map_to_atominfo(reactants)
    map_to_atom_p, map_to_mol_idx_p, map_to_atom_idx_p = get_map_to_atominfo(products)

    bond_changes, rbonds, pbonds = extract_bond_changes(reactants, products)
    reactant_present_changes = get_reactant_present_changes(bond_changes)
    product_present_changes = get_product_present_changes(bond_changes)

    atom_fg_by_mol = [get_atom_functional_groups(m) for m in reactants]
    base_h_changes = extract_atom_h_changes(reactants, products)

    # Stage A candidates: bond change replacement
    bond_change_replacements = {}
    for bc in reactant_present_changes:
        bond_change_replacements[bc] = find_bond_change_replacements(
            reactants=reactants,
            map_to_mol_idx=map_to_mol_idx_r,
            map_to_atom_idx=map_to_atom_idx_r,
            atom_fg_by_mol=atom_fg_by_mol,
            bond_change=bc,
        )

    # For stage B exclusion: all reactant-present-change endpoints
    reactant_change_endpoint_set = set()
    for a, b, bo, ao in reactant_present_changes:
        reactant_change_endpoint_set.add(a)
        reactant_change_endpoint_set.add(b)

    # Stage B candidates: atom replacements in formed-like changes that are not "non reactant-present-change endpoint"
    atom_replacements = {}
    for x, y, bo, ao in product_present_changes:
        for amap in (x, y):
            if amap not in reactant_change_endpoint_set:
                atom_replacements[amap] = find_atom_replacements(
                    reactants=reactants,
                    map_to_mol_idx=map_to_mol_idx_r,
                    map_to_atom_idx=map_to_atom_idx_r,
                    atom_fg_by_mol=atom_fg_by_mol,
                    target_map=amap,
                    forbidden_maps=reactant_change_endpoint_set,
                )

    all_side_reactions = []

    # Stage A: Global bond change replacement
    stageA_side_reactions = generate_global_bond_change_replacement_side_reactions(
        reactants=reactants,
        bond_changes=bond_changes,
        bond_change_replacements=bond_change_replacements,
        base_h_changes=base_h_changes,
    )
    all_side_reactions.extend(stageA_side_reactions)

    # Stage B: Global formed-like atom replacement
    stageB_side_reactions = generate_global_formed_replacement_side_reactions(
        reactants=reactants,
        bond_changes=bond_changes,
        atom_replacements=atom_replacements,
        base_h_changes=base_h_changes,
    )
    all_side_reactions.extend(stageB_side_reactions)

    return {
        "reactant_smis": reactant_smis,
        "product_smis": product_smis,
        "bond_changes": bond_changes,
        "base_h_changes": base_h_changes,
        "bond_change_replacements": bond_change_replacements,
        "atom_replacements": atom_replacements,
        "side_reactions": all_side_reactions,
    }


# ============================================================
# Print
# ============================================================

def format_reaction_smiles_from_products(reactant_smis, product_smis):
    left = ".".join(sorted(reactant_smis))
    right = ".".join(sorted(product_smis))
    return left + ">>" + right


def main():
    result = generate_side_reactions(rxn_smi)

    print("=" * 120)
    print("Original reaction")
    print(rxn_smi)
    print("=" * 120)

    print("\n[1] Bond changes")
    for bc in result["bond_changes"]:
        print(bc)

    print("\n[2] Explicit H changes in the main reaction")
    print(result["base_h_changes"])

    print("\n[3] Stage A candidates: bond change replacement")
    for k, v in result["bond_change_replacements"].items():
        print(f"{k} -> {v}")

    print("\n[4] Stage B candidates: atom replacements in formed-like changes that are not \"non reactant-present-change endpoint\"")
    for k, v in result["atom_replacements"].items():
        print(f"{k} -> {v}")

        print("\n[5] Side reaction candidates")
    if len(result["side_reactions"]) == 0:
        print("No side reaction candidates found.")
        return

    for i, sr in enumerate(result["side_reactions"], 1):
        print("-" * 120)
        print(f"Candidate #{i}")
        print("type             :", sr["type"])

        if "replaced_operation" in sr:
            print("replaced_op      :", sr["replaced_operation"])
        if "replaced_atom" in sr:
            print("replaced_atom    :", sr["replaced_atom"])

        print("replace_map       :", sr["replace_map"])
        print("bond_changes      :", sr["bond_changes"])
        print("mapped_h_changes  :", sr["mapped_h_changes"])
        print("products          :", sr["products"])

        side_rxn_smi = format_reaction_smiles_from_products(
            result["reactant_smis"],
            sr["products"]
        )
        print("side_rxn_smi      :", side_rxn_smi)


if __name__ == "__main__":
    main()