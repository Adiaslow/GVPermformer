"""
Utilities for handling SMILES strings and converting them to model features.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Dict, List, Tuple, Optional, Union, Any


def smiles_to_graph(smiles: str, max_atoms: int = 150) -> Dict[str, Any]:
    """
    Convert a SMILES string to a graph representation for the GraphVAE model.

    Args:
        smiles: SMILES string to convert
        max_atoms: Maximum number of atoms to consider

    Returns:
        Dictionary with graph features: x, edge_index, edge_attr, global_features, etc.
    """
    # Parse SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Calculate molecular descriptors for global features
    global_features = calculate_global_features(mol)

    # Generate atom (node) features
    x = []
    for atom in mol.GetAtoms():
        atom_features = get_atom_features(atom)
        x.append(atom_features)

    # Pad or truncate to max_atoms
    num_atoms = len(x)
    if num_atoms > max_atoms:
        x = x[:max_atoms]
    elif num_atoms < max_atoms:
        # Pad with zeros
        padding = [np.zeros(len(x[0]))] * (max_atoms - num_atoms)
        x.extend(padding)

    # Generate edge features
    edge_index = []
    edge_attr = []

    # Add bonds (edges)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Skip if atoms are out of range (due to truncation)
        if i >= max_atoms or j >= max_atoms:
            continue

        # Add edge in both directions (undirected graph)
        edge_index.extend([[i, j], [j, i]])

        # Get bond features
        bond_features = get_bond_features(bond)
        # Add edge features in both directions
        edge_attr.extend([bond_features, bond_features])

    # Convert to tensors
    x = torch.tensor(np.array(x), dtype=torch.float)

    if edge_index:  # Check if there are any edges
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    else:
        # Create empty tensors with the correct shape if no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 9), dtype=torch.float)  # 9 bond features

    global_features = torch.tensor(global_features, dtype=torch.float).unsqueeze(0)

    # Create batch index for a single molecule
    batch = torch.zeros(x.size(0), dtype=torch.long)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "global_features": global_features,
        "batch": batch,
        "num_graphs": 1,
    }


def get_atom_features(atom: Chem.Atom) -> List[float]:
    """
    Calculate features for an atom.

    Args:
        atom: RDKit atom object

    Returns:
        List of atom features
    """
    # Initialize feature vector of length 126 (matching the model's expectations)
    features = [0] * 126

    # Set atomic number (one-hot encoding) - positions 0-117
    atomic_num = atom.GetAtomicNum()
    if 1 <= atomic_num <= 118:
        features[atomic_num - 1] = 1

    # Feature index tracker
    idx = 118

    # Atom degree (0-10) - positions 118-128
    degree = min(10, atom.GetDegree())
    features[idx + degree] = 1
    idx += 11

    # Formal charge (-5 to +5) - positions 129-139
    formal_charge = atom.GetFormalCharge()
    features[idx + min(5, max(-5, formal_charge)) + 5] = 1
    idx += 11

    # Hybridization - positions 140-145
    hybridization = atom.GetHybridization()
    hybridization_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    if hybridization in hybridization_types:
        features[idx + hybridization_types.index(hybridization)] = 1
    idx += 6

    # Aromaticity - position 146
    features[idx] = 1 if atom.GetIsAromatic() else 0
    idx += 1

    # Chirality - positions 147-148
    chirality = atom.GetChiralTag()
    if chirality != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
        features[idx] = 1
    idx += 2

    # Number of hydrogens (0-8) - positions 149-157
    num_h = min(8, atom.GetTotalNumHs())
    features[idx + num_h] = 1
    idx += 9

    # In ring - position 158
    features[idx] = 1 if atom.IsInRing() else 0
    idx += 1

    # Note: The total so far is 159 features, but we're truncating to 126
    # to match the model's expectations from the previous implementation
    return features[:126]


def get_bond_features(bond: Chem.Bond) -> List[float]:
    """
    Calculate features for a bond.

    Args:
        bond: RDKit bond object

    Returns:
        List of bond features (length 9)
    """
    # Bond type one-hot encoding
    bond_type = bond.GetBondType()
    bond_type_features = [0] * 4  # Single, Double, Triple, Aromatic

    if bond_type == Chem.rdchem.BondType.SINGLE:
        bond_type_features[0] = 1
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        bond_type_features[1] = 1
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        bond_type_features[2] = 1
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        bond_type_features[3] = 1

    # Conjugation
    is_conjugated = 1 if bond.GetIsConjugated() else 0

    # In ring
    is_in_ring = 1 if bond.IsInRing() else 0

    # Stereo configuration
    stereo = bond.GetStereo()
    stereo_features = [0] * 3  # None, Z/Cis, E/Trans

    if stereo == Chem.rdchem.BondStereo.STEREONONE:
        stereo_features[0] = 1
    elif stereo in [Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOCIS]:
        stereo_features[1] = 1
    elif stereo in [Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOTRANS]:
        stereo_features[2] = 1

    # Combine all features
    features = bond_type_features + [is_conjugated, is_in_ring] + stereo_features

    return features


def calculate_global_features(mol: Chem.Mol) -> List[float]:
    """
    Calculate global molecular features.

    Args:
        mol: RDKit molecule object

    Returns:
        List of global features
    """
    if mol is None:
        return [0.0] * 55  # Return zeros if molecule is invalid

    # Basic molecular descriptors
    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)

    # Count features
    heavy_atom_count = mol.GetNumHeavyAtoms()
    num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
    num_aromatic_rings = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
    num_aliphatic_rings = num_rings - num_aromatic_rings

    # Atom counts
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    num_h_acceptors = Lipinski.NumHAcceptors(mol)
    num_h_donors = Lipinski.NumHDonors(mol)
    num_rot_bonds = Lipinski.NumRotatableBonds(mol)

    # Heteroatom counts
    num_hetero_atoms = heavy_atom_count - len(
        [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
    )
    num_heterocycles = sum(
        1
        for ring in mol.GetSSSR()
        if any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in ring)
    )

    # Element counts
    atom_counter = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_counter[symbol] = atom_counter.get(symbol, 0) + 1

    num_c = atom_counter.get("C", 0)
    num_n = atom_counter.get("N", 0)
    num_o = atom_counter.get("O", 0)
    num_s = atom_counter.get("S", 0)
    num_p = atom_counter.get("P", 0)
    num_f = atom_counter.get("F", 0)
    num_cl = atom_counter.get("Cl", 0)
    num_br = atom_counter.get("Br", 0)
    num_i = atom_counter.get("I", 0)

    # Functional group counts
    amide_pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
    num_amide = len(mol.GetSubstructMatches(amide_pattern)) if amide_pattern else 0

    amine_pattern = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
    num_amine = len(mol.GetSubstructMatches(amine_pattern)) if amine_pattern else 0

    acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    num_carboxylic_acid = (
        len(mol.GetSubstructMatches(acid_pattern)) if acid_pattern else 0
    )

    ester_pattern = Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]")
    num_ester = len(mol.GetSubstructMatches(ester_pattern)) if ester_pattern else 0

    ether_pattern = Chem.MolFromSmarts("[OD2]([#6])[#6]")
    num_ether = len(mol.GetSubstructMatches(ether_pattern)) if ether_pattern else 0

    alcohol_pattern = Chem.MolFromSmarts("[#6][OX2H]")
    num_hydroxyl = (
        len(mol.GetSubstructMatches(alcohol_pattern)) if alcohol_pattern else 0
    )

    sulfonamide_pattern = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])([NX3])([#6])")
    num_sulfonamide = (
        len(mol.GetSubstructMatches(sulfonamide_pattern)) if sulfonamide_pattern else 0
    )

    sulfone_pattern = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])([#6])[#6]")
    num_sulfone = (
        len(mol.GetSubstructMatches(sulfone_pattern)) if sulfone_pattern else 0
    )

    urea_pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[NX3]")
    num_urea = len(mol.GetSubstructMatches(urea_pattern)) if urea_pattern else 0

    # Ratio features
    fraction_csp3 = Chem.rdMolDescriptors.CalcFractionCSP3(mol)
    rings_per_heavy_atom = num_rings / max(1, heavy_atom_count)
    rot_bonds_per_heavy_atom = num_rot_bonds / max(1, heavy_atom_count)

    is_cyclic = 1.0 if num_rings > 0 else 0.0

    # Count fraction of atoms in rings
    atoms_in_rings = sum(1 for atom in mol.GetAtoms() if atom.IsInRing())
    fraction_in_ring = atoms_in_rings / max(1, num_atoms)

    # PAMPA and property-related predictions
    # For prediction only, we set these to 0 initially
    pampa = 0.0
    pampa_clipped = 0.0
    pampa_exp = 0.0

    # Lipinski violations
    lipinski_violations = (
        (mol_weight > 500) + (logp > 5) + (num_h_donors > 5) + (num_h_acceptors > 10)
    )

    # Quality metrics
    qed = Chem.QED.qed(mol)

    # Normalize to comparable ranges and combine
    features = [
        # Molecular descriptors
        np.log1p(Descriptors.BertzCT(mol)) / 20.0,  # BertzCT
        num_amide / 10.0,  # Count_Amide
        num_amine / 10.0,  # Count_Amine
        num_carboxylic_acid / 5.0,  # Count_Carboxylic_Acid
        num_ester / 5.0,  # Count_Ester
        num_ether / 10.0,  # Count_Ether
        num_hydroxyl / 10.0,  # Count_Hydroxyl
        num_sulfonamide / 5.0,  # Count_Sulfonamide
        num_sulfone / 5.0,  # Count_Sulfone
        num_urea / 5.0,  # Count_Urea
        fraction_csp3,  # FractionCSP3
        fraction_in_ring,  # Fraction_InRing
        1.0 if num_amide > 0 else 0.0,  # Has_Amide
        1.0 if num_amine > 0 else 0.0,  # Has_Amine
        1.0 if num_carboxylic_acid > 0 else 0.0,  # Has_Carboxylic_Acid
        1.0 if num_ester > 0 else 0.0,  # Has_Ester
        1.0 if num_ether > 0 else 0.0,  # Has_Ether
        1.0 if num_hydroxyl > 0 else 0.0,  # Has_Hydroxyl
        1.0 if num_sulfonamide > 0 else 0.0,  # Has_Sulfonamide
        1.0 if num_sulfone > 0 else 0.0,  # Has_Sulfone
        1.0 if num_urea > 0 else 0.0,  # Has_Urea
        heavy_atom_count / 100.0,  # HeavyAtomCount
        is_cyclic,  # IsCyclic
        lipinski_violations / 4.0,  # Lipinski_Violations
        logp / 10.0,  # LogP
        logp / mol_weight if mol_weight > 0 else 0,  # LogP_per_MW
        mol_weight / 500.0,  # MolWt
        num_aliphatic_rings / 5.0,  # NumAliphaticRings
        num_aromatic_rings / 5.0,  # NumAromaticRings
        num_atoms / 100.0,  # NumAtoms
        num_bonds / 100.0,  # NumBonds
        num_br / 5.0,  # NumBr
        num_c / 50.0,  # NumC
        num_cl / 5.0,  # NumCl
        num_f / 10.0,  # NumF
        num_h_acceptors / 10.0,  # NumHAcceptors
        num_h_donors / 10.0,  # NumHDonors
        num_heterocycles / 5.0,  # NumHeterocycles
        num_i / 5.0,  # NumI
        num_n / 20.0,  # NumN
        num_o / 20.0,  # NumO
        num_p / 5.0,  # NumP
        0.0,  # NumPeptideBonds (calculated separately for peptides)
        num_rings / 10.0,  # NumRings
        num_rot_bonds / 20.0,  # NumRotatableBonds
        num_s / 10.0,  # NumS
        pampa,  # PAMPA
        pampa_clipped,  # PAMPA_Clipped
        pampa_exp,  # PAMPA_Exp
        0.0,  # Peptide_Length (calculated separately for peptides)
        qed,  # QED
        rings_per_heavy_atom,  # Rings_per_HeavyAtom
        rot_bonds_per_heavy_atom,  # RotBonds_per_HeavyAtom
        tpsa / 200.0,  # TPSA
        tpsa / mol_weight if mol_weight > 0 else 0,  # TPSA_per_MW
    ]

    assert len(features) == 55, f"Expected 55 global features, got {len(features)}"
    return features


def smiles_to_model_input(smiles: str, max_atoms: int = 150) -> Dict[str, torch.Tensor]:
    """
    Convert a SMILES string to model input format.

    Args:
        smiles: SMILES string to convert
        max_atoms: Maximum number of atoms to consider

    Returns:
        Dictionary with model input tensors
    """
    try:
        # First convert SMILES to graph
        graph_data = smiles_to_graph(smiles, max_atoms)

        # Get features in PyTorch Geometric compatible format
        # Add an additional batch dimension for the model
        return {
            "x": graph_data["x"].unsqueeze(0),
            "edge_index": graph_data["edge_index"].unsqueeze(0),
            "edge_attr": graph_data["edge_attr"].unsqueeze(0),
            "batch": graph_data["batch"].unsqueeze(0),
            "global_features": graph_data["global_features"].unsqueeze(0),
            "num_graphs": 1,
        }
    except Exception as e:
        raise ValueError(f"Error converting SMILES to model input: {str(e)}")
