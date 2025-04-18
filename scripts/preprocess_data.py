#!/usr/bin/env python
# scripts/preprocess_data.py

"""
Script to preprocess molecular data and generate visualizations.
Includes molecular graph featurization and cyclic peptide specific features.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.Descriptors import (
    ExactMolWt,
    MolLogP,
    NumHAcceptors,
    NumHDonors,
    NumRotatableBonds,
    TPSA,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from tqdm import tqdm

from src.preprocessing import MolecularGraphPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Atom feature dimensions
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

# Bond feature dimensions
BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "stereo": list(range(6)),
    "is_conjugated": [True, False],
}


def one_hot_encode(value: int, choices: List) -> List[int]:
    """One-hot encode a value given a list of choices."""
    encoding = [0] * len(choices)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        pass
    return encoding


def get_atom_features(atom: Chem.Atom) -> List[int]:
    """Get atom features as a one-hot encoded list."""
    features = []
    features.extend(one_hot_encode(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"]))
    features.extend(one_hot_encode(atom.GetDegree(), ATOM_FEATURES["degree"]))
    features.extend(
        one_hot_encode(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
    )
    features.extend(one_hot_encode(atom.GetChiralTag(), ATOM_FEATURES["chiral_tag"]))
    features.extend(one_hot_encode(atom.GetTotalNumHs(), ATOM_FEATURES["num_Hs"]))
    features.extend(
        one_hot_encode(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
    )

    features.extend(
        [
            atom.GetIsAromatic(),
            atom.IsInRing(),
            atom.GetMass(),
            atom.GetExplicitValence(),
            atom.GetImplicitValence(),
        ]
    )

    return features


def get_bond_features(bond: Chem.Bond) -> List[int]:
    """Get bond features as a one-hot encoded list."""
    features = []
    features.extend(one_hot_encode(bond.GetBondType(), BOND_FEATURES["bond_type"]))
    features.extend(one_hot_encode(bond.GetStereo(), BOND_FEATURES["stereo"]))
    features.extend(
        one_hot_encode(bond.GetIsConjugated(), BOND_FEATURES["is_conjugated"])
    )

    features.extend(
        [
            bond.IsInRing(),
            bond.GetIsAromatic(),
        ]
    )

    return features


def count_amide_bonds(mol: Chem.Mol) -> int:
    """Count the number of amide bonds in a molecule."""
    pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
    return len(mol.GetSubstructMatches(pattern))


def get_largest_ring_size(mol: Chem.Mol) -> int:
    """Get the size of the largest ring in the molecule."""
    rings = mol.GetRingInfo().AtomRings()
    return max([len(ring) for ring in rings]) if rings else 0


def get_cyclic_peptide_features(mol: Chem.Mol) -> Dict[str, float]:
    """Extract cyclic peptide specific features."""
    return {
        "n_amide_bonds": count_amide_bonds(mol),
        "largest_ring_size": get_largest_ring_size(mol),
        "n_rotatable_bonds": AllChem.CalcNumRotatableBonds(mol),
        "n_hbd": AllChem.CalcNumHBD(mol),
        "n_hba": AllChem.CalcNumHBA(mol),
        "mw": AllChem.CalcExactMolWt(mol),
        "logp": MolLogP(mol),
        "tpsa": TPSA(mol),
    }


def get_molecular_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Calculate molecular descriptors for a molecule using RDKit."""
    return {
        "n_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "n_hbd": Descriptors.NumHDonors(mol),
        "n_hba": Descriptors.NumHAcceptors(mol),
        "mw": Descriptors.ExactMolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
    }


def mol_to_graph_data(mol: Chem.Mol, pampa: float) -> Data:
    """Convert a molecule to a PyTorch Geometric Data object."""
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    # Get edge features and indices
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]  # Add both directions

        features = get_bond_features(bond)
        edge_features += [features, features]  # Add features for both directions

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(
            (0, len(get_bond_features(mol.GetBonds()[0]))), dtype=torch.float
        )

    # Get global features
    rdkit_features = get_molecular_descriptors(mol)
    cp_features = get_cyclic_peptide_features(mol)
    global_features = torch.tensor(
        np.concatenate(
            [
                np.array(list(rdkit_features.values())),
                np.array(
                    [
                        cp_features["n_amide_bonds"],
                        cp_features["largest_ring_size"],
                        cp_features["n_rotatable_bonds"],
                        cp_features["n_hbd"],
                        cp_features["n_hba"],
                        cp_features["mw"],
                    ]
                ),
            ]
        ),
        dtype=torch.float,
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_features=global_features,
        y=torch.tensor([pampa], dtype=torch.float),
    )


def parse_split_ratio(split_ratio_str: str) -> Tuple[float, float, float]:
    """Parse a comma-separated string of three floats that sum to 1.0."""
    try:
        train, val, test = map(float, split_ratio_str.split(","))
        if not (0 <= train <= 1 and 0 <= val <= 1 and 0 <= test <= 1):
            raise ValueError("Split ratios must be between 0 and 1")
        if abs(train + val + test - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1")
        return train, val, test
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def main():
    """Run the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess molecular data for PAMPA prediction"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input CSV file containing SMILES and PAMPA data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data and visualizations",
    )
    parser.add_argument(
        "--smiles_column",
        type=str,
        default="SMILES",
        help="Name of the column containing SMILES strings",
    )
    parser.add_argument(
        "--property_column",
        type=str,
        default="PAMPA",
        help="Name of the column containing property values",
    )
    parser.add_argument(
        "--split_ratio",
        type=str,
        default="0.8,0.1,0.1",
        help="Comma-separated train,val,test split ratios (must sum to 1)",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="Delimiter used in the CSV file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting preprocessing pipeline")
    logger.info(f"Input file: {args.data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"SMILES column: {args.smiles_column}")
    logger.info(f"Property column: {args.property_column}")
    logger.info(f"Split ratio (train/val/test): {args.split_ratio}")
    logger.info(f"Delimiter: {args.delimiter}")
    logger.info(f"Random seed: {args.seed}")

    # 1. Load and filter data
    logger.info("Loading and filtering data...")
    df = pd.read_csv(args.data_path, delimiter=args.delimiter)
    original_count = len(df)

    # Filter out missing values
    df = df.dropna(subset=[args.smiles_column, args.property_column])
    after_na_count = len(df)
    logger.info(
        f"Filtered out {original_count - after_na_count} compounds with missing values"
    )

    # Filter out compounds with PAMPA = -10 (invalid/placeholder values)
    df = df[df[args.property_column] > -10]
    final_count = len(df)
    logger.info(
        f"Filtered out {after_na_count - final_count} compounds with PAMPA = -10"
    )
    logger.info(f"Final number of compounds: {final_count}")

    # Log PAMPA value distribution
    pampa_stats = df[args.property_column].describe()
    logger.info("\nPAMPA value distribution:")
    logger.info(f"Mean: {pampa_stats['mean']:.2f}")
    logger.info(f"Std: {pampa_stats['std']:.2f}")
    logger.info(f"Min: {pampa_stats['min']:.2f}")
    logger.info(f"Max: {pampa_stats['max']:.2f}")
    logger.info(f"25%: {pampa_stats['25%']:.2f}")
    logger.info(f"50%: {pampa_stats['50%']:.2f}")
    logger.info(f"75%: {pampa_stats['75%']:.2f}\n")

    # Initialize preprocessor
    preprocessor = MolecularGraphPreprocessor(
        smiles_column=args.smiles_column,
        property_column=args.property_column,
        split_ratio=tuple(map(float, args.split_ratio.split(","))),
        delimiter=args.delimiter,
    )

    # Process the dataset
    preprocessor.df = df
    preprocessor.process_dataset(args.data_path, output_dir)

    logger.info("Preprocessing completed successfully")
    logger.info(f"Processed data saved to {output_dir}")


if __name__ == "__main__":
    main()
