"""
Utility module for converting SMILES strings to graph features that can be used
with the GraphVAE model for predictions.
"""

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from typing import Dict, Tuple, List, Optional, Union


class SmilesConverter:
    """
    Convert SMILES strings to graph features compatible with the GraphVAE model.
    """

    def __init__(self, max_atoms: int = 50):
        """
        Initialize the SMILES converter.

        Args:
            max_atoms: Maximum number of atoms to consider in a molecule
        """
        self.max_atoms = max_atoms
        # Hybridization types for atom feature extraction
        self.hybridization_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]

    def convert(self, smiles: str) -> Dict[str, torch.Tensor]:
        """
        Convert a SMILES string to graph features.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Dict containing node features, edge index, edge attributes, and global features
        """
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            # Return empty tensors if the molecule is invalid
            return {
                "node_features": torch.zeros(0, 126, dtype=torch.float),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "edge_attr": torch.zeros(0, 9, dtype=torch.float),
                "global_features": torch.zeros(1, 17, dtype=torch.float),
            }

        # Check if molecule has too many atoms
        if mol.GetNumAtoms() > self.max_atoms:
            print(
                f"Warning: Molecule has {mol.GetNumAtoms()} atoms, which exceeds the maximum of {self.max_atoms}"
            )
            # You could either truncate or return empty tensors
            return {
                "node_features": torch.zeros(0, 126, dtype=torch.float),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "edge_attr": torch.zeros(0, 9, dtype=torch.float),
                "global_features": torch.zeros(1, 17, dtype=torch.float),
            }

        # Create a DataFrame row with molecular descriptors
        row = self._calculate_descriptors(smiles, mol)

        # Extract features
        node_features = self._get_node_features(mol)
        edge_index, edge_attr = self._get_edge_features(mol)
        global_features = self._get_global_features(row)

        # Return dictionary with all features
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "global_features": global_features,
        }

    def _calculate_descriptors(self, smiles: str, mol: Chem.Mol) -> pd.Series:
        """
        Calculate molecular descriptors from a molecule.

        Args:
            smiles: SMILES string of the molecule
            mol: RDKit molecule object

        Returns:
            pd.Series containing molecular descriptors
        """
        # Create a dictionary to store descriptors
        descriptors = {}

        # Add SMILES to descriptors
        descriptors["SMILES"] = smiles

        # Basic molecular properties
        descriptors["MolWt"] = Descriptors.MolWt(mol)
        descriptors["HeavyAtomCount"] = mol.GetNumHeavyAtoms()
        descriptors["NumRotatableBonds"] = Descriptors.NumRotatableBonds(mol)
        descriptors["NumHAcceptors"] = Descriptors.NumHAcceptors(mol)
        descriptors["NumHDonors"] = Descriptors.NumHDonors(mol)
        descriptors["NumRings"] = Descriptors.RingCount(mol)

        # Electronic properties
        descriptors["LogP"] = Descriptors.MolLogP(mol)
        descriptors["TPSA"] = Descriptors.TPSA(mol)

        # Return as pandas Series
        return pd.Series(descriptors)

    def _get_node_features(self, mol: Chem.Mol) -> torch.Tensor:
        """
        Extract node features from a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            torch.Tensor: Node features tensor of shape [num_atoms, 126]
        """
        if mol is None:
            return torch.zeros(0, 126, dtype=torch.float)

        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, 126, dtype=torch.float)

        for atom_idx, atom in enumerate(mol.GetAtoms()):
            feature_idx = 0

            # One-hot encoding of atom type (94 features)
            atom_type = atom.GetAtomicNum() - 1  # Hydrogen = 0
            if 0 <= atom_type < 94:
                features[atom_idx, atom_type] = 1
            feature_idx += 94

            # One-hot encoding of atom degree (11 features)
            degree = min(atom.GetDegree(), 10)
            features[atom_idx, feature_idx + degree] = 1
            feature_idx += 11

            # One-hot encoding of formal charge (11 features)
            formal_charge = atom.GetFormalCharge()
            # Shift to range [0, 10] from [-5, 5]
            charge_idx = min(max(formal_charge + 5, 0), 10)
            features[atom_idx, feature_idx + charge_idx] = 1
            feature_idx += 11

            # One-hot encoding of hybridization (6 features)
            hybridization = atom.GetHybridization()
            hyb_idx = (
                self.hybridization_types.index(hybridization)
                if hybridization in self.hybridization_types
                else 5
            )
            features[atom_idx, feature_idx + hyb_idx] = 1
            feature_idx += 6

            # Remaining 4 features (4 features)
            # Aromaticity (1 feature)
            features[atom_idx, feature_idx] = 1 if atom.GetIsAromatic() else 0
            feature_idx += 1

            # Chirality (1 feature)
            features[atom_idx, feature_idx] = (
                1
                if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
                else 0
            )
            feature_idx += 1

            # Number of H atoms (1 feature)
            features[atom_idx, feature_idx] = (
                atom.GetTotalNumHs() / 4.0
            )  # Normalize by typical max
            feature_idx += 1

            # Atom is in ring (1 feature)
            features[atom_idx, feature_idx] = 1 if atom.IsInRing() else 0

        assert (
            features.shape[1] == 126
        ), f"Expected 126 features, got {features.shape[1]}"
        return features

    def _get_edge_features(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract edge features from a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            tuple: (edge_index, edge_attr) where edge_index is a tensor of shape [2, num_edges]
                  and edge_attr is a tensor of shape [num_edges, 9]
        """
        if mol is None:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(
                0, 9, dtype=torch.float
            )

        num_atoms = mol.GetNumAtoms()
        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Enhanced bond features (9 features total)
            edge_feature = [0] * 9

            # Bond type (first 4 positions)
            bond_type = bond.GetBondTypeAsDouble()
            if bond_type == 1:
                edge_feature[0] = 1  # Single
            elif bond_type == 2:
                edge_feature[1] = 1  # Double
            elif bond_type == 3:
                edge_feature[2] = 1  # Triple
            elif bond_type == 1.5:
                edge_feature[3] = 1  # Aromatic

            # Bond is conjugated
            edge_feature[4] = int(bond.GetIsConjugated())

            # Bond is in ring
            edge_feature[5] = int(bond.IsInRing())

            # Bond stereochemistry
            stereo = bond.GetStereo()
            if (
                stereo == Chem.rdchem.BondStereo.STEREOZ
                or stereo == Chem.rdchem.BondStereo.STEREOCIS
            ):
                edge_feature[6] = 1  # Z/cis
            elif (
                stereo == Chem.rdchem.BondStereo.STEREOE
                or stereo == Chem.rdchem.BondStereo.STEREOTRANS
            ):
                edge_feature[7] = 1  # E/trans
            else:
                edge_feature[8] = 1  # None or unspecified

            # Add edges in both directions
            edge_indices.append([i, j])
            edge_indices.append([j, i])

            edge_attrs.append(edge_feature)
            edge_attrs.append(edge_feature)

        if not edge_indices:
            # If no bonds, create self-loops
            edge_indices = [[i, i] for i in range(num_atoms)]
            edge_attrs = [[1, 0, 0, 0, 0, 0, 0, 0, 1] for _ in range(num_atoms)]

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        return edge_index, edge_attr

    def _get_global_features(self, row: pd.Series) -> torch.Tensor:
        """
        Extract global molecular features from pre-calculated descriptors.

        Args:
            row: Series containing molecule data and descriptors

        Returns:
            torch.Tensor: Global features vector
        """
        # Extract features from the descriptor row
        feature_list = []

        # Physical properties
        feature_list.append(float(row.get("MolWt", 0.0)))
        feature_list.append(float(row.get("HeavyAtomCount", 0.0)))
        feature_list.append(float(row.get("NumRotatableBonds", 0.0)))
        feature_list.append(float(row.get("NumHAcceptors", 0.0)))
        feature_list.append(float(row.get("NumHDonors", 0.0)))
        feature_list.append(float(row.get("NumRings", 0.0)))

        # Electronic properties
        feature_list.append(float(row.get("LogP", 0.0)))
        feature_list.append(float(row.get("TPSA", 0.0)))

        # Normalize typical value ranges
        # Simple min-max normalization based on typical ranges
        normalized_features = []

        # MolWt (typical range 0-500)
        normalized_features.append(min(feature_list[0] / 500.0, 1.0))

        # HeavyAtomCount (typical range 0-50)
        normalized_features.append(min(feature_list[1] / 50.0, 1.0))

        # NumRotatableBonds (typical range 0-15)
        normalized_features.append(min(feature_list[2] / 15.0, 1.0))

        # NumHAcceptors (typical range 0-10)
        normalized_features.append(min(feature_list[3] / 10.0, 1.0))

        # NumHDonors (typical range 0-5)
        normalized_features.append(min(feature_list[4] / 5.0, 1.0))

        # NumRings (typical range 0-6)
        normalized_features.append(min(feature_list[5] / 6.0, 1.0))

        # LogP (typical range -3 to 7, normalize to 0-1)
        normalized_features.append((feature_list[6] + 3.0) / 10.0)

        # TPSA (typical range 0-140)
        normalized_features.append(min(feature_list[7] / 140.0, 1.0))

        # Repeat values to match the expected global feature size (17)
        # This ensures compatibility with the model
        while len(normalized_features) < 17:
            normalized_features.append(0.0)

        return torch.tensor(normalized_features, dtype=torch.float).unsqueeze(0)
