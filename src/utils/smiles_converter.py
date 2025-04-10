"""
Utility class for converting SMILES strings to molecular graph features
suitable for use with the GraphVAE model.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, List, Optional, Tuple, Union


class SmilesConverter:
    """
    Converts SMILES strings to molecular graph features suitable for the GraphVAE model.

    This class handles the conversion of SMILES strings to molecular graphs with
    the appropriate node (atom) and edge (bond) features required by the model.
    """

    def __init__(self, num_node_features: int = 126, num_edge_features: int = 9):
        """
        Initialize the SMILES converter with feature dimensions.

        Args:
            num_node_features: Number of atom features to generate
            num_edge_features: Number of bond features to generate
        """
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        # Define atom feature mappings
        self.atom_types = [
            "C",
            "N",
            "O",
            "S",
            "F",
            "Si",
            "P",
            "Cl",
            "Br",
            "Mg",
            "Na",
            "Ca",
            "Fe",
            "As",
            "Al",
            "I",
            "B",
            "V",
            "K",
            "Tl",
            "Yb",
            "Sb",
            "Sn",
            "Ag",
            "Pd",
            "Co",
            "Se",
            "Ti",
            "Zn",
            "H",
            "Li",
            "Ge",
            "Cu",
            "Au",
            "Ni",
            "Cd",
            "In",
            "Mn",
            "Zr",
            "Cr",
            "Pt",
            "Hg",
            "Pb",
            "W",
            "Ru",
            "Nb",
            "Re",
            "Te",
            "Rh",
            "Tc",
            "Ba",
            "Bi",
            "Hf",
            "Mo",
            "U",
            "Sm",
            "Os",
            "Ir",
            "Ce",
            "Gd",
            "Ga",
            "Cs",
            "unknown",
        ]

        self.atom_degrees = list(range(11))
        self.atom_formal_charges = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        self.atom_hybridizations = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
        ]
        self.atom_num_hs = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Define bond feature mappings
        self.bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        self.bond_stereo = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
        ]

    def smiles_to_graph(self, smiles: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convert a SMILES string to a molecular graph with features.

        Args:
            smiles: SMILES string representing a molecule

        Returns:
            Dictionary containing node_features, edge_index, edge_features, and batch indices.
            Returns None if the SMILES cannot be converted to a valid molecule.
        """
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogen atoms to molecule
        mol = Chem.AddHs(mol)

        # Check if molecule is valid and has atoms
        if mol is None or mol.GetNumAtoms() == 0:
            return None

        # Calculate atom features
        node_features = self._get_node_features(mol)

        # Calculate edge indices and features
        edge_index, edge_features = self._get_edge_features(mol)

        # If empty graph (no atoms or bonds), return None
        if node_features.shape[0] == 0 or edge_index.shape[1] == 0:
            return None

        # Create a batch index (all zeros for a single molecule)
        batch = torch.zeros(node_features.shape[0], dtype=torch.long)

        return {
            "x": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_features,
            "batch": batch,
        }

    def _get_node_features(self, mol: Chem.Mol) -> torch.Tensor:
        """
        Calculate atom (node) features for the molecule.

        Args:
            mol: RDKit molecule

        Returns:
            Tensor of shape [num_atoms, num_node_features] with atom features
        """
        if mol is None:
            return torch.zeros((0, self.num_node_features), dtype=torch.float)

        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, self.num_node_features), dtype=np.float32)

        # Track feature index
        feature_idx = 0

        for atom_idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)

            # Atom type one-hot encoding
            atom_type = atom.GetSymbol()
            if atom_type in self.atom_types:
                type_idx = self.atom_types.index(atom_type)
            else:
                type_idx = self.atom_types.index("unknown")
            features[atom_idx, feature_idx + type_idx] = 1.0
            feature_idx += len(self.atom_types)

            # Atom degree one-hot encoding
            degree = min(atom.GetDegree(), len(self.atom_degrees) - 1)
            features[atom_idx, feature_idx + degree] = 1.0
            feature_idx += len(self.atom_degrees)

            # Formal charge one-hot encoding
            charge_idx = (
                self.atom_formal_charges.index(atom.GetFormalCharge())
                if atom.GetFormalCharge() in self.atom_formal_charges
                else self.atom_formal_charges.index(0)
            )
            features[atom_idx, feature_idx + charge_idx] = 1.0
            feature_idx += len(self.atom_formal_charges)

            # Hybridization one-hot encoding
            hybrid_idx = (
                self.atom_hybridizations.index(atom.GetHybridization())
                if atom.GetHybridization() in self.atom_hybridizations
                else self.atom_hybridizations.index(
                    Chem.rdchem.HybridizationType.UNSPECIFIED
                )
            )
            features[atom_idx, feature_idx + hybrid_idx] = 1.0
            feature_idx += len(self.atom_hybridizations)

            # Aromaticity
            features[atom_idx, feature_idx] = 1.0 if atom.GetIsAromatic() else 0.0
            feature_idx += 1

            # Chirality
            features[atom_idx, feature_idx] = (
                1.0
                if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
                else 0.0
            )
            feature_idx += 1

            # Number of hydrogen atoms
            num_hs_idx = min(atom.GetTotalNumHs(), len(self.atom_num_hs) - 1)
            features[atom_idx, feature_idx + num_hs_idx] = 1.0
            feature_idx += len(self.atom_num_hs)

            # Is atom in ring
            features[atom_idx, feature_idx] = 1.0 if atom.IsInRing() else 0.0
            feature_idx += 1

            # Remaining features set to 0 by default
            # This ensures we always have num_node_features dimensions

        # Ensure we have the correct number of features
        assert (
            feature_idx <= self.num_node_features
        ), f"Feature index {feature_idx} exceeds num_node_features {self.num_node_features}"

        return torch.tensor(features, dtype=torch.float)

    def _get_edge_features(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate bond (edge) indices and features for the molecule.

        Args:
            mol: RDKit molecule

        Returns:
            Tuple of (edge_index, edge_features):
            - edge_index: Tensor of shape [2, num_edges] with source and target indices
            - edge_features: Tensor of shape [num_edges, num_edge_features] with bond features
        """
        if mol is None:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                (0, self.num_edge_features), dtype=torch.float
            )

        num_bonds = mol.GetNumBonds()
        if num_bonds == 0:
            # If no bonds, create self-loops for each atom
            num_atoms = mol.GetNumAtoms()
            edge_index = torch.zeros((2, num_atoms), dtype=torch.long)
            for i in range(num_atoms):
                edge_index[0, i] = i
                edge_index[1, i] = i
            return edge_index, torch.zeros(
                (num_atoms, self.num_edge_features), dtype=torch.float
            )

        # Initialize edge index and features
        # We double the number of bonds because we need to represent edges in both directions
        edge_index = torch.zeros((2, num_bonds * 2), dtype=torch.long)
        edge_features = torch.zeros(
            (num_bonds * 2, self.num_edge_features), dtype=torch.float
        )

        for bond_idx in range(num_bonds):
            bond = mol.GetBondWithIdx(bond_idx)

            # Get indices of atoms in bond
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Add edge in both directions
            edge_index[0, bond_idx * 2] = i
            edge_index[1, bond_idx * 2] = j
            edge_index[0, bond_idx * 2 + 1] = j
            edge_index[1, bond_idx * 2 + 1] = i

            # Calculate bond features
            features = np.zeros(self.num_edge_features, dtype=np.float32)
            feature_idx = 0

            # Bond type one-hot encoding
            bond_type_idx = (
                self.bond_types.index(bond.GetBondType())
                if bond.GetBondType() in self.bond_types
                else 0
            )  # Default to single bond
            features[feature_idx + bond_type_idx] = 1.0
            feature_idx += len(self.bond_types)

            # Bond is conjugated
            features[feature_idx] = 1.0 if bond.GetIsConjugated() else 0.0
            feature_idx += 1

            # Bond is in ring
            features[feature_idx] = 1.0 if bond.IsInRing() else 0.0
            feature_idx += 1

            # Bond stereo one-hot encoding
            stereo_idx = (
                self.bond_stereo.index(bond.GetStereo())
                if bond.GetStereo() in self.bond_stereo
                else 0
            )  # Default to STEREONONE
            features[feature_idx + stereo_idx] = 1.0

            # Set same features for both directions
            edge_features[bond_idx * 2] = torch.tensor(features, dtype=torch.float)
            edge_features[bond_idx * 2 + 1] = torch.tensor(features, dtype=torch.float)

        return edge_index, edge_features

    def batch_smiles_to_graphs(self, smiles_list: List[str]) -> Dict[str, torch.Tensor]:
        """
        Convert a list of SMILES strings to batched graph data.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary containing batched graph data with:
            - x: Node features
            - edge_index: Edge indices
            - edge_attr: Edge features
            - batch: Batch indices for nodes
            - valid_mask: Boolean mask indicating which SMILES were valid
        """
        valid_graphs = []
        valid_mask = torch.zeros(len(smiles_list), dtype=torch.bool)

        for i, smiles in enumerate(smiles_list):
            graph_data = self.smiles_to_graph(smiles)
            if graph_data is not None:
                valid_graphs.append(graph_data)
                valid_mask[i] = True

        if not valid_graphs:
            # Return empty tensors if no valid graphs
            return {
                "x": torch.zeros((0, self.num_node_features), dtype=torch.float),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros(
                    (0, self.num_edge_features), dtype=torch.float
                ),
                "batch": torch.zeros(0, dtype=torch.long),
                "valid_mask": valid_mask,
            }

        # Combine all graphs into a single batch
        batch_size = len(valid_graphs)

        # Concatenate node features
        all_x = []
        all_edge_index = []
        all_edge_attr = []
        all_batch = []

        offset = 0
        for i, graph in enumerate(valid_graphs):
            num_nodes = graph["x"].shape[0]

            all_x.append(graph["x"])

            # Adjust edge indices by the offset
            edge_index = graph["edge_index"].clone()
            edge_index += offset
            all_edge_index.append(edge_index)

            all_edge_attr.append(graph["edge_attr"])

            # Create batch tensor (all nodes from this graph belong to batch i)
            batch = torch.full((num_nodes,), i, dtype=torch.long)
            all_batch.append(batch)

            offset += num_nodes

        # Concatenate everything
        batch_data = {
            "x": torch.cat(all_x, dim=0),
            "edge_index": torch.cat(all_edge_index, dim=1),
            "edge_attr": torch.cat(all_edge_attr, dim=0),
            "batch": torch.cat(all_batch, dim=0),
            "valid_mask": valid_mask,
        }

        return batch_data
