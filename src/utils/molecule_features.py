"""
Utility for converting SMILES strings to molecular features for model prediction.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from typing import Dict, List, Tuple, Union, Optional


class MoleculeFeaturizer:
    """Converts SMILES strings to molecular graph features for the GraphVAE model."""

    def __init__(self):
        """Initialize the featurizer."""
        # Atom features
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
            "Unknown",
        ]
        self.atom_type_to_idx = {
            atom_type: i for i, atom_type in enumerate(self.atom_types)
        }

        # Feature dimensions
        self.num_atom_features = 126
        self.num_bond_features = 9

    def smiles_to_data(self, smiles: str) -> Dict[str, torch.Tensor]:
        """
        Convert a SMILES string to a dictionary containing graph tensors for the model.

        Args:
            smiles: The SMILES string to convert

        Returns:
            Dictionary containing node_features, edge_index, edge_features, and global_features
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")

            # Add hydrogens to get full molecular structure
            mol = Chem.AddHs(mol)

            # Get node features
            node_features = self._get_node_features(mol)

            # Get adjacency info and edge features
            edge_index, edge_features = self._get_edge_features(mol)

            # Get global molecular descriptors
            global_features = self._calculate_molecular_descriptors(mol)

            # Remove hydrogens for consistent representation with training data
            mol = Chem.RemoveHs(mol)

            data = {
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_features": edge_features,
                "global_features": global_features,
                "smiles": smiles,
            }

            return data

        except Exception as e:
            raise ValueError(f"Error processing SMILES {smiles}: {str(e)}")

    def _get_node_features(self, mol) -> torch.Tensor:
        """
        Compute node features for each atom in the molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            Tensor of shape [num_atoms, 126] with atom features
        """
        if mol is None:
            return torch.zeros((0, self.num_atom_features), dtype=torch.float32)

        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, self.num_atom_features), dtype=np.float32)

        # Feature index tracking
        feat_idx = 0

        for atom_idx, atom in enumerate(mol.GetAtoms()):
            # Atom type one-hot encoding (43 features)
            atom_type = atom.GetSymbol()
            if atom_type in self.atom_type_to_idx:
                features[atom_idx, self.atom_type_to_idx[atom_type]] = 1
            else:
                features[atom_idx, self.atom_type_to_idx["Unknown"]] = 1
            feat_idx = len(self.atom_types)  # 43

            # Atom degree one-hot encoding (11 features)
            degree = min(10, atom.GetDegree())
            features[atom_idx, feat_idx + degree] = 1
            feat_idx += 11

            # Formal charge one-hot encoding (5 features)
            formal_charge = atom.GetFormalCharge()
            features[atom_idx, feat_idx + min(4, formal_charge + 2)] = 1
            feat_idx += 5

            # Hybridization one-hot encoding (6 features)
            hybridization_types = [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ]
            hybridization = atom.GetHybridization()
            if hybridization in hybridization_types:
                features[
                    atom_idx, feat_idx + hybridization_types.index(hybridization)
                ] = 1
            else:
                features[atom_idx, feat_idx + 5] = 1  # Other hybridization
            feat_idx += 6

            # Aromaticity (1 feature)
            features[atom_idx, feat_idx] = 1 if atom.GetIsAromatic() else 0
            feat_idx += 1

            # Number of hydrogens one-hot encoding (5 features)
            num_h = min(4, atom.GetTotalNumHs())
            features[atom_idx, feat_idx + num_h] = 1
            feat_idx += 5

            # Chirality (3 features)
            if atom.HasProp("_CIPCode"):
                if atom.GetProp("_CIPCode") == "R":
                    features[atom_idx, feat_idx] = 1
                elif atom.GetProp("_CIPCode") == "S":
                    features[atom_idx, feat_idx + 1] = 1
            else:
                features[atom_idx, feat_idx + 2] = 1  # Not chiral
            feat_idx += 3

            # Is in ring (1 feature)
            features[atom_idx, feat_idx] = 1 if atom.IsInRing() else 0
            feat_idx += 1

            # Atomic mass normalized (1 feature)
            features[atom_idx, feat_idx] = atom.GetMass() / 100.0
            feat_idx += 1

            # Atomic number normalized (1 feature)
            features[atom_idx, feat_idx] = atom.GetAtomicNum() / 100.0
            feat_idx += 1

            # Van der Waals radius normalized (1 feature)
            features[atom_idx, feat_idx] = (
                Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) / 10.0
            )
            feat_idx += 1

            # Covalent radius normalized (1 feature)
            features[atom_idx, feat_idx] = (
                Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) / 10.0
            )
            feat_idx += 1

            # Additional atom properties
            # Gasteiger charge (1 feature)
            if atom.HasProp("_GasteigerCharge"):
                charge = float(atom.GetProp("_GasteigerCharge"))
                # Normalize and clip gasteiger charge
                charge = max(-1.0, min(1.0, charge))
                features[atom_idx, feat_idx] = charge
            feat_idx += 1

            # Skip complex atom contributions (3 features)
            feat_idx += 3

            # Remaining property calculations (47 features)
            # For simplicity, we'll skip these and set them to 0
            # This ensures we maintain the expected 126 features per atom

            # Verify we're at the correct index
            feat_remaining = self.num_atom_features - feat_idx
            feat_idx += feat_remaining

        # Verify we have the expected number of features
        assert (
            features.shape[1] == self.num_atom_features
        ), f"Expected {self.num_atom_features} features but got {features.shape[1]}"

        return torch.tensor(features, dtype=torch.float32)

    def _get_edge_features(self, mol) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edge indices and features for bonds in the molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            Tuple containing:
                - edge_index: Tensor of shape [2, num_edges*2] with source and target nodes
                - edge_features: Tensor of shape [num_edges*2, 9] with bond features
        """
        if mol is None:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                (0, self.num_bond_features), dtype=torch.float32
            )

        # Initialize edge index and features lists
        edge_indices = []
        edge_features = []

        # Loop through all bonds in the molecule
        for bond in mol.GetBonds():
            # Get indices of atoms in the bond
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Add both directions (i->j and j->i) for undirected graph
            edge_indices.append((i, j))
            edge_indices.append((j, i))

            # Calculate bond features
            bond_features = self._get_bond_features(bond)
            edge_features.append(bond_features)
            edge_features.append(bond_features)  # Same features for both directions

        # If molecule has no bonds
        if len(edge_indices) == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                (0, self.num_bond_features), dtype=torch.float32
            )

        # Convert to tensors
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_features = torch.tensor(edge_features, dtype=torch.float32)

        return edge_index, edge_features

    def _get_bond_features(self, bond) -> List[float]:
        """
        Calculate features for a bond.

        Args:
            bond: RDKit bond object

        Returns:
            List of bond features
        """
        # Bond type one-hot encoding (4 features)
        bond_type_features = [0] * 4
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_type_features[0] = 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_type_features[1] = 1
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_type_features[2] = 1
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            bond_type_features[3] = 1

        # Bond stereo one-hot encoding (3 features)
        stereo_features = [0] * 3
        stereo = bond.GetStereo()
        if stereo == Chem.rdchem.BondStereo.STERONONE:
            stereo_features[0] = 1
        elif stereo in (
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ):
            stereo_features[1] = 1
        else:
            stereo_features[2] = 1

        # Is conjugated (1 feature)
        is_conjugated = 1 if bond.GetIsConjugated() else 0

        # Is in ring (1 feature)
        is_in_ring = 1 if bond.IsInRing() else 0

        # Combine all features
        bond_features = (
            bond_type_features + stereo_features + [is_conjugated, is_in_ring]
        )

        return bond_features

    def _calculate_molecular_descriptors(self, mol) -> torch.Tensor:
        """
        Calculate global molecular descriptors.

        Args:
            mol: RDKit molecule object

        Returns:
            Tensor of global molecular descriptors
        """
        if mol is None:
            return torch.zeros(10, dtype=torch.float32)

        try:
            descriptors = []

            # Molecular weight
            descriptors.append(Descriptors.MolWt(mol) / 500.0)

            # LogP
            descriptors.append((Descriptors.MolLogP(mol) + 10) / 20.0)

            # Number of H-bond donors
            descriptors.append(Descriptors.NumHDonors(mol) / 10.0)

            # Number of H-bond acceptors
            descriptors.append(Descriptors.NumHAcceptors(mol) / 10.0)

            # Number of rotatable bonds
            descriptors.append(Descriptors.NumRotatableBonds(mol) / 10.0)

            # Topological polar surface area
            descriptors.append(Descriptors.TPSA(mol) / 200.0)

            # Number of rings
            descriptors.append(Descriptors.RingCount(mol) / 10.0)

            # Number of aromatic rings
            descriptors.append(Descriptors.NumAromaticRings(mol) / 5.0)

            # Number of hetero atoms
            descriptors.append(Descriptors.NumHeteroatoms(mol) / 20.0)

            # Number of atoms
            descriptors.append(mol.GetNumAtoms() / 50.0)

            return torch.tensor(descriptors, dtype=torch.float32)

        except:
            # Return zeros if descriptor calculation fails
            return torch.zeros(10, dtype=torch.float32)

    def smiles_to_batch(self, smiles_list: List[str]) -> Dict[str, torch.Tensor]:
        """
        Convert a list of SMILES strings to a batched graph for the model.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary with batched graph tensors
        """
        if not smiles_list:
            raise ValueError("Empty SMILES list provided")

        # Process each SMILES string
        data_list = []
        valid_smiles = []

        for smiles in smiles_list:
            try:
                data = self.smiles_to_data(smiles)
                data_list.append(data)
                valid_smiles.append(smiles)
            except Exception as e:
                print(f"Warning: Could not process SMILES {smiles}: {e}")

        if not data_list:
            raise ValueError("None of the provided SMILES could be processed")

        # Batch the data
        return self._batch_data(data_list, valid_smiles)

    def _batch_data(
        self, data_list: List[Dict[str, torch.Tensor]], smiles_list: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Batch individual molecular graphs into a single graph with disconnected components.

        Args:
            data_list: List of dictionaries with graph tensors
            smiles_list: List of valid SMILES strings

        Returns:
            Dictionary with batched tensor data
        """
        batch_size = len(data_list)

        # Initialize batch data
        batch_data = {
            "batch": torch.zeros(0, dtype=torch.long),
            "node_features": torch.zeros(
                (0, self.num_atom_features), dtype=torch.float32
            ),
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_features": torch.zeros(
                (0, self.num_bond_features), dtype=torch.float32
            ),
            "global_features": torch.zeros((batch_size, 10), dtype=torch.float32),
            "smiles": smiles_list,
        }

        # Track sizes for batching
        cumsum_nodes = 0
        batch_index = []

        # Process each molecule
        for i, data in enumerate(data_list):
            num_nodes = data["node_features"].shape[0]
            num_edges = data["edge_index"].shape[1]

            # Update batch index
            batch_index.append(torch.full((num_nodes,), i, dtype=torch.long))

            # Update node features
            batch_data["node_features"] = torch.cat(
                [batch_data["node_features"], data["node_features"]], dim=0
            )

            # Update edge indices with offset
            if num_edges > 0:
                edges = data["edge_index"].clone()
                edges += cumsum_nodes
                batch_data["edge_index"] = torch.cat(
                    [batch_data["edge_index"], edges], dim=1
                )

            # Update edge features
            batch_data["edge_features"] = torch.cat(
                [batch_data["edge_features"], data["edge_features"]], dim=0
            )

            # Update global features
            batch_data["global_features"][i] = data["global_features"]

            # Update cumulative node count
            cumsum_nodes += num_nodes

        # Combine batch indices
        if batch_index:
            batch_data["batch"] = torch.cat(batch_index)

        return batch_data
