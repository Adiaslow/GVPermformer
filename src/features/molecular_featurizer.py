"""
Molecular featurizer module for converting chemical structures to graph representations.
Includes atom and bond feature extraction and graph construction.
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import Dict, List, Tuple, Optional, Any, Union
import networkx as nx


class MolecularFeaturizer:
    """
    Molecular featurizer that transforms RDKit molecules into graph representations.

    Features extracted include atom properties, bond properties, and graph topology.
    """

    def __init__(
        self,
        atom_features: List[str] = None,
        bond_features: List[str] = None,
        use_3d: bool = False,
    ):
        """
        Initialize the molecular featurizer.

        Args:
            atom_features: List of atom features to extract
            bond_features: List of bond features to extract
            use_3d: Whether to use 3D coordinates (requires molecules with 3D conformers)
        """
        # Default atom features if none provided
        self.atom_features = atom_features or [
            "atomic_num",
            "formal_charge",
            "hybridization",
            "aromatic",
            "num_hydrogen",
            "degree",
            "chirality",
            "in_ring",
        ]

        # Default bond features if none provided
        self.bond_features = bond_features or [
            "bond_type",
            "conjugated",
            "in_ring",
            "stereo",
        ]

        self.use_3d = use_3d

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a molecule into a graph representation.

        Args:
            sample: Dictionary containing molecule data

        Returns:
            Updated sample with graph representation added
        """
        mol = sample["mol"]

        if self.use_3d and mol.GetNumConformers() == 0:
            # Add 3D coordinates if needed and not present
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)

        # Create graph representation
        graph = self._molecule_to_graph(mol)

        # Add graph to sample
        sample["graph"] = graph

        return sample

    def _molecule_to_graph(self, mol) -> Dict[str, torch.Tensor]:
        """
        Convert an RDKit molecule to a graph representation.

        Args:
            mol: RDKit molecule object

        Returns:
            Dictionary with node features, edge features, and edge indices
        """
        # Extract atom (node) features
        num_atoms = mol.GetNumAtoms()
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self._extract_atom_features(atom))

        # Convert to tensor
        node_features = torch.tensor(atom_features, dtype=torch.float)

        # Extract bond (edge) features and connectivity
        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            # Get atom indices
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            # Add edges in both directions for undirected graph
            edge_indices.extend([[start, end], [end, start]])

            # Extract bond features (same for both directions)
            bond_feats = self._extract_bond_features(bond)
            edge_features.extend([bond_feats, bond_feats])

        # Convert to tensors
        if edge_indices:  # Only if molecule has bonds
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(self.bond_features)), dtype=torch.float)

        # Add 3D coordinates if requested
        if self.use_3d and mol.GetNumConformers() > 0:
            conformer = mol.GetConformer()
            positions = torch.tensor(
                [
                    [
                        conformer.GetAtomPosition(i).x,
                        conformer.GetAtomPosition(i).y,
                        conformer.GetAtomPosition(i).z,
                    ]
                    for i in range(num_atoms)
                ],
                dtype=torch.float,
            )
        else:
            positions = None

        # Create graph dictionary
        graph = {
            "x": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": num_atoms,
        }

        if positions is not None:
            graph["pos"] = positions

        return graph

    def _extract_atom_features(self, atom) -> List[float]:
        """
        Extract features for a single atom.

        Args:
            atom: RDKit atom object

        Returns:
            List of atom features
        """
        features = []

        # Process each requested feature
        for feat_name in self.atom_features:
            if feat_name == "atomic_num":
                # One-hot encoding of common elements in peptides (H, C, N, O, S, etc.)
                element = atom.GetAtomicNum()
                # Map to common elements: H, C, N, O, S, P, F, Cl, Br, I, other
                element_map = {
                    1: 0,
                    6: 1,
                    7: 2,
                    8: 3,
                    16: 4,
                    15: 5,
                    9: 6,
                    17: 7,
                    35: 8,
                    53: 9,
                }
                one_hot = [0] * 11
                one_hot[element_map.get(element, 10)] = 1
                features.extend(one_hot)

            elif feat_name == "formal_charge":
                # Formal charge as a scalar feature
                charge = atom.GetFormalCharge()
                # Clip to range [-2, 2]
                features.append(max(-2, min(2, charge)) / 2.0)

            elif feat_name == "hybridization":
                # One-hot encoding of hybridization
                hyb_type = atom.GetHybridization()
                hyb_types = [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                ]
                one_hot = [int(hyb_type == h) for h in hyb_types]
                features.extend(one_hot)

            elif feat_name == "aromatic":
                # Boolean indicating aromaticity
                features.append(int(atom.GetIsAromatic()))

            elif feat_name == "num_hydrogen":
                # Total number of hydrogens (implicit and explicit)
                h_count = atom.GetTotalNumHs()
                # Normalize to [0, 1] assuming max 4 hydrogens
                features.append(min(h_count, 4) / 4.0)

            elif feat_name == "degree":
                # Degree (number of directly bonded neighbors)
                degree = atom.GetDegree()
                # Normalize to [0, 1] assuming max 6 neighbors
                features.append(min(degree, 6) / 6.0)

            elif feat_name == "chirality":
                # Chirality as one-hot
                chiral_tag = atom.GetChiralTag()
                chiral_tags = [
                    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                ]
                one_hot = [int(chiral_tag == c) for c in chiral_tags]
                features.extend(one_hot)

            elif feat_name == "in_ring":
                # Boolean indicating if atom is in a ring
                features.append(int(atom.IsInRing()))

        return features

    def _extract_bond_features(self, bond) -> List[float]:
        """
        Extract features for a single bond.

        Args:
            bond: RDKit bond object

        Returns:
            List of bond features
        """
        features = []

        # Process each requested feature
        for feat_name in self.bond_features:
            if feat_name == "bond_type":
                # One-hot encoding of bond type
                bond_type = bond.GetBondType()
                bond_types = [
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC,
                ]
                one_hot = [int(bond_type == b) for b in bond_types]
                features.extend(one_hot)

            elif feat_name == "conjugated":
                # Boolean indicating conjugation
                features.append(int(bond.GetIsConjugated()))

            elif feat_name == "in_ring":
                # Boolean indicating if bond is in a ring
                features.append(int(bond.IsInRing()))

            elif feat_name == "stereo":
                # One-hot encoding of stereo configuration
                stereo = bond.GetStereo()
                stereo_types = [
                    Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE,
                    Chem.rdchem.BondStereo.STEREOCIS,
                    Chem.rdchem.BondStereo.STEREOTRANS,
                ]
                one_hot = [int(stereo == s) for s in stereo_types]
                features.extend(one_hot)

        return features


class GlobalFeatureExtractor:
    """
    Extract global molecular features that capture overall properties.
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize the global feature extractor.

        Args:
            normalize: Whether to normalize features
        """
        self.normalize = normalize

        # List of RDKit descriptors to compute
        self.descriptors = [
            ("MolWt", Descriptors.MolWt),
            ("LogP", Descriptors.MolLogP),
            ("NumRotatableBonds", Descriptors.NumRotatableBonds),
            ("NumHDonors", Descriptors.NumHDonors),
            ("NumHAcceptors", Descriptors.NumHAcceptors),
            ("TPSA", Descriptors.TPSA),
            ("NumRings", Descriptors.RingCount),
            ("FractionCSP3", Descriptors.FractionCSP3),
            ("NumAromaticRings", Descriptors.NumAromaticRings),
            ("NumAliphaticRings", Descriptors.NumAliphaticRings),
            ("NumHeavyAtoms", Descriptors.HeavyAtomCount),
        ]

        # Default feature scales for normalization
        self.feature_scales = {
            "MolWt": (300.0, 800.0),  # Expected range for peptides
            "LogP": (-2.0, 5.0),
            "NumRotatableBonds": (0.0, 20.0),
            "NumHDonors": (0.0, 10.0),
            "NumHAcceptors": (0.0, 15.0),
            "TPSA": (40.0, 200.0),
            "NumRings": (0.0, 6.0),
            "FractionCSP3": (0.0, 1.0),
            "NumAromaticRings": (0.0, 5.0),
            "NumAliphaticRings": (0.0, 5.0),
            "NumHeavyAtoms": (20.0, 60.0),
        }

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract global features for a molecule.

        Args:
            sample: Dictionary containing molecule data

        Returns:
            Updated sample with global features added
        """
        mol = sample["mol"]

        # Calculate descriptors
        features = []
        for name, func in self.descriptors:
            value = func(mol)

            # Normalize if requested
            if self.normalize:
                min_val, max_val = self.feature_scales.get(name, (0.0, 1.0))
                if max_val > min_val:
                    value = (value - min_val) / (max_val - min_val)
                    # Clip to [0, 1]
                    value = max(0.0, min(1.0, value))

            features.append(value)

        # Add to sample
        sample["global_features"] = torch.tensor(features, dtype=torch.float)

        return sample


class CombinedFeaturizer:
    """
    Combine multiple featurizers into a single transform.
    """

    def __init__(self, transforms: List):
        """
        Initialize with a list of transforms.

        Args:
            transforms: List of featurizer transforms to apply
        """
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all transforms sequentially.

        Args:
            sample: Dictionary containing molecule data

        Returns:
            Updated sample with all features added
        """
        for t in self.transforms:
            sample = t(sample)

        return sample
