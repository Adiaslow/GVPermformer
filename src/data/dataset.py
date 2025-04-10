"""
Dataset module for loading and preprocessing cyclic peptide data.
Handles CSV parsing, transformations, and PyTorch Dataset creation.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.GraphDescriptors as GraphDescriptors
import networkx as nx
from rdkit.Chem import rdmolops
from typing import Dict, List, Tuple, Optional, Union
import logging


class CyclicPeptideDataset(Dataset):
    """
    Dataset class for handling cyclic peptide permeability data.

    This class handles loading the raw CSV data, preprocessing molecular structures,
    and converting to graph-based representations suitable for the model.
    """

    def __init__(
        self,
        csv_file: str,
        smiles_col: str = "SMILES",
        target_col: str = "Permeability",
        transform=None,
        test_mode: bool = False,
    ):
        """
        Initialize dataset with CSV file containing SMILES and permeability data.

        Args:
            csv_file: Path to CSV file with peptide data
            smiles_col: Name of column containing SMILES strings
            target_col: Name of column containing permeability values
            transform: Optional transform to apply to samples
            test_mode: If True, runs in test mode (no target values required)
        """
        self.data_path = csv_file
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.transform = transform
        self.test_mode = test_mode

        self.df = pd.read_csv(csv_file)
        # Filter out invalid rows
        self._filter_invalid_data()

        # Store SMILES and permeability data
        self.smiles_list = self.df[smiles_col].values
        if not test_mode:
            self.permeability = self.df[target_col].values

        # Precompute molecules for faster access
        self.mol_list = [Chem.MolFromSmiles(s) for s in self.smiles_list]

        # Remove entries with invalid molecules
        valid_indices = [i for i, mol in enumerate(self.mol_list) if mol is not None]
        self.smiles_list = self.smiles_list[valid_indices]
        self.mol_list = [self.mol_list[i] for i in valid_indices]
        if not test_mode:
            self.permeability = self.permeability[valid_indices]

        # Log dataset statistics
        logging.info(f"Loaded {len(self.mol_list)} valid molecules from {csv_file}")

    def _filter_invalid_data(self):
        """Filter out rows with missing SMILES or target values."""
        # Remove rows with missing SMILES
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=[self.smiles_col])

        # Remove rows with missing targets in training mode
        if not self.test_mode:
            self.df = self.df.dropna(subset=[self.target_col])

        # Log filtering statistics
        if len(self.df) < initial_len:
            logging.info(f"Filtered out {initial_len - len(self.df)} invalid rows")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.mol_list)

    def __getitem__(self, idx: int):
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing molecule data and target value
        """
        mol = self.mol_list[idx]

        # Create sample dictionary
        sample = {
            "mol": mol,
            "smiles": self.smiles_list[idx],
        }

        # Add target value if not in test mode
        if not self.test_mode:
            sample["target"] = torch.tensor(self.permeability[idx], dtype=torch.float)

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)

        return sample


def get_dataloader(
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform=None,
    test_mode: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the specified dataset.

    Args:
        csv_path: Path to CSV file
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for DataLoader
        transform: Optional transform to apply to samples
        test_mode: If True, runs in test mode
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader object
    """
    dataset = CyclicPeptideDataset(
        csv_file=csv_path, transform=transform, test_mode=test_mode
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_molecules,
    )


def collate_molecules(batch: List[Dict]):
    """
    Custom collate function for batching molecular data.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched data dictionary
    """
    # Extract keys from batch
    keys = batch[0].keys()

    # Initialize batch dictionary
    batched = {key: [] for key in keys}

    # Add samples to batch
    for sample in batch:
        for key in keys:
            batched[key].append(sample[key])

    # Convert target to tensor if present
    if "target" in batched:
        batched["target"] = torch.stack(batched["target"])

    return batched


class MoleculeDataset(Dataset):
    """
    Dataset for molecular graphs with properties for the Graph VAE Transformer.

    This dataset handles loading molecules from SMILES strings and converting
    them into graph representations suitable for the Graph VAE Transformer model.
    """

    def __init__(
        self,
        csv_file: str,
        smiles_col: str = "smiles",
        property_cols: list = None,
        max_atoms: int = 50,
        n_features: int = 94,
        filter_pampa: bool = True,
        pampa_threshold: float = -9.0,
    ):
        """
        Initialize the MoleculeDataset.

        Args:
            csv_file: Path to the CSV file containing SMILES and properties
            smiles_col: Name of the column containing SMILES strings
            property_cols: List of column names for molecular properties
            max_atoms: Maximum number of atoms to consider in molecules
            n_features: Number of atom features to use
            filter_pampa: Whether to filter out entries with PAMPA below threshold
            pampa_threshold: Threshold for filtering PAMPA values
        """
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.smiles_col = smiles_col
        self.property_cols = property_cols or []
        self.max_atoms = max_atoms
        self.n_features = n_features

        # Filter by PAMPA value if requested and if 'PAMPA' is in property_cols
        if filter_pampa and "PAMPA" in self.property_cols:
            pampa_idx = self.property_cols.index("PAMPA")
            print(f"Filtering out entries with PAMPA values below {pampa_threshold}")
            original_count = len(self.data)
            self.data = self.data[self.data["PAMPA"] >= pampa_threshold]
            filtered_count = len(self.data)
            print(
                f"Removed {original_count - filtered_count} entries with low PAMPA values"
            )
            print(f"Remaining data size: {filtered_count}")

        # Calculate additional molecular descriptors for feature engineering
        self._calculate_additional_descriptors()

        # Filter out invalid molecules after calculating descriptors
        self._filter_invalid_molecules()

    def _filter_invalid_molecules(self):
        """Filter out invalid molecules from the dataset"""
        valid_mols = []
        print("Filtering out invalid or too large molecules...")
        for i, row in self.data.iterrows():
            try:
                mol = Chem.MolFromSmiles(row[self.smiles_col])
                if mol is not None and mol.GetNumAtoms() <= self.max_atoms:
                    valid_mols.append(i)
            except:
                # Skip any errors when processing molecules
                continue

        print(f"Starting with {len(self.data)} entries")
        self.data = self.data.loc[valid_mols].reset_index(drop=True)
        print(f"Filtered to {len(self.data)} valid molecules")

    def _calculate_additional_descriptors(self):
        """Calculate additional molecular descriptors for filtering and property prediction."""
        print("Calculating additional molecular descriptors...")

        # Initialize descriptor columns
        descriptors = []

        for i, row in self.data.iterrows():
            mol = Chem.MolFromSmiles(row[self.smiles_col])
            if mol is None:
                # If molecule is invalid, add placeholder values
                descriptors.append(
                    {
                        "MolWt": 0.0,
                        "LogP": 0.0,
                        "NumHDonors": 0,
                        "NumHAcceptors": 0,
                        "NumRotatableBonds": 0,
                        "NumRings": 0,
                        "TPSA": 0.0,
                        "QED": 0.0,
                        "FractionCSP3": 0.0,
                        "HeavyAtomCount": 0,
                    }
                )
                continue

            # Calculate basic molecular descriptors for each molecule
            try:
                descriptors.append(
                    {
                        "MolWt": Descriptors.MolWt(mol),
                        "LogP": Descriptors.MolLogP(mol),
                        "NumHDonors": Descriptors.NumHDonors(mol),
                        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
                        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
                        "NumRings": rdMolDescriptors.CalcNumRings(mol),
                        "TPSA": Descriptors.TPSA(mol),
                        "QED": QED.qed(mol),
                        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
                        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
                    }
                )
            except:
                # If descriptor calculation fails, add placeholder values
                descriptors.append(
                    {
                        "MolWt": 0.0,
                        "LogP": 0.0,
                        "NumHDonors": 0,
                        "NumHAcceptors": 0,
                        "NumRotatableBonds": 0,
                        "NumRings": 0,
                        "TPSA": 0.0,
                        "QED": 0.0,
                        "FractionCSP3": 0.0,
                        "HeavyAtomCount": 0,
                    }
                )

        # Add descriptors as new columns
        descriptor_df = pd.DataFrame(descriptors)
        self.data = pd.concat([self.data, descriptor_df], axis=1)

        print("Completed calculation of molecular descriptors")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        smiles = row[self.smiles_col]
        mol = Chem.MolFromSmiles(smiles)

        # Get graph representation
        x = self._get_node_features(mol)
        edge_index, edge_attr = self._get_edge_features(mol)

        # Extract global molecular features
        global_features = self._get_global_features(row)

        # Get properties if available
        properties = None
        if self.property_cols:
            prop_values = []
            for col in self.property_cols:
                # Get the property value and handle NaN
                value = row[col]
                if pd.isna(value):
                    # Replace NaN with 0.0 for now
                    value = 0.0
                prop_values.append(value)

            properties = torch.tensor(prop_values, dtype=torch.float)

        sample = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "global_features": global_features,
            "num_nodes": x.size(0),
            "smiles": smiles,
        }

        if properties is not None:
            sample["properties"] = properties

        return sample

    def _get_node_features(self, mol):
        """
        Extract enhanced node features from a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            torch.Tensor: Node features matrix of shape [num_atoms, n_features]
        """
        if mol is None:
            return torch.zeros(0, 126)  # Fixed dimension

        num_atoms = mol.GetNumAtoms()
        # Use exactly 126 features per atom
        features = torch.zeros(num_atoms, 126)

        # Track feature index for clarity
        feature_offset = 0

        # Predefined lists for one-hot encoding
        hybridization_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP2D,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.OTHER,
        ]

        for atom_idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)
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
                hybridization_types.index(hybridization)
                if hybridization in hybridization_types
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

    def _get_edge_features(self, mol):
        """
        Extract enhanced edge features from a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            tuple: (edge_index, edge_attr) where edge_index is a tensor of shape [2, num_edges]
                  and edge_attr is a tensor of shape [num_edges, num_edge_features]
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

            # Enhanced bond features (9 features total):
            # - Bond type (one-hot, 4 features: single, double, triple, aromatic)
            # - Bond is conjugated (1 feature)
            # - Bond is in ring (1 feature)
            # - Bond stereochemistry (one-hot, 3 features: none, Z/cis, E/trans)
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

    def _get_graph_descriptors(self, mol):
        """
        Calculate graph-based descriptors for the molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            Dictionary of graph-based descriptors
        """
        if mol is None or mol.GetNumAtoms() == 0:
            return {}

        graph_features = {}

        # 1. BASIC GRAPH DESCRIPTORS FROM RDKIT
        # Handle potential exceptions for each descriptor
        try:
            if mol.GetNumBonds() > 0:
                graph_features["BalabanJ"] = GraphDescriptors.BalabanJ(mol)
            else:
                graph_features["BalabanJ"] = 0
        except:
            graph_features["BalabanJ"] = 0

        try:
            graph_features["BertzCT"] = GraphDescriptors.BertzCT(mol)
        except:
            graph_features["BertzCT"] = 0

        try:
            graph_features["Chi0"] = GraphDescriptors.Chi0(mol)
        except:
            graph_features["Chi0"] = 0

        try:
            graph_features["Chi0v"] = rdMolDescriptors.CalcChi0v(mol)
        except:
            graph_features["Chi0v"] = 0

        try:
            graph_features["Chi1"] = GraphDescriptors.Chi1(mol)
        except:
            graph_features["Chi1"] = 0

        try:
            graph_features["Kappa1"] = rdMolDescriptors.CalcKappa1(mol)
        except:
            graph_features["Kappa1"] = 0

        try:
            graph_features["Kappa2"] = rdMolDescriptors.CalcKappa2(mol)
        except:
            graph_features["Kappa2"] = 0

        try:
            graph_features["Kappa3"] = rdMolDescriptors.CalcKappa3(mol)
        except:
            graph_features["Kappa3"] = 0

        # 2. ADVANCED GRAPH METRICS USING NETWORKX
        try:
            # Convert molecule to NetworkX graph
            G = nx.Graph()

            # Add nodes with atomic number as node attribute
            for atom in mol.GetAtoms():
                G.add_node(
                    atom.GetIdx(),
                    atomic_num=atom.GetAtomicNum(),
                    formal_charge=atom.GetFormalCharge(),
                    is_aromatic=int(atom.GetIsAromatic()),
                    in_ring=int(atom.IsInRing()),
                )

            # Add bonds as edges with bond type as edge attribute
            for bond in mol.GetBonds():
                G.add_edge(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond_type=bond.GetBondTypeAsDouble(),
                    is_aromatic=int(bond.GetIsAromatic()),
                    in_ring=int(bond.IsInRing()),
                )

            if len(G.nodes) > 0:
                # Calculate centrality measures
                degree_cent = nx.degree_centrality(G)
                graph_features["MaxDegreeCentrality"] = (
                    max(degree_cent.values()) if degree_cent else 0
                )
                graph_features["MeanDegreeCentrality"] = (
                    sum(degree_cent.values()) / len(degree_cent) if degree_cent else 0
                )

                if len(G.nodes) > 1:  # Need at least 2 nodes for these metrics
                    try:
                        # Betweenness centrality
                        between_cent = nx.betweenness_centrality(G)
                        graph_features["MaxBetweennessCentrality"] = (
                            max(between_cent.values()) if between_cent else 0
                        )
                        graph_features["MeanBetweennessCentrality"] = (
                            sum(between_cent.values()) / len(between_cent)
                            if between_cent
                            else 0
                        )

                        # Closeness centrality (for connected graphs)
                        if nx.is_connected(G):
                            close_cent = nx.closeness_centrality(G)
                            graph_features["MaxClosenessCentrality"] = (
                                max(close_cent.values()) if close_cent else 0
                            )
                            graph_features["MeanClosenessCentrality"] = (
                                sum(close_cent.values()) / len(close_cent)
                                if close_cent
                                else 0
                            )
                        else:
                            # For disconnected graphs, calculate on largest connected component
                            largest_cc = max(nx.connected_components(G), key=len)
                            subgraph = G.subgraph(largest_cc)
                            close_cent = nx.closeness_centrality(subgraph)
                            graph_features["MaxClosenessCentrality"] = (
                                max(close_cent.values()) if close_cent else 0
                            )
                            graph_features["MeanClosenessCentrality"] = (
                                sum(close_cent.values()) / len(close_cent)
                                if close_cent
                                else 0
                            )
                            graph_features["LargestComponentRatio"] = len(
                                largest_cc
                            ) / len(G.nodes)
                    except:
                        # Default values if calculation fails
                        graph_features["MaxBetweennessCentrality"] = 0
                        graph_features["MeanBetweennessCentrality"] = 0
                        graph_features["MaxClosenessCentrality"] = 0
                        graph_features["MeanClosenessCentrality"] = 0
                        graph_features["LargestComponentRatio"] = 1

                # Graph structure metrics
                graph_features["GraphDensity"] = nx.density(G)

                # Number of connected components
                num_components = nx.number_connected_components(G)
                graph_features["ConnectedComponents"] = num_components

                # Calculate cycle-specific metrics
                cycles = nx.cycle_basis(G)
                graph_features["NumCycles"] = len(cycles)
                cycle_lengths = [len(cycle) for cycle in cycles]
                graph_features["MeanCycleLength"] = (
                    sum(cycle_lengths) / len(cycle_lengths) if cycle_lengths else 0
                )
                graph_features["MaxCycleLength"] = (
                    max(cycle_lengths) if cycle_lengths else 0
                )

                # Graph diameter and radius (for connected graphs)
                if nx.is_connected(G):
                    graph_features["GraphDiameter"] = nx.diameter(G)
                    graph_features["GraphRadius"] = nx.radius(G)
                    # Eccentricity - distance from node to farthest node
                    eccentricity = nx.eccentricity(G)
                    graph_features["MeanEccentricity"] = sum(
                        eccentricity.values()
                    ) / len(eccentricity)
                else:
                    largest_cc = max(nx.connected_components(G), key=len)
                    subgraph = G.subgraph(largest_cc)
                    graph_features["GraphDiameter"] = nx.diameter(subgraph)
                    graph_features["GraphRadius"] = nx.radius(subgraph)
                    eccentricity = nx.eccentricity(subgraph)
                    graph_features["MeanEccentricity"] = sum(
                        eccentricity.values()
                    ) / len(eccentricity)

                # Clustering coefficient - measure of node clustering
                clustering = nx.clustering(G)
                graph_features["MeanClusteringCoeff"] = (
                    sum(clustering.values()) / len(clustering) if clustering else 0
                )

            # Graph spectral properties
            if len(G.nodes) > 1:
                try:
                    # Get adjacency matrix eigenvalues
                    A = nx.adjacency_matrix(G).todense()
                    eigenvalues = np.linalg.eigvals(A)
                    # Energy is sum of absolute eigenvalues
                    graph_features["GraphEnergy"] = sum(abs(eigenvalues))
                    # Spectral radius is largest eigenvalue
                    graph_features["SpectralRadius"] = max(abs(eigenvalues))
                    # HOMO-LUMO gap approximation from graph theory
                    sorted_eigenvalues = sorted(eigenvalues)
                    mid = len(sorted_eigenvalues) // 2
                    graph_features["EigenvalueFrontierGap"] = (
                        abs(sorted_eigenvalues[mid] - sorted_eigenvalues[mid - 1])
                        if len(sorted_eigenvalues) > 1
                        else 0
                    )
                except:
                    graph_features["GraphEnergy"] = 0
                    graph_features["SpectralRadius"] = 0
                    graph_features["EigenvalueFrontierGap"] = 0

        except Exception as e:
            # Default values if NetworkX calculations fail
            graph_features["MaxDegreeCentrality"] = 0
            graph_features["MeanDegreeCentrality"] = 0
            graph_features["MaxBetweennessCentrality"] = 0
            graph_features["MeanBetweennessCentrality"] = 0
            graph_features["MaxClosenessCentrality"] = 0
            graph_features["MeanClosenessCentrality"] = 0
            graph_features["GraphDensity"] = 0
            graph_features["ConnectedComponents"] = 1
            graph_features["NumCycles"] = 0
            graph_features["MeanCycleLength"] = 0
            graph_features["MaxCycleLength"] = 0
            graph_features["GraphDiameter"] = 0
            graph_features["GraphRadius"] = 0
            graph_features["MeanEccentricity"] = 0
            graph_features["MeanClusteringCoeff"] = 0
            graph_features["GraphEnergy"] = 0
            graph_features["SpectralRadius"] = 0
            graph_features["EigenvalueFrontierGap"] = 0
            graph_features["LargestComponentRatio"] = 1

        # 3. RING AND AROMATIC SYSTEM ANALYSIS
        ring_info = mol.GetRingInfo()
        ring_count = ring_info.NumRings()
        graph_features["NumRingSystems"] = ring_count

        # Analyze aromatic systems
        aromatic_rings = []
        aliphatic_rings = []

        # Get rings as atom indices
        for i in range(ring_count):
            ring_atoms = ring_info.AtomRings()[i]
            is_aromatic = all(
                mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_atoms
            )
            if is_aromatic:
                aromatic_rings.append(ring_atoms)
            else:
                aliphatic_rings.append(ring_atoms)

        graph_features["NumAromaticSystems"] = len(aromatic_rings)
        graph_features["NumAliphaticRings"] = len(aliphatic_rings)
        graph_features["FractionAromaticRings"] = (
            len(aromatic_rings) / ring_count if ring_count > 0 else 0
        )

        # Analyze fused ring systems
        try:
            # Calculate fusion ratio from existing ring data
            if ring_count > 1:
                # Flatten all ring atoms
                all_ring_atoms = [
                    atom for ring in ring_info.AtomRings() for atom in ring
                ]
                # Count unique ring atoms
                unique_ring_atoms = len(set(all_ring_atoms))
                total_ring_atoms = len(all_ring_atoms)
                graph_features["RingFusionRatio"] = (
                    (total_ring_atoms - unique_ring_atoms) / total_ring_atoms
                    if total_ring_atoms > 0
                    else 0
                )
            else:
                graph_features["RingFusionRatio"] = 0
        except:
            graph_features["RingFusionRatio"] = 0

        return graph_features

    def _get_global_features(self, row):
        """
        Extract global molecular features for a molecule.

        Args:
            row: DataFrame row containing molecule data and descriptors

        Returns:
            Dictionary of features
        """
        if row is None:
            return torch.zeros(1, dtype=torch.float)

        features_dict = {}

        # Get SMILES and create molecule object
        smiles = row[self.smiles_col]
        mol = Chem.MolFromSmiles(smiles)

        # Helper function to calculate descriptors safely
        def calc_descriptor(molecule, descriptor_fn, default=0.0):
            try:
                if molecule is None:
                    return default
                return descriptor_fn(molecule)
            except:
                return default

        # 1. BASIC MOLECULAR PROPERTIES
        features_dict["MolecularWeight"] = calc_descriptor(mol, Descriptors.MolWt)
        features_dict["HeavyAtomCount"] = mol.GetNumHeavyAtoms() if mol else 0
        features_dict["NumBonds"] = mol.GetNumBonds() if mol else 0

        # Get graph-based descriptors and add them to the features dictionary
        if mol is not None:
            graph_descriptors = self._get_graph_descriptors(mol)
            features_dict.update(graph_descriptors)

        # 2. ELECTRONIC PROPERTIES
        features_dict["LogP"] = calc_descriptor(mol, Descriptors.MolLogP)
        features_dict["TPSA"] = calc_descriptor(mol, Descriptors.TPSA)
        # Add MR (Molar Refractivity) - important for permeability
        features_dict["MR"] = calc_descriptor(mol, Descriptors.MolMR)
        # Add LabuteASA - Molecular surface area
        features_dict["LabuteASA"] = calc_descriptor(mol, Descriptors.LabuteASA)
        # Add PEOE and GASTEIGER charges - using individual computation
        features_dict["PEOE_VSA"] = calc_descriptor(
            mol, lambda m: sum(Descriptors.PEOE_VSA_(m)) if m else 0
        )
        features_dict["SMR_VSA"] = calc_descriptor(
            mol, lambda m: sum(Descriptors.SMR_VSA_(m)) if m else 0
        )
        features_dict["SlogP_VSA"] = calc_descriptor(
            mol, lambda m: sum(Descriptors.SlogP_VSA_(m)) if m else 0
        )

        # Additional electronic properties important for permeability
        features_dict["NumHDonors"] = calc_descriptor(mol, Descriptors.NumHDonors)
        features_dict["NumHAcceptors"] = calc_descriptor(mol, Descriptors.NumHAcceptors)
        features_dict["NumRotatableBonds"] = calc_descriptor(
            mol, Descriptors.NumRotatableBonds
        )
        features_dict["NumRings"] = calc_descriptor(mol, rdMolDescriptors.CalcNumRings)
        features_dict["NumAromaticRings"] = calc_descriptor(
            mol, rdMolDescriptors.CalcNumAromaticRings
        )
        features_dict["NumAliphaticRings"] = calc_descriptor(
            mol, rdMolDescriptors.CalcNumAliphaticRings
        )
        features_dict["NumHeteroatoms"] = calc_descriptor(mol, Lipinski.NumHeteroatoms)
        features_dict["NumSaturatedRings"] = calc_descriptor(
            mol, rdMolDescriptors.CalcNumSaturatedRings
        )
        features_dict["NumHeterocycles"] = calc_descriptor(
            mol, rdMolDescriptors.CalcNumHeterocycles
        )
        features_dict["NumAromaticHeterocycles"] = calc_descriptor(
            mol, rdMolDescriptors.CalcNumAromaticHeterocycles
        )
        features_dict["NumSaturatedHeterocycles"] = calc_descriptor(
            mol, rdMolDescriptors.CalcNumSaturatedHeterocycles
        )
        features_dict["NumAliphaticHeterocycles"] = calc_descriptor(
            mol, rdMolDescriptors.CalcNumAliphaticHeterocycles
        )

        # 3. CONSTITUTIONAL INDICES
        # Atom type counts (common elements in organic molecules)
        if mol:
            atoms = mol.GetAtoms()
            atom_counts = {}
            for atom in atoms:
                symbol = atom.GetSymbol()
                if symbol in atom_counts:
                    atom_counts[symbol] += 1
                else:
                    atom_counts[symbol] = 1

            # Add atom counts as features
            for symbol in ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]:
                features_dict[f"Num{symbol}"] = atom_counts.get(symbol, 0)
        else:
            for symbol in ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]:
                features_dict[f"Num{symbol}"] = 0

        # 4. DRUG-LIKENESS PARAMETERS - Compute directly with RDKit
        features_dict["QED"] = calc_descriptor(mol, QED.qed)
        features_dict["FractionCSP3"] = calc_descriptor(
            mol, rdMolDescriptors.CalcFractionCSP3
        )

        # Add RDKit molecular complexity
        features_dict["HallKierAlpha"] = calc_descriptor(mol, Descriptors.HallKierAlpha)

        # Drug-likeness scores beyond Lipinski
        features_dict["qed_weightedProperties"] = calc_descriptor(
            mol, lambda m: QED.properties(m)[0] if m else 0
        )
        features_dict["qed_PAINS"] = calc_descriptor(
            mol, lambda m: QED.properties(m)[1] if m else 0
        )
        features_dict["qed_Inorganic"] = calc_descriptor(
            mol, lambda m: QED.properties(m)[2] if m else 0
        )

        # Lipinski's Rule of 5 properties
        features_dict["Lipinski_Violations"] = 0
        if features_dict["MolecularWeight"] > 500:
            features_dict["Lipinski_Violations"] += 1
        if features_dict["LogP"] > 5:
            features_dict["Lipinski_Violations"] += 1
        if features_dict["NumHDonors"] > 5:
            features_dict["Lipinski_Violations"] += 1
        if features_dict["NumHAcceptors"] > 10:
            features_dict["Lipinski_Violations"] += 1

        # 5. TOPOLOGICAL INDICES - Compute directly with RDKit
        features_dict["Ipc"] = 0  # Remove CalcIpc as it's not available
        features_dict["Chi2v"] = calc_descriptor(
            mol, rdMolDescriptors.CalcChi2v
        )  # Connectivity index
        features_dict["Chi3v"] = calc_descriptor(
            mol, rdMolDescriptors.CalcChi3v
        )  # Connectivity index
        features_dict["Chi4v"] = calc_descriptor(
            mol, rdMolDescriptors.CalcChi4v
        )  # Connectivity index

        # Added 3D-based descriptors (if possible)
        if mol:
            try:
                mol_3d = Chem.AddHs(mol)
                # Try to compute 3D coordinates
                success = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                if success == 0:  # Successful embedding
                    # Add 3D descriptors
                    features_dict["PMI1"] = calc_descriptor(
                        mol_3d, rdMolDescriptors.CalcPMI1
                    )
                    features_dict["PMI2"] = calc_descriptor(
                        mol_3d, rdMolDescriptors.CalcPMI2
                    )
                    features_dict["PMI3"] = calc_descriptor(
                        mol_3d, rdMolDescriptors.CalcPMI3
                    )
                    features_dict["NPR1"] = calc_descriptor(
                        mol_3d, rdMolDescriptors.CalcNPR1
                    )
                    features_dict["NPR2"] = calc_descriptor(
                        mol_3d, rdMolDescriptors.CalcNPR2
                    )
                    features_dict["Asphericity"] = calc_descriptor(
                        mol_3d, rdMolDescriptors.CalcAsphericity
                    )
                    features_dict["Eccentricity"] = calc_descriptor(
                        mol_3d, rdMolDescriptors.CalcEccentricity
                    )
                    features_dict["InertialShapeFactor"] = calc_descriptor(
                        mol_3d, rdMolDescriptors.CalcInertialShapeFactor
                    )
                    features_dict["RadiusOfGyration"] = calc_descriptor(
                        mol_3d, rdMolDescriptors.CalcRadiusOfGyration
                    )
                else:
                    # Default values if 3D embedding fails
                    features_dict["PMI1"] = 0.0
                    features_dict["PMI2"] = 0.0
                    features_dict["PMI3"] = 0.0
                    features_dict["NPR1"] = 0.0
                    features_dict["NPR2"] = 0.0
                    features_dict["Asphericity"] = 0.0
                    features_dict["Eccentricity"] = 0.0
                    features_dict["InertialShapeFactor"] = 0.0
                    features_dict["RadiusOfGyration"] = 0.0
            except:
                # Default values if 3D embedding fails
                features_dict["PMI1"] = 0.0
                features_dict["PMI2"] = 0.0
                features_dict["PMI3"] = 0.0
                features_dict["NPR1"] = 0.0
                features_dict["NPR2"] = 0.0
                features_dict["Asphericity"] = 0.0
                features_dict["Eccentricity"] = 0.0
                features_dict["InertialShapeFactor"] = 0.0
                features_dict["RadiusOfGyration"] = 0.0
        else:
            # Default values if molecule is None
            features_dict["PMI1"] = 0.0
            features_dict["PMI2"] = 0.0
            features_dict["PMI3"] = 0.0
            features_dict["NPR1"] = 0.0
            features_dict["NPR2"] = 0.0
            features_dict["Asphericity"] = 0.0
            features_dict["Eccentricity"] = 0.0
            features_dict["InertialShapeFactor"] = 0.0
            features_dict["RadiusOfGyration"] = 0.0

        # 6. PAMPA-SPECIFIC FEATURES
        # Get PAMPA from properties if available, otherwise set to 0
        if "PAMPA" in row and pd.notna(row["PAMPA"]):
            pampa_value = row["PAMPA"]
        else:
            pampa_value = 0.0

        features_dict["PAMPA"] = pampa_value
        features_dict["PAMPA_Clipped"] = max(pampa_value, -10.0)
        features_dict["PAMPA_Exp"] = np.exp(pampa_value)

        # 7. CALCULATED RATIOS AND NORMALIZED FEATURES
        if features_dict["MolecularWeight"] > 0:
            features_dict["LogP_per_MW"] = (
                features_dict["LogP"] / features_dict["MolecularWeight"]
            )
            features_dict["TPSA_per_MW"] = (
                features_dict["TPSA"] / features_dict["MolecularWeight"]
            )
            # New ratio for permeability prediction
            features_dict["MR_per_MW"] = (
                features_dict["MR"] / features_dict["MolecularWeight"]
                if "MR" in features_dict
                else 0
            )
        else:
            features_dict["LogP_per_MW"] = 0
            features_dict["TPSA_per_MW"] = 0
            features_dict["MR_per_MW"] = 0

        if features_dict["HeavyAtomCount"] > 0:
            features_dict["Rings_per_HeavyAtom"] = (
                features_dict["NumRings"] / features_dict["HeavyAtomCount"]
            )
            features_dict["RotBonds_per_HeavyAtom"] = (
                features_dict["NumRotatableBonds"] / features_dict["HeavyAtomCount"]
            )
            # New for permeability prediction
            features_dict["LogP_per_HeavyAtom"] = (
                features_dict["LogP"] / features_dict["HeavyAtomCount"]
            )
            features_dict["TPSA_per_HeavyAtom"] = (
                features_dict["TPSA"] / features_dict["HeavyAtomCount"]
            )
        else:
            features_dict["Rings_per_HeavyAtom"] = 0
            features_dict["RotBonds_per_HeavyAtom"] = 0
            features_dict["LogP_per_HeavyAtom"] = 0
            features_dict["TPSA_per_HeavyAtom"] = 0

        # 8. FRAGMENT-BASED PROPERTIES
        if mol:
            # Count functional groups of interest - especially for permeability
            smarts_patterns = {
                "Amide": "[NX3][CX3](=[OX1])",
                "Amine": "[NX3;H2,H1,H0;!$(NC=O)]",
                "Carboxylic_Acid": "[CX3](=O)[OX2H1]",
                "Ester": "[#6][CX3](=O)[OX2H0][#6]",
                "Ether": "[OD2]([#6])[#6]",
                "Hydroxyl": "[OX2H]",
                "Sulfonamide": "[SX4](=[OX1])(=[OX1])([NX3])",
                "Sulfone": "[SX4](=[OX1])(=[OX1])([#6])[#6]",
                "Urea": "[NX3][CX3](=[OX1])[NX3]",
                # New patterns important for membrane interactions
                "Guanidine": "[NX3][CX3](=[NX2])[NX3]",
                "Phosphate": "[PX4](=[OX1])([OX2])",
                "Nitro": "[NX3](=[OX1])(=[OX1])",
                "Halogen": "[F,Cl,Br,I]",
                # Additional peptide-relevant groups
                "BetaLactam": "[C]1[C](=O)[N][C]1",
                "Macrocycle": "[r{8-}]",
                "Phenol": "[OH][c]",
                "Thiol": "[SH]",
                "Imidazole": "c1cnc[nH]1",
                "Indole": "c1ccc2c(c1)c[nH]c2",
            }

            for name, smarts in smarts_patterns.items():
                try:
                    patt = Chem.MolFromSmarts(smarts)
                    if patt:
                        matches = mol.GetSubstructMatches(patt)
                        features_dict[f"Has_{name}"] = 1 if matches else 0
                        features_dict[f"Count_{name}"] = len(matches)
                    else:
                        features_dict[f"Has_{name}"] = 0
                        features_dict[f"Count_{name}"] = 0
                except:
                    features_dict[f"Has_{name}"] = 0
                    features_dict[f"Count_{name}"] = 0
        else:
            for name in [
                "Amide",
                "Amine",
                "Carboxylic_Acid",
                "Ester",
                "Ether",
                "Hydroxyl",
                "Sulfonamide",
                "Sulfone",
                "Urea",
                "Guanidine",
                "Phosphate",
                "Nitro",
                "Halogen",
                "BetaLactam",
                "Macrocycle",
                "Phenol",
                "Thiol",
                "Imidazole",
                "Indole",
            ]:
                features_dict[f"Has_{name}"] = 0
                features_dict[f"Count_{name}"] = 0

        # 9. CYCLIC VS. ACYCLIC CHARACTER
        if mol and mol.GetNumAtoms() > 0:
            atoms_in_ring = sum(1 for a in mol.GetAtoms() if a.IsInRing())
            features_dict["Fraction_InRing"] = atoms_in_ring / mol.GetNumAtoms()
        else:
            features_dict["Fraction_InRing"] = 0

        # 10. PEPTIDE-SPECIFIC FEATURES
        if mol:
            # Check for cyclic structure
            is_cyclic = mol.GetRingInfo().NumRings() > 0
            features_dict["IsCyclic"] = 1 if is_cyclic else 0

            # Count peptide bonds
            patt = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[CX4]")
            if patt:
                matches = mol.GetSubstructMatches(patt)
                features_dict["NumPeptideBonds"] = len(matches)
            else:
                features_dict["NumPeptideBonds"] = 0

            # Number of stereocenters - important for 3D structure
            features_dict["NumStereocenters"] = calc_descriptor(
                mol, rdMolDescriptors.CalcNumAtomStereoCenters
            )
            features_dict["NumUnspecifiedStereocenters"] = calc_descriptor(
                mol, rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters
            )

            # Add acid and basic group counts (replace Lipinski.NumAcidGroups)
            features_dict["NumAcidGroups"] = 0
            acid_patterns = {
                "carboxylic_acid": "[CX3](=O)[OX2H1]",
                "sulfonic_acid": "[SX4](=[OX1])(=[OX1])[OX2H1]",
                "phosphonic_acid": "[PX4](=[OX1])([OX2H1])",
                "boronic_acid": "[BX3]([OX2H1])[OX2H1]",
            }
            for name, smarts in acid_patterns.items():
                patt = Chem.MolFromSmarts(smarts)
                if patt:
                    features_dict["NumAcidGroups"] += len(mol.GetSubstructMatches(patt))

            # Add basic group counts (replace Lipinski.NumBasicGroups)
            features_dict["NumBasicGroups"] = 0
            basic_patterns = {
                "amine": "[NX3;H2,H1,H0;!$(NC=O);!$(N=C)]",
                "guanidine": "[NX3][CX3](=[NX2])[NX3]",
                "imidazole": "c1cnc[nH]1",
            }
            for name, smarts in basic_patterns.items():
                patt = Chem.MolFromSmarts(smarts)
                if patt:
                    features_dict["NumBasicGroups"] += len(
                        mol.GetSubstructMatches(patt)
                    )
        else:
            features_dict["IsCyclic"] = 0
            features_dict["NumPeptideBonds"] = 0
            features_dict["NumStereocenters"] = 0
            features_dict["NumUnspecifiedStereocenters"] = 0
            features_dict["NumAcidGroups"] = 0
            features_dict["NumBasicGroups"] = 0

        # Convert dictionary to list in a consistent order
        feature_names = sorted(features_dict.keys())
        feature_values = [features_dict[name] for name in feature_names]

        # Store feature names for reference (first time only)
        if not hasattr(self, "global_feature_names"):
            self.global_feature_names = feature_names
            print(f"Global feature space dimension: {len(feature_names)}")
            print(f"Global features: {', '.join(feature_names)}")

        return torch.tensor(feature_values, dtype=torch.float)

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for batching graph data.

        Args:
            batch: List of samples from __getitem__

        Returns:
            dict: Batched data
        """
        batch_dict = {
            "x": [],
            "edge_index": [],
            "edge_attr": [],
            "global_features": [],
            "batch": [],
            "smiles": [],
        }

        if "properties" in batch[0]:
            batch_dict["properties"] = []

        cumulative_nodes = 0
        for i, sample in enumerate(batch):
            num_nodes = sample["num_nodes"]

            batch_dict["x"].append(sample["x"])
            batch_dict["edge_index"].append(sample["edge_index"] + cumulative_nodes)
            batch_dict["edge_attr"].append(sample["edge_attr"])
            batch_dict["global_features"].append(sample["global_features"])
            batch_dict["batch"].append(torch.full((num_nodes,), i, dtype=torch.long))
            batch_dict["smiles"].append(sample["smiles"])

            if "properties" in sample:
                batch_dict["properties"].append(sample["properties"])

            cumulative_nodes += num_nodes

        # Concatenate tensors
        batch_dict["x"] = torch.cat(batch_dict["x"], dim=0)
        batch_dict["edge_index"] = torch.cat(batch_dict["edge_index"], dim=1)
        batch_dict["edge_attr"] = torch.cat(batch_dict["edge_attr"], dim=0)
        batch_dict["batch"] = torch.cat(batch_dict["batch"], dim=0)
        batch_dict["global_features"] = torch.stack(
            batch_dict["global_features"], dim=0
        )

        if "properties" in batch_dict:
            batch_dict["properties"] = torch.stack(batch_dict["properties"], dim=0)

        return batch_dict
