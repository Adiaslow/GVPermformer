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

from src.utils.smiles_to_features import smiles_to_graph_data


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
    Features are computed only once during initialization from SMILES strings.
    """

    def __init__(
        self,
        csv_file: str,
        smiles_col: str = "smiles",
        target_col: str = "permeability",
        max_atoms: int = 100,
        filter_pampa: bool = True,
        pampa_threshold: float = -9.0,
        use_edge_features: bool = True,
        use_enhanced_features: bool = True,
        property_prediction: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            csv_file: Path to the CSV file containing SMILES and target values
            smiles_col: Name of the column containing SMILES strings
            target_col: Name of the column containing target values
            max_atoms: Maximum number of atoms allowed in a molecule
            filter_pampa: Whether to filter out entries with PAMPA values below threshold
            pampa_threshold: Threshold for filtering PAMPA values
            use_edge_features: Whether to use edge features in the graph
            use_enhanced_features: Whether to use enhanced node/edge features
            property_prediction: Whether this is a property prediction task
        """
        self.csv_file = csv_file
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.max_atoms = max_atoms
        self.filter_pampa = filter_pampa
        self.pampa_threshold = pampa_threshold
        self.use_edge_features = use_edge_features
        self.use_enhanced_features = use_enhanced_features
        self.property_prediction = property_prediction

        # Load data
        self.data = pd.read_csv(self.csv_file)

        # Filter by PAMPA threshold if requested
        if self.filter_pampa:
            print(f"Filtering out entries with PAMPA values below {pampa_threshold}")
            self.data = self.data[self.data["permeability"] >= pampa_threshold]

        # Filter out invalid molecules and those with too many atoms
        self._filter_invalid_molecules()

        # Extract required data
        self.smiles_list = self.data[self.smiles_col].tolist()

        # Store property values separately to ensure complete separation
        # from the molecular structure features
        self.property_values = None
        if self.property_prediction:
            self.property_values = self.data[self.target_col].values

        # Pre-compute features for all valid molecules - using ONLY SMILES strings
        # This ensures PAMPA values are not leaked into the input features
        self._precompute_features()

    def _filter_invalid_molecules(self):
        """Filter out invalid molecules from the dataset"""
        valid_indices = []
        print("Filtering out invalid or too large molecules...")

        for i, smiles in enumerate(self.data[self.smiles_col]):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and mol.GetNumAtoms() <= self.max_atoms:
                    valid_indices.append(i)
            except:
                # Skip any errors when processing molecules
                continue

        print(f"Filtered out {len(self.data) - len(valid_indices)} invalid molecules")
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        print(f"Remaining data size: {len(self.data)}")

    def _precompute_features(self):
        """
        Precompute molecular graph features for all molecules in the dataset.
        This is done once at initialization to avoid recomputing features during training.
        IMPORTANT: Only SMILES strings are used for feature generation to prevent data leakage.
        """
        print("Precomputing molecular features...")
        self.features = []

        try:
            # Import the parallel processing function
            from src.utils.smiles_to_features import batch_smiles_to_features_parallel

            # Use parallel processing for faster feature computation
            # Process all SMILES strings at once
            batch_features = batch_smiles_to_features_parallel(
                self.smiles_list,
                n_jobs=min(8, max(1, os.cpu_count() - 1)),  # Use at most 8 cores
                chunk_size=min(
                    100, len(self.smiles_list)
                ),  # Adjust chunk size based on data
                optimize_speed=True,  # Optimize for speed
            )

            if batch_features:
                # Split the batched features back into individual molecule features
                num_molecules = len(self.smiles_list)
                unique_batch_indices = torch.unique(batch_features["batch"])

                for idx in range(len(self.smiles_list)):
                    # If this molecule wasn't processed successfully, add an empty feature set
                    if idx not in unique_batch_indices:
                        print(f"Warning: Unable to process molecule {idx}")
                        self.features.append(
                            {
                                "node_features": torch.zeros(0, 142),
                                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                                "edge_attr": torch.zeros(0, 16),
                                "global_features": torch.zeros(1, 32),
                                "smiles": self.smiles_list[idx],
                            }
                        )
                        continue

                    # Extract features for this molecule from the batch
                    batch_mask = batch_features["batch"] == idx
                    edge_mask = (
                        batch_features["edge_batch"] == idx
                        if "edge_batch" in batch_features
                        else None
                    )

                    molecule_features = {
                        "node_features": batch_features["node_features"][batch_mask],
                        "edge_index": None,  # Will be set below
                        "edge_attr": None,  # Will be set below
                        "global_features": batch_features["global_features"][
                            idx : idx + 1
                        ],
                        "smiles": self.smiles_list[idx],
                    }

                    # Handle edge features
                    if (
                        edge_mask is not None
                        and batch_features["edge_attr"].shape[0] > 0
                    ):
                        # Get edges for this molecule
                        molecule_features["edge_attr"] = batch_features["edge_attr"][
                            edge_mask
                        ]

                        # Get corresponding edge indices and adjust to start from 0
                        edge_indices = batch_features["edge_index"][:, edge_mask]

                        # Renumber nodes to start from 0
                        node_indices = torch.nonzero(batch_mask).squeeze()
                        if len(node_indices.shape) == 0:  # Handle single node case
                            node_indices = node_indices.unsqueeze(0)

                        # Create mapping from original indices to [0, num_nodes-1]
                        idx_map = {
                            int(node_indices[i]): i for i in range(len(node_indices))
                        }

                        # Remap edge indices
                        new_edge_indices = torch.zeros_like(edge_indices)
                        for i in range(edge_indices.shape[1]):
                            new_edge_indices[0, i] = idx_map[int(edge_indices[0, i])]
                            new_edge_indices[1, i] = idx_map[int(edge_indices[1, i])]

                        molecule_features["edge_index"] = new_edge_indices
                    else:
                        # Create empty edge tensors if no edges
                        molecule_features["edge_index"] = torch.zeros(
                            2, 0, dtype=torch.long
                        )
                        molecule_features["edge_attr"] = torch.zeros(0, 16)

                    self.features.append(molecule_features)
            else:
                # Fallback to sequential processing if parallel processing fails
                print(
                    "Warning: Parallel processing failed, falling back to sequential processing"
                )
                for smiles in self.smiles_list:
                    # Convert SMILES to graph features - nothing else used as input
                    graph_data = smiles_to_graph_data(smiles)
                    self.features.append(graph_data)
        except Exception as e:
            print(f"Warning: Error during parallel feature computation: {e}")
            print("Falling back to sequential processing")
            # Fallback to sequential processing
            for smiles in self.smiles_list:
                try:
                    # Convert SMILES to graph features - nothing else used as input
                    graph_data = smiles_to_graph_data(smiles)
                    self.features.append(graph_data)
                except Exception as e:
                    print(f"Warning: Error processing molecule: {e}")
                    # Add empty feature set on error
                    self.features.append(
                        {
                            "node_features": torch.zeros(0, 142),
                            "edge_index": torch.zeros(2, 0, dtype=torch.long),
                            "edge_attr": torch.zeros(0, 16),
                            "global_features": torch.zeros(1, 32),
                            "smiles": smiles,
                        }
                    )

        print(f"Precomputed features for {len(self.features)} molecules")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.smiles_list)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing molecule graph features and target value
        """
        # Get precomputed features (derived ONLY from SMILES)
        sample = self.features[idx]

        # Add SMILES string
        sample["smiles"] = self.smiles_list[idx]

        # Add properties as target values
        # This is the ONLY place where PAMPA values are used,
        # and they're strictly kept as target values
        if self.property_values is not None:
            sample["target"] = torch.tensor(
                self.property_values[idx], dtype=torch.float
            )

        return sample

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for batching molecular data.

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched data dictionary
        """
        # Extract keys from first item
        keys = list(batch[0].keys())

        # Initialize batch dictionary
        batched = {}

        # Handle graph data fields specifically
        if "node_features" in keys:
            # Initialize graph-specific fields
            graph_batch = {
                "node_features": [],
                "edge_index": [],
                "edge_attr": [],
                "global_features": [],
                "batch": [],  # Node batch indices
                "edge_batch": [],  # Edge batch indices
            }

            node_offset = 0
            valid_graphs = 0

            for i, sample in enumerate(batch):
                # Get number of nodes in this graph
                num_nodes = sample["node_features"].shape[0]

                # Skip empty graphs
                if num_nodes == 0:
                    continue

                # Add node features
                graph_batch["node_features"].append(sample["node_features"])

                # Add edge indices with offset
                if sample["edge_index"].shape[1] > 0:
                    edge_index = sample["edge_index"].clone()
                    edge_index += node_offset
                    graph_batch["edge_index"].append(edge_index)

                    # Add edge attributes
                    graph_batch["edge_attr"].append(sample["edge_attr"])

                    # Add edge batch indices
                    num_edges = sample["edge_attr"].shape[0]
                    edge_batch = torch.full(
                        (num_edges,), valid_graphs, dtype=torch.long
                    )
                    graph_batch["edge_batch"].append(edge_batch)

                # Add global features
                if "global_features" in sample:
                    graph_batch["global_features"].append(sample["global_features"])

                # Add node batch indices
                batch_idx = torch.full((num_nodes,), valid_graphs, dtype=torch.long)
                graph_batch["batch"].append(batch_idx)

                # Update node offset and valid graph counter
                node_offset += num_nodes
                valid_graphs += 1

            # Concatenate tensors
            for key in graph_batch:
                if graph_batch[key]:
                    if key == "edge_index":
                        batched[key] = torch.cat(graph_batch[key], dim=1)
                    else:
                        try:
                            batched[key] = torch.cat(graph_batch[key], dim=0)
                        except:
                            # If concatenation fails, skip this key
                            print(f"Warning: Failed to concatenate {key}")
                            continue
                else:
                    # Empty tensors if no valid data
                    if key == "node_features":
                        batched[key] = torch.zeros((0, 142), dtype=torch.float)
                    elif key == "edge_index":
                        batched[key] = torch.zeros((2, 0), dtype=torch.long)
                    elif key == "edge_attr":
                        batched[key] = torch.zeros((0, 16), dtype=torch.float)
                    elif key == "global_features":
                        batched[key] = torch.zeros((0, 32), dtype=torch.float)
                    else:
                        batched[key] = torch.zeros((0,), dtype=torch.long)

        # Handle non-graph data (smiles, target, etc.)
        for key in keys:
            if key not in batched:
                if key == "smiles":
                    batched[key] = [sample[key] for sample in batch]
                elif key == "target":
                    try:
                        targets = [
                            sample[key] for sample in batch if sample[key] is not None
                        ]
                        if targets:
                            batched[key] = torch.stack(targets)
                        else:
                            batched[key] = torch.tensor([])
                    except:
                        print(f"Warning: Failed to process targets")
                        batched[key] = torch.tensor([])

        return batched
