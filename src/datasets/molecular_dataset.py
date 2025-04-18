"""
Module for PyTorch Geometric dataset implementation for molecular data.
"""

import torch
from torch_geometric.data import Data, Dataset
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class MolecularDataset(Dataset):
    """PyTorch Geometric dataset for molecular data."""

    def __init__(
        self,
        graph_features: List,
        custom_features: List[Dict[str, Any]],
        descriptor_features: np.ndarray,
        targets: np.ndarray,
        transform: Optional[callable] = None,
    ):
        """
        Initialize the dataset.

        Args:
            graph_features: List of graph features from DeepChem
            custom_features: List of dictionaries containing custom molecular features
            descriptor_features: Array of RDKit descriptor features
            targets: Array of target values
            transform: Optional transform to be applied to the data
        """
        super().__init__(transform=transform)
        self.graph_features = graph_features
        self.custom_features = custom_features
        self.descriptor_features = descriptor_features
        self.targets = targets

        # Convert custom features to tensor format
        self.process_custom_features()

    def process_custom_features(self):
        """Convert custom features dictionary to tensor format."""
        # Get all unique keys from custom features
        all_keys = set()
        for features in self.custom_features:
            all_keys.update(features.keys())

        # Convert to sorted list for consistent ordering
        self.feature_keys = sorted(all_keys)

        # Create tensor for custom features
        custom_features_list = []
        for features in self.custom_features:
            features_array = [features.get(key, 0.0) for key in self.feature_keys]
            custom_features_list.append(features_array)

        self.custom_features_tensor = torch.tensor(
            custom_features_list, dtype=torch.float32
        )

    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.graph_features)

    def get(self, idx: int) -> Data:
        """
        Get a single graph with its features.

        Args:
            idx: Index of the graph to get

        Returns:
            PyTorch Geometric Data object
        """
        # Get graph features
        graph_data = self.graph_features[idx]

        # Create PyG Data object
        data = Data(
            x=torch.tensor(graph_data.node_features, dtype=torch.float32),
            edge_index=torch.tensor(graph_data.edge_index, dtype=torch.long)
            .t()
            .contiguous(),
            edge_attr=(
                torch.tensor(graph_data.edge_features, dtype=torch.float32)
                if hasattr(graph_data, "edge_features")
                else None
            ),
        )

        # Add custom features
        data.custom_features = self.custom_features_tensor[idx]

        # Add descriptor features
        data.descriptor_features = torch.tensor(
            self.descriptor_features[idx], dtype=torch.float32
        )

        # Add target
        data.y = torch.tensor(self.targets[idx], dtype=torch.float32)

        return data
