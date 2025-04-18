"""
Module for splitting datasets into train, validation, and test sets.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class DatasetSplitter:
    """Class for splitting and preprocessing datasets."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the splitter with configuration.

        Args:
            config: Dictionary containing splitting configuration
        """
        self.config = config
        self._setup_scaler()

    def _setup_scaler(self):
        """Set up the feature scaler based on configuration."""
        scale_method = self.config["preprocessing"]["scale_method"]
        if scale_method == "standard":
            self.scaler = StandardScaler()
        elif scale_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scale_method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scale_method}")

    def split_dataset(
        self,
        features: Tuple[List, List[Dict[str, Any]], np.ndarray],
        targets: np.ndarray,
    ) -> Dict[str, Tuple]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            features: Tuple of (graph features, custom features, descriptor features)
            targets: Array of target values

        Returns:
            Dictionary containing split datasets
        """
        split_config = self.config["splits"]

        # First split: train + validation vs test
        temp_size = split_config["val_size"] + split_config["train_size"]

        # Unpack features
        graph_features, custom_features, descriptor_features = features

        if split_config["stratify"]:
            # Create bins for stratification
            target_bins = np.digitize(
                targets, bins=np.linspace(targets.min(), targets.max(), 10)
            )
            stratify = target_bins
        else:
            stratify = None

        # First split
        (
            train_val_graph,
            test_graph,
            train_val_custom,
            test_custom,
            train_val_desc,
            test_desc,
            train_val_targets,
            test_targets,
        ) = train_test_split(
            graph_features,
            custom_features,
            descriptor_features,
            targets,
            test_size=split_config["test_size"],
            random_state=split_config["random_state"],
            stratify=stratify,
        )

        # Second split: train vs validation
        if split_config["stratify"]:
            target_bins = np.digitize(
                train_val_targets,
                bins=np.linspace(train_val_targets.min(), train_val_targets.max(), 10),
            )
            stratify = target_bins

        val_size = split_config["val_size"] / temp_size

        (
            train_graph,
            val_graph,
            train_custom,
            val_custom,
            train_desc,
            val_desc,
            train_targets,
            val_targets,
        ) = train_test_split(
            train_val_graph,
            train_val_custom,
            train_val_desc,
            train_val_targets,
            test_size=val_size,
            random_state=split_config["random_state"],
            stratify=stratify,
        )

        # Scale descriptor features if configured
        if self.config["preprocessing"]["normalize_features"]:
            train_desc = self.scaler.fit_transform(train_desc)
            val_desc = self.scaler.transform(val_desc)
            test_desc = self.scaler.transform(test_desc)

        return {
            "train": (train_graph, train_custom, train_desc, train_targets),
            "val": (val_graph, val_custom, val_desc, val_targets),
            "test": (test_graph, test_custom, test_desc, test_targets),
        }
