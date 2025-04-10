"""
DataModule for handling molecular graph data in PyTorch Lightning.
"""

import os
import torch
from typing import Optional, Dict, List, Tuple
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.data.dataset import MoleculeDataset


class MoleculeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for molecular graph data.

    Handles data loading, splitting, and preparation of dataloaders
    for training, validation, and testing.
    """

    def __init__(
        self,
        data_path: str,
        smiles_col: str = "smiles",
        property_cols: List[str] = None,
        batch_size: int = 32,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        num_workers: int = 4,
        max_atoms: int = 50,
        seed: int = 42,
        filter_pampa: bool = True,
        pampa_threshold: float = -9.0,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
    ):
        """
        Initialize the MoleculeDataModule.

        Args:
            data_path: Path to CSV file with SMILES and properties
            smiles_col: Name of column containing SMILES strings
            property_cols: List of column names for properties to predict
            batch_size: Batch size for dataloaders
            train_val_test_split: Proportions for train/val/test split
            num_workers: Number of worker processes for data loading
            max_atoms: Maximum number of atoms to consider in molecules
            seed: Random seed for reproducibility
            filter_pampa: Whether to filter out entries with PAMPA below threshold
            pampa_threshold: Threshold for filtering PAMPA values
            pin_memory: Whether to use pinned memory for faster CPU to GPU transfers
            prefetch_factor: Number of batches to prefetch (if num_workers > 0)
        """
        super().__init__()
        self.data_path = data_path
        self.smiles_col = smiles_col
        self.property_cols = property_cols or []
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.max_atoms = max_atoms
        self.seed = seed
        self.filter_pampa = filter_pampa
        self.pampa_threshold = pampa_threshold
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        # Check that split proportions sum to 1
        assert sum(train_val_test_split) == 1.0, "Split proportions must sum to 1"

        # Initialize datasets to None
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Download or prepare the data if needed.
        This method is called once per node.
        """
        # Check if file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage (fit, test, predict).
        This method is called on every GPU.

        Args:
            stage: Stage to set up ('fit', 'test', 'predict', or None)
        """
        # Create the dataset if it doesn't exist
        if self.dataset is None:
            self.dataset = MoleculeDataset(
                csv_file=self.data_path,
                smiles_col=self.smiles_col,
                property_cols=self.property_cols,
                max_atoms=self.max_atoms,
                filter_pampa=self.filter_pampa,
                pampa_threshold=self.pampa_threshold,
            )

        # Split the dataset if not already done
        if (
            self.train_dataset is None
            and self.val_dataset is None
            and self.test_dataset is None
        ):
            self._split_dataset()

    def _split_dataset(self):
        """
        Split the dataset into train, validation, and test sets.
        """
        dataset_size = len(self.dataset)
        train_size = int(self.train_val_test_split[0] * dataset_size)
        val_size = int(self.train_val_test_split[1] * dataset_size)
        test_size = dataset_size - train_size - val_size

        # Set generator for reproducibility
        generator = torch.Generator().manual_seed(self.seed)

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size], generator=generator
        )

        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

    def train_dataloader(self):
        """
        Create dataloader for training data.

        Returns:
            DataLoader for training
        """
        pin_memory = self.pin_memory if torch.backends.mps.is_available() else False
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_fn,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=prefetch_factor,
        )

    def val_dataloader(self):
        """
        Create dataloader for validation data.

        Returns:
            DataLoader for validation
        """
        pin_memory = self.pin_memory if torch.backends.mps.is_available() else False
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=prefetch_factor,
        )

    def test_dataloader(self):
        """
        Create dataloader for test data.

        Returns:
            DataLoader for testing
        """
        pin_memory = self.pin_memory if torch.backends.mps.is_available() else False
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=prefetch_factor,
        )

    def get_node_features(self):
        """
        Get the number of node features.

        Returns:
            int: Number of node features
        """
        # Return enhanced feature dimension (126 features per node)
        return 126

    def get_edge_features(self):
        """
        Get the number of edge features.

        Returns:
            int: Number of edge features
        """
        # Return enhanced edge feature dimension (9 features per edge)
        return 9

    def get_global_features(self):
        """
        Get the number of global molecular features.

        Returns:
            int: Number of global features
        """
        # Create a temporary dataset to determine the global feature dimension
        if not hasattr(self, "global_feature_dim"):
            if self.dataset is None:
                # Setup dataset if not already done
                self.dataset = MoleculeDataset(
                    csv_file=self.data_path,
                    smiles_col=self.smiles_col,
                    property_cols=self.property_cols,
                    max_atoms=self.max_atoms,
                    filter_pampa=self.filter_pampa,
                    pampa_threshold=self.pampa_threshold,
                )

            # Get the first molecule to determine feature dimension
            sample = self.dataset[0]
            global_features = sample["global_features"]
            self.global_feature_dim = global_features.shape[0]
            print(f"Dynamic global feature dimension: {self.global_feature_dim}")

        return self.global_feature_dim

    def get_num_properties(self):
        """
        Get the number of properties to predict.

        Returns:
            int: Number of properties
        """
        return len(self.property_cols)
