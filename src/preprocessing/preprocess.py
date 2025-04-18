# src/preprocessing/preprocess.py

"""
Main module for preprocessing molecular data.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

from src.preprocessing.data_loader import DataLoader
from src.preprocessing.molecular_featurizer import MolecularFeaturizer
from src.preprocessing.dataset_splitter import DatasetSplitter
from src.datasets.molecular_dataset import MolecularDataset


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_datasets(
    config_path: str,
) -> Tuple[MolecularDataset, MolecularDataset, MolecularDataset]:
    """
    Create train, validation, and test datasets.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load configuration
    config = load_config(config_path)

    # Initialize components
    data_loader = DataLoader(config)
    featurizer = MolecularFeaturizer(config)
    splitter = DatasetSplitter(config)

    # Load data
    data = data_loader.load_data()
    smiles, targets = data_loader.get_features_and_targets()

    # Generate features
    features = featurizer.featurize(smiles)

    # Split dataset
    splits = splitter.split_dataset(features, targets)

    # Create PyTorch Geometric datasets
    datasets = {}
    for split_name, split_data in splits.items():
        datasets[split_name] = MolecularDataset(
            graph_features=split_data[0],
            custom_features=split_data[1],
            descriptor_features=split_data[2],
            targets=split_data[3],
        )

    return datasets["train"], datasets["val"], datasets["test"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess molecular data for the model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing_config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(args.config)

    print(f"Created datasets:")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
