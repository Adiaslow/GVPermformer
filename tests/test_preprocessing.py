# tests/test_preprocessing.py

"""
Tests for the molecular graph preprocessing module.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

from src.preprocessing import MolecularGraphPreprocessor


@pytest.fixture
def sample_data():
    """Create a temporary CSV file with sample data."""
    # Sample SMILES strings and PAMPA values
    data = {
        "SMILES": [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CC1=CC=C(C=C1)O",  # p-Cresol
            "C1=CC=NC=C1",  # Pyridine
            "CC(C)(C)C1=CC(=C(C(=C1)O)C(C)(C)C)C(C)(C)C",  # BHT
            "CC1=CC=CC=C1N",  # o-Toluidine
            "CC(=O)N",  # Acetamide
            "C1=CC=C(C=C1)O",  # Phenol
            "CC(=O)O",  # Acetic acid
        ],
        "PAMPA": [0.5, 0.7, 0.3, 0.9, 0.4, 0.6, 0.8, 0.2],
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        return f.name


@pytest.fixture
def preprocessor(sample_data):
    """Create a preprocessor instance with sample data."""
    return MolecularGraphPreprocessor(
        data_path=sample_data,
        smiles_column="SMILES",
        property_column="PAMPA",
        split_ratio=(0.5, 0.25, 0.25),  # Adjusted split ratio for small dataset
    )


def test_initialization(preprocessor):
    """Test preprocessor initialization."""
    assert preprocessor.smiles_column == "SMILES"
    assert preprocessor.property_column == "PAMPA"
    assert preprocessor.split_ratio == (0.5, 0.25, 0.25)
    assert len(preprocessor.df) == 8


def test_atom_features(preprocessor):
    """Test atom feature extraction."""
    mol = Chem.MolFromSmiles("CC")  # Ethane
    atom = mol.GetAtomWithIdx(0)  # First carbon atom
    features = preprocessor.get_atom_features(atom)

    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert len(features) > 0


def test_bond_features(preprocessor):
    """Test bond feature extraction."""
    mol = Chem.MolFromSmiles("C=C")  # Ethene
    bond = mol.GetBondWithIdx(0)  # Double bond
    features = preprocessor.get_bond_features(bond)

    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert len(features) > 0


def test_smiles_to_graph(preprocessor):
    """Test conversion of SMILES to graph data."""
    smiles = "CC(=O)O"  # Acetic acid
    property_value = 0.5
    data = preprocessor.smiles_to_graph(smiles, property_value)

    assert data is not None
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.edge_index, torch.Tensor)
    assert isinstance(data.edge_attr, torch.Tensor)
    assert isinstance(data.y, torch.Tensor)
    assert data.y.item() == property_value


def test_smiles_to_graph_from_df(preprocessor):
    """Test conversion of SMILES to graph data using value from DataFrame."""
    smiles = preprocessor.df["SMILES"].iloc[0]
    data = preprocessor.smiles_to_graph(smiles)  # No property value provided

    assert data is not None
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.edge_index, torch.Tensor)
    assert isinstance(data.edge_attr, torch.Tensor)
    assert isinstance(data.y, torch.Tensor)
    assert data.y.item() == preprocessor.df["PAMPA"].iloc[0]


def test_invalid_smiles(preprocessor):
    """Test handling of invalid SMILES strings."""
    smiles = "invalid_smiles"
    property_value = 0.5
    data = preprocessor.smiles_to_graph(smiles, property_value)
    assert data is None


def test_process_dataset(preprocessor, tmp_path):
    """Test dataset processing and splitting."""
    train_data, val_data, test_data = preprocessor.process_dataset(output_dir=tmp_path)

    # Check that data was split correctly
    total_samples = len(train_data) + len(val_data) + len(test_data)
    assert total_samples > 0
    assert len(train_data) >= 3  # At least 3 samples in training
    assert len(val_data) >= 1  # At least 1 sample in validation
    assert len(test_data) >= 1  # At least 1 sample in test
    assert total_samples == 8  # Total number of valid molecules

    # Check that files were saved
    assert (tmp_path / "train_data.pt").exists()
    assert (tmp_path / "val_data.pt").exists()
    assert (tmp_path / "test_data.pt").exists()
    assert (tmp_path / "property_distribution.png").exists()
    assert (tmp_path / "molecule_size_distribution.png").exists()


def test_visualization(preprocessor, tmp_path):
    """Test visualization functions."""
    # Test property distribution plot
    preprocessor.plot_property_distribution(
        preprocessor.property_column, save_path=tmp_path / "property_dist.png"
    )
    assert (tmp_path / "property_dist.png").exists()

    # Test molecule size distribution plot
    preprocessor.plot_molecule_size_distribution(tmp_path / "size_dist.png")
    assert (tmp_path / "size_dist.png").exists()

    # Test molecule visualization
    smiles = preprocessor.df["SMILES"].iloc[0]
    preprocessor.visualize_molecule(smiles, tmp_path / "molecule.png")
    assert (tmp_path / "molecule.png").exists()


def test_data_consistency(preprocessor, tmp_path):
    """Test consistency of processed data."""
    train_data, val_data, test_data = preprocessor.process_dataset(output_dir=tmp_path)

    # Test a sample from each split
    for dataset in [train_data, val_data, test_data]:
        if len(dataset) > 0:
            sample = dataset[0]
            assert hasattr(sample, "x")
            assert hasattr(sample, "edge_index")
            assert hasattr(sample, "edge_attr")
            assert hasattr(sample, "y")
            assert sample.x.dim() == 2  # [num_nodes, num_features]
            assert sample.edge_index.dim() == 2  # [2, num_edges]
            assert sample.edge_attr.dim() == 2  # [num_edges, num_features]
            assert sample.y.dim() == 1  # [1]
