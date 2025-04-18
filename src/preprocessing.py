# src/preprocessing.py

"""
Preprocessing module for molecular graph data with visualization capabilities.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from numpy.typing import NDArray
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from torch_geometric.data import Data
else:
    try:
        from torch_geometric.data import Data
    except ImportError:
        Data = object  # type: ignore
        print("Warning: torch_geometric not found. Some functionality may be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
AtomFeatures = NDArray[np.float32]
BondFeatures = NDArray[np.float32]
MoleculeGraph = Data

# Constants for atom/bond features
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),  # All possible atomic numbers
    "degree": [0, 1, 2, 3, 4, 5, 6],
    "formal_charge": [-3, -2, -1, 0, 1, 2, 3],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "stereo": [0, 1, 2, 3, 4, 5],
    "is_conjugated": [False, True],
}


class MolecularGraphPreprocessor:
    """Class for preprocessing molecular data into graph format."""

    def __init__(
        self,
        smiles_column: str = "SMILES",
        property_column: str = "PAMPA",
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        delimiter: str = ",",
    ):
        """Initialize the preprocessor.

        Args:
            smiles_column: Name of the column containing SMILES strings
            property_column: Name of the column containing property values
            split_ratio: Tuple of (train, validation, test) split ratios
            delimiter: Delimiter used in the CSV file
        """
        self.smiles_column = smiles_column
        self.property_column = property_column
        self.split_ratio = split_ratio
        self.delimiter = delimiter
        self.df = None
        self.logger = logging.getLogger(__name__)

        # Initialize scalers for feature normalization
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.global_scaler = StandardScaler()

        # Track feature dimensions
        self.feature_dims = {
            "node_feature_dim": None,
            "edge_feature_dim": None,
            "num_node_types": len(ATOM_FEATURES["atomic_num"]),
            "num_edge_types": len(BOND_FEATURES["bond_type"]),
        }

        # Validate split ratio
        if not isinstance(split_ratio, tuple) or len(split_ratio) != 3:
            raise ValueError("split_ratio must be a tuple of three floats")
        if not all(isinstance(r, float) for r in split_ratio):
            raise ValueError("split_ratio values must be floats")
        if not np.isclose(sum(split_ratio), 1.0):
            raise ValueError("split_ratio values must sum to 1.0")
        if not all(0 <= r <= 1 for r in split_ratio):
            raise ValueError("split_ratio values must be between 0 and 1")

    def process_dataset(self, input_path: str, output_dir: Path) -> None:
        """Process the entire dataset and save train/val/test splits.

        Args:
            input_path: Path to input CSV file
            output_dir: Directory to save processed data
        """
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Data file not found: {input_path}")

        # Read dataset
        try:
            self.df = pd.read_csv(input_path, delimiter=self.delimiter)
            if self.df is None or self.df.empty:
                raise ValueError("Empty DataFrame")
        except Exception as e:
            self.logger.error(f"Error reading data file: {e}")
            raise

        # Validate required columns
        if self.smiles_column not in self.df.columns:
            raise ValueError(f"SMILES column '{self.smiles_column}' not found")
        if self.property_column not in self.df.columns:
            raise ValueError(f"Property column '{self.property_column}' not found")

        # Process each molecule and collect features for normalization
        processed_data = []
        node_features_list = []
        edge_features_list = []
        global_features_list = []

        for _, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Processing molecules"
        ):
            smiles = row[self.smiles_column]
            property_value = row[self.property_column]

            # Skip invalid property values
            if pd.isna(property_value) or property_value <= -10:
                continue

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    self.logger.warning(f"Failed to parse SMILES: {smiles}")
                    continue

                data = self.smiles_to_graph(smiles, property_value)
                if data is not None:
                    processed_data.append(data)
                    # Collect features for normalization
                    node_features_list.append(data.x.numpy())
                    if data.edge_attr is not None:
                        edge_features_list.append(data.edge_attr.numpy())
                    if hasattr(data, "global_features"):
                        global_features_list.append(data.global_features.numpy())

            except Exception as e:
                self.logger.warning(f"Error processing {smiles}: {str(e)}")
                continue

        self.logger.info(f"Successfully processed {len(processed_data)} molecules")

        # Update feature dimensions
        if processed_data:
            self.feature_dims.update(
                {
                    "node_feature_dim": processed_data[0].x.shape[1],
                    "edge_feature_dim": (
                        processed_data[0].edge_attr.shape[1]
                        if processed_data[0].edge_attr is not None
                        else 0
                    ),
                    "global_feature_dim": (
                        processed_data[0].global_features.shape[0]
                        if hasattr(processed_data[0], "global_features")
                        else 0
                    ),
                }
            )

        # Fit scalers on training data
        if node_features_list:
            self.node_scaler.fit(np.vstack(node_features_list))
        if edge_features_list:
            self.edge_scaler.fit(np.vstack(edge_features_list))
        if global_features_list:
            self.global_scaler.fit(np.vstack(global_features_list))

        # Create stratified splits
        y = np.array([data.y.item() for data in processed_data])

        # Create bins for stratification using quantiles to ensure equal number of samples per bin
        n_bins = 5  # Reduced number of bins to ensure enough samples per bin
        bins = pd.qcut(y, q=n_bins, duplicates="drop")
        y_binned = bins.codes  # Get the bin indices

        # Log distribution of bins
        bin_counts = pd.Series(y_binned).value_counts().sort_index()
        self.logger.info("\nPAMPA value bin distribution:")
        for bin_idx, count in bin_counts.items():
            bin_range = bins.categories[bin_idx]
            self.logger.info(
                f"Bin {bin_idx} ({bin_range.left:.2f} to {bin_range.right:.2f}): {count} samples"
            )

        # First split: train vs (val+test)
        train_ratio, val_ratio, test_ratio = self.split_ratio
        temp_ratio = val_ratio + test_ratio

        X_train, X_temp, y_train, y_temp = train_test_split(
            processed_data,
            y_binned,
            train_size=train_ratio,
            stratify=y_binned,
            random_state=42,
        )

        # Second split: val vs test
        val_size = val_ratio / temp_ratio
        X_val, X_test, _, _ = train_test_split(
            X_temp,
            y_temp,
            train_size=val_size,
            stratify=y_temp,
            random_state=42,
        )

        self.logger.info(f"\nTrain set: {len(X_train)} molecules")
        self.logger.info(f"Validation set: {len(X_val)} molecules")
        self.logger.info(f"Test set: {len(X_test)} molecules")

        # Save feature statistics and dimensions
        stats = {
            "n_total": len(processed_data),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "property_mean": float(np.mean(y)),
            "property_std": float(np.std(y)),
            "property_min": float(np.min(y)),
            "property_max": float(np.max(y)),
            "feature_dimensions": self.feature_dims,
            "feature_statistics": {
                "node_features": {
                    "mean": (
                        self.node_scaler.mean_.tolist() if node_features_list else None
                    ),
                    "scale": (
                        self.node_scaler.scale_.tolist() if node_features_list else None
                    ),
                },
                "edge_features": (
                    {
                        "mean": (
                            self.edge_scaler.mean_.tolist()
                            if edge_features_list
                            else None
                        ),
                        "scale": (
                            self.edge_scaler.scale_.tolist()
                            if edge_features_list
                            else None
                        ),
                    }
                    if edge_features_list
                    else None
                ),
                "global_features": (
                    {
                        "mean": (
                            self.global_scaler.mean_.tolist()
                            if global_features_list
                            else None
                        ),
                        "scale": (
                            self.global_scaler.scale_.tolist()
                            if global_features_list
                            else None
                        ),
                    }
                    if global_features_list
                    else None
                ),
            },
        }

        # Save processed data and statistics
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(X_train, output_dir / "train_data.pt")
        torch.save(X_val, output_dir / "val_data.pt")
        torch.save(X_test, output_dir / "test_data.pt")

        with open(output_dir / "preprocessing_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Save scalers only if they have been fitted
        scalers_dict = {}
        if node_features_list:
            scalers_dict["node_scaler"] = self.node_scaler
        if edge_features_list:
            scalers_dict["edge_scaler"] = self.edge_scaler
        if global_features_list:
            scalers_dict["global_scaler"] = self.global_scaler

        if scalers_dict:
            torch.save(scalers_dict, output_dir / "feature_scalers.pt")

        self.logger.info(f"Successfully processed {len(processed_data)} molecules")
        self.logger.info(f"Feature dimensions: {self.feature_dims}")
        self.logger.info(
            f"Saved processed data, statistics, and scalers to {output_dir}"
        )

    def plot_property_distribution(
        self,
        property_column: str,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
    ) -> None:
        """Plot the distribution of a molecular property.

        Args:
            property_column: Column name of the property to plot
            save_path: Path to save the plot (optional)
            title: Title for the plot (optional)
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=property_column, bins=50)
        plt.title(f"Distribution of {property_column} Values")
        plt.xlabel(property_column)
        plt.ylabel("Count")

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved property distribution plot to {save_path}")
        plt.close()

    def plot_molecule_size_distribution(
        self,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot the distribution of molecule sizes in the dataset.

        Args:
            save_path: Path to save the plot (optional)
        """
        sizes = []
        for smiles in self.df[self.smiles_column]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                sizes.append(mol.GetNumAtoms())

        plt.figure(figsize=(10, 6))
        sns.histplot(sizes, bins=50)
        plt.title("Distribution of Molecule Sizes")
        plt.xlabel("Number of Atoms")
        plt.ylabel("Count")

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved molecule size distribution plot to {save_path}")
        plt.close()

    def visualize_molecule(
        self,
        smiles: str,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Visualize a molecule from its SMILES string.

        Args:
            smiles: SMILES string of the molecule
            save_path: Path to save the visualization (optional)
        """
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            logger.warning(f"Could not parse molecule")
            return

        img = Draw.MolToImage(mol)

        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved molecule visualization to {save_path}")
        plt.close()

    def get_atom_features(self, atom: Chem.Atom) -> AtomFeatures:
        """Extract features for a given atom.

        Args:
            atom: RDKit atom object

        Returns:
            Array of atom features
        """
        features = []

        # Add atomic number (one-hot)
        atomic_num = atom.GetAtomicNum()
        atomic_num_enc = [int(atomic_num == x) for x in ATOM_FEATURES["atomic_num"]]
        features.extend(atomic_num_enc)

        # Add degree (one-hot)
        degree = atom.GetDegree()
        degree_enc = [int(degree == x) for x in ATOM_FEATURES["degree"]]
        features.extend(degree_enc)

        # Add formal charge (one-hot)
        formal_charge = atom.GetFormalCharge()
        formal_charge_enc = [
            int(formal_charge == x) for x in ATOM_FEATURES["formal_charge"]
        ]
        features.extend(formal_charge_enc)

        # Add chirality (one-hot)
        chiral_tag = int(atom.GetChiralTag())
        chiral_enc = [int(chiral_tag == x) for x in ATOM_FEATURES["chiral_tag"]]
        features.extend(chiral_enc)

        # Add number of hydrogens (one-hot)
        num_h = atom.GetTotalNumHs()
        num_h_enc = [int(num_h == x) for x in ATOM_FEATURES["num_Hs"]]
        features.extend(num_h_enc)

        # Add hybridization (one-hot)
        hybridization = atom.GetHybridization()
        hybrid_enc = [int(hybridization == x) for x in ATOM_FEATURES["hybridization"]]
        features.extend(hybrid_enc)

        # Add boolean features
        features.append(int(atom.IsInRing()))
        features.append(int(atom.GetIsAromatic()))

        return np.array(features, dtype=np.float32)

    def get_bond_features(self, bond: Chem.Bond) -> BondFeatures:
        """Extract features for a given bond.

        Args:
            bond: RDKit bond object

        Returns:
            Array of bond features
        """
        features = []

        # Add bond type (one-hot)
        bond_type = bond.GetBondType()
        bond_type_enc = [int(bond_type == x) for x in BOND_FEATURES["bond_type"]]
        features.extend(bond_type_enc)

        # Add stereochemistry (one-hot)
        stereo = int(bond.GetStereo())
        stereo_enc = [int(stereo == x) for x in BOND_FEATURES["stereo"]]
        features.extend(stereo_enc)

        # Add conjugation
        features.append(int(bond.GetIsConjugated()))

        return np.array(features, dtype=np.float32)

    def smiles_to_graph(
        self, smiles: str, property_value: Optional[float] = None
    ) -> Optional[Data]:
        """Convert a SMILES string to a PyG Data object.

        Args:
            smiles: SMILES string of the molecule
            property_value: Property value for the molecule

        Returns:
            PyG Data object or None if conversion fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"Failed to parse SMILES: {smiles}")
                return None

            # Get property value from DataFrame if not provided
            if property_value is None:
                if self.df is not None and not self.df.empty:
                    mask = self.df[self.smiles_column] == smiles
                    if mask.any():
                        property_value = float(
                            self.df.loc[mask, self.property_column].iloc[0]
                        )
                    else:
                        self.logger.warning(f"SMILES {smiles} not found in DataFrame")
                        return None
                else:
                    self.logger.warning("DataFrame not loaded or empty")
                    return None

            # Get atom features
            num_atoms = mol.GetNumAtoms()
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append(self.get_atom_features(atom))
            atom_features = np.array(atom_features)

            # Normalize node features if scaler is fitted
            if hasattr(self.node_scaler, "mean_"):
                atom_features = self.node_scaler.transform(atom_features)
            atom_features = torch.tensor(atom_features, dtype=torch.float)

            # Get bond features and edge indices
            edge_indices = []
            edge_features = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices += [[i, j], [j, i]]
                edge_features += [self.get_bond_features(bond)] * 2

            if edge_features:
                edge_features = np.array(edge_features)
                # Normalize edge features if scaler is fitted
                if hasattr(self.edge_scaler, "mean_"):
                    edge_features = self.edge_scaler.transform(edge_features)
                edge_features = torch.tensor(edge_features, dtype=torch.float)
                edge_indices = (
                    torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                )
            else:
                edge_features = None
                edge_indices = torch.empty((2, 0), dtype=torch.long)

            # Create the Data object
            data = Data(
                x=atom_features,
                edge_index=edge_indices,
                edge_attr=edge_features,
                y=torch.tensor([property_value], dtype=torch.float),
                num_nodes=num_atoms,
            )

            return data

        except Exception as e:
            self.logger.warning(f"Error processing molecule {smiles}: {str(e)}")
            return None
