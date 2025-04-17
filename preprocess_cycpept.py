#!/usr/bin/env python
# preprocess_cycpept.py

"""
This script preprocesses the cyclic peptide dataset by:
1. Loading the CSV file containing SMILES and permeability data
2. Filtering by PAMPA threshold and removing duplicates
3. Computing molecular features using RDKit descriptors
4. Splitting into train, test, val subsets with k-fold stratification
5. Creating visualizations of property distributions
6. Saving processed PyTorch Geometric datasets
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, GraphDescriptors  # type: ignore
from rdkit.Chem.Descriptors import rdMolDescriptors  # type: ignore
from rdkit.Chem import EState  # type: ignore
from rdkit.Chem.EState import EState_VSA  # type: ignore
from rdkit.Chem import Fragments  # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    Union,
    TypeVar,
    Sequence,
    cast,
    Protocol,
    Generic,
    Callable,
)
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data  # type: ignore
from numpy.typing import ArrayLike, NDArray
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_regression,
)
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem.rdchem import Mol
from sklearn.impute import SimpleImputer

# Type definitions
T = TypeVar("T")
R = TypeVar("R")
MolFeatures = Dict[str, Union[float, int]]
ArrayLike = Union[np.ndarray, pd.Series]
MolList = List[Optional[Mol]]
FeatureDF = pd.DataFrame

# Column names as constants
SMILES_COL: str = "SMILES"
PERMEABILITY_COL: str = "Permeability"

# Suppress RDKit warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Key molecular properties for stratification and visualization
KEY_PROPERTIES = [
    "ExactMolWt",  # Molecular Weight
    "MolLogP",  # LogP
    "TPSA",  # Topological Polar Surface Area
    "NumRotatableBonds",  # Number of Rotatable Bonds
]


class FixtureFunction(Generic[T, R]):
    def __call__(self, arg: T) -> R: ...


class MolProcessor(Protocol):
    """Protocol for molecule processing functions."""

    def __call__(self, mol: Optional[Mol]) -> Any: ...


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate the input CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = [SMILES_COL, PERMEABILITY_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def filter_data(df: pd.DataFrame, pampa_threshold: float = -9.0) -> pd.DataFrame:
    """Filter dataset by PAMPA threshold and remove duplicates."""
    print(f"Initial dataset size: {len(df)}")

    # Remove rows with missing values
    df = df.dropna(subset=[SMILES_COL, PERMEABILITY_COL])
    print(f"After removing missing values: {len(df)}")

    # Convert permeability to numeric, dropping any non-numeric values
    df[PERMEABILITY_COL] = pd.to_numeric(df[PERMEABILITY_COL], errors="coerce")
    df = df.dropna(subset=[PERMEABILITY_COL])
    print(f"After converting permeability to numeric: {len(df)}")

    # Filter by PAMPA threshold
    df = df[df[PERMEABILITY_COL] > pampa_threshold]
    print(f"After filtering PAMPA values below {pampa_threshold}: {len(df)}")

    # Remove duplicates based on SMILES
    df = df.drop_duplicates(subset=[SMILES_COL])
    print(f"After removing duplicates: {len(df)}")

    # Reset index after filtering
    df = df.reset_index(drop=True)

    return df


def compute_molecular_features(mol: Chem.Mol) -> Optional[Dict[str, float]]:
    """Compute essential molecular descriptors for a molecule using RDKit.

    Focuses on key physicochemical and topological properties that are most
    relevant for drug-like molecules and peptides.
    """
    try:
        features = {}

        # Constitutional descriptors
        features.update(
            {
                "ExactMolWt": Descriptors.ExactMolWt(mol),  # type: ignore
                "FractionCSP3": Descriptors.FractionCSP3(mol),  # type: ignore
                "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),  # type: ignore
                "NHOHCount": Descriptors.NHOHCount(mol),  # type: ignore
                "NOCount": Descriptors.NOCount(mol),  # type: ignore
                "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),  # type: ignore
                "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),  # type: ignore
                "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),  # type: ignore
                "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),  # type: ignore
                "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),  # type: ignore
                "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),  # type: ignore
                "RingCount": Descriptors.RingCount(mol),  # type: ignore
            }
        )

        # Topological descriptors
        features.update(
            {
                "BalabanJ": Descriptors.BalabanJ(mol),  # type: ignore
                "BertzCT": Descriptors.BertzCT(mol),  # type: ignore
                "Chi0v": GraphDescriptors.Chi0v(mol),  # type: ignore
                "Chi1v": GraphDescriptors.Chi1v(mol),  # type: ignore
                "Chi2v": GraphDescriptors.Chi2v(mol),  # type: ignore
                "Chi3v": GraphDescriptors.Chi3v(mol),  # type: ignore
                "Chi4v": GraphDescriptors.Chi4v(mol),  # type: ignore
                "HallKierAlpha": GraphDescriptors.HallKierAlpha(mol),  # type: ignore
            }
        )

        # Surface area and physicochemical descriptors
        features.update(
            {
                "LabuteASA": Descriptors.LabuteASA(mol),  # type: ignore
                "TPSA": Descriptors.TPSA(mol),  # type: ignore
                "MolLogP": Descriptors.MolLogP(mol),  # type: ignore
                "MolMR": Descriptors.MolMR(mol),  # type: ignore
            }
        )

        return features

    except Exception as e:
        print(f"Error computing features for molecule: {e}")
        return None


def create_graph_data(mol: Optional[Mol]) -> Optional[Data]:
    """Create PyTorch Geometric graph data from RDKit molecule."""
    if not isinstance(mol, Mol):
        return None

    # Compute node features
    node_features = compute_node_features(mol)
    if node_features is None:
        return None

    # Convert node features to tensor
    node_feature_list = []
    for atom in mol.GetAtoms():
        features = []
        for feat_name in [
            "AtomicNum",
            "Degree",
            "TotalNumHs",
            "ImplicitValence",
            "IsAromatic",
            "IsInRing",
            "FormalCharge",
            "Hybridization",
        ]:
            features.append(node_features[feat_name][atom.GetIdx()])
        node_feature_list.append(features)
    node_features_tensor = torch.tensor(node_feature_list, dtype=torch.float)

    # Get bond information for edges
    edges: List[List[int]] = []
    edge_features: List[List[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = float(bond.GetBondTypeAsDouble())
        is_conjugated = float(bond.GetIsConjugated())
        is_in_ring = float(bond.IsInRing())
        is_aromatic = float(bond.GetIsAromatic())

        # Add edges in both directions
        edges.extend([[i, j], [j, i]])
        edge_features.extend([[bond_type, is_conjugated, is_in_ring, is_aromatic]] * 2)

    if not edges:
        return None

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=node_features_tensor, edge_index=edge_index, edge_attr=edge_attr)


def create_multi_property_stratification(
    df: pd.DataFrame, n_bins: int = 3
) -> pd.Series:
    """Create stratification bins based on multiple molecular properties."""
    properties = ["Permeability", "ExactMolWt", "MolLogP", "TPSA"]
    bins = {}

    for prop in properties:
        # Use fewer bins and handle edge cases with duplicates
        try:
            bins[prop] = pd.qcut(df[prop], n_bins, labels=False, duplicates="drop")
        except ValueError:
            # If qcut fails, fall back to simple median split
            bins[prop] = (df[prop] > df[prop].median()).astype(int)

    # Combine bins into a single stratification key
    strat_key = pd.Series(index=df.index, dtype=str)
    for prop in properties:
        strat_key = strat_key.astype(str) + bins[prop].astype(str)

    return strat_key


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create and save visualizations of molecular properties."""
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Plot distributions of key properties
    for prop in KEY_PROPERTIES + ["Permeability"]:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=prop, kde=True)
        plt.title(f"Distribution of {prop}")
        plt.savefig(os.path.join(output_dir, "plots", f"{prop}_dist.png"))
        plt.close()

    # Create correlation heatmap for key properties
    plt.figure(figsize=(12, 10))
    correlation_data = df[KEY_PROPERTIES + ["Permeability"]]
    sns.heatmap(correlation_data.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Property Correlations")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "correlation_heatmap.png"))
    plt.close()


def create_split_visualizations(
    df: pd.DataFrame,
    train_idx: Union[List[int], np.ndarray, pd.Index],
    val_idx: Union[List[int], np.ndarray, pd.Index],
    test_idx: Union[List[int], np.ndarray, pd.Index],
    output_dir: str,
) -> None:
    """Create visualizations comparing property distributions across splits."""
    properties = ["ExactMolWt", "MolLogP", "TPSA", "NumRotatableBonds", "Permeability"]
    stats_data = []

    # Create violin plots for each property
    for prop in properties:
        plt.figure(figsize=(10, 6))
        data_dict = {
            "Train": df.iloc[train_idx][prop],
            "Validation": df.iloc[val_idx][prop],
            "Test": df.iloc[test_idx][prop],
        }

        # Create violin plot
        sns.violinplot(data=data_dict)
        plt.title(f"Distribution of {prop} across splits")
        plt.ylabel(prop)
        plt.xticks(rotation=45)

        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prop}_distribution.png"))
        plt.close()

        # Calculate statistics
        for split_name, indices in [
            ("Train", train_idx),
            ("Validation", val_idx),
            ("Test", test_idx),
        ]:
            split_data = df.iloc[indices][prop]
            stats_data.append(
                {
                    "Property": prop,
                    "Split": split_name,
                    "Mean": split_data.mean(),
                    "Std": split_data.std(),
                    "Min": split_data.min(),
                    "Max": split_data.max(),
                    "Median": split_data.median(),
                }
            )

    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(
        os.path.join(output_dir, "distribution_statistics.csv"), index=False
    )


def aggregate_features_by_molecule(features_df, mol_ids):
    """Aggregate node/edge features by molecule using mean, std, min, and max.

    Args:
        features_df: DataFrame containing node/edge features
        mol_ids: Series containing molecule IDs for each node/edge

    Returns:
        DataFrame with aggregated features per molecule
    """
    agg_funcs = ["mean", "std", "min", "max"]
    agg_df = features_df.groupby(mol_ids).agg(agg_funcs)

    # Flatten column names
    agg_df.columns = [
        f"{col[0]}_{func}"
        for col, func in zip(agg_df.columns, agg_funcs * len(features_df.columns))
    ]
    return agg_df


def select_features_by_type(
    features_df: FeatureDF,
    target: ArrayLike,
    output_dir: str,
    feature_type: str,
    n_features: int = 5,
) -> Tuple[FeatureDF, List[str]]:
    """Select top features using multiple selection methods.

    Uses a combination of:
    1. Mutual Information with target
    2. Random Forest importance
    3. Correlation analysis (removing highly correlated features)
    """
    # Handle missing values using median imputation
    imputer = SimpleImputer(strategy="median")
    features_imputed = imputer.fit_transform(features_df)
    features_imputed_df = pd.DataFrame(features_imputed, columns=features_df.columns)
    target_array = np.asarray(target, dtype=np.float64)

    # 1. Mutual Information Selection
    mi_selector = SelectKBest(score_func=mutual_info_regression, k="all")
    mi_selector.fit(features_imputed_df, target_array)
    mi_scores = pd.DataFrame(
        {"Feature": features_df.columns, "MI_Score": mi_selector.scores_}
    )
    mi_scores = mi_scores.sort_values("MI_Score", ascending=False)

    # 2. Random Forest Importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(features_imputed_df, target_array)
    rf_scores = pd.DataFrame(
        {"Feature": features_df.columns, "RF_Score": rf.feature_importances_}
    )
    rf_scores = rf_scores.sort_values("RF_Score", ascending=False)

    # 3. Correlation Analysis
    correlation_matrix = features_imputed_df.corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [
        column for column in upper_tri.columns if any(upper_tri[column] > 0.95)
    ]

    # Combine scores from MI and RF
    combined_scores = pd.merge(mi_scores, rf_scores, on="Feature")
    combined_scores["Combined_Score"] = (
        combined_scores["MI_Score"] * combined_scores["RF_Score"]
    ) ** 0.5  # Geometric mean
    combined_scores = combined_scores.sort_values("Combined_Score", ascending=False)

    # Remove highly correlated features, keeping the one with higher combined score
    final_features = []
    for feature in combined_scores["Feature"]:
        if len(final_features) >= n_features:
            break

        if (
            feature not in high_corr_features
            or feature in combined_scores["Feature"].head(n_features // 2).values
        ):
            final_features.append(feature)

    # Create feature selection directory
    feature_dir = os.path.join(output_dir, "feature_selection")
    os.makedirs(feature_dir, exist_ok=True)

    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, annot=True)
    plt.title(f"{feature_type.capitalize()} Feature Correlations")
    plt.tight_layout()
    plt.savefig(os.path.join(feature_dir, f"{feature_type}_correlation_heatmap.png"))
    plt.close()

    # Create feature importance plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    # Mutual Information plot
    sns.barplot(data=mi_scores.head(10), x="MI_Score", y="Feature", ax=ax1)
    ax1.set_title(f"Top 10 {feature_type.capitalize()} Features by Mutual Information")

    # Random Forest plot
    sns.barplot(data=rf_scores.head(10), x="RF_Score", y="Feature", ax=ax2)
    ax2.set_title(
        f"Top 10 {feature_type.capitalize()} Features by Random Forest Importance"
    )

    # Combined score plot
    sns.barplot(data=combined_scores.head(10), x="Combined_Score", y="Feature", ax=ax3)
    ax3.set_title(f"Top 10 {feature_type.capitalize()} Features by Combined Score")

    plt.tight_layout()
    plt.savefig(os.path.join(feature_dir, f"{feature_type}_feature_importance.png"))
    plt.close()

    # Save detailed scores
    combined_scores.to_csv(
        os.path.join(feature_dir, f"{feature_type}_feature_scores.csv"), index=False
    )

    # Save correlation analysis
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > 0.95:
                high_corr_pairs.append(
                    {
                        "Feature1": correlation_matrix.columns[i],
                        "Feature2": correlation_matrix.columns[j],
                        "Correlation": correlation_matrix.iloc[i, j],
                    }
                )

    if high_corr_pairs:
        pd.DataFrame(high_corr_pairs).to_csv(
            os.path.join(feature_dir, f"{feature_type}_high_correlations.csv"),
            index=False,
        )

    # Return selected features
    return features_imputed_df[final_features], final_features


def process_molecules(smiles_series: pd.Series) -> MolList:
    """Process a pandas Series of SMILES strings into RDKit molecules."""
    return [
        Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
        for smiles in smiles_series
    ]


def compute_node_features(mol: Chem.Mol) -> Optional[Dict[str, List[float]]]:
    """Compute node (atom) features for a molecule."""
    try:
        features = {
            "AtomicNum": [],
            "Degree": [],
            "TotalNumHs": [],
            "ImplicitValence": [],
            "IsAromatic": [],
            "IsInRing": [],
            "FormalCharge": [],
            "Hybridization": [],
        }

        for atom in mol.GetAtoms():
            features["AtomicNum"].append(float(atom.GetAtomicNum()))
            features["Degree"].append(float(atom.GetDegree()))
            features["TotalNumHs"].append(float(atom.GetTotalNumHs()))
            features["ImplicitValence"].append(float(atom.GetImplicitValence()))
            features["IsAromatic"].append(float(atom.GetIsAromatic()))
            features["IsInRing"].append(float(atom.IsInRing()))
            features["FormalCharge"].append(float(atom.GetFormalCharge()))
            features["Hybridization"].append(float(atom.GetHybridization()))

        return features
    except Exception as e:
        print(f"Error computing node features: {e}")
        return None


def compute_edge_features(mol: Chem.Mol) -> Optional[Dict[str, List[float]]]:
    """Compute edge (bond) features for a molecule."""
    try:
        features = {
            "BondType": [],
            "IsConjugated": [],
            "IsInRing": [],
            "IsAromatic": [],
        }

        for bond in mol.GetBonds():
            features["BondType"].append(float(bond.GetBondTypeAsDouble()))
            features["IsConjugated"].append(float(bond.GetIsConjugated()))
            features["IsInRing"].append(float(bond.IsInRing()))
            features["IsAromatic"].append(float(bond.GetIsAromatic()))

        return features
    except Exception as e:
        print(f"Error computing edge features: {e}")
        return None


def aggregate_features(features: Dict[str, List[float]]) -> Dict[str, float]:
    """Aggregate features using statistical measures."""
    aggregated = {}
    for feat_name, feat_values in features.items():
        if feat_values:
            values = np.array(feat_values)
            aggregated[f"{feat_name}_mean"] = float(np.mean(values))
            aggregated[f"{feat_name}_std"] = float(np.std(values))
            aggregated[f"{feat_name}_min"] = float(np.min(values))
            aggregated[f"{feat_name}_max"] = float(np.max(values))
    return aggregated


def process_dataset(input_csv: str, output_dir: str, n_splits: int = 5) -> None:
    """Process the dataset and create train/val/test splits."""
    # Delete existing output directory if it exists
    if os.path.exists(output_dir):
        print(f"Removing existing output directory: {output_dir}")
        import shutil

        shutil.rmtree(output_dir)

    # Create output directory and raw subdirectory
    print(f"Creating new output directory: {output_dir}")
    os.makedirs(output_dir)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir)

    # Read and preprocess data
    print("Loading data...")
    df = pd.read_csv(input_csv)
    initial_size = len(df)
    print(f"Initial dataset size: {initial_size}")

    # Remove missing values and convert permeability to numeric
    df = df.dropna(subset=[SMILES_COL, PERMEABILITY_COL])
    df[PERMEABILITY_COL] = pd.to_numeric(df[PERMEABILITY_COL], errors="coerce")
    print(f"Dataset size after removing missing values: {len(df)}")

    # Filter PAMPA values below -9.0 (detection limit)
    df = df[df[PERMEABILITY_COL] > -9.0]
    print(f"Dataset size after filtering PAMPA values: {len(df)}")

    # Remove duplicates
    df = df.drop_duplicates(subset=[SMILES_COL])
    print(f"Dataset size after removing duplicates: {len(df)}")

    # Process molecules and create graph data
    print("Processing molecules and creating graph data...")
    mols = process_molecules(df[SMILES_COL])
    valid_indices = [i for i, mol in enumerate(mols) if mol is not None]

    # Process each valid molecule and save to raw directory
    for idx in tqdm(valid_indices):
        mol = mols[idx]
        if mol is not None:
            graph_data = create_graph_data(mol)
            if graph_data is not None:
                torch.save(graph_data, os.path.join(raw_dir, f"mol_{idx}.pt"))

    # Convert permeability to numeric and create bins for stratification
    df[PERMEABILITY_COL] = pd.to_numeric(df[PERMEABILITY_COL], errors="coerce")
    df["permeability_bin"] = pd.qcut(
        df[PERMEABILITY_COL], q=3, labels=["low", "medium", "high"]
    )

    # Compute molecular features
    print("Computing molecular features...")
    mols = process_molecules(df[SMILES_COL])
    valid_indices = [i for i, mol in enumerate(mols) if mol is not None]

    # Filter out invalid molecules
    df = df.iloc[valid_indices].reset_index(drop=True)
    mols = [mol for mol in mols if mol is not None]

    # Compute features for valid molecules
    global_features = []
    for mol in tqdm(mols):
        features = compute_molecular_features(mol)
        if features is not None:
            global_features.append(features)

    # Create feature dataframes with the correct feature set
    feature_names = list(global_features[0].keys()) if global_features else []
    global_features_df = pd.DataFrame(global_features, columns=feature_names)

    # Compute node and edge features
    print("Computing node and edge features...")
    node_features_list = []
    edge_features_list = []

    for mol in tqdm(mols):
        if mol is not None:
            # Compute and aggregate node features
            node_feats = compute_node_features(mol)
            if node_feats:
                node_features_list.append(aggregate_features(node_feats))
            else:
                continue

            # Compute and aggregate edge features
            edge_feats = compute_edge_features(mol)
            if edge_feats:
                edge_features_list.append(aggregate_features(edge_feats))
            else:
                continue

    # Convert to DataFrames
    node_features_df = pd.DataFrame(node_features_list)
    edge_features_df = pd.DataFrame(edge_features_list)

    # Fill any missing values with 0
    node_features_df = node_features_df.fillna(0)
    edge_features_df = edge_features_df.fillna(0)

    # Perform feature selection for each feature type
    selected_node_features_df, selected_node_features = select_features_by_type(
        node_features_df, df[PERMEABILITY_COL].values, output_dir, "node", n_features=10
    )

    selected_edge_features_df, selected_edge_features = select_features_by_type(
        edge_features_df, df[PERMEABILITY_COL].values, output_dir, "edge", n_features=8
    )

    selected_global_features_df, selected_global_features = select_features_by_type(
        global_features_df,
        df[PERMEABILITY_COL].values,
        output_dir,
        "global",
        n_features=5,
    )

    # Update split_info with selected features
    split_info = {
        "n_splits": n_splits,
        "global_features": selected_global_features,
        "node_features": selected_node_features,
        "edge_features": selected_edge_features,
        "n_node_features": len(selected_node_features),
        "n_edge_features": len(selected_edge_features),
        "max_nodes": max(len(mol.GetAtoms()) for mol in mols if mol is not None),
    }

    # Save split information
    with open(os.path.join(output_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # Create splits using StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Process each fold
    for fold, (train_val_idx, test_idx) in enumerate(
        skf.split(df, df["permeability_bin"])
    ):
        print(f"Processing fold {fold + 1}/{n_splits}")

        # Split train_val into train and validation
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.1,
            stratify=df.iloc[train_val_idx]["permeability_bin"],
            random_state=42,
        )

        # Create fold directory
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Split data
        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]
        test_data = df.iloc[test_idx]

        # Create visualizations for this fold
        create_split_visualizations(df, train_idx, val_idx, test_idx, fold_dir)

        # Update graph data creation to include selected features
        for split_name, split_data, indices in [
            ("train", train_data, train_idx),
            ("val", val_data, val_idx),
            ("test", test_data, test_idx),
        ]:
            split_dir = os.path.join(fold_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            for i, (idx, row) in enumerate(split_data.iterrows()):
                mol = mols[idx]
                if mol is None:
                    continue

                graph_data = create_graph_data(mol)
                if graph_data is None:
                    continue

                # Add selected features
                graph_data.global_features = torch.tensor(
                    selected_global_features_df.iloc[idx].values, dtype=torch.float
                )
                graph_data.node_features = torch.tensor(
                    selected_node_features_df.iloc[idx].values, dtype=torch.float
                )
                graph_data.edge_features = torch.tensor(
                    selected_edge_features_df.iloc[idx].values, dtype=torch.float
                )
                graph_data.y = torch.tensor([row[PERMEABILITY_COL]], dtype=torch.float)

                torch.save(graph_data, os.path.join(split_dir, f"mol_{i}.pt"))

            print(f"Processed {split_name} set for fold {fold}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_cycpept.py <input_csv> <output_dir>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]

    process_dataset(input_csv, output_dir)
