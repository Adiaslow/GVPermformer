"""
Data preprocessing module for molecular graph data.
Handles data filtering, splitting, and feature computation.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, SaltRemover
import json
import torch
from tqdm import tqdm
from src.utils.molecule_featurizer import MoleculeFeaturizer

logger = logging.getLogger(__name__)


def sanitize_molecule(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Sanitize and standardize a molecule.
    Uses a more lenient approach to avoid losing valid molecules.

    Args:
        mol: RDKit molecule object

    Returns:
        Sanitized molecule or None if failed
    """
    if mol is None:
        return None

    try:
        # Try with original molecule first
        orig_mol = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(orig_mol)
            return orig_mol
        except:
            pass

        # Try removing salts and get largest fragment
        remover = SaltRemover.SaltRemover()
        stripped_mol = remover.StripMol(mol, dontRemoveEverything=True)

        # Try sanitization with different levels on stripped molecule
        try:
            Chem.SanitizeMol(stripped_mol)
            return stripped_mol
        except:
            try:
                # Try again with more lenient parameters
                Chem.SanitizeMol(
                    stripped_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE
                )
                return stripped_mol
            except:
                try:
                    # Try minimal sanitization
                    Chem.SanitizeMol(stripped_mol, sanitizeOps=Chem.SANITIZE_PROPERTIES)
                    return stripped_mol
                except:
                    # If all attempts fail with stripped molecule, try original again with lenient params
                    try:
                        Chem.SanitizeMol(
                            orig_mol,
                            sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE,
                        )
                        return orig_mol
                    except:
                        try:
                            Chem.SanitizeMol(
                                orig_mol, sanitizeOps=Chem.SANITIZE_PROPERTIES
                            )
                            return orig_mol
                        except:
                            return None

    except Exception as e:
        logger.debug(f"Molecule sanitization failed: {e}")
        return None


def validate_molecule(smiles: str) -> bool:
    """
    Validate if a SMILES string represents a valid molecule.
    Includes sanitization checks.

    Args:
        smiles: SMILES string to validate

    Returns:
        bool: True if molecule is valid
    """
    if not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return sanitize_molecule(mol) is not None


def detect_outliers(
    series: pd.Series, method: str = "iqr", threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers in a series using various methods.

    Args:
        series: Input series
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        Boolean series indicating outliers
    """
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        return (series < lower) | (series > upper)
    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def compute_molecular_properties(smiles: str) -> Dict[str, float]:
    """
    Compute basic molecular properties for stratification.

    Args:
        smiles: SMILES string

    Returns:
        Dictionary of properties
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    try:
        return {
            "MW": Descriptors.ExactMolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
        }
    except:
        return {}


def filter_data(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    target_col: str = "Permeability",
    pampa_threshold: Optional[float] = None,
    min_atoms: int = 3,
    max_atoms: int = 500,
    outlier_method: str = "iqr",
    outlier_threshold: float = 3.0,
    remove_outliers: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Filter dataset based on various criteria.
    Handles duplicates, invalid molecules, and size constraints.
    All filtering is done before any preprocessing to prevent data leakage.

    Args:
        df: Input DataFrame
        smiles_col: Column containing SMILES strings
        target_col: Column containing target values
        pampa_threshold: Optional threshold for filtering PAMPA values
        min_atoms: Minimum number of atoms allowed
        max_atoms: Maximum number of atoms allowed
        outlier_method: Method for outlier detection ('iqr' or 'zscore')
        outlier_threshold: Threshold for outlier detection
        remove_outliers: Whether to remove detected outliers

    Returns:
        Tuple of (filtered DataFrame, filtering statistics)
    """
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    stats = {"initial_size": len(df)}
    logger.info(f"Initial dataset size: {stats['initial_size']}")

    # Remove rows with missing values first
    df = df.dropna(subset=[smiles_col, target_col])
    stats["after_missing"] = len(df)
    logger.info(f"After removing NaN: {len(df)}")

    # Convert target to numeric and remove invalid values
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    stats["after_numeric_conversion"] = len(df)
    logger.info(f"After converting target to numeric: {len(df)}")

    # Remove invalid values in target column
    df = df[~df[target_col].isin([float("inf"), float("-inf")])]
    stats["after_invalid_values"] = len(df)
    logger.info(f"After removing invalid target values: {len(df)}")

    # Initial SMILES validation and canonicalization
    valid_mols = df[smiles_col].apply(lambda s: Chem.MolFromSmiles(s))
    df = df[valid_mols.notna()]
    df["canonical_smiles"] = df[smiles_col].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
    )

    # Remove duplicates
    df = df.drop_duplicates(subset=["canonical_smiles"], keep="first")
    stats["after_duplicates"] = len(df)
    logger.info(f"After removing duplicates: {len(df)}")

    # Handle outliers
    outliers = detect_outliers(
        df[target_col], method=outlier_method, threshold=outlier_threshold
    )
    stats["outliers_detected"] = outliers.sum()
    logger.info(f"Detected {outliers.sum()} outliers in {target_col}")

    if remove_outliers:
        df = df[~outliers]
        stats["after_outliers"] = len(df)
        logger.info(f"After removing outliers: {len(df)}")

    # Sanitize molecules with more lenient approach
    sanitized_mols = df[smiles_col].apply(
        lambda s: sanitize_molecule(Chem.MolFromSmiles(s))
    )
    df = df[sanitized_mols.notna()]
    stats["after_sanitization"] = len(df)
    logger.info(f"After sanitization: {len(df)}")

    # Count atoms for each molecule
    atom_counts = df[smiles_col].apply(lambda s: len(Chem.MolFromSmiles(s).GetAtoms()))
    size_mask = (atom_counts >= min_atoms) & (atom_counts <= max_atoms)
    df = df[size_mask]
    stats["after_size_filter"] = len(df)
    logger.info(f"After size filtering: {len(df)}")

    # Apply PAMPA threshold if specified
    if pampa_threshold is not None:
        pampa_mask = df[target_col] >= pampa_threshold
        df = df[pampa_mask]
        stats["after_pampa"] = len(df)
        logger.info(f"After PAMPA threshold filtering: {len(df)}")

    # Compute molecular properties for stratification
    logger.info("Computing molecular properties...")
    mol_props = df[smiles_col].apply(compute_molecular_properties)
    prop_df = pd.DataFrame.from_records(mol_props.values)

    # Add properties to main DataFrame
    for col in prop_df.columns:
        df[f"mol_{col}"] = prop_df[col]

    # Remove any rows where property computation failed
    df = df.dropna(subset=[f"mol_{col}" for col in prop_df.columns])
    stats["final_size"] = len(df)
    logger.info(f"Final dataset size after all filtering: {len(df)}")

    # Reset index after all filtering
    df = df.reset_index(drop=True)

    # Log reduction statistics
    stats["total_reduction"] = 100 * (1 - len(df) / stats["initial_size"])
    logger.info(f"Total reduction: {stats['total_reduction']:.1f}%")
    logger.info(
        "Filtering steps complete. Final dataset is clean and contains no duplicates."
    )

    # Clean up temporary columns
    if "canonical_smiles" in df.columns:
        df = df.drop(columns=["canonical_smiles"])

    return df, stats


def create_stratification_bins(
    df: pd.DataFrame,
    target_col: str,
    n_bins: int = 10,
    additional_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create bins for stratified splitting based on target values
    and optional additional molecular properties.
    Ensures each bin has at least 2 members.

    Args:
        df: Input DataFrame
        target_col: Main column to use for stratification
        n_bins: Maximum number of bins for stratification
        additional_cols: Additional columns to consider for stratification

    Returns:
        DataFrame with added stratification bin columns
    """
    df = df.copy()

    # Function to create bins with at least 2 members
    def create_safe_bins(series, max_bins):
        n = max_bins
        while n > 1:
            try:
                bins = pd.qcut(series, q=n, labels=False, duplicates="drop")
                # Check if each bin has at least 2 members
                if bins.value_counts().min() >= 2:
                    return bins
            except:
                pass
            n = n - 1
        # Fallback to binary split if all else fails
        return pd.qcut(series, q=2, labels=False, duplicates="drop")

    # Create bins for target column
    df["target_bin"] = create_safe_bins(df[target_col], n_bins)

    # Create bins for additional columns if specified
    if additional_cols:
        for col in additional_cols:
            if col in df.columns:
                df[f"{col}_bin"] = create_safe_bins(
                    df[col], min(n_bins, len(df[col].unique()))
                )

        # Combine bins into a single stratification column
        bin_cols = ["target_bin"] + [
            f"{col}_bin" for col in additional_cols if f"{col}_bin" in df.columns
        ]

        # Create combined stratification key
        df["strat_bin"] = df[bin_cols].apply(lambda x: "_".join(x.astype(str)), axis=1)

        # If any combined bin has less than 2 members, fall back to target_bin only
        if df["strat_bin"].value_counts().min() < 2:
            logger.warning(
                "Combined stratification resulted in bins with < 2 members. Using target bins only."
            )
            df["strat_bin"] = df["target_bin"]
    else:
        df["strat_bin"] = df["target_bin"]

    return df


def split_data_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify_col: Optional[str] = None,
    random_state: int = 42,
) -> List[Dict[str, pd.DataFrame]]:
    """
    Split data into train/val/test sets using k-fold cross-validation.

    Args:
        df: Input DataFrame
        n_splits: Number of folds for cross-validation
        val_size: Fraction of data for validation
        test_size: Fraction of data for testing
        stratify_col: Column to use for stratified splitting
        random_state: Random seed

    Returns:
        List of dictionaries containing train/val/test splits for each fold
    """
    # First, split off the test set
    if stratify_col is not None:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[stratify_col],
            random_state=random_state,
        )
    else:
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

    # Create k-fold splits of the remaining data
    if stratify_col is not None:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(train_val_df, train_val_df[stratify_col])
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(train_val_df)

    # Create splits for each fold
    fold_splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        fold_splits.append(
            {"train": train_df, "val": val_df, "test": test_df, "fold": fold_idx}
        )

        # Log split sizes
        logger.info(
            f"Fold {fold_idx}: "
            f"Train={len(train_df)}, "
            f"Val={len(val_df)}, "
            f"Test={len(test_df)}"
        )

    return fold_splits


def process_and_save_split(df: pd.DataFrame, output_dir: str, split_name: str):
    """Process molecules in a data split and save as PyTorch Geometric data files."""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize featurizer
    featurizer = MoleculeFeaturizer()

    processed_data = []
    skipped = 0
    failed = 0

    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Processing {split_name} set"
    ):
        try:
            # Convert SMILES to graph data with comprehensive features
            graph_data = featurizer.featurize(row["SMILES"])
            if graph_data is not None:
                # Add target value
                graph_data.y = torch.tensor(
                    row["Permeability"], dtype=torch.float
                ).unsqueeze(0)
                # Save to file
                torch.save(graph_data, os.path.join(output_dir, f"graph_{idx}.pt"))
                processed_data.append({"idx": idx, "smiles": row["SMILES"]})
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing molecule {idx}: {e}")
            failed += 1

    print(f"\n{split_name} set statistics:")
    print(f"Successfully processed: {len(processed_data)}")
    print(f"Failed: {failed}")
    print(f"Success rate: {len(processed_data)/(len(processed_data)+failed)*100:.2f}%")

    # Save metadata
    metadata = {
        "num_samples": len(processed_data),
        "failed_samples": failed,
        "processed_indices": [p["idx"] for p in processed_data],
        "smiles": [p["smiles"] for p in processed_data],
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def preprocess_and_split(
    input_file: str,
    output_dir: str,
    smiles_col: str = "SMILES",
    target_col: str = "Permeability",
    val_size: float = 0.15,
    test_size: float = 0.15,
    n_splits: int = 5,
    n_bins: int = 10,
    pampa_threshold: Optional[float] = -9.0,
    min_atoms: int = 3,
    max_atoms: int = 500,
    random_state: int = 42,
    property_cols: Optional[List[str]] = None,
    outlier_method: str = "iqr",
    outlier_threshold: float = 3.0,
    remove_outliers: bool = True,
    stratify_props: bool = True,
) -> Dict[str, Union[str, Dict]]:
    """
    Preprocess data and create k-fold cross-validation splits.
    Converts molecules directly to PyTorch Geometric data files with rich featurization.

    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save output files
        smiles_col: Column name containing SMILES strings
        target_col: Column name containing target values
        val_size: Fraction of data for validation
        test_size: Fraction of data for testing
        n_splits: Number of folds for cross-validation
        n_bins: Number of bins for stratification
        pampa_threshold: Threshold for filtering PAMPA values
        min_atoms: Minimum number of atoms allowed
        max_atoms: Maximum number of atoms allowed
        random_state: Random seed
        property_cols: Additional property columns to keep
        outlier_method: Method for outlier detection
        outlier_threshold: Threshold for outlier detection
        remove_outliers: Whether to remove outliers
        stratify_props: Whether to use molecular properties for stratification

    Returns:
        Dictionary with paths to output files and preprocessing statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load and filter data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    # Filter data before splitting
    df, filter_stats = filter_data(
        df,
        smiles_col=smiles_col,
        target_col=target_col,
        pampa_threshold=pampa_threshold,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        outlier_method=outlier_method,
        outlier_threshold=outlier_threshold,
        remove_outliers=remove_outliers,
    )

    # Convert numpy integers to Python integers in filter_stats
    filter_stats = {
        k: int(v) if isinstance(v, np.integer) else v for k, v in filter_stats.items()
    }

    # Determine columns to use for stratification
    strat_cols = ["mol_MW", "mol_LogP", "mol_TPSA"] if stratify_props else None

    # Create stratification bins
    df = create_stratification_bins(
        df, target_col, n_bins=n_bins, additional_cols=strat_cols
    )

    # Create k-fold splits
    fold_splits = split_data_kfold(
        df,
        n_splits=n_splits,
        val_size=val_size,
        test_size=test_size,
        stratify_col="strat_bin",
        random_state=random_state,
    )

    # Process and save each fold
    output_paths = {}
    for fold_data in fold_splits:
        fold_idx = fold_data["fold"]
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # Process and save each split
        for split_name in ["train", "val", "test"]:
            split_df = fold_data[split_name]
            split_dir = os.path.join(fold_dir, split_name)
            process_and_save_split(split_df, split_dir, f"Fold {fold_idx} {split_name}")

            if fold_idx == 0:  # Store paths for first fold
                output_paths[split_name] = split_dir

    # Save configuration
    config = {
        "n_splits": n_splits,
        "val_size": val_size,
        "test_size": test_size,
        "random_state": random_state,
        "n_bins": n_bins,
        "pampa_threshold": pampa_threshold,
        "min_atoms": min_atoms,
        "max_atoms": max_atoms,
        "outlier_method": outlier_method,
        "outlier_threshold": outlier_threshold,
        "remove_outliers": remove_outliers,
        "stratify_props": stratify_props,
        "filter_stats": filter_stats,
    }

    config_path = os.path.join(output_dir, "preprocessing_info.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return {"paths": output_paths, "stats": filter_stats}
