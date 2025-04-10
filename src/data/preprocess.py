"""
Data preprocessing module for cyclic peptide permeability prediction.
Handles data cleaning, feature extraction, and train/validation/test splitting.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List, Dict
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_data(
    df: pd.DataFrame, smiles_col: str = "SMILES", target_col: str = "Permeability"
) -> pd.DataFrame:
    """
    Clean and prepare the raw data.

    Args:
        df: Input DataFrame
        smiles_col: Column name containing SMILES strings
        target_col: Column name containing permeability values

    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning dataset with {len(df)} entries")

    # Drop rows with missing SMILES or permeability values
    df = df.dropna(subset=[smiles_col, target_col])
    logger.info(f"After dropping NAs: {len(df)} entries")

    # Validate SMILES
    valid_smiles = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(True)
        else:
            valid_smiles.append(False)

    df["valid_smiles"] = valid_smiles
    df = df[df["valid_smiles"]].drop(columns=["valid_smiles"])
    logger.info(f"After SMILES validation: {len(df)} entries")

    # Convert target to numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    logger.info(f"After numeric conversion: {len(df)} entries")

    return df


def add_molecular_descriptors(
    df: pd.DataFrame, smiles_col: str = "SMILES"
) -> pd.DataFrame:
    """
    Add molecular descriptors to the DataFrame.

    Args:
        df: Input DataFrame
        smiles_col: Column name containing SMILES strings

    Returns:
        DataFrame with added descriptors
    """
    logger.info("Adding molecular descriptors")

    # Initialize descriptor columns
    descriptors = []

    # Calculate descriptors for each molecule
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)

        # Skip invalid molecules
        if mol is None:
            descriptors.append(None)
            continue

        # Calculate descriptors
        desc_dict = {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumRings": Descriptors.RingCount(mol),
            "AromaticRings": Descriptors.NumAromaticRings(mol),
            "FractionCSP3": Descriptors.FractionCSP3(mol),
            "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
            "NumAtoms": mol.GetNumAtoms(),
        }

        descriptors.append(desc_dict)

    # Convert list of dictionaries to DataFrame
    desc_df = pd.DataFrame(descriptors)

    # Join with original DataFrame
    for col in desc_df.columns:
        df[f"Desc_{col}"] = desc_df[col]

    logger.info(f"Added {len(desc_df.columns)} molecular descriptors")

    return df


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify_col: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Args:
        df: Input DataFrame
        val_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        stratify_col: Column to use for stratified splitting
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Splitting data with val_size={val_size}, test_size={test_size}")

    # Determine stratification
    stratify = None
    if stratify_col is not None and stratify_col in df.columns:
        stratify = df[stratify_col]
        logger.info(f"Using {stratify_col} for stratified splitting")

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=stratify, random_state=random_state
    )

    # Update stratify for second split
    if stratify is not None:
        stratify = train_val_df[stratify_col]

    # Second split: separate validation set
    # Calculate adjusted validation size
    adjusted_val_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=stratify,
        random_state=random_state,
    )

    logger.info(
        f"Split result - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}"
    )

    return train_df, val_df, test_df


def create_stratification_bins(
    df: pd.DataFrame, target_col: str = "Permeability", n_bins: int = 10
) -> pd.DataFrame:
    """
    Create bins for stratified sampling.

    Args:
        df: Input DataFrame
        target_col: Column containing target values to bin
        n_bins: Number of bins to create

    Returns:
        DataFrame with added stratification column
    """
    logger.info(f"Creating {n_bins} stratification bins for {target_col}")

    # Create copy to avoid modifying original
    df = df.copy()

    # Create bins
    df["strat_bin"] = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates="drop")

    # Log bin distribution
    bin_counts = df["strat_bin"].value_counts().sort_index()
    logger.info(f"Bin distribution: {bin_counts.to_dict()}")

    return df


def preprocess_and_split(
    input_file: str,
    output_dir: str,
    smiles_col: str = "SMILES",
    target_col: str = "Permeability",
    val_size: float = 0.15,
    test_size: float = 0.15,
    add_descriptors: bool = True,
    stratify: bool = True,
    n_bins: int = 10,
    random_state: int = 42,
) -> Dict[str, str]:
    """
    Preprocess data and split into train/val/test sets.

    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save output files
        smiles_col: Column name containing SMILES strings
        target_col: Column name containing permeability values
        val_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        add_descriptors: Whether to add molecular descriptors
        stratify: Whether to use stratified splitting
        n_bins: Number of bins for stratification
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with paths to output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    # Clean data
    df = clean_data(df, smiles_col=smiles_col, target_col=target_col)

    # Add descriptors if requested
    if add_descriptors:
        df = add_molecular_descriptors(df, smiles_col=smiles_col)

    # Create stratification bins if requested
    stratify_col = None
    if stratify:
        df = create_stratification_bins(df, target_col=target_col, n_bins=n_bins)
        stratify_col = "strat_bin"

    # Split data
    train_df, val_df, test_df = split_data(
        df,
        val_size=val_size,
        test_size=test_size,
        stratify_col=stratify_col,
        random_state=random_state,
    )

    # Save to files
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved preprocessed data to {output_dir}")

    return {"train": train_path, "val": val_path, "test": test_path}


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Preprocess peptide permeability data")

    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument(
        "--output_dir", type=str, default="processed_data", help="Output directory"
    )
    parser.add_argument(
        "--smiles_col", type=str, default="SMILES", help="SMILES column name"
    )
    parser.add_argument(
        "--target_col", type=str, default="Permeability", help="Target column name"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.15, help="Validation set size"
    )
    parser.add_argument("--test_size", type=float, default=0.15, help="Test set size")
    parser.add_argument(
        "--add_descriptors", action="store_true", help="Add molecular descriptors"
    )
    parser.add_argument(
        "--no_stratify", action="store_true", help="Disable stratified splitting"
    )
    parser.add_argument(
        "--n_bins", type=int, default=10, help="Number of bins for stratification"
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Process data
    output_paths = preprocess_and_split(
        input_file=args.input,
        output_dir=args.output_dir,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        val_size=args.val_size,
        test_size=args.test_size,
        add_descriptors=args.add_descriptors,
        stratify=not args.no_stratify,
        n_bins=args.n_bins,
        random_state=args.random_state,
    )

    # Print output paths
    for name, path in output_paths.items():
        print(f"{name.capitalize()} data saved to: {path}")


if __name__ == "__main__":
    main()
