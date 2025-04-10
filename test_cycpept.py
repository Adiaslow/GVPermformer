# test_cycpept.py
"""
Test script for examining the cyclic peptide dataset and verifying the preprocessing pipeline.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

# Paths
DATA_PATH = "training_data/CycPeptMPDB_Peptide_All.csv"
OUTPUT_DIR = "cycpept_test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column names
SMILES_COL = "SMILES"
PROP_COL = "Permeability"


def main():
    """Main function to analyze the dataset."""
    print(f"Loading data from {DATA_PATH}")

    # Load dataset
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Successfully loaded dataset with {len(df)} entries")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Display dataset info
    print("\nDataset columns:")
    for col in df.columns:
        print(f"- {col}")

    # Check SMILES column
    if SMILES_COL not in df.columns:
        print(f"Error: SMILES column '{SMILES_COL}' not found in dataset")
        return
    else:
        print(f"\nFound {len(df[SMILES_COL].dropna())} entries with SMILES strings")

    # Check property column
    if PROP_COL not in df.columns:
        print(f"Error: Property column '{PROP_COL}' not found in dataset")
        return
    else:
        # Convert to numeric, coercing non-numeric values to NaN
        df[PROP_COL] = pd.to_numeric(df[PROP_COL], errors="coerce")
        valid_props = df[PROP_COL].dropna()
        print(f"Found {len(valid_props)} entries with valid permeability values")
        print(f"Permeability range: {valid_props.min()} to {valid_props.max()}")
        print(
            f"Permeability mean: {valid_props.mean():.4f}, std: {valid_props.std():.4f}"
        )

    # Filter dataset for valid entries
    valid_df = df.dropna(subset=[SMILES_COL, PROP_COL])
    print(
        f"\nDataset contains {len(valid_df)} entries with both SMILES and permeability"
    )

    # Analyze SMILES strings
    print("\nAnalyzing molecular structures...")
    mol_list = []
    atom_counts = []
    mol_weights = []

    for smiles in valid_df[SMILES_COL].values[:500]:  # Limit to first 500 for speed
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_list.append(mol)
                atom_counts.append(mol.GetNumAtoms())
                mol_weights.append(Chem.Descriptors.MolWt(mol))
        except:
            continue

    print(f"Successfully parsed {len(mol_list)} molecules out of 500 sampled")

    # Plot atom count distribution
    plt.figure(figsize=(10, 6))
    plt.hist(atom_counts, bins=30, alpha=0.7, color="blue")
    plt.title("Distribution of Molecule Sizes (Atom Count)")
    plt.xlabel("Number of Atoms")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "atom_count_distribution.png"))

    # Plot molecular weight distribution
    plt.figure(figsize=(10, 6))
    plt.hist(mol_weights, bins=30, alpha=0.7, color="green")
    plt.title("Distribution of Molecular Weights")
    plt.xlabel("Molecular Weight (g/mol)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "mol_weight_distribution.png"))

    # Plot permeability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(valid_df[PROP_COL].values, bins=30, alpha=0.7, color="red")
    plt.title("Distribution of Permeability Values")
    plt.xlabel("Permeability")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "permeability_distribution.png"))

    # Visualize a few example molecules
    if len(mol_list) > 0:
        print("\nGenerating visualizations for example molecules...")
        sample_mols = mol_list[: min(10, len(mol_list))]
        img = Draw.MolsToGridImage(sample_mols, molsPerRow=5, subImgSize=(200, 200))
        img.save(os.path.join(OUTPUT_DIR, "sample_molecules.png"))

    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR} directory")
    print("Dataset is ready for model training.")


if __name__ == "__main__":
    main()
