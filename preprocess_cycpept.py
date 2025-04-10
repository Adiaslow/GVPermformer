# preprocess_cycpept.py
"""
Preprocess and analyze the cyclic peptide dataset.
This script prepares the dataset for model training by:
1. Filtering invalid molecules
2. Computing molecular features
3. Splitting into train/validation sets
4. Saving processed data
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw

# Configuration
INPUT_PATH = "training_data/CycPeptMPDB_Peptide_All.csv"
OUTPUT_DIR = "processed_cycpept_data"
SMILES_COL = "SMILES"
PERMEABILITY_COL = "Permeability"
MAX_ATOMS = 100
TRAIN_RATIO = 0.8
SEED = 42

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "validation"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

# Set random seed
np.random.seed(SEED)


def compute_molecular_features(mol):
    """Compute features for a molecule."""
    features = {}

    # Basic features
    features["num_atoms"] = mol.GetNumAtoms()
    features["num_bonds"] = mol.GetNumBonds()
    features["mol_weight"] = Descriptors.MolWt(mol)
    features["logp"] = Descriptors.MolLogP(mol)
    features["tpsa"] = Descriptors.TPSA(mol)
    features["hba"] = Descriptors.NumHAcceptors(mol)
    features["hbd"] = Descriptors.NumHDonors(mol)
    features["rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)
    features["rings"] = len(mol.GetRingInfo().AtomRings())

    # Compute 2D fingerprints
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    features["fingerprint"] = list(fp)

    return features


def main():
    """Main function to preprocess the dataset."""
    print(f"Loading dataset from {INPUT_PATH}")

    try:
        df = pd.read_csv(INPUT_PATH)
        print(f"Successfully loaded dataset with {len(df)} entries")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Check columns
    if SMILES_COL not in df.columns:
        print(f"Error: SMILES column '{SMILES_COL}' not found in dataset")
        sys.exit(1)
    if PERMEABILITY_COL not in df.columns:
        print(f"Error: Permeability column '{PERMEABILITY_COL}' not found in dataset")
        sys.exit(1)

    # Convert permeability values to numeric, dropping any non-numeric entries
    df[PERMEABILITY_COL] = pd.to_numeric(df[PERMEABILITY_COL], errors="coerce")
    df = df.dropna(subset=[SMILES_COL, PERMEABILITY_COL])

    print(f"Dataset contains {len(df)} entries with both SMILES and permeability")
    print(
        f"Permeability range: {df[PERMEABILITY_COL].min()} to {df[PERMEABILITY_COL].max()}"
    )
    print(
        f"Permeability mean: {df[PERMEABILITY_COL].mean():.4f}, std: {df[PERMEABILITY_COL].std():.4f}"
    )

    # Process molecules
    print("\nProcessing molecules...")
    processed_data = []
    failed_count = 0
    large_mol_count = 0

    for i, row in df.iterrows():
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} molecules...")

        smiles = row[SMILES_COL]
        permeability = row[PERMEABILITY_COL]

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed_count += 1
                continue

            if mol.GetNumAtoms() > MAX_ATOMS:
                large_mol_count += 1
                continue

            # Compute features
            features = compute_molecular_features(mol)

            # Add data
            processed_data.append(
                {"smiles": smiles, "permeability": permeability, "features": features}
            )

        except Exception as e:
            failed_count += 1
            if i < 10:  # Print first few errors for debugging
                print(f"Error processing molecule {i} ({smiles}): {e}")

    print(f"Successfully processed {len(processed_data)} molecules")
    print(f"Failed to process {failed_count} molecules")
    print(
        f"Skipped {large_mol_count} molecules that exceed the maximum atom count ({MAX_ATOMS})"
    )

    # Convert to DataFrame for analysis
    analysis_df = pd.DataFrame(
        [
            {
                "smiles": item["smiles"],
                "permeability": item["permeability"],
                "num_atoms": item["features"]["num_atoms"],
                "mol_weight": item["features"]["mol_weight"],
                "logp": item["features"]["logp"],
                "tpsa": item["features"]["tpsa"],
                "rings": item["features"]["rings"],
            }
            for item in processed_data
        ]
    )

    # Save processed data
    print("\nSaving processed data...")

    # Shuffle and split data
    indices = np.random.permutation(len(processed_data))
    train_size = int(TRAIN_RATIO * len(processed_data))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = [processed_data[i] for i in train_indices]
    val_data = [processed_data[i] for i in val_indices]

    print(f"Training set: {len(train_data)} molecules")
    print(f"Validation set: {len(val_data)} molecules")

    # Save splits
    train_df = pd.DataFrame(
        [
            {"smiles": item["smiles"], "permeability": item["permeability"]}
            for item in train_data
        ]
    )
    val_df = pd.DataFrame(
        [
            {"smiles": item["smiles"], "permeability": item["permeability"]}
            for item in val_data
        ]
    )

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train", "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "validation", "val_data.csv"), index=False)

    # Save analysis data
    analysis_df.to_csv(os.path.join(OUTPUT_DIR, "molecule_features.csv"), index=False)

    # Create visualizations
    print("\nGenerating visualizations...")

    # Plot feature distributions
    for feature in ["num_atoms", "mol_weight", "logp", "tpsa", "rings"]:
        plt.figure(figsize=(10, 6))
        plt.hist(analysis_df[feature], bins=30, alpha=0.7)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.savefig(
            os.path.join(OUTPUT_DIR, "visualizations", f"{feature}_distribution.png")
        )
        plt.close()

    # Plot permeability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(analysis_df["permeability"], bins=30, alpha=0.7, color="red")
    plt.title("Distribution of Permeability Values")
    plt.xlabel("Permeability")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(
        os.path.join(OUTPUT_DIR, "visualizations", "permeability_distribution.png")
    )
    plt.close()

    # Plot correlation between features and permeability
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(["num_atoms", "mol_weight", "logp", "tpsa"]):
        plt.subplot(2, 2, i + 1)
        plt.scatter(analysis_df[feature], analysis_df["permeability"], alpha=0.5)
        plt.title(f"{feature} vs Permeability")
        plt.xlabel(feature)
        plt.ylabel("Permeability")
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "visualizations", "feature_correlations.png"))
    plt.close()

    # Visualize some example molecules
    mols = []
    perms = []
    for i in range(min(10, len(processed_data))):
        mol = Chem.MolFromSmiles(processed_data[i]["smiles"])
        if mol:
            mols.append(mol)
            perms.append(f"{processed_data[i]['permeability']:.2f}")

    if mols:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(300, 300),
            legends=[f"Perm: {p}" for p in perms],
        )
        img.save(os.path.join(OUTPUT_DIR, "visualizations", "example_molecules.png"))

    print("\nPreprocessing complete!")
    print(f"All data saved to {OUTPUT_DIR} directory")
    print(
        f"Use the processed training data at {os.path.join(OUTPUT_DIR, 'train', 'train_data.csv')}"
    )
    print(
        f"and validation data at {os.path.join(OUTPUT_DIR, 'validation', 'val_data.csv')}"
    )


if __name__ == "__main__":
    main()
