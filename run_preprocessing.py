# run_preprocessing.py

"""
Script to run preprocessing of cyclic peptide data.
"""

import logging
from src.data.preprocess import preprocess_and_split


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run preprocessing
    result = preprocess_and_split(
        input_file="training_data/CycPeptMPDB_Peptide_All.csv",
        output_dir="processed_cycpept_data",
        smiles_col="SMILES",
        target_col="Permeability",
        val_size=0.1,
        test_size=0.1,
        n_splits=5,
        n_bins=10,
        pampa_threshold=-9.0,
        min_atoms=3,
        max_atoms=500,
        random_state=42,
        outlier_method="iqr",
        outlier_threshold=3.0,
        remove_outliers=True,
        stratify_props=True,
    )

    print("\nPreprocessing complete!")
    print("Statistics:")
    for key, value in result["stats"].items():
        print(f"{key}: {value}")

    print("\nOutput files can be found in: processed_cycpept_data/")


if __name__ == "__main__":
    main()
