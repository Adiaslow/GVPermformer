# scripts/clean_cycpept_data.py

"""
Script to clean CycPeptMPDB data by removing duplicates based on Structurally_Unique_ID
and keeping only SMILES and PAMPA columns.
"""

import pandas as pd


def clean_cycpept_data(input_file: str, output_file: str) -> None:
    """
    Clean CycPeptMPDB data by removing duplicates and keeping specific columns.

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output DSV file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Remove duplicates based on Structurally_Unique_ID
    df = df.drop_duplicates(subset=["Structurally_Unique_ID"])

    # Keep only SMILES and PAMPA columns
    df = df[["SMILES", "PAMPA"]]

    # Save as DSV file (using | as delimiter)
    df.to_csv(output_file, sep="|", index=False)

    print(f"Original shape: {len(pd.read_csv(input_file))} rows")
    print(f"After removing duplicates: {len(df)} rows")
    print(f"Saved cleaned data to {output_file}")


if __name__ == "__main__":
    input_file = "training_data/CycPeptMPDB_Peptide_All.csv"
    output_file = "training_data/CycPeptMPDB_cleaned.dsv"
    clean_cycpept_data(input_file, output_file)
