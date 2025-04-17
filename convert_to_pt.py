# convert_to_pt.py

"""
Script to convert preprocessed CSV files to PyTorch graph data files.
Uses comprehensive molecular featurization combining deepchem and graph-based features.
"""

import os
import pandas as pd
import torch
from tqdm import tqdm
from src.utils.molecule_featurizer import MoleculeFeaturizer


def convert_csv_to_pt(input_dir: str):
    """Convert CSV files to PyTorch graph data files with rich featurization."""
    # Initialize featurizer
    featurizer = MoleculeFeaturizer()
    print("Initialized comprehensive molecular featurizer")

    # Process each fold
    for fold_idx in range(5):
        fold_dir = os.path.join(input_dir, f"fold_{fold_idx}")

        # Process each split
        for split in ["train", "val", "test"]:
            csv_path = os.path.join(fold_dir, f"{split}.csv")
            output_dir = os.path.join(fold_dir, split)
            os.makedirs(output_dir, exist_ok=True)

            # Load CSV data
            df = pd.read_csv(csv_path)
            print(f"\nProcessing {csv_path}")
            print(f"Found {len(df)} molecules")

            # Convert each molecule with rich featurization
            successful = 0
            failed = 0
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                try:
                    # Convert SMILES to graph data with comprehensive features
                    graph_data = featurizer.featurize(row["SMILES"])
                    if graph_data is not None:
                        # Add target value
                        graph_data.y = torch.tensor(
                            row["Permeability"], dtype=torch.float
                        ).unsqueeze(0)
                        # Save to file
                        torch.save(
                            graph_data, os.path.join(output_dir, f"graph_{idx}.pt")
                        )
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Error processing molecule {idx}: {e}")
                    failed += 1

            print(f"\nSplit {split} statistics:")
            print(f"Successfully processed: {successful}")
            print(f"Failed: {failed}")
            print(f"Success rate: {successful/(successful+failed)*100:.2f}%")


def main():
    input_dir = "processed_cycpept_data"
    print(
        f"Converting CSV files in {input_dir} to PyTorch format with rich featurization"
    )
    convert_csv_to_pt(input_dir)
    print("\nConversion complete!")


if __name__ == "__main__":
    main()
