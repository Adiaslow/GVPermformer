# example_predict.py
"""
Example script demonstrating how to use the GraphVAE model for property prediction.
"""

import os
import subprocess
import pandas as pd
from typing import List


def main():
    """Run example predictions using the trained GraphVAE model."""
    # Create directory for example outputs
    os.makedirs("examples", exist_ok=True)

    # Example SMILES strings
    example_smiles = [
        # Aspirin
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        # Tramadol
        "CCC1(C)C2=C(CC3CC(O)C(C)C(O3)C2)CCN1C",
        # Dexamethasone
        "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
        # Spironolactone
        "CC(=O)SC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C",
        # Salbutamol
        "CC(C)(C)NCC(O)C1=CC(=C(C=C1)O)CO",
    ]

    # Save example SMILES to file
    with open("examples/sample_molecules.txt", "w") as f:
        for smiles in example_smiles:
            f.write(f"{smiles}\n")

    print("Example 1: Single molecule prediction")
    print("-------------------------------------")
    print(f"SMILES: {example_smiles[0]} (Aspirin)")

    # Single molecule prediction example
    single_cmd = f"python predict_properties.py --model_path checkpoints/latest.ckpt --smiles '{example_smiles[0]}' --device auto"
    print(f"Running command: {single_cmd}")
    subprocess.run(single_cmd, shell=True)

    print("\n\nExample 2: Batch prediction from file")
    print("-------------------------------------")

    # Batch prediction example
    batch_cmd = "python predict_properties.py --model_path checkpoints/latest.ckpt --smiles_file examples/sample_molecules.txt --output examples/predictions.csv --extract_latent --device auto"
    print(f"Running command: {batch_cmd}")
    subprocess.run(batch_cmd, shell=True)

    # Display results
    if os.path.exists("examples/predictions.csv"):
        results = pd.read_csv("examples/predictions.csv")

        # Format the latent column for display if it exists
        latent_cols = [col for col in results.columns if col.startswith("latent_")]
        display_df = results.drop(columns=latent_cols, errors="ignore")

        print("\nPrediction results:")
        print(display_df)

        if latent_cols:
            print(
                f"\nLatent space has {len(latent_cols)} dimensions (not shown in table)"
            )
    else:
        print(
            "\nNo results file found. Make sure the model path is correct and the model is trained."
        )

    print("\nNotes:")
    print("- Make sure you have a trained model checkpoint available")
    print("- Update the model path in the commands if needed")
    print("- You can add more SMILES strings to the examples list or to the file")


if __name__ == "__main__":
    main()
