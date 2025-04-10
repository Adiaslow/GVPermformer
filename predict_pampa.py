# predict_pampa.py
"""
Command-line script to predict PAMPA permeability from SMILES strings.
This script demonstrates how to use the trained GraphVAE model for prediction.
"""

import os
import argparse
import torch
from src.models.graph_vae import GraphVAE
from src.utils.smiles_utils import smiles_to_model_input


def load_model(checkpoint_path):
    """Load a trained GraphVAE model from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load the model from checkpoint
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = GraphVAE.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Using device: {device}")
    return model


def predict_from_smiles(model, smiles):
    """Predict PAMPA permeability from a SMILES string."""
    try:
        # Simplified approach: use the model's predict_pampa method
        pampa_value = model.predict_pampa(smiles)
        return pampa_value
    except Exception as e:
        print(f"Error predicting for SMILES {smiles}: {str(e)}")
        return None


def predict_from_file(model, input_file, output_file):
    """Predict PAMPA permeability for SMILES strings in a file."""
    results = []

    # Read SMILES from input file
    with open(input_file, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(smiles_list)} SMILES strings...")

    # Process each SMILES
    for i, smiles in enumerate(smiles_list):
        if i % 10 == 0:
            print(f"Processing {i}/{len(smiles_list)}...")

        pampa_value = predict_from_smiles(model, smiles)
        results.append((smiles, pampa_value))

    # Write results to output file
    with open(output_file, "w") as f:
        f.write("SMILES,PAMPA_Prediction\n")
        for smiles, pampa in results:
            f.write(f"{smiles},{pampa if pampa is not None else 'N/A'}\n")

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict PAMPA permeability from SMILES strings"
    )

    # Add arguments
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint file"
    )
    parser.add_argument("--smiles", help="SMILES string to predict")
    parser.add_argument(
        "--input_file", help="File containing SMILES strings (one per line)"
    )
    parser.add_argument(
        "--output_file", default="predictions.csv", help="Output file for predictions"
    )

    args = parser.parse_args()

    # Load the model
    model = load_model(args.checkpoint)

    # Process input
    if args.smiles:
        # Predict for a single SMILES string
        pampa_value = predict_from_smiles(model, args.smiles)
        print(f"SMILES: {args.smiles}")
        print(f"Predicted PAMPA: {pampa_value}")

    elif args.input_file:
        # Predict for SMILES strings in a file
        predict_from_file(model, args.input_file, args.output_file)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
