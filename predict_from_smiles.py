# predict_from_smiles.py
"""
Script to predict properties from SMILES strings using a trained GraphVAE model.
"""

import os
import torch
import argparse
from typing import List, Optional
import pytorch_lightning as pl

from src.models.graph_vae import GraphVAE
from src.utils.smiles_to_features import smiles_to_graph_data, batch_smiles_to_features


def load_model(checkpoint_path: str) -> GraphVAE:
    """
    Load a trained GraphVAE model from a checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint

    Returns:
        Loaded GraphVAE model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load the model
    model = GraphVAE.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Move to GPU if available
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    print(f"Model loaded from {checkpoint_path} (on {device})")
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )

    return model


def predict_from_smiles(
    model: GraphVAE, smiles_list: List[str]
) -> Optional[torch.Tensor]:
    """
    Predict properties for a list of SMILES strings.

    Args:
        model: Trained GraphVAE model
        smiles_list: List of SMILES strings

    Returns:
        Tensor of predicted properties or None if no valid predictions
    """
    # Convert SMILES to features
    batch = batch_smiles_to_features(smiles_list)
    if batch is None:
        print("No valid molecules found in the input")
        return None

    # Move to the same device as the model
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Make predictions
    with torch.no_grad():
        predictions = model.predict_properties(batch)

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Predict properties from SMILES strings using a trained GraphVAE model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--smiles", type=str, nargs="+", help="SMILES strings to predict properties for"
    )
    parser.add_argument(
        "--file", type=str, help="File containing SMILES strings (one per line)"
    )
    parser.add_argument(
        "--output", type=str, help="Path to save predictions (CSV format)"
    )

    args = parser.parse_args()

    # Check input
    if not args.smiles and not args.file:
        parser.error("Please provide either --smiles or --file")

    # Load model
    model = load_model(args.checkpoint)

    # Get SMILES strings
    smiles_list = []
    if args.smiles:
        smiles_list.extend(args.smiles)

    if args.file:
        with open(args.file, "r") as f:
            for line in f:
                smiles = line.strip()
                if smiles:
                    smiles_list.append(smiles)

    print(f"Processing {len(smiles_list)} SMILES strings...")

    # Make predictions
    predictions = predict_from_smiles(model, smiles_list)

    # Display and save results
    if predictions is not None:
        # Print predictions
        for i, (smiles, pred) in enumerate(zip(smiles_list, predictions)):
            print(f"{i+1}. {smiles}: {pred.item():.4f}")

        # Save predictions to file if requested
        if args.output:
            import pandas as pd

            df = pd.DataFrame(
                {
                    "SMILES": smiles_list,
                    "Prediction": predictions.cpu().numpy().flatten(),
                }
            )
            df.to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
