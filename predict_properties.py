#!/usr/bin/env python
# predict_properties.py
"""
Script for predicting properties of molecules using a trained GraphVAE model.

Usage:
  python predict_properties.py --model_path path/to/checkpoint.ckpt --smiles "CC(=O)Oc1ccccc1C(=O)O"
  python predict_properties.py --model_path path/to/checkpoint.ckpt --smiles_file molecules.txt --output predictions.csv
"""

import os
import sys
import argparse
import torch
import pandas as pd
import logging
from typing import List, Dict, Union, Optional

from src.models.graph_vae import GraphVAE
from src.utils.molecule_features import MoleculeFeaturizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict properties for molecules using a trained GraphVAE model"
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--smiles", type=str, help="SMILES string of the molecule to predict"
    )
    input_group.add_argument(
        "--smiles_file",
        type=str,
        help="File containing SMILES strings (one per line)",
    )

    # Optional arguments
    parser.add_argument(
        "--output", type=str, help="Path to save prediction results CSV"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--extract_latent",
        action="store_true",
        help="Extract and include latent representation in output",
    )

    return parser.parse_args()


def load_model(model_path: str, device: Union[str, torch.device] = "auto") -> GraphVAE:
    """
    Load a trained GraphVAE model from a checkpoint.

    Args:
        model_path: Path to the model checkpoint file
        device: Device to load the model onto (auto, cpu, cuda, mps)

    Returns:
        Loaded GraphVAE model
    """
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")

    # Load the checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Load checkpoint to CPU first for better compatibility
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # Extract hyperparameters from the checkpoint
    hparams = checkpoint.get("hyper_parameters", {})
    if not hparams:
        raise ValueError("Could not find hyperparameters in checkpoint")

    # Create the model
    model = GraphVAE(
        node_dim=hparams.get("node_dim", 126),
        edge_dim=hparams.get("edge_dim", 9),
        hidden_dim=hparams.get("hidden_dim", 256),
        latent_dim=hparams.get("latent_dim", 64),
        num_layers=hparams.get("num_layers", 3),
        dropout=hparams.get("dropout", 0.1),
        use_batch_norm=hparams.get("use_batch_norm", True),
        use_huber_loss=hparams.get("use_huber_loss", False),
    )

    # Load the state dict
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        f"Model loaded successfully with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters"
    )

    return model


def predict_from_smiles(
    model: GraphVAE,
    smiles_list: List[str],
    device: torch.device,
    extract_latent: bool = False,
) -> pd.DataFrame:
    """
    Predict properties for a list of SMILES strings.

    Args:
        model: Loaded GraphVAE model
        smiles_list: List of SMILES strings to predict
        device: Device to run inference on
        extract_latent: Whether to include latent representation in output

    Returns:
        DataFrame with SMILES and predicted properties
    """
    # Create featurizer
    featurizer = MoleculeFeaturizer()

    # Process SMILES
    results = []
    latent_vectors = []
    valid_smiles = []

    # Process in batches (if list is long)
    batch_size = 32
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i : i + batch_size]

        try:
            # Convert SMILES to molecular graph batch
            batch_data = featurizer.smiles_to_batch(batch_smiles)

            # Move data to device
            batch_data = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_data.items()
            }

            # Run inference
            with torch.no_grad():
                output = model(batch_data)

                # Get predictions
                predictions = output.get("property_pred", None)

                # Get latent representation if requested
                if extract_latent:
                    z = output.get("z", None)
                    if z is not None:
                        latent_vectors.extend(z.cpu().numpy())

            # Add results
            if predictions is not None:
                for j, smiles in enumerate(batch_data["smiles"]):
                    valid_smiles.append(smiles)
                    pred_value = predictions[j].item()
                    results.append({"SMILES": smiles, "Predicted_PAMPA": pred_value})

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            for smiles in batch_smiles:
                try:
                    # Try individual SMILES to identify which ones fail
                    featurizer.smiles_to_data(smiles)
                    logger.info(
                        f"  - SMILES {smiles} processed individually but failed in batch"
                    )
                except Exception as e2:
                    logger.error(f"  - Error with SMILES {smiles}: {e2}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Add latent representations if requested
    if extract_latent and latent_vectors:
        latent_df = pd.DataFrame(
            latent_vectors,
            columns=[f"latent_{i}" for i in range(len(latent_vectors[0]))],
        )
        df = pd.concat([df, latent_df], axis=1)

    return df


def main():
    """Main function for prediction script."""
    # Parse arguments
    args = parse_args()

    # Load model
    model = load_model(args.model_path, device=args.device)
    device = next(model.parameters()).device

    # Get SMILES strings
    if args.smiles:
        smiles_list = [args.smiles]
    else:
        # Read SMILES from file
        if not os.path.exists(args.smiles_file):
            logger.error(f"Error: SMILES file {args.smiles_file} not found")
            return 1

        with open(args.smiles_file, "r") as f:
            smiles_list = [line.strip() for line in f if line.strip()]

    logger.info(f"Predicting properties for {len(smiles_list)} molecules...")

    # Predict properties
    results = predict_from_smiles(
        model=model,
        smiles_list=smiles_list,
        device=device,
        extract_latent=args.extract_latent,
    )

    # Print results to stdout if single SMILES
    if len(results) == 1 and args.smiles:
        logger.info("\nPrediction results:")
        for col in results.columns:
            if col == "SMILES":
                continue
            elif col.startswith("latent_"):
                continue  # Skip latent representations in terminal output
            logger.info(f"{col}: {results.iloc[0][col]:.4f}")

        if args.extract_latent:
            latent_cols = [col for col in results.columns if col.startswith("latent_")]
            if latent_cols:
                logger.info(f"\nLatent representation (dimension: {len(latent_cols)}):")
                latent_str = "[" + ", ".join(
                    f"{results.iloc[0][col]:.4f}" for col in latent_cols[:5]
                )
                if len(latent_cols) > 5:
                    latent_str += f", ... ({len(latent_cols)-5} more dimensions)"
                latent_str += "]"
                logger.info(latent_str)

    # Save results to file if specified
    if args.output:
        results.to_csv(args.output, index=False)
        logger.info(f"\nResults saved to {args.output}")

        # Print summary
        logger.info(f"\nPrediction summary:")
        logger.info(f"  - Total molecules: {len(smiles_list)}")
        logger.info(f"  - Successfully predicted: {len(results)}")
        if len(results) < len(smiles_list):
            logger.info(f"  - Failed molecules: {len(smiles_list) - len(results)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
