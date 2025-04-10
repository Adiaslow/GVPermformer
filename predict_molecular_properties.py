# predict_molecular_properties.py
"""
Command-line utility for predicting molecular properties using GraphVAE model.

This script provides a simple interface for predicting properties of molecules
using a trained Graph Variational Autoencoder (GraphVAE) model. It supports
both single SMILES input and batch processing from files.
"""

import os
import sys
import argparse
import logging
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_available_device() -> torch.device:
    """Determine the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def preprocess_smiles(smiles: str) -> str:
    """Basic preprocessing for SMILES strings."""
    # Remove any whitespace or newlines
    return smiles.strip()


def read_smiles_from_file(file_path: str) -> List[str]:
    """Read SMILES strings from a file, one per line."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as f:
        smiles_list = [preprocess_smiles(line) for line in f if line.strip()]

    if not smiles_list:
        raise ValueError(f"No valid SMILES strings found in {file_path}")

    return smiles_list


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load a trained GraphVAE model from a checkpoint.

    Args:
        model_path: Path to the model checkpoint file
        device: Device to load the model onto

    Returns:
        Loaded model
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        # Attempt to dynamically import required modules
        try:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from src.models.graph_vae import GraphVAE
            from src.utils.molecule_features import MoleculeFeaturizer
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            logger.error(
                "Make sure you're running the script from the project root directory"
            )
            sys.exit(1)

        # Load checkpoint to CPU first for better compatibility
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

        # Extract hyperparameters from the checkpoint
        hparams = checkpoint.get("hyper_parameters", {})
        if not hparams:
            logger.warning(
                "Could not find hyperparameters in checkpoint, using defaults"
            )
            hparams = {}

        # Create the model with hyperparameters
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

        return model, MoleculeFeaturizer()

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        sys.exit(1)


def predict_properties(
    model: torch.nn.Module,
    featurizer,
    smiles_list: List[str],
    device: torch.device,
    batch_size: int = 32,
    extract_latent: bool = False,
) -> pd.DataFrame:
    """
    Predict properties for a list of SMILES strings.

    Args:
        model: Loaded model
        featurizer: Molecule featurizer
        smiles_list: List of SMILES strings to predict
        device: Device to run inference on
        batch_size: Number of molecules to process at once
        extract_latent: Whether to include latent representation in output

    Returns:
        DataFrame with SMILES and predicted properties
    """
    # Track results
    all_results = []

    # Process in batches to avoid memory issues
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(smiles_list)-1)//batch_size + 1} with {len(batch_smiles)} molecules"
        )

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

                # Extract results
                batch_results = []

                # Get predictions
                property_pred = output.get("property_pred", None)
                latent_vectors = output.get("z", None) if extract_latent else None

                for j, smiles in enumerate(batch_data["smiles"]):
                    result = {"SMILES": smiles}

                    # Add property prediction if available
                    if property_pred is not None:
                        result["Predicted_PAMPA"] = property_pred[j].item()

                    # Add reconstruction loss if available
                    if "loss" in output:
                        result["Reconstruction_Loss"] = output["loss"][j].item()

                    # Add latent vector if requested
                    if extract_latent and latent_vectors is not None:
                        for k, val in enumerate(latent_vectors[j].cpu().numpy()):
                            result[f"latent_{k}"] = val

                    batch_results.append(result)

                all_results.extend(batch_results)

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Try to process individual SMILES to identify problematic ones
            for smiles in batch_smiles:
                try:
                    # See if this specific SMILES can be processed
                    featurizer.smiles_to_data(smiles)
                    logger.info(
                        f"  - SMILES processed individually but failed in batch: {smiles}"
                    )
                except Exception as e2:
                    logger.error(f"  - Invalid SMILES: {smiles}")

    if not all_results:
        logger.error("No valid predictions were generated")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    return df


def visualize_predictions(df: pd.DataFrame, output_prefix: str = None) -> None:
    """Generate visualizations of the predictions."""
    try:
        import matplotlib.pyplot as plt
        from rdkit import Chem
        from rdkit.Chem import Draw

        # Only proceed if we have matplotlib and rdkit
        if len(df) == 0:
            return

        # Create folder for visualizations
        if output_prefix:
            vis_dir = os.path.join(os.path.dirname(output_prefix), "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
        else:
            vis_dir = "visualizations"
            os.makedirs(vis_dir, exist_ok=True)

        # 1. Histogram of predicted values
        if "Predicted_PAMPA" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df["Predicted_PAMPA"], bins=20, alpha=0.7)
            plt.xlabel("Predicted PAMPA")
            plt.ylabel("Count")
            plt.title("Distribution of Predicted PAMPA Values")
            plt.grid(alpha=0.3)

            hist_path = os.path.join(vis_dir, "pampa_distribution.png")
            plt.savefig(hist_path)
            plt.close()
            logger.info(f"Saved histogram to {hist_path}")

        # 2. Generate molecule images for up to 10 molecules
        sample_size = min(10, len(df))
        if sample_size > 0:
            # Select a sample of molecules - include highest and lowest predicted values
            if "Predicted_PAMPA" in df.columns:
                # Get indices of highest and lowest predictions
                sorted_df = df.sort_values("Predicted_PAMPA")
                low_idx = sorted_df.index[: sample_size // 2]
                high_idx = sorted_df.index[-sample_size // 2 :]
                sample_idx = list(low_idx) + list(high_idx)
                sample_df = df.loc[sample_idx].copy()
            else:
                sample_df = df.sample(sample_size).copy()

            # Generate molecule images
            mols = []
            labels = []
            for i, row in sample_df.iterrows():
                try:
                    mol = Chem.MolFromSmiles(row["SMILES"])
                    if mol:
                        mols.append(mol)
                        if "Predicted_PAMPA" in row:
                            labels.append(f"PAMPA: {row['Predicted_PAMPA']:.2f}")
                        else:
                            labels.append("")
                except:
                    logger.warning(
                        f"Could not generate image for SMILES: {row['SMILES']}"
                    )

            if mols:
                img = Draw.MolsToGridImage(
                    mols, molsPerRow=2, subImgSize=(300, 300), legends=labels
                )
                img_path = os.path.join(vis_dir, "molecule_samples.png")
                img.save(img_path)
                logger.info(f"Saved molecule images to {img_path}")

    except ImportError:
        logger.warning("Visualization requires matplotlib and rdkit - skipping")
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")


def main():
    """Main entry point for the prediction script."""
    parser = argparse.ArgumentParser(
        description="Predict properties of molecules using a trained GraphVAE model"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )

    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--smiles", type=str, help="SMILES string of the molecule to predict"
    )
    input_group.add_argument(
        "--smiles_file",
        type=str,
        help="Path to file containing SMILES strings (one per line)",
    )

    # Output arguments
    parser.add_argument(
        "--output", type=str, help="Path to save prediction results as CSV"
    )

    # Additional options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing multiple molecules",
    )
    parser.add_argument(
        "--extract_latent",
        action="store_true",
        help="Extract latent representations alongside predictions",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of the predictions",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = get_available_device()
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load model
    model, featurizer = load_model(args.model_path, device)

    # Get SMILES strings
    if args.smiles:
        smiles_list = [preprocess_smiles(args.smiles)]
        logger.info(f"Processing single SMILES: {smiles_list[0]}")
    else:
        logger.info(f"Reading SMILES from file: {args.smiles_file}")
        smiles_list = read_smiles_from_file(args.smiles_file)
        logger.info(f"Loaded {len(smiles_list)} SMILES strings")

    # Predict properties
    results = predict_properties(
        model=model,
        featurizer=featurizer,
        smiles_list=smiles_list,
        device=device,
        batch_size=args.batch_size,
        extract_latent=args.extract_latent,
    )

    # Check if we have any results
    if results.empty:
        logger.error("No valid predictions generated, exiting")
        return 1

    # Display results
    if len(results) == 1:
        # Single molecule result
        logger.info("\nPrediction Results:")
        for col in results.columns:
            if col == "SMILES":
                continue
            elif col.startswith("latent_"):
                continue  # Skip latent dimensions in console output
            logger.info(f"  {col}: {results.iloc[0][col]:.4f}")

        if args.extract_latent:
            latent_cols = [col for col in results.columns if col.startswith("latent_")]
            if latent_cols:
                logger.info(f"\nLatent Representation ({len(latent_cols)} dimensions):")
                latent_preview = ", ".join(
                    f"{results.iloc[0][col]:.4f}" for col in latent_cols[:5]
                )
                if len(latent_cols) > 5:
                    latent_preview += f", ... ({len(latent_cols)-5} more)"
                logger.info(f"  [{latent_preview}]")
    else:
        # Multiple results - show summary
        logger.info("\nPrediction Summary:")
        logger.info(f"  Total molecules processed: {len(smiles_list)}")
        logger.info(f"  Successful predictions: {len(results)}")

        if "Predicted_PAMPA" in results.columns:
            logger.info("\nPAMPA Prediction Statistics:")
            logger.info(f"  Min: {results['Predicted_PAMPA'].min():.4f}")
            logger.info(f"  Max: {results['Predicted_PAMPA'].max():.4f}")
            logger.info(f"  Mean: {results['Predicted_PAMPA'].mean():.4f}")
            logger.info(f"  Median: {results['Predicted_PAMPA'].median():.4f}")

    # Save results if specified
    if args.output:
        results.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")

    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations...")
        visualize_predictions(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
