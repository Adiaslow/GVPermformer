# src/models/evaluate.py
"""
Evaluation module for Graph VAE Transformer model.
Handles model assessment, performance visualization, and latent space exploration.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import summarize

from src.data.dataset import get_dataloader
from src.features.molecular_featurizer import (
    MolecularFeaturizer,
    GlobalFeatureExtractor,
    CombinedFeaturizer,
)
from src.models.train import GraphVAETransformerLightning, collate_batch
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def load_model(
    checkpoint_path: str, map_location: str = "auto"
) -> GraphVAETransformerLightning:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        map_location: Device to load model on ('cpu', 'cuda', 'mps', or 'auto')

    Returns:
        Loaded model
    """
    # Determine device
    if map_location == "auto":
        if torch.backends.mps.is_available():
            map_location = "mps"
        elif torch.cuda.is_available():
            map_location = "cuda"
        else:
            map_location = "cpu"

    print(f"Loading model on {map_location} device")

    # Load model from checkpoint
    model = GraphVAETransformerLightning.load_from_checkpoint(
        checkpoint_path=checkpoint_path, map_location=map_location
    )

    model.eval()
    return model


def evaluate_model(
    model: GraphVAETransformerLightning,
    test_csv: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, float]:
    """
    Evaluate model on test data.

    Args:
        model: Trained model
        test_csv: Path to test CSV file
        batch_size: Batch size for evaluation
        num_workers: Number of dataloader workers

    Returns:
        Dictionary of evaluation metrics
    """
    # Setup featurizer
    mol_featurizer = MolecularFeaturizer()
    global_featurizer = GlobalFeatureExtractor()
    combined_featurizer = CombinedFeaturizer([mol_featurizer, global_featurizer])

    # Create test dataloader
    test_loader = get_dataloader(
        csv_path=test_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=combined_featurizer,
        shuffle=False,
    )

    # Set model to evaluation mode
    model.eval()

    # Lists to store predictions and targets
    all_predictions = []
    all_targets = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to appropriate device
            device = next(model.parameters()).device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            outputs = model(batch)

            # Get predictions and targets
            predictions = outputs["permeability"].cpu().numpy()
            targets = batch["target"].view(-1, 1).cpu().numpy()

            # Store predictions and targets
            all_predictions.extend(predictions)
            all_targets.extend(targets)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    # Print metrics
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")

    # Return metrics
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": all_predictions,
        "targets": all_targets,
    }


def plot_predictions(
    predictions: np.ndarray, targets: np.ndarray, save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot predicted vs actual values.

    Args:
        predictions: Predicted values
        targets: Actual values
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot data
    ax.scatter(targets, predictions, alpha=0.6)

    # Add perfect prediction line
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    ax.plot([min_val, max_val], [min_val, max_val], "r--")

    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    # Add metrics to plot
    ax.text(
        0.05,
        0.95,
        f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Set labels and title
    ax.set_xlabel("Actual Permeability")
    ax.set_ylabel("Predicted Permeability")
    ax.set_title("Predicted vs Actual Permeability")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


def visualize_latent_space(
    model: GraphVAETransformerLightning,
    test_csv: str,
    batch_size: int = 32,
    num_workers: int = 4,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
    """
    Visualize latent space using t-SNE.

    Args:
        model: Trained model
        test_csv: Path to test CSV file
        batch_size: Batch size for evaluation
        num_workers: Number of dataloader workers
        save_path: Optional path to save the plot

    Returns:
        Tuple of (figure, latent_vectors, targets)
    """
    # Setup featurizer
    mol_featurizer = MolecularFeaturizer()
    global_featurizer = GlobalFeatureExtractor()
    combined_featurizer = CombinedFeaturizer([mol_featurizer, global_featurizer])

    # Create test dataloader
    test_loader = get_dataloader(
        csv_path=test_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=combined_featurizer,
        shuffle=False,
    )

    # Set model to evaluation mode
    model.eval()

    # Lists to store latent vectors and targets
    all_latent_vectors = []
    all_targets = []
    all_smiles = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to appropriate device
            device = next(model.parameters()).device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            outputs = model(batch)

            # Get latent vectors and targets
            latent_vectors = outputs["z"].cpu().numpy()
            targets = batch["target"].view(-1, 1).cpu().numpy()

            # Store latent vectors and targets
            all_latent_vectors.extend(latent_vectors)
            all_targets.extend(targets)

            # Store SMILES strings if available
            if "smiles" in batch:
                all_smiles.extend(batch["smiles"])

    # Convert to numpy arrays
    all_latent_vectors = np.array(all_latent_vectors)
    all_targets = np.array(all_targets).flatten()

    # Apply t-SNE for dimensionality reduction
    print("Applying t-SNE to latent vectors...")
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(all_latent_vectors) - 1)
    )
    tsne_result = tsne.fit_transform(all_latent_vectors)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot t-SNE points colored by target value
    scatter = ax.scatter(
        tsne_result[:, 0],
        tsne_result[:, 1],
        c=all_targets,
        cmap="viridis",
        alpha=0.8,
        s=50,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Permeability")

    # Set labels and title
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("Latent Space Visualization (t-SNE)")

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    # Return figure and data
    return fig, all_latent_vectors, all_targets


def plot_latent_traversal(
    model: GraphVAETransformerLightning,
    test_csv: str,
    dimension: int = 0,
    num_points: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot latent space traversal for a selected dimension.

    Args:
        model: Trained model
        test_csv: Path to test CSV file containing examples
        dimension: Latent dimension to traverse
        num_points: Number of points along traversal
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    # Setup featurizer
    mol_featurizer = MolecularFeaturizer()
    global_featurizer = GlobalFeatureExtractor()
    combined_featurizer = CombinedFeaturizer([mol_featurizer, global_featurizer])

    # Load a single example
    dataloader = get_dataloader(
        csv_path=test_csv,
        batch_size=1,
        num_workers=0,
        transform=combined_featurizer,
        shuffle=True,
    )

    # Get a single batch
    batch = next(iter(dataloader))

    # Move batch to appropriate device
    device = next(model.parameters()).device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    # Forward pass to get latent vector
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        z = outputs["z"].clone()

    # Check if dimension is valid
    if dimension >= z.shape[1]:
        raise ValueError(f"Dimension {dimension} is out of bounds (0-{z.shape[1]-1})")

    # Get range of values to traverse
    latent_dim = z.shape[1]
    z_min = -3.0  # Assume standard normal distribution in latent space
    z_max = 3.0
    values = np.linspace(z_min, z_max, num_points)

    # Create figure for traversal
    fig, axes = plt.subplots(1, num_points, figsize=(num_points * 2, 4))

    # Traverse the dimension
    permeability_values = []
    for i, val in enumerate(values):
        # Create a copy of the latent vector
        z_new = z.clone()

        # Modify the selected dimension
        z_new[0, dimension] = val

        # Generate prediction from modified latent vector
        with torch.no_grad():
            permeability = model.predict_permeability(z_new)
            permeability_values.append(permeability.item())

    # Plot permeability values along traversal
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(values, permeability_values, "o-", linewidth=2)
    ax.set_xlabel(f"Latent Dimension {dimension} Value")
    ax.set_ylabel("Predicted Permeability")
    ax.set_title(f"Latent Dimension {dimension} Traversal")
    ax.grid(True, linestyle="--", alpha=0.7)

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


def analyze_latent_dimensions(
    model: GraphVAETransformerLightning,
    test_csv: str,
    batch_size: int = 32,
    num_workers: int = 4,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Analyze sensitivity of each latent dimension for permeability prediction.

    Args:
        model: Trained model
        test_csv: Path to test CSV file
        batch_size: Batch size for evaluation
        num_workers: Number of dataloader workers
        save_dir: Optional directory to save the plots

    Returns:
        DataFrame with dimension sensitivity analysis
    """
    # Setup featurizer
    mol_featurizer = MolecularFeaturizer()
    global_featurizer = GlobalFeatureExtractor()
    combined_featurizer = CombinedFeaturizer([mol_featurizer, global_featurizer])

    # Create test dataloader
    test_loader = get_dataloader(
        csv_path=test_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=combined_featurizer,
        shuffle=False,
    )

    # Set model to evaluation mode
    model.eval()

    # Get latent dimension
    latent_dim = model.model.latent_dim

    # Initialize results
    dimension_sensitivity = []

    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Process examples
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Only process the first few batches
            if batch_idx >= 5:
                break

            # Move batch to appropriate device
            device = next(model.parameters()).device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            outputs = model(batch)
            z = outputs["z"]
            original_permeability = outputs["permeability"]

            # Analyze each dimension
            for dim in range(latent_dim):
                # Create a copy of the latent vector
                z_plus = z.clone()
                z_minus = z.clone()

                # Modify the selected dimension
                z_plus[:, dim] += 1.0  # Increase by 1 standard deviation
                z_minus[:, dim] -= 1.0  # Decrease by 1 standard deviation

                # Generate predictions from modified latent vectors
                perm_plus = model.predict_permeability(z_plus)
                perm_minus = model.predict_permeability(z_minus)

                # Calculate sensitivity for each example in batch
                for i in range(len(z)):
                    sensitivity = torch.abs(perm_plus[i] - perm_minus[i]) / 2.0

                    # Store sensitivity values
                    dimension_sensitivity.append(
                        {
                            "dimension": dim,
                            "batch_idx": batch_idx,
                            "example_idx": i,
                            "sensitivity": sensitivity.item(),
                            "original_permeability": original_permeability[i].item(),
                        }
                    )

    # Convert to DataFrame
    sensitivity_df = pd.DataFrame(dimension_sensitivity)

    # Summarize by dimension
    dim_summary = (
        sensitivity_df.groupby("dimension")["sensitivity"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    dim_summary = dim_summary.sort_values("mean", ascending=False)

    # Print dimension summary
    print("Latent Dimension Sensitivity Analysis:")
    print(dim_summary)

    # Create summary plot
    if save_dir:
        plt.figure(figsize=(12, 6))
        sns.barplot(x="dimension", y="mean", data=dim_summary, yerr=dim_summary["std"])
        plt.xlabel("Latent Dimension")
        plt.ylabel("Sensitivity (Mean ± Std)")
        plt.title("Latent Dimension Sensitivity for Permeability Prediction")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "dimension_sensitivity.png"),
            dpi=300,
            bbox_inches="tight",
        )

    return dim_summary


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Evaluate Graph VAE Transformer model")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test_csv", type=str, required=True, help="Path to test CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation",
        help="Directory to save output",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run evaluation on (cpu, cuda, mps, auto)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Print model summary
    print(summarize(model))

    # Evaluate model
    results = evaluate_model(
        model=model, test_csv=args.test_csv, batch_size=args.batch_size
    )

    # Plot predictions
    fig = plot_predictions(
        predictions=results["predictions"],
        targets=results["targets"],
        save_path=os.path.join(args.output_dir, "predictions.png"),
    )

    # Visualize latent space
    fig, latent_vectors, targets = visualize_latent_space(
        model=model,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        save_path=os.path.join(args.output_dir, "latent_space.png"),
    )

    # Analyze latent dimensions
    dim_summary = analyze_latent_dimensions(
        model=model,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        save_dir=args.output_dir,
    )

    # Plot latent traversal for most sensitive dimension
    top_dim = int(dim_summary.iloc[0]["dimension"])
    fig = plot_latent_traversal(
        model=model,
        test_csv=args.test_csv,
        dimension=top_dim,
        save_path=os.path.join(args.output_dir, f"dimension_{top_dim}_traversal.png"),
    )

    # Save results to CSV
    results_df = pd.DataFrame(
        {"actual": results["targets"], "predicted": results["predictions"]}
    )
    results_df.to_csv(
        os.path.join(args.output_dir, "prediction_results.csv"), index=False
    )

    # Save metrics to text file
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"MSE: {results['mse']:.6f}\n")
        f.write(f"RMSE: {results['rmse']:.6f}\n")
        f.write(f"MAE: {results['mae']:.6f}\n")
        f.write(f"R²: {results['r2']:.6f}\n")

    print(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
