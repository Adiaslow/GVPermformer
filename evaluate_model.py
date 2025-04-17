"""
Comprehensive evaluation script for GraphVAE model.
Analyzes model performance, generates visualizations, and produces insightful plots.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rdkit import Chem
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.models.graph_vae import GraphVAE
from src.data.cycpept_dataset import CycPeptDataset
from src.utils.visualization_utils import (
    visualize_molecule,
    visualize_molecules_grid,
    visualize_latent_space,
    plot_training_history,
    plot_property_correlations,
)
from torch.utils.data import DataLoader, random_split


def get_device():
    """
    Determine the device to use for loading the model.

    Returns:
        torch.device: The determined device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model(model_path, device=None):
    """
    Load a trained model from a checkpoint file.

    Args:
        model_path (str): Path to the model checkpoint file
        device (torch.device, optional): Device to load the model on. Defaults to None.

    Returns:
        tuple: (model, device) - The loaded model and the device it's loaded to
    """
    import os
    import torch
    import sys

    sys.path.append("/Users/Adam/GVPermformer")

    # Import locally to avoid circular imports
    try:
        from src.models.graph_vae import GraphVAE, GraphEncoder, GraphDecoder
        from src.config.config_cycpept import Config
        from src.utils.device_utils import get_device
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        raise

    # Set device if not provided
    if device is None:
        device = get_device()

    print(f"Loading model from {model_path} to device {device}")

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

    # Extract configuration
    config_dict = checkpoint.get("config", {})
    print(f"Model configuration: {config_dict}")

    # Initialize config with loaded values
    config = Config()

    # Set device in config for later use
    config.device = device

    # Extract necessary parameters for model initialization
    node_features = config_dict.get("node_features", 126)
    edge_features = config_dict.get("edge_features", 9)
    hidden_dim = config_dict.get("hidden_dim", 256)
    latent_dim = config_dict.get("latent_dim", 64)

    # Create a complete model
    model = GraphVAE(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        property_prediction=True,
        use_mps_optimizations=(device.type == "mps"),
    )

    # Load model weights
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Error loading model state dict: {e}")
        print("Attempting to load with strict=False")
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("Model loaded with strict=False")
        except Exception as e2:
            print(f"Error loading model with strict=False: {e2}")
            raise

    model.to(device)
    model.eval()

    return model, config


def create_test_dataloader(data_csv, batch_size=32):
    """
    Create a dataloader for testing the model.

    Args:
        data_csv (str): Path to the CSV file containing test data
        batch_size (int, optional): Batch size for the dataloader. Defaults to 32.

    Returns:
        tuple: (dataloader, property_names) - Test dataloader and list of property names
    """
    import sys
    import torch
    from torch.utils.data import DataLoader

    sys.path.append("/Users/Adam/GVPermformer")

    try:
        from src.data.dataset import CycPeptDataset
        from src.utils.collate import collate_fn
    except ImportError:
        # Fallback for collate_fn if not found in module
        def collate_fn(batch):
            """
            Custom collate function for batching graph data.

            Args:
                batch: List of data samples

            Returns:
                dict: Batched data
            """
            if not batch:
                return {}

            # Check if we're dealing with dictionaries
            if isinstance(batch[0], dict):
                result = {key: [] for key in batch[0].keys()}

                # Collect all items by key
                for data in batch:
                    for key, value in data.items():
                        result[key].append(value)

                # Convert lists to tensors where possible
                for key in result:
                    if isinstance(result[key][0], torch.Tensor):
                        try:
                            result[key] = torch.stack(result[key])
                        except:
                            # If tensors can't be stacked (different sizes), keep as list
                            pass
                return result

            # If batch is not dictionary-based, return as is
            return batch

    print(f"Loading test data from {data_csv}")

    # Create dataset with property prediction enabled
    try:
        dataset = CycPeptDataset(
            csv_file=data_csv,
            transform=None,
            filter_invalid=True,
            property_prediction=True,
        )
        print(f"Created dataset with {len(dataset)} samples")

        # Get property names if available
        property_names = (
            dataset.property_names if hasattr(dataset, "property_names") else []
        )
        print(f"Property names: {property_names}")

    except Exception as e:
        print(f"Error creating dataset: {e}")
        raise

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return dataloader, property_names


def evaluate_predictions(model, test_loader, device, output_dir):
    """Evaluate model predictions and generate performance plots."""
    # Lists to store predictions and ground truth
    all_predictions = []
    all_targets = []
    all_latent_vectors = []

    # Evaluate model
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get predictions
            outputs = model(batch)

            # Get latent vectors - assuming model's encode method returns z_mean and z_logvar
            z_mean, _ = model.encode(batch)

            # Store predictions and targets
            if "target" in batch:
                predictions = outputs["prop_pred"].cpu().numpy()
                targets = batch["target"].cpu().numpy()
                all_predictions.extend(predictions)
                all_targets.extend(targets)

            # Store latent vectors
            all_latent_vectors.append(z_mean.cpu().numpy())

    # Convert to numpy arrays
    if all_predictions:
        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()

    all_latent_vectors = np.vstack(all_latent_vectors)

    # Calculate metrics if we have predictions and targets
    if all_predictions.size > 0:
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)

        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R²: {r2:.4f}")

        # Plot predictions vs ground truth
        plt.figure(figsize=(10, 8))
        plt.scatter(all_targets, all_predictions, alpha=0.6)

        # Plot perfect prediction line
        min_val = min(np.min(all_targets), np.min(all_predictions))
        max_val = max(np.max(all_targets), np.max(all_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        # Add metrics to plot
        plt.text(
            0.05,
            0.95,
            f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.xlabel("Ground Truth PAMPA")
        plt.ylabel("Predicted PAMPA")
        plt.title("Predicted vs. Actual PAMPA Values")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "prediction_scatter.png"), dpi=300)
        plt.close()

        # Create residual plot
        residuals = all_predictions - all_targets
        plt.figure(figsize=(10, 8))
        plt.scatter(all_targets, residuals, alpha=0.6)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Ground Truth PAMPA")
        plt.ylabel("Residuals (Predicted - Actual)")
        plt.title("Residual Plot")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save residual plot
        plt.savefig(os.path.join(output_dir, "residual_plot.png"), dpi=300)
        plt.close()

        # Create distribution plot
        plt.figure(figsize=(12, 8))
        sns.histplot(all_targets, color="blue", label="Actual", kde=True, alpha=0.5)
        sns.histplot(
            all_predictions, color="red", label="Predicted", kde=True, alpha=0.5
        )
        plt.legend()
        plt.xlabel("PAMPA Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Actual vs. Predicted PAMPA Values")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save distribution plot
        plt.savefig(os.path.join(output_dir, "distribution_plot.png"), dpi=300)
        plt.close()

    # Visualize latent space
    visualize_latent_space(
        all_latent_vectors,
        colors=all_targets if all_predictions.size > 0 else None,
        method="pca",
        n_components=2,
        figsize=(10, 8),
        title="Latent Space Visualization (PCA)",
        colorbar_label="PAMPA Value" if all_predictions.size > 0 else None,
        save_path=os.path.join(output_dir, "latent_space_pca.png"),
        show=False,
    )

    # Visualize latent space with t-SNE
    visualize_latent_space(
        all_latent_vectors,
        colors=all_targets if all_predictions.size > 0 else None,
        method="tsne",
        n_components=2,
        figsize=(10, 8),
        title="Latent Space Visualization (t-SNE)",
        colorbar_label="PAMPA Value" if all_predictions.size > 0 else None,
        save_path=os.path.join(output_dir, "latent_space_tsne.png"),
        show=False,
    )

    # Save latent vectors and predictions for further analysis
    if all_predictions.size > 0:
        results_df = pd.DataFrame({"Actual": all_targets, "Predicted": all_predictions})
        results_df.to_csv(
            os.path.join(output_dir, "prediction_results.csv"), index=False
        )

    # Save latent vectors
    latent_df = pd.DataFrame(all_latent_vectors)
    latent_df.to_csv(os.path.join(output_dir, "latent_vectors.csv"), index=False)

    # Return results
    if all_predictions.size > 0:
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "predictions": all_predictions,
            "targets": all_targets,
            "latent_vectors": all_latent_vectors,
        }
    else:
        return {"latent_vectors": all_latent_vectors}


def analyze_latent_dimensions(latent_vectors, targets=None, output_dir=None):
    """Analyze the importance of different latent dimensions."""
    if targets is None or len(targets) == 0:
        print("No targets provided for latent dimension analysis")
        return

    # Correlation with target
    correlations = []
    for i in range(latent_vectors.shape[1]):
        corr = np.corrcoef(latent_vectors[:, i], targets)[0, 1]
        correlations.append((i, corr, abs(corr)))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[2], reverse=True)

    # Plot top 10 dimensions
    top_dims = correlations[:10]

    plt.figure(figsize=(12, 8))
    indices = [t[0] for t in top_dims]
    corrs = [t[1] for t in top_dims]

    plt.bar(range(len(indices)), corrs)
    plt.xticks(range(len(indices)), [f"Dim {i}" for i in indices])
    plt.xlabel("Latent Dimensions")
    plt.ylabel("Correlation with Target")
    plt.title("Top 10 Latent Dimensions by Correlation with PAMPA")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if output_dir:
        plt.savefig(
            os.path.join(output_dir, "latent_dimension_correlations.png"), dpi=300
        )
    plt.close()

    # Plot scatter plots for top 5 dimensions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (dim_idx, corr, _) in enumerate(top_dims[:6]):
        ax = axes[i]
        ax.scatter(latent_vectors[:, dim_idx], targets, alpha=0.6)
        ax.set_xlabel(f"Dimension {dim_idx}")
        ax.set_ylabel("PAMPA Value")
        ax.set_title(f"Dim {dim_idx} (r = {corr:.3f})")
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "top_dimension_scatters.png"), dpi=300)
    plt.close()

    # Save correlations
    corr_df = pd.DataFrame(
        correlations, columns=["Dimension", "Correlation", "Abs_Correlation"]
    )
    if output_dir:
        corr_df.to_csv(
            os.path.join(output_dir, "latent_dimension_correlations.csv"), index=False
        )

    return corr_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GraphVAE model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to saved model checkpoint"
    )
    parser.add_argument(
        "--data_csv", type=str, required=True, help="Path to test data CSV"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, config = load_model(args.model_path)

    # Load data
    test_loader, property_names = create_test_dataloader(args.data_csv, args.batch_size)

    # Evaluate predictions
    results = evaluate_predictions(model, test_loader, config.device, args.output_dir)

    # Analyze latent dimensions
    if "targets" in results:
        analyze_latent_dimensions(
            results["latent_vectors"], results["targets"], args.output_dir
        )

    print(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
