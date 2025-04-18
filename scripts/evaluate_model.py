# scripts/evaluate_model.py

"""
Script for evaluating a trained GVPermformer model and generating visualizations.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    # Successfully imported torch_geometric
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logging.warning(
        "torch_geometric is not available. Some functionality may be limited."
    )

    # Define placeholder classes for type hints
    class Data:
        pass

    class Batch:
        pass

    class PyGDataLoader:
        pass


# Import project modules
import sys

sys.path.append(".")  # Add the project root to the path
from src.model import GraphTransformerVAE


def setup_logging(output_dir: str) -> None:
    """
    Setup logging configuration.

    Args:
        output_dir: Directory to save log file
    """
    log_file = os.path.join(output_dir, "evaluation.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GVPermformer model"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing the processed data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to run evaluation on (e.g., "cuda:0", "cpu")',
    )

    return parser.parse_args()


def load_data(data_dir: str, batch_size: int) -> Optional[PyGDataLoader]:
    """
    Load the test dataset.

    Args:
        data_dir: Directory containing processed data
        batch_size: Batch size for data loader

    Returns:
        DataLoader for the test dataset
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        logging.error("torch_geometric is required to load graph data")
        return None

    test_data_path = os.path.join(data_dir, "test_data.pt")
    if not os.path.exists(test_data_path):
        logging.error(f"Test data not found at {test_data_path}")
        return None

    try:
        # Add basic Data class to safe globals, which worked in our test
        torch.serialization.add_safe_globals([Data])

        # Load with weights_only=False which worked in our test
        test_dataset = torch.load(test_data_path, weights_only=False)
        logging.info(f"Loaded test dataset with {len(test_dataset)} samples")

        return PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        logging.error(f"Error loading test data: {str(e)}")
        return None


def load_trained_model(model_path: str, device: str) -> Optional[GraphTransformerVAE]:
    """
    Load the trained model from checkpoint.

    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on

    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        logging.error(f"Model checkpoint not found at {model_path}")
        return None

    try:
        # Add basic Data class to safe globals if available
        if TORCH_GEOMETRIC_AVAILABLE:
            torch.serialization.add_safe_globals([Data])

        # Load with weights_only=False which worked in our test
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Extract model parameters from checkpoint - using dimensions from the error message
        model = GraphTransformerVAE(
            node_feature_dim=148,  # From error: copying a param with shape torch.Size([256, 148])
            edge_feature_dim=11,  # From error: copying a param with shape torch.Size([256, 11])
            hidden_dim=256,  # From error: multiple mentions of dimension 256
            latent_dim=128,  # From error: copying a param with shape torch.Size([128, 256])
            num_layers=6,  # From error: unexpected keys up to encoder_layers.5
            num_heads=8,  # Keeping this the same
            num_node_types=148,  # Same as node_feature_dim
            num_edge_types=4,  # From error: copying a param with shape torch.Size([4, 512])
            num_property_features=1,
            property_hidden_dim=64,
            dropout=0.1,
            use_positional_encoding=True,
            beta=0.001,
            lambda_prop=1.5,
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logging.info(f"Loaded model from {model_path}")
        logging.info(
            f"Model dimensions from checkpoint: node_feature_dim=148, edge_feature_dim=11, hidden_dim=256, latent_dim=128"
        )

        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None


def evaluate_model(
    model: GraphTransformerVAE, data_loader: PyGDataLoader, device: str
) -> Dict[str, Any]:
    """
    Evaluate the model on test data.

    Args:
        model: Trained model
        data_loader: DataLoader for test data
        device: Device to run evaluation on

    Returns:
        Dictionary containing evaluation metrics and outputs
    """
    model.eval()

    # Initialize lists to store results
    all_true_values = []
    all_predicted_values = []
    all_z_mean = []
    all_z_logvar = []
    all_latent_z = []
    all_smiles = []
    all_attention_weights = []
    all_node_features = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(device)

            # Forward pass - using the model's expected input format
            # Note: From error, we see the model expects node_features, edge_features, edge_index, mask
            outputs = model(
                node_features=batch.x,
                edge_features=batch.edge_attr,
                edge_index=batch.edge_index,
                mask=batch.batch,
            )

            if outputs is None:
                continue

            # Extract outputs from the dictionary returned by the model
            node_logits = outputs.get("node_logits")
            edge_logits = outputs.get("edge_logits")
            z_mean = outputs.get("mu")
            z_logvar = outputs.get("logvar")
            z = outputs.get("z")
            property_pred = outputs.get("predicted_properties")

            # Store results
            if hasattr(batch, "y") and batch.y is not None:
                all_true_values.extend(batch.y.cpu().numpy())
                all_predicted_values.extend(property_pred.cpu().numpy())

            all_z_mean.append(z_mean.cpu().numpy())
            all_z_logvar.append(z_logvar.cpu().numpy())
            all_latent_z.append(z.cpu().numpy())

            # Store attention weights if available
            if hasattr(model, "get_attention_weights"):
                attention = model.get_attention_weights()
                if attention is not None:
                    all_attention_weights.append(attention.cpu().numpy())

            # Store node features for visualization
            if hasattr(batch, "x"):
                all_node_features.append(batch.x.cpu().numpy())

            # Store SMILES if available
            if hasattr(batch, "smiles"):
                all_smiles.extend(batch.smiles)

    # Concatenate results
    all_z_mean = np.vstack(all_z_mean) if all_z_mean else np.array([])
    all_z_logvar = np.vstack(all_z_logvar) if all_z_logvar else np.array([])
    all_latent_z = np.vstack(all_latent_z) if all_latent_z else np.array([])
    all_node_features = (
        np.vstack(all_node_features) if all_node_features else np.array([])
    )

    # Calculate metrics
    metrics = {}
    if all_true_values and all_predicted_values:
        all_true_values = np.array(all_true_values)
        all_predicted_values = np.array(all_predicted_values)

        metrics["mse"] = mean_squared_error(all_true_values, all_predicted_values)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(all_true_values, all_predicted_values)
        metrics["r2"] = r2_score(all_true_values, all_predicted_values)

        logging.info(f"Evaluation metrics:")
        logging.info(f"  MSE: {metrics['mse']:.4f}")
        logging.info(f"  RMSE: {metrics['rmse']:.4f}")
        logging.info(f"  MAE: {metrics['mae']:.4f}")
        logging.info(f"  R²: {metrics['r2']:.4f}")

    return {
        "metrics": metrics,
        "true_values": all_true_values,
        "predicted_values": all_predicted_values,
        "z_mean": all_z_mean,
        "z_logvar": all_z_logvar,
        "latent_z": all_latent_z,
        "attention_weights": all_attention_weights,
        "node_features": all_node_features,
        "smiles": all_smiles,
    }


def create_visualizations(results: Dict[str, Any], output_dir: str) -> None:
    """
    Create and save visualizations of the evaluation results.

    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set Seaborn style
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = [10, 8]

    # 1. True vs Predicted Values
    if len(results["true_values"]) > 0 and len(results["predicted_values"]) > 0:
        plt.figure()

        # Convert to 1D arrays if they're not already
        true_values = np.array(results["true_values"]).flatten()
        pred_values = np.array(results["predicted_values"]).flatten()

        max_val = max(np.max(true_values), np.max(pred_values))
        min_val = min(np.min(true_values), np.min(pred_values))

        # Create DataFrame for plotting
        plot_data = pd.DataFrame(
            {"true_values": true_values, "predicted_values": pred_values}
        )

        # Plot the scatter plot
        sns.scatterplot(
            data=plot_data, x="true_values", y="predicted_values", alpha=0.6
        )

        # Plot the identity line
        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        plt.xlabel("True PAMPA")
        plt.ylabel("Predicted PAMPA")
        plt.title("True vs Predicted PAMPA Values")

        # Add metrics as text
        if "metrics" in results and results["metrics"]:
            metrics_text = f"RMSE: {results['metrics']['rmse']:.4f}\n"
            metrics_text += f"MAE: {results['metrics']['mae']:.4f}\n"
            metrics_text += f"R²: {results['metrics']['r2']:.4f}"
            plt.figtext(
                0.15,
                0.8,
                metrics_text,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "true_vs_predicted.png"), dpi=300)
        plt.close()

    # 2. Error Distribution
    if len(results["true_values"]) > 0 and len(results["predicted_values"]) > 0:
        errors = results["predicted_values"] - results["true_values"]

        plt.figure()
        sns.histplot(data=errors, kde=True)
        plt.xlabel("Prediction Error")
        plt.ylabel("Count")
        plt.title("Distribution of Prediction Errors")
        plt.axvline(x=0, color="r", linestyle="--")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=300)
        plt.close()

    # 3. Latent Space Visualization using PCA
    if results["latent_z"].size > 0:
        # Apply PCA to reduce dimensionality to 2D
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(results["latent_z"])

        plt.figure()
        if len(results["true_values"]) > 0:
            scatter = plt.scatter(
                latent_2d[:, 0],
                latent_2d[:, 1],
                c=results["true_values"],
                cmap="viridis",
                alpha=0.7,
            )
            plt.colorbar(scatter, label="PAMPA Value")
        else:
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7)

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.title("PCA of Latent Space")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "latent_space_pca.png"), dpi=300)
        plt.close()

    # 4. Latent Space Visualization using t-SNE (if there are enough samples)
    if results["latent_z"].size > 0 and results["latent_z"].shape[0] >= 5:
        try:
            # Apply t-SNE to reduce dimensionality to 2D
            tsne = TSNE(n_components=2, random_state=42)
            latent_tsne = tsne.fit_transform(results["latent_z"])

            plt.figure()
            if len(results["true_values"]) > 0:
                scatter = plt.scatter(
                    latent_tsne[:, 0],
                    latent_tsne[:, 1],
                    c=results["true_values"],
                    cmap="viridis",
                    alpha=0.7,
                )
                plt.colorbar(scatter, label="PAMPA Value")
            else:
                plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.7)

            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.title("t-SNE of Latent Space")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "latent_space_tsne.png"), dpi=300)
            plt.close()
        except Exception as e:
            logging.warning(f"Failed to generate t-SNE visualization: {str(e)}")

    # 5. Save evaluation report as markdown
    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("# Model Evaluation Report\n\n")

        f.write("## Metrics\n\n")
        if "metrics" in results and results["metrics"]:
            f.write(f"- Mean Squared Error (MSE): {results['metrics']['mse']:.4f}\n")
            f.write(
                f"- Root Mean Squared Error (RMSE): {results['metrics']['rmse']:.4f}\n"
            )
            f.write(f"- Mean Absolute Error (MAE): {results['metrics']['mae']:.4f}\n")
            f.write(f"- R² Score: {results['metrics']['r2']:.4f}\n\n")
        else:
            f.write("No metrics available.\n\n")

        f.write("## Visualizations\n\n")
        f.write("The following visualizations were generated:\n\n")

        if len(results["true_values"]) > 0 and len(results["predicted_values"]) > 0:
            f.write(
                "1. **True vs Predicted Values** - Scatter plot comparing true and predicted PAMPA values\n"
            )
            f.write(
                "2. **Error Distribution** - Histogram showing the distribution of prediction errors\n"
            )

        if results["latent_z"].size > 0:
            f.write(
                "3. **PCA of Latent Space** - 2D projection of the latent space using PCA\n"
            )

        if results["latent_z"].size > 0 and results["latent_z"].shape[0] >= 5:
            f.write(
                "4. **t-SNE of Latent Space** - 2D projection of the latent space using t-SNE\n"
            )

        f.write("\n## Dataset Information\n\n")
        f.write(f"- Number of test samples: {len(results['true_values'])}\n")

        if results["latent_z"].size > 0:
            f.write(f"- Latent dimension: {results['latent_z'].shape[1]}\n")


def save_predictions(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save the model predictions to a CSV file.

    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save the predictions
    """
    if len(results["true_values"]) > 0 and len(results["predicted_values"]) > 0:
        # Flatten arrays if they're not already 1D
        true_values = np.array(results["true_values"]).flatten()
        predicted_values = np.array(results["predicted_values"]).flatten()

        predictions_df = pd.DataFrame(
            {
                "true_value": true_values,
                "predicted_value": predicted_values,
                "error": predicted_values - true_values,
            }
        )

        if results["smiles"] and len(results["smiles"]) == len(true_values):
            predictions_df["smiles"] = results["smiles"]

        predictions_path = os.path.join(output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        logging.info(f"Saved predictions to {predictions_path}")


def save_latent_representations(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save the latent space representations to a file.

    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save the latent representations
    """
    if results["latent_z"].size > 0:
        # Ensure latent_z is properly shaped
        latent_z = np.vstack(results["latent_z"])

        latent_df = pd.DataFrame(
            latent_z,
            columns=[f"z_{i}" for i in range(latent_z.shape[1])],
        )

        if len(results["true_values"]) > 0:
            # Flatten true values if needed
            true_values = np.array(results["true_values"]).flatten()

            # Make sure the length matches - truncate if necessary
            if len(true_values) == len(latent_z):
                latent_df["true_value"] = true_values
            elif len(true_values) > len(latent_z):
                latent_df["true_value"] = true_values[: len(latent_z)]
            else:
                logging.warning(
                    f"Mismatch between latent_z length ({len(latent_z)}) and true_values length ({len(true_values)})"
                )

        if results["smiles"] and len(results["smiles"]) == len(latent_z):
            latent_df["smiles"] = results["smiles"]

        latent_path = os.path.join(output_dir, "latent_representations.csv")
        latent_df.to_csv(latent_path, index=False)
        logging.info(f"Saved latent representations to {latent_path}")


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    setup_logging(args.output_dir)

    # Log arguments
    logging.info("Evaluation parameters:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load data
    test_loader = load_data(args.data_dir, args.batch_size)
    if test_loader is None:
        logging.error("Failed to load test data. Exiting.")
        return

    # Load model
    model = load_trained_model(args.model_path, device)
    if model is None:
        logging.error("Failed to load model. Exiting.")
        return

    # Evaluate model
    results = evaluate_model(model, test_loader, device)

    # Create visualizations
    create_visualizations(results, args.output_dir)

    # Save predictions
    save_predictions(results, args.output_dir)

    # Save latent representations
    save_latent_representations(results, args.output_dir)

    logging.info(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
