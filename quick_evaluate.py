#!/usr/bin/env python3
# quick_evaluate.py
"""
# relative/path/to/file: quick_evaluate.py
Quickly evaluate a saved GraphVAE model checkpoint on a dataset.
This script loads a saved model checkpoint and evaluates it on a dataset.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import json

# Add the project directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from local modules
from src.data.cycpept_dataset import CycPeptDataset
from src.data.dataset import MoleculeDataset
from src.models.graph_vae import GraphVAE


def get_device():
    """
    Get the appropriate device for model inference, prioritizing Apple Metal.

    Returns:
        torch.device: Device to use for inference (mps or cpu)
    """
    if torch.backends.mps.is_available():
        # Use Metal Performance Shaders on Apple Silicon
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for inference")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        print("Using CPU for inference")

    return device


def load_model_for_evaluation(model_path: str, device):
    """
    Load a saved model from a checkpoint file using the actual GraphVAE architecture.

    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on

    Returns:
        Loaded model and configuration details
    """
    print(f"Loading model from {model_path} to {device}")

    try:
        # Perform a warmup to initialize MPS device
        if device.type == "mps":
            x = torch.randn(1, 1).to(device)
            y = x + x
    except Exception as e:
        print(f"MPS warmup failed (non-critical): {e}")

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model state dict from the checkpoint
    if "model_state_dict" in checkpoint:
        # This is the standard format saved by our training script
        state_dict = checkpoint["model_state_dict"]

        # Check if model_config is available in the checkpoint
        if "model_config" in checkpoint:
            # Use the saved configuration
            config = checkpoint["model_config"]
            node_features = config["node_features"]
            edge_features = config["edge_features"]
            hidden_dim = config["hidden_dim"]
            latent_dim = config["latent_dim"]
            property_prediction = config["property_prediction"]
            print("Using model configuration stored in checkpoint")
        else:
            # Try to infer model parameters from the state dict
            node_features = 126  # Default
            edge_features = 9  # Default
            hidden_dim = 256  # Default
            latent_dim = 64  # Default

            # Check if encoder module weights exist
            if "encoder.node_encoder.0.weight" in state_dict:
                node_features = state_dict["encoder.node_encoder.0.weight"].size(1)

            if "encoder.edge_encoder.0.weight" in state_dict:
                edge_features = state_dict["encoder.edge_encoder.0.weight"].size(1)

            # Determine hidden dimension
            if "encoder.node_encoder.0.weight" in state_dict:
                hidden_dim = state_dict["encoder.node_encoder.0.weight"].size(0)

            # Determine latent dimension
            if "encoder.fc_mean.weight" in state_dict:
                latent_dim = state_dict["encoder.fc_mean.weight"].size(0)

            # Check if property prediction is enabled
            property_prediction = "property_predictor.0.weight" in state_dict
            print("Inferred model configuration from state dictionary")
    else:
        # The checkpoint might be just the state dict itself
        state_dict = checkpoint

        # Try to infer model parameters from the state dict
        node_features = 126  # Default
        edge_features = 9  # Default
        hidden_dim = 256  # Default
        latent_dim = 64  # Default

        # Check if encoder module weights exist
        if "encoder.node_encoder.0.weight" in state_dict:
            node_features = state_dict["encoder.node_encoder.0.weight"].size(1)

        if "encoder.edge_encoder.0.weight" in state_dict:
            edge_features = state_dict["encoder.edge_encoder.0.weight"].size(1)

        # Determine hidden dimension
        if "encoder.node_encoder.0.weight" in state_dict:
            hidden_dim = state_dict["encoder.node_encoder.0.weight"].size(0)

        # Determine latent dimension
        if "encoder.fc_mean.weight" in state_dict:
            latent_dim = state_dict["encoder.fc_mean.weight"].size(0)

        # Check if property prediction is enabled
        property_prediction = "property_predictor.0.weight" in state_dict
        print("Using raw state dictionary, inferring configuration")

    # Create the model with the appropriate parameters
    model = GraphVAE(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        property_prediction=property_prediction,
        use_mps_optimizations=(device.type == "mps"),
    )

    # Display model configuration
    print("Model configuration from checkpoint:")
    print(f"  Node features: {node_features}")
    print(f"  Edge features: {edge_features}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Property prediction: {property_prediction}")

    # Load the weights
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(
                f"Warning: Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}"
            )
        if unexpected_keys:
            print(
                f"Warning: Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}"
            )
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None

    # Move model to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model


def create_test_dataloader(data_csv: str, batch_size: int = 32):
    """
    Create a dataloader for testing with the model using the same test split as during training.

    Args:
        data_csv: Path to the CSV file with test data
        batch_size: Batch size for the dataloader

    Returns:
        DataLoader object
    """
    print(f"Loading dataset from {data_csv}")

    try:
        # Create the full dataset just like in training
        full_dataset = CycPeptDataset(
            csv_file=data_csv,
            smiles_col="SMILES",
            property_cols=["PAMPA"],
            max_atoms=500,
            pampa_threshold=-9.0,
            use_enhanced_atom_features=True,
        )

        print(f"Dataset loaded with {len(full_dataset)} samples")

        # Use the same split ratios and seed as in training
        train_val_test_split = (0.8, 0.1, 0.1)  # Same as in training
        random_seed = 42  # Same as in training

        # Calculate split sizes
        train_size = int(train_val_test_split[0] * len(full_dataset))
        val_size = int(train_val_test_split[1] * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        # Split dataset using the same random seed as in training
        from torch.utils.data import random_split

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )

        print(
            f"Split dataset: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}"
        )
        print(f"Using the test set ({len(test_dataset)} samples) for evaluation")

        # Create dataloader for the test set
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=(
                full_dataset.collate_fn
                if hasattr(full_dataset, "collate_fn")
                else MoleculeDataset.collate_fn
            ),
        )

        return dataloader
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Failed to create test dataloader: {e}")


def evaluate_model(model, dataloader, device, output_dir):
    """
    Evaluate the model on the provided dataloader.

    Args:
        model: The model to evaluate
        dataloader: DataLoader with test data
        device: Device to run evaluation on
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation metrics
    """
    print("\nStarting model evaluation...")
    model.eval()

    # Store results
    all_smiles = []
    all_targets = []
    all_predictions = []
    batch_times = []
    successful_batches = 0

    # Get property scaler if available
    property_scaler = None
    dataset = (
        dataloader.dataset.dataset
        if hasattr(dataloader.dataset, "dataset")
        else dataloader.dataset
    )
    if hasattr(dataset, "property_scaler"):
        property_scaler = dataset.property_scaler
        print(f"Found property scaler in dataset: {type(property_scaler).__name__}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Skip empty batches
            if not batch:
                print(f"Skipping batch {batch_idx}: Empty batch")
                continue

            # Start timing
            start_time = time.time()

            # Move batch to the appropriate device and adapt the data format
            batch_on_device = {}

            # Map the keys to match what the model expects
            if "node_features" in batch and "x" not in batch:
                batch_on_device["x"] = batch["node_features"].to(device)

            # Add all other keys normally
            for k, v in batch.items():
                if (
                    k != "node_features"
                ):  # Skip node_features as it's already handled above
                    if isinstance(v, torch.Tensor):
                        batch_on_device[k] = v.to(device)
                    else:
                        batch_on_device[k] = v

            # Forward pass
            try:
                # Run the model - properly formatted for GraphVAE
                outputs = model(batch_on_device)

                # Get property predictions if available
                if "prop_pred" in outputs and outputs["prop_pred"] is not None:
                    predictions = outputs["prop_pred"].cpu().numpy()
                else:
                    # If no property predictions, use zeros
                    batch_size = 1
                    if "smiles" in batch:
                        batch_size = len(batch["smiles"])
                    else:
                        # Estimate batch size from number of unique batch indices
                        if "batch" in batch:
                            batch_size = len(torch.unique(batch["batch"]))

                    predictions = np.zeros((batch_size, 1))

                # Store results
                if "smiles" in batch:
                    all_smiles.extend(batch["smiles"])
                if "target" in batch:
                    all_targets.append(batch["target"].cpu().numpy())
                all_predictions.append(predictions)

                # Calculate batch processing time
                end_time = time.time()
                batch_time = end_time - start_time
                batch_times.append(batch_time)
                successful_batches += 1

                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    avg_time = sum(batch_times[-10:]) / min(10, len(batch_times))
                    print(
                        f"Processed {successful_batches} batches, avg time: {avg_time:.4f}s per batch"
                    )

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback

                traceback.print_exc()
                continue

    # Check if we have any valid predictions
    if len(all_predictions) == 0:
        print("No valid predictions were generated. Evaluation failed.")
        return {"error": "No valid predictions"}

    # Concatenate results
    if all_targets:
        all_targets = np.concatenate(all_targets, axis=0)
    else:
        all_targets = np.array([])

    all_predictions = np.concatenate(all_predictions, axis=0)

    # Unscale predictions if a property scaler exists
    if property_scaler is not None:
        try:
            print("Unscaling predictions using dataset's property scaler...")

            # Make a copy of the scaled predictions
            scaled_predictions = all_predictions.copy()

            # Try to unscale predictions
            if hasattr(property_scaler, "inverse_transform"):
                all_predictions = property_scaler.inverse_transform(all_predictions)
                print(f"Predictions unscaled using inverse_transform method")
            elif hasattr(property_scaler, "inverse"):
                all_predictions = property_scaler.inverse(all_predictions)
                print(f"Predictions unscaled using inverse method")

            # Report scaling effect
            print(
                f"Scaling effect: min/max before: {scaled_predictions.min():.4f}/{scaled_predictions.max():.4f}, "
                f"after: {all_predictions.min():.4f}/{all_predictions.max():.4f}"
            )
        except Exception as e:
            print(f"Error unscaling predictions: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("No property scaler found. Using raw predictions for evaluation.")

    # Calculate metrics if we have targets
    results = {}
    if len(all_targets) > 0:
        # MSE
        mse = mean_squared_error(all_targets, all_predictions)
        results["mse"] = float(mse)

        # RMSE
        rmse = np.sqrt(mse)
        results["rmse"] = float(rmse)

        # MAE
        mae = mean_absolute_error(all_targets, all_predictions)
        results["mae"] = float(mae)

        # R²
        r2 = r2_score(all_targets, all_predictions)
        results["r2"] = float(r2)

        print(f"Evaluation metrics:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")

        # Create a scatter plot of predicted vs actual values
        plt.figure(figsize=(10, 8))
        plt.scatter(all_targets, all_predictions, alpha=0.5)
        plt.plot(
            [all_targets.min(), all_targets.max()],
            [all_targets.min(), all_targets.max()],
            "r--",
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs Actual Values")
        plt.text(
            0.05,
            0.95,
            f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(output_dir, "prediction_plot.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Prediction plot saved to {plot_path}")

        # Save the metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

        # Save the predictions
        predictions_df = pd.DataFrame(
            {
                "SMILES": all_smiles,
                "Actual": all_targets.flatten(),
                "Predicted": all_predictions.flatten(),
            }
        )
        predictions_path = os.path.join(output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

    print("Evaluation completed")
    return results


def main():
    """
    Main function to evaluate a model on the test set from training.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a saved GraphVAE model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default="training_data/CycPeptMPDB_Peptide_All.csv",
        help="Path to the data CSV (default is the training data file)",
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

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get device
    device = get_device()

    # Load model
    try:
        model = load_model_for_evaluation(args.model_path, device)
        if model is None:
            print("Failed to load model. Exiting.")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create dataloader from the test split
    try:
        print(
            f"Creating test set using the same split configuration as during training"
        )
        dataloader = create_test_dataloader(args.data_csv, args.batch_size)
    except Exception as e:
        print(f"Error creating test dataloader: {e}")
        return

    # Evaluate model
    try:
        print(f"Evaluating model on the test set")
        results = evaluate_model(model, dataloader, device, args.output_dir)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"Evaluation results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
