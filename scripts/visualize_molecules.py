# scripts/visualize_molecules.py

"""
Generate molecular visualizations from the trained model.
Includes latent space interpolation and property-conditioned generation.
"""

import argparse
import json
import logging
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.model import GraphTransformerVAE
import torch.serialization
from torch_geometric.data import Data, HeteroData

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_trained_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """
    Load the trained model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on

    Returns:
        Trained model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Add safe globals for PyTorch Geometric data structures
    torch.serialization.add_safe_globals([Data, HeteroData])

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model parameters from checkpoint
    model_params = checkpoint.get("model_params", {})

    # Create model with same parameters
    model = GraphTransformerVAE(
        node_feature_dim=model_params.get("node_feature_dim"),
        edge_feature_dim=model_params.get("edge_feature_dim"),
        hidden_dim=model_params.get("hidden_dim", 128),
        latent_dim=model_params.get("latent_dim", 64),
        num_layers=4,  # Default value
        num_heads=8,  # Default value
        num_node_types=model_params.get(
            "node_feature_dim"
        ),  # Assuming this is the same
        num_edge_types=model_params.get(
            "edge_feature_dim"
        ),  # Assuming this is the same
        num_property_features=model_params.get("num_property_features", 1),
        property_hidden_dim=64,  # Default value
        dropout=0.1,  # Default value
        use_positional_encoding=True,
        beta=model_params.get("beta", 0.001),
        lambda_prop=model_params.get("lambda_prop", 1.5),
    ).to(device)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Model loaded successfully (from epoch {checkpoint['epoch']})")
    return model


def visualize_sample_molecules(test_loader, model, device, output_dir, num_samples=5):
    """
    Visualize sample molecules from the test set along with their reconstructions.

    Args:
        test_loader: DataLoader for test dataset
        model: Trained model
        device: Device for computation
        output_dir: Directory to save visualizations
        num_samples: Number of molecules to visualize
    """
    logger.info(f"Generating visualizations for {num_samples} sample molecules...")

    # Get a batch from test loader
    batch = next(iter(test_loader)).to(device)

    # Select a few samples
    indices = np.random.choice(
        batch.num_graphs, min(num_samples, batch.num_graphs), replace=False
    )

    # Create directory for sample reconstructions
    sample_dir = output_dir / "sample_reconstructions"
    sample_dir.mkdir(exist_ok=True, parents=True)

    # Process each sample
    with torch.no_grad():
        # Forward pass
        outputs = model(
            node_features=batch.x,
            edge_features=batch.edge_attr,
            edge_index=batch.edge_index,
            mask=batch.batch,
        )

        # Get node reconstructions (apply sigmoid to convert logits to probabilities)
        node_recon = torch.sigmoid(outputs["node_logits"])

    # For each selected sample, visualize original and reconstruction
    max_atoms = 30  # Maximum number of atoms to visualize

    for idx in indices:
        # Find molecule corresponding to this index
        node_mask = batch.batch == idx
        num_nodes = node_mask.sum().item()

        if num_nodes > max_atoms:
            logger.warning(
                f"Molecule {idx} has {num_nodes} atoms, which exceeds the maximum of {max_atoms}. Skipping."
            )
            continue

        # Get original and reconstructed node features
        orig_nodes = batch.x[node_mask].cpu().numpy()
        recon_nodes = node_recon[node_mask].cpu().numpy()

        # Get edges
        edge_mask = batch.batch[batch.edge_index[0]] == idx
        orig_edges = batch.edge_attr[edge_mask].cpu().numpy()

        # Get property value
        prop_value = batch.y[idx].item()
        pred_prop = outputs["predicted_properties"][idx].item()

        # Create visualization
        plt.figure(figsize=(12, 6))

        # Original node features
        plt.subplot(1, 2, 1)
        plt.title(f"Original (PAMPA: {prop_value:.4f})")
        plt.imshow(orig_nodes.T, aspect="auto", cmap="viridis")
        plt.colorbar(shrink=0.8)
        plt.xlabel("Atom Index")
        plt.ylabel("Feature Index")

        # Reconstructed node features
        plt.subplot(1, 2, 2)
        plt.title(f"Reconstructed (Pred PAMPA: {pred_prop:.4f})")
        plt.imshow(recon_nodes.T, aspect="auto", cmap="viridis")
        plt.colorbar(shrink=0.8)
        plt.xlabel("Atom Index")
        plt.ylabel("Feature Index")

        plt.tight_layout()
        plt.savefig(sample_dir / f"molecule_{idx}_features.png", dpi=300)
        plt.close()

        # Additional visualization: latent space
        latent_vec = outputs["z"][idx].cpu().numpy()

        plt.figure(figsize=(10, 4))
        plt.title(f"Latent Representation for Molecule {idx}")
        plt.plot(latent_vec)
        plt.xlabel("Latent Dimension")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(sample_dir / f"molecule_{idx}_latent.png", dpi=300)
        plt.close()

    logger.info(f"Sample visualizations saved to {sample_dir}")


def explore_latent_space(model, test_loader, device, output_dir, num_points=5):
    """
    Explore and visualize the latent space by interpolating between molecules.

    Args:
        model: Trained model
        test_loader: DataLoader for test dataset
        device: Device for computation
        output_dir: Directory to save visualizations
        num_points: Number of interpolation points
    """
    logger.info("Exploring latent space through interpolation...")

    # Create directory for latent space exploration
    latent_dir = output_dir / "latent_space"
    latent_dir.mkdir(exist_ok=True, parents=True)

    # Get a batch from test loader
    batch = next(iter(test_loader)).to(device)

    # Select two distinct molecules with similar node counts
    with torch.no_grad():
        # Forward pass
        outputs = model(
            node_features=batch.x,
            edge_features=batch.edge_attr,
            edge_index=batch.edge_index,
            mask=batch.batch,
        )

        # Get latent vectors
        latent_vecs = outputs["z"].cpu().numpy()

        # Get property values
        prop_values = batch.y.cpu().numpy()

    # Find pairs of molecules with similar structure but different properties
    node_counts = []
    for i in range(batch.num_graphs):
        node_counts.append((batch.batch == i).sum().item())

    node_counts = np.array(node_counts)

    # Sort molecules by property value
    sorted_idx = np.argsort(prop_values.squeeze())

    # Find molecules with similar node counts but different properties
    found_pair = False
    for i in range(len(sorted_idx) // 3):
        idx1 = sorted_idx[i]  # Low property value
        for j in range(len(sorted_idx) - len(sorted_idx) // 3, len(sorted_idx)):
            idx2 = sorted_idx[j]  # High property value
            if abs(node_counts[idx1] - node_counts[idx2]) <= 2:
                found_pair = True
                break
        if found_pair:
            break

    if not found_pair:
        logger.warning(
            "Couldn't find molecules with similar node counts but different properties. Using first and last instead."
        )
        idx1 = sorted_idx[0]
        idx2 = sorted_idx[-1]

    # Get latent vectors for selected molecules
    z1 = latent_vecs[idx1]
    z2 = latent_vecs[idx2]

    logger.info(
        f"Interpolating between molecule {idx1} (PAMPA={prop_values[idx1][0]:.4f}) and "
        f"molecule {idx2} (PAMPA={prop_values[idx2][0]:.4f})"
    )

    # Create interpolation points
    alphas = np.linspace(0, 1, num_points)
    interp_zs = []

    for alpha in alphas:
        interp_z = (1 - alpha) * z1 + alpha * z2
        interp_zs.append(interp_z)

    # Convert interpolated points to tensor
    interp_zs = torch.tensor(np.array(interp_zs), dtype=torch.float32).to(device)

    # Set property targets for different values
    property_targets = (
        torch.linspace(float(prop_values[idx1]), float(prop_values[idx2]), num_points)
        .unsqueeze(1)
        .to(device)
    )

    # Decode interpolated points
    with torch.no_grad():
        # Forward through decoder parts
        preds = []
        for z, prop in zip(interp_zs, property_targets):
            # Decode single sample (expand dimensions)
            z = z.unsqueeze(0)
            prop = prop.unsqueeze(0)

            # Predict property from latent
            pred_prop = model.property_predictor(z)
            preds.append(pred_prop.item())

            # Decode node features
            num_nodes = min(node_counts[idx1], node_counts[idx2])
            node_logits, _ = model.decode(z, prop, num_nodes)

            # Apply sigmoid to convert logits to probabilities
            node_features = torch.sigmoid(node_logits)

            # Create visualization
            plt.figure(figsize=(8, 6))
            plt.imshow(node_features.cpu().numpy().T, aspect="auto", cmap="viridis")
            plt.colorbar(shrink=0.8)
            plt.title(
                f"Interpolation α={alpha:.2f}, Target PAMPA={prop.item():.4f}, Pred={pred_prop.item():.4f}"
            )
            plt.xlabel("Node Index")
            plt.ylabel("Feature Index")
            plt.tight_layout()
            plt.savefig(latent_dir / f"interp_{alpha:.2f}.png", dpi=300)
            plt.close()

    # Create summary plot
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, [p.item() for p in property_targets], "b-", label="Target PAMPA")
    plt.plot(alphas, preds, "r--", label="Predicted PAMPA")
    plt.xlabel("Interpolation Factor (α)")
    plt.ylabel("PAMPA Value")
    plt.title("PAMPA Values During Latent Space Interpolation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(latent_dir / "interpolation_summary.png", dpi=300)
    plt.close()

    logger.info(f"Latent space exploration saved to {latent_dir}")


def generate_molecules_with_properties(model, device, output_dir, property_values=None):
    """
    Generate molecules conditioned on specific property values.

    Args:
        model: Trained model
        device: Device for computation
        output_dir: Directory to save visualizations
        property_values: List of property values to condition on, or None to use defaults
    """
    logger.info("Generating molecules conditioned on property values...")

    # Create directory for generated molecules
    gen_dir = output_dir / "property_conditioned"
    gen_dir.mkdir(exist_ok=True, parents=True)

    # Default property values if not provided
    if property_values is None:
        property_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Number of molecules to generate per property value
    num_per_value = 3
    latent_dim = model.latent_dim

    # Generate molecules for each property value
    for prop_val in property_values:
        logger.info(f"Generating molecules with PAMPA = {prop_val:.2f}")

        # Create directory for this property value
        prop_dir = gen_dir / f"pampa_{prop_val:.2f}"
        prop_dir.mkdir(exist_ok=True, parents=True)

        for i in range(num_per_value):
            # Sample random latent vector
            z = torch.randn(1, latent_dim).to(device)

            # Create property tensor
            prop = torch.tensor([[prop_val]], dtype=torch.float32).to(device)

            with torch.no_grad():
                # Generate molecule with this property
                node_logits, _ = model.decode(
                    z, prop, num_nodes=20
                )  # Assuming molecules with ~20 nodes

                # Apply sigmoid to convert logits to probabilities
                node_features = torch.sigmoid(node_logits)

                # Predict property from latent
                pred_prop = model.property_predictor(z).item()

                # Create visualization
                plt.figure(figsize=(8, 6))
                plt.imshow(node_features.cpu().numpy().T, aspect="auto", cmap="viridis")
                plt.colorbar(shrink=0.8)
                plt.title(
                    f"Generated Molecule: Target PAMPA={prop_val:.2f}, Pred={pred_prop:.4f}"
                )
                plt.xlabel("Node Index")
                plt.ylabel("Feature Index")
                plt.tight_layout()
                plt.savefig(prop_dir / f"molecule_{i}.png", dpi=300)
                plt.close()

    logger.info(f"Generated molecules saved to {gen_dir}")


def main():
    """Run the visualization pipeline."""
    parser = argparse.ArgumentParser(
        description="Visualize molecules from GVPermformer model"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/processed", help="Path to processed data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, mps, cpu)"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure file handler for logging
    file_handler = logging.FileHandler(output_dir / "visualization.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Set device
    if args.device:
        device = torch.device(args.device)
        logger.info(f"Using specified device: {args.device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Metal for acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA for acceleration")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for computation")

    # Load test data
    logger.info("Loading test data...")
    data_dir = Path(args.data_dir)

    # Add safe globals for PyTorch Geometric data structures
    torch.serialization.add_safe_globals([Data, HeteroData])

    test_data = torch.load(data_dir / "test_data.pt", weights_only=False)

    # Create test data loader
    test_loader = DataLoader(
        test_data, batch_size=args.num_samples * 2, shuffle=True, num_workers=0
    )

    # Load trained model
    model = load_trained_model(Path(args.model_path), device)

    # Visualize sample molecules
    visualize_sample_molecules(test_loader, model, device, output_dir, args.num_samples)

    # Explore latent space through interpolation
    explore_latent_space(model, test_loader, device, output_dir)

    # Generate molecules conditioned on property values
    generate_molecules_with_properties(model, device, output_dir)

    logger.info("Visualization completed successfully!")


if __name__ == "__main__":
    main()
