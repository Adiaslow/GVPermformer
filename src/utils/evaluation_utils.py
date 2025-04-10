"""
Utility functions for evaluating Graph VAE model performance.
Includes metrics calculation, latent space visualization, and result analysis.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import Draw, AllChem


def calculate_metrics(predictions, targets):
    """
    Calculate regression metrics for property prediction.

    Args:
        predictions: Numpy array of predicted values
        targets: Numpy array of true values

    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def plot_prediction_scatter(predictions, targets, save_path=None):
    """
    Create scatter plot of predicted vs actual values.

    Args:
        predictions: Numpy array of predicted values
        targets: Numpy array of true values
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    metrics = calculate_metrics(predictions, targets)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot scatter with density coloring
    scatter = ax.scatter(targets, predictions, alpha=0.6)

    # Add perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--")

    # Add metrics text
    text = f"RMSE: {metrics['rmse']:.4f}\nMAE: {metrics['mae']:.4f}\nR²: {metrics['r2']:.4f}"
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Set labels and title
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predicted vs Actual")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


def visualize_latent_space(z, labels=None, save_path=None):
    """
    Visualize 2D projection of latent space using t-SNE.

    Args:
        z: Latent vectors (n_samples, latent_dim)
        labels: Optional color labels for points
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    # Apply t-SNE for dimensionality reduction
    print("Applying t-SNE to latent vectors...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z) - 1))
    z_2d = tsne.fit_transform(z)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot points
    if labels is not None:
        scatter = ax.scatter(
            z_2d[:, 0], z_2d[:, 1], c=labels, cmap="viridis", alpha=0.7, s=50
        )
        plt.colorbar(scatter, ax=ax, label="Property Value")
    else:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.7, s=50)

    # Set labels
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("Latent Space Visualization (t-SNE)")

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


def analyze_latent_dimensions(model, test_loader, save_dir=None):
    """
    Analyze the contribution of each latent dimension to property prediction.

    Args:
        model: Trained model
        test_loader: DataLoader with test data
        save_dir: Directory to save plots

    Returns:
        DataFrame with dimension analysis
    """
    device = next(model.parameters()).device
    latent_dim = model.latent_dim

    # Create storage for sensitivity analysis
    sensitivities = []

    # Process batches
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get baseline outputs
            outputs = model(batch)
            z_base = outputs["z"]

            if model.property_prediction:
                base_props = outputs["properties"]

                # For each dimension
                for dim in range(latent_dim):
                    # Increase dimension by 1 std
                    z_plus = z_base.clone()
                    z_minus = z_base.clone()

                    z_plus[:, dim] += 1.0
                    z_minus[:, dim] -= 1.0

                    # Get property predictions for modified latent vectors
                    prop_plus = model.property_predictor(z_plus)
                    prop_minus = model.property_predictor(z_minus)

                    # Calculate sensitivity for each sample
                    for i in range(len(z_base)):
                        delta = torch.abs(prop_plus[i] - prop_minus[i]).mean().item()
                        sensitivities.append(
                            {"dimension": dim, "sample_idx": i, "sensitivity": delta}
                        )

    # Convert to DataFrame
    if sensitivities:
        df = pd.DataFrame(sensitivities)

        # Summarize by dimension
        summary = (
            df.groupby("dimension")["sensitivity"].agg(["mean", "std"]).reset_index()
        )
        summary = summary.sort_values("mean", ascending=False)

        # Plot dimension sensitivity
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.figure(figsize=(12, 6))
            sns.barplot(x="dimension", y="mean", data=summary, yerr=summary["std"])
            plt.xlabel("Latent Dimension")
            plt.ylabel("Sensitivity (Mean ± Std)")
            plt.title("Latent Dimension Contribution to Property Prediction")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, "dimension_sensitivity.png"),
                dpi=300,
                bbox_inches="tight",
            )

        return summary

    return None


def latent_space_interpolation(model, smiles1, smiles2, num_points=10, property_idx=0):
    """
    Interpolate between two molecules in the latent space.

    Args:
        model: Trained GraphVAE model
        smiles1: SMILES string of first molecule
        smiles2: SMILES string of second molecule
        num_points: Number of interpolation points
        property_idx: Index of property to predict (if applicable)

    Returns:
        Dict with interpolation data
    """
    from rdkit import Chem
    from src.data.dataset import MoleculeDataset

    device = next(model.parameters()).device

    # Convert SMILES to molecular graphs
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return None

    # Create dataset instance to process molecules
    dataset = MoleculeDataset(csv_file="dummy", smiles_col="smiles")

    # Get graph representations
    graph1 = dataset._get_mol_graph(mol1)
    graph2 = dataset._get_mol_graph(mol2)

    # Move to device
    for key in graph1:
        if isinstance(graph1[key], torch.Tensor):
            graph1[key] = graph1[key].to(device)
            graph2[key] = graph2[key].to(device)

    # Add batch dimension
    batch1 = {
        "batch": torch.zeros(graph1["x"].size(0), dtype=torch.long, device=device)
    }
    batch2 = {
        "batch": torch.zeros(graph2["x"].size(0), dtype=torch.long, device=device)
    }

    graph1.update(batch1)
    graph2.update(batch2)

    # Encode molecules
    model.eval()
    with torch.no_grad():
        # Get latent representations
        mu1, _ = model.encoder(
            graph1["x"], graph1["edge_index"], graph1["edge_attr"], graph1["batch"]
        )
        mu2, _ = model.encoder(
            graph2["x"], graph2["edge_index"], graph2["edge_attr"], graph2["batch"]
        )

        # Interpolate in latent space
        alphas = np.linspace(0, 1, num_points)
        interpolated_z = []
        properties = []

        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            interpolated_z.append(z.cpu().numpy())

            # Predict property if available
            if model.property_prediction:
                prop = model.property_predictor(z)
                properties.append(prop[0, property_idx].item())

    # Create visualizations
    mol_images = [Draw.MolToImage(mol1)]
    for i in range(1, num_points - 1):
        # This is just a placeholder - in reality, we would want to
        # decode the latent vector to a molecular graph, but our simplified
        # model doesn't fully implement this
        mol_images.append(None)
    mol_images.append(Draw.MolToImage(mol2))

    return {
        "z": np.array(interpolated_z).squeeze(),
        "properties": properties if properties else None,
        "mol_images": mol_images,
        "smiles1": smiles1,
        "smiles2": smiles2,
    }


def save_model_summary(model, output_path):
    """
    Save model summary to text file.

    Args:
        model: PyTorch model
        output_path: Path to save summary

    Returns:
        None
    """
    from io import StringIO
    import sys

    # Redirect stdout to capture summary
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # Print model details
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    print("\nModel Architecture:")
    print(model)

    # Print hyperparameters if available
    if hasattr(model, "hparams"):
        print("\nHyperparameters:")
        for key, value in model.hparams.items():
            print(f"  {key}: {value}")

    # Restore stdout and get the summary
    sys.stdout = old_stdout
    summary = mystdout.getvalue()

    # Save to file
    with open(output_path, "w") as f:
        f.write(summary)

    print(f"Model summary saved to {output_path}")


def visualize_attention_weights(model, batch, layer_idx=0, head_idx=0, save_path=None):
    """
    Visualize attention weights for a molecular graph.

    Args:
        model: GraphVAE model with transformer components
        batch: Batch of graph data
        layer_idx: Index of transformer layer to visualize
        head_idx: Index of attention head to visualize
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    # This is a placeholder for a more sophisticated implementation
    # In a real implementation, we would extract attention weights from the model
    # Here we'll just create a placeholder visualization

    # Get attention weights
    # This requires modifying the model to return attention weights
    # For now, we'll create dummy weights
    num_nodes = batch["x"].size(0)
    attention_weights = torch.rand(num_nodes, num_nodes)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot attention weights
    im = ax.imshow(attention_weights.cpu().numpy(), cmap="viridis")

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set labels
    ax.set_title(f"Attention Weights (Layer {layer_idx}, Head {head_idx})")
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Node Index")

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig
