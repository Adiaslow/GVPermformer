# scripts/train_model.py

"""
Training script for the GVPermformer model.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
import torch.serialization
from torch_geometric.data import Data, HeteroData
import torch.nn.functional as F

from configs.model_config import ModelConfig
from src.model import GraphTransformerVAE

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(data_dir: Path):
    """Load preprocessed data and statistics."""
    # Add safe globals for PyTorch Geometric data structures
    torch.serialization.add_safe_globals([Data, HeteroData])

    train_data = torch.load(data_dir / "train_data.pt", weights_only=False)
    val_data = torch.load(data_dir / "val_data.pt", weights_only=False)
    test_data = torch.load(data_dir / "test_data.pt", weights_only=False)

    with open(data_dir / "preprocessing_stats.json", "r") as f:
        stats = json.load(f)

    return train_data, val_data, test_data, stats


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    prop_loss_sum = 0

    # Create progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass with batched data
        outputs = model(
            node_features=batch.x,
            edge_features=batch.edge_attr,
            edge_index=batch.edge_index,
            mask=batch.batch,
        )

        # Compute losses
        recon_loss = F.binary_cross_entropy_with_logits(outputs["node_logits"], batch.x)
        if outputs["edge_logits"] is not None:
            recon_loss += F.binary_cross_entropy_with_logits(
                outputs["edge_logits"], batch.edge_attr
            )

        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()
        )
        kl_loss = kl_loss / batch.num_graphs  # Normalize by batch size

        # Property prediction loss with weighting
        prop_loss_raw = F.mse_loss(
            outputs["predicted_properties"].squeeze(),
            batch.y.squeeze(),
            reduction="none",
        )
        # Apply weighting based on value ranges - higher weight for extreme values
        weights = 1.0 + torch.abs(batch.y.squeeze())
        prop_loss = (prop_loss_raw * weights).mean()

        # Total loss (using model parameters for weights)
        loss = recon_loss + model.beta * kl_loss + model.lambda_prop * prop_loss

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update running sums
        total_loss += loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        prop_loss_sum += prop_loss.item()

        # Update progress bar
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{model.beta.item() * kl_loss.item():.4f}",
                "prop": f"{model.lambda_prop.item() * prop_loss.item():.4f}",
                "lr": f"{lr:.6f}",
            }
        )

    # Calculate average losses
    num_batches = len(train_loader)
    return {
        "total": total_loss / num_batches,
        "recon": recon_loss_sum / num_batches,
        "kl": kl_loss_sum / num_batches,
        "prop": prop_loss_sum / num_batches,
    }


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    prop_loss_sum = 0

    # Create progress bar
    pbar = tqdm(val_loader, desc="Validating", leave=False)

    for batch in pbar:
        batch = batch.to(device)

        # Forward pass with batched data
        outputs = model(
            node_features=batch.x,
            edge_features=batch.edge_attr,
            edge_index=batch.edge_index,
            mask=batch.batch,
        )

        # Compute losses (same as training)
        recon_loss = F.binary_cross_entropy_with_logits(outputs["node_logits"], batch.x)
        if outputs["edge_logits"] is not None:
            recon_loss += F.binary_cross_entropy_with_logits(
                outputs["edge_logits"], batch.edge_attr
            )

        kl_loss = -0.5 * torch.sum(
            1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()
        )
        kl_loss = kl_loss / batch.num_graphs

        # Property prediction loss with weighting
        prop_loss_raw = F.mse_loss(
            outputs["predicted_properties"].squeeze(),
            batch.y.squeeze(),
            reduction="none",
        )
        # Apply weighting based on value ranges - higher weight for extreme values
        weights = 1.0 + torch.abs(batch.y.squeeze())
        prop_loss = (prop_loss_raw * weights).mean()

        # Total loss (using model parameters for weights)
        loss = recon_loss + model.beta * kl_loss + model.lambda_prop * prop_loss

        # Update running sums
        total_loss += loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        prop_loss_sum += prop_loss.item()

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{model.beta.item() * kl_loss.item():.4f}",
                "prop": f"{model.lambda_prop.item() * prop_loss.item():.4f}",
            }
        )

    # Calculate average losses
    num_batches = len(val_loader)
    return {
        "total": total_loss / num_batches,
        "recon": recon_loss_sum / num_batches,
        "kl": kl_loss_sum / num_batches,
        "prop": prop_loss_sum / num_batches,
    }


# KL loss annealing function
def get_kl_weight(epoch, max_epochs, min_weight=0.001, max_weight=0.01):
    """
    Gradually increase the KL weight from min_weight to max_weight.
    Reaches max_weight at 2/3 of max_epochs.
    """
    return min(
        max_weight,
        min_weight + (max_weight - min_weight) * (epoch / (max_epochs * 2 / 3)),
    )


def main():
    """Run the training pipeline."""
    parser = argparse.ArgumentParser(description="Train the GVPermformer model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing preprocessed data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, mps, cpu)"
    )
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    data_dir = Path(args.data_dir)
    train_data, val_data, test_data, stats = load_data(data_dir)

    # Create data loaders
    train_loader = PyGDataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = PyGDataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Initialize model
    logger.info("Initializing model...")

    # Set device - prioritize MPS (Metal) on Apple Silicon, then CUDA, then CPU
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

    # Create model with parameters from config
    model = GraphTransformerVAE(
        node_feature_dim=stats["feature_dimensions"]["node_feature_dim"],
        edge_feature_dim=stats["feature_dimensions"]["edge_feature_dim"],
        hidden_dim=256,  # From ModelConfig defaults
        latent_dim=128,  # From ModelConfig defaults
        num_layers=6,  # From ModelConfig defaults
        num_heads=8,  # From ModelConfig defaults
        num_node_types=stats["feature_dimensions"]["num_node_types"],
        num_edge_types=stats["feature_dimensions"]["num_edge_types"],
        num_property_features=1,  # PAMPA prediction
        property_hidden_dim=64,  # From ModelConfig defaults
        dropout=0.1,
        use_positional_encoding=True,
        beta=0.001,  # Start with very low KL weight
        lambda_prop=1.5,  # Slightly increase property prediction weight
    ).to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    patience = 15  # Increased patience for early stopping
    patience_counter = 0

    # Create a progress bar for epochs
    epoch_pbar = tqdm(range(args.epochs), desc="Training Progress")

    for epoch in epoch_pbar:
        # Update KL weight with annealing
        kl_weight = get_kl_weight(epoch, args.epochs)
        model.beta = nn.Parameter(
            torch.tensor(kl_weight, device=device), requires_grad=False
        )

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Update learning rate scheduler
        scheduler.step(val_metrics["total"])

        # Update epoch progress bar
        epoch_pbar.set_postfix(
            {
                "epoch": f"{epoch + 1}/{args.epochs}",
                "train_loss": f'{train_metrics["total"]:.4f}',
                "val_loss": f'{val_metrics["total"]:.4f}',
                "patience": f"{patience_counter}/{patience}",
                "beta": f"{model.beta.item():.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:6f}",
            }
        )

        # Log detailed metrics
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(
            f"Train Loss: {train_metrics['total']:.4f} (Recon: {train_metrics['recon']:.4f}, "
            f"KL: {train_metrics['kl']:.4f}, Prop: {train_metrics['prop']:.4f})"
        )
        logger.info(
            f"Val Loss: {val_metrics['total']:.4f} (Recon: {val_metrics['recon']:.4f}, "
            f"KL: {val_metrics['kl']:.4f}, Prop: {val_metrics['prop']:.4f})"
        )
        logger.info(
            f"KL weight (beta): {model.beta.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save checkpoint if validation loss improved
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            patience_counter = 0

            # Create a dictionary with model parameters instead of config
            model_params = {
                "node_feature_dim": model.node_feature_dim,
                "edge_feature_dim": model.edge_feature_dim,
                "hidden_dim": model.hidden_dim,
                "latent_dim": model.latent_dim,
                "num_property_features": model.num_property_features,
                "beta": model.beta.item(),
                "lambda_prop": model.lambda_prop.item(),
            }

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_metrics["total"],
                    "val_loss": val_metrics["total"],
                    "model_params": model_params,
                },
                output_dir / "best_model.pt",
            )
            logger.info("Saved new best model checkpoint\n")
        else:
            patience_counter += 1
            logger.info(f"No improvement, patience: {patience_counter}/{patience}\n")
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs\n")
                break

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
