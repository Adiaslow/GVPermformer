"""
Command-line interface for training and evaluating the Graph VAE Transformer model.
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.config import config_from_args
from src.models.graph_vae import GraphVAE
from src.data.data_module import MoleculeDataModule
from src.utils.training_utils import set_seed


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate Graph VAE Transformer model"
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to CSV file with SMILES and properties",
    )
    data_group.add_argument(
        "--smiles_col",
        type=str,
        default="smiles",
        help="Name of column containing SMILES strings",
    )
    data_group.add_argument(
        "--property_cols",
        nargs="+",
        default=["logP", "QED", "SA"],
        help="List of property column names to predict",
    )
    data_group.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    data_group.add_argument(
        "--max_atoms", type=int, default=50, help="Maximum number of atoms in molecules"
    )
    data_group.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension size in model"
    )
    model_group.add_argument(
        "--latent_dim", type=int, default=64, help="Latent space dimension size"
    )
    model_group.add_argument(
        "--num_layers", type=int, default=3, help="Number of graph convolution layers"
    )
    model_group.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    model_group.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    model_group.add_argument(
        "--beta", type=float, default=0.5, help="Weight for KL divergence loss term"
    )
    model_group.add_argument(
        "--property_prediction", action="store_true", help="Enable property prediction"
    )

    # Training arguments
    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of training epochs"
    )
    training_group.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    training_group.add_argument(
        "--weight_decay", type=float, default=1e-6, help="Weight decay for optimizer"
    )
    training_group.add_argument(
        "--early_stopping", action="store_true", help="Enable early stopping"
    )
    training_group.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )
    training_group.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    training_group.add_argument(
        "--log_dir", type=str, default="logs", help="Directory to save training logs"
    )

    # Hardware arguments
    hardware_group = parser.add_argument_group("Hardware")
    hardware_group.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator to use (cpu, gpu, mps, tpu, auto)",
    )
    hardware_group.add_argument(
        "--devices", type=int, default=1, help="Number of devices to use"
    )
    hardware_group.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Precision for training (16, 32, 64, bf16)",
    )

    # Misc arguments
    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )
    misc_group.add_argument(
        "--debug", action="store_true", help="Enable debug mode (fast dev run)"
    )
    misc_group.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for reproducibility",
    )

    return parser.parse_args()


def main():
    """Main function to run training."""
    # Parse arguments
    args = parse_args()

    # Create configuration
    config = config_from_args(args)

    # Set random seed for reproducibility
    set_seed(config.data.random_seed)

    # Create data module
    data_module = MoleculeDataModule(
        data_path=config.data.data_path,
        smiles_col=config.data.smiles_col,
        property_cols=config.data.property_cols,
        batch_size=config.data.batch_size,
        train_val_test_split=config.data.train_val_test_split,
        num_workers=config.data.num_workers,
        max_atoms=config.data.max_atoms,
        seed=config.data.random_seed,
    )

    # Setup data module to get feature dimensions
    data_module.prepare_data()
    data_module.setup()

    # Create model
    model = GraphVAE(
        node_features=data_module.get_node_features(),
        edge_features=data_module.get_edge_features(),
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        learning_rate=config.training.learning_rate,
        property_prediction=config.model.property_prediction,
        num_properties=(
            data_module.get_num_properties() if config.model.property_prediction else 0
        ),
        beta=config.model.beta,
        max_atoms=config.data.max_atoms,
    )

    # Create callbacks
    callbacks = []

    # Checkpoint callback
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.training.checkpoint_dir,
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=config.training.save_top_k,
        monitor=config.training.monitor,
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    if config.training.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor=config.training.monitor,
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stopping_callback)

    # Create logger
    os.makedirs(config.training.log_dir, exist_ok=True)
    logger = TensorBoardLogger(
        save_dir=config.training.log_dir,
        name=config.project_name,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.precision,
        callbacks=callbacks,
        logger=logger,
        deterministic=config.deterministic,
        log_every_n_steps=config.training.log_every_n_steps,
        fast_dev_run=args.debug,
    )

    # Train model
    trainer.fit(model, data_module)

    # Test model
    trainer.test(model, data_module)

    # Save final model
    final_model_path = os.path.join(
        config.training.checkpoint_dir, f"{config.project_name}_final.pt"
    )
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Return best model path
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved to {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    main()
