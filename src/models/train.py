"""
Training script for the Graph VAE Transformer model.
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.config import get_config
from src.models.graph_vae import GraphVAE
from src.data.data_module import MoleculeDataModule
from src.utils.training_utils import set_seed


def train(config_path=None):
    """
    Train the Graph VAE model using the specified configuration.

    Args:
        config_path: Path to a YAML configuration file

    Returns:
        Path to the best model checkpoint
    """
    # Load configuration
    config = get_config()

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
    parser = argparse.ArgumentParser(description="Train Graph VAE model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()

    train(args.config)
