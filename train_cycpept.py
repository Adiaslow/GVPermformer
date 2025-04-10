# train_cycpept.py
"""
Script to train the Graph VAE model on cyclic peptide data.
Optimized for Apple Metal GPU acceleration with enhanced molecular feature engineering.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.config import Config, DataConfig, ModelConfig, TrainingConfig
from src.models.graph_vae import GraphVAE
from src.data.data_module import MoleculeDataModule
from src.utils.training_utils import set_seed
from src.train_and_evaluate import train_model, evaluate_model, generate_samples


def main():
    # Create configuration
    config = Config()

    # Project name
    config.project_name = "GraphVAE_CycPept_Enhanced"

    # Data configuration
    config.data = DataConfig()
    config.data.data_path = "training_data/CycPeptMPDB_Peptide_All.csv"
    config.data.smiles_col = "SMILES"
    config.data.property_cols = ["PAMPA"]  # Predicting PAMPA permeability
    config.data.batch_size = 32  # Optimized batch size for Metal
    config.data.train_val_test_split = (0.8, 0.1, 0.1)
    config.data.max_atoms = 150  # Increased for larger peptides
    config.data.num_workers = (
        2  # Metal benefits from fewer workers but more efficient ones
    )
    config.data.pin_memory = True  # Enable pinned memory for faster transfers
    config.data.prefetch_factor = 2  # Prefetch batches
    config.data.random_seed = 42

    # PAMPA filtering (remove low-quality measurements)
    config.data.filter_pampa = True  # Filter out entries with PAMPA below threshold
    config.data.pampa_threshold = -9.0  # Threshold for filtering PAMPA values

    # Model configuration
    config.model = ModelConfig()
    # Increased hidden dimensions to handle the enhanced feature set
    config.model.hidden_dim = 256  # Increased from 128 for enhanced features
    config.model.latent_dim = 64  # Increased from 32 for better representation capacity
    config.model.dropout = 0.1  # Reduced dropout for better MPS performance
    config.model.beta = 0.5
    config.model.property_prediction = True
    config.model.weight_decay = 1e-5  # Better regularization

    # Add new parameters for feature attention and Huber loss
    config.model.use_huber_loss = True  # Use Huber loss for property prediction
    config.model.use_feature_attention = True  # Enable feature-wise attention

    # Training configuration
    config.training = TrainingConfig()
    config.training.max_epochs = 30  # Increased for better convergence
    config.training.learning_rate = 1e-3  # Higher learning rate for initial training
    config.training.patience = 10  # Increased patience for better convergence

    # Gradient accumulation for larger effective batch size
    config.training.accumulate_grad_batches = 2

    # Gradient clipping to prevent exploding gradients
    config.training.gradient_clip_val = 0.5

    # Validation check frequency
    config.training.val_check_interval = 0.5  # Check validation set twice per epoch

    # OneCycleLR scheduler for better convergence
    config.training.lr_scheduler = "one_cycle"

    # Create output directories
    output_dir = "outputs/cycpept_model_enhanced"
    os.makedirs(output_dir, exist_ok=True)

    # Set checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Metal-specific optimizations
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available! Using Metal GPU.")
        # Enable async compilation for better performance
        if hasattr(torch._dynamo.config, "async_compile"):
            torch._dynamo.config.async_compile = True

        # Pre-allocate memory for better performance
        torch.mps.empty_cache()
    else:
        print("MPS not available. Using CPU.")

    print("\n" + "=" * 70)
    print("TRAINING GRAPH VAE WITH ENHANCED MOLECULAR FEATURE ENGINEERING")
    print("-" * 70)
    print(f"• Filtering PAMPA values below {config.data.pampa_threshold}")
    print(f"• Using enhanced atom features (126 features per atom)")
    print(f"• Using enhanced bond features (9 features per bond)")
    print(f"• Training with {config.model.hidden_dim} hidden dimensions")
    print(f"• Using Huber loss: {config.model.use_huber_loss}")
    print(f"• Using feature attention: {config.model.use_feature_attention}")
    print(f"• Using {config.training.lr_scheduler} scheduler")
    print(f"• Gradient accumulation: {config.training.accumulate_grad_batches} batches")
    print(f"• Predicting target: {config.data.property_cols}")
    print("=" * 70 + "\n")

    # Set random seed for reproducibility
    set_seed(config.data.random_seed)

    # Train model - pass Config object and output directory
    model, data_module, best_checkpoint_path = train_model(config, output_dir)

    # Evaluate model
    evaluation_metrics = evaluate_model(model, data_module, output_dir)

    # Generate samples
    generate_samples(model, data_module, num_samples=20, output_dir=output_dir)


if __name__ == "__main__":
    main()
