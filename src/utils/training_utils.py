"""
Utility functions for model training.
Includes random seed setting, dataloader creation, and callback configuration.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)
    print(f"Random seed set to {seed}")


def create_dataloaders(
    train_dataset, val_dataset, batch_size: int = 32, num_workers: int = 4
):
    """
    Create PyTorch DataLoaders for training and validation datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_callbacks(
    checkpoint_dir: str,
    patience: int = 10,
    monitor: str = "val_loss",
    save_top_k: int = 3,
):
    """
    Create PyTorch Lightning callbacks for training.

    Args:
        checkpoint_dir: Directory to save checkpoints
        patience: Patience for early stopping
        monitor: Metric to monitor for checkpointing
        save_top_k: Number of best models to save

    Returns:
        list: List of callbacks
    """
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=save_top_k,
        monitor=monitor,
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=True,
        mode="min",
    )
    callbacks.append(early_stopping_callback)

    return callbacks
