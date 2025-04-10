"""
Configuration module for the Graph VAE Transformer project.
Defines default settings and parameter configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class DataConfig:
    """Data configuration parameters."""

    # Path to the data
    data_path: str = "data/processed/molecules.csv"

    # Column names
    smiles_col: str = "smiles"
    property_cols: List[str] = field(default_factory=lambda: ["logP", "QED", "SA"])

    # Data splitting
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    random_seed: int = 42

    # Molecule processing
    max_atoms: int = 50

    # Dataloader settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = False
    prefetch_factor: int = 2

    # PAMPA filtering
    filter_pampa: bool = False
    pampa_threshold: float = -9.0


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    # Model architecture
    hidden_dim: int = 256
    latent_dim: int = 64
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1

    # VAE parameters
    beta: float = 0.5  # KL divergence weight

    # Property prediction
    property_prediction: bool = True

    # Feature enhancement
    use_huber_loss: bool = False  # Use Huber loss for property prediction
    use_feature_attention: bool = False  # Use feature-wise attention


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    # Training settings
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    lr_scheduler: str = (
        "reduce_on_plateau"  # 'reduce_on_plateau', 'cosine', 'step', 'one_cycle'
    )
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    val_check_interval: float = (
        1.0  # Check validation every N epochs or fraction of epoch
    )

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.0001

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3
    save_last: bool = False
    monitor: str = "val_loss"

    # Logging
    log_dir: str = "logs"
    log_every_n_steps: int = 10


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""

    # Project name
    project_name: str = "GraphVAETransformer"

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Hardware settings
    accelerator: str = "auto"  # 'cpu', 'gpu', 'mps', 'tpu', 'auto'
    devices: int = 1
    precision: str = "32"  # '16', '32', '64', 'bf16'

    # Reproducibility
    deterministic: bool = True


def get_config(overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Get configuration with optional overrides.

    Args:
        overrides: Dictionary of configuration overrides

    Returns:
        Config object with applied overrides
    """
    config = Config()

    if overrides:
        for key, value in overrides.items():
            if "." in key:
                # Handle nested configs (e.g., "data.batch_size")
                main_key, sub_key = key.split(".", 1)
                if hasattr(config, main_key) and hasattr(
                    getattr(config, main_key), sub_key
                ):
                    setattr(getattr(config, main_key), sub_key, value)
            else:
                # Handle top-level configs
                if hasattr(config, key):
                    setattr(config, key, value)

    return config


def config_from_args(args) -> Config:
    """
    Create configuration from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Config object with values from arguments
    """
    overrides = {}

    # Convert args namespace to dictionary
    arg_dict = vars(args)

    # Map arg keys to config keys
    mapping = {
        "data_path": "data.data_path",
        "smiles_col": "data.smiles_col",
        "property_cols": "data.property_cols",
        "batch_size": "data.batch_size",
        "max_atoms": "data.max_atoms",
        "num_workers": "data.num_workers",
        "random_seed": "data.random_seed",
        "hidden_dim": "model.hidden_dim",
        "latent_dim": "model.latent_dim",
        "num_layers": "model.num_layers",
        "num_heads": "model.num_heads",
        "dropout": "model.dropout",
        "beta": "model.beta",
        "property_prediction": "model.property_prediction",
        "max_epochs": "training.max_epochs",
        "learning_rate": "training.learning_rate",
        "weight_decay": "training.weight_decay",
        "early_stopping": "training.early_stopping",
        "patience": "training.patience",
        "checkpoint_dir": "training.checkpoint_dir",
        "log_dir": "training.log_dir",
        "accelerator": "accelerator",
        "devices": "devices",
        "precision": "precision",
        "deterministic": "deterministic",
    }

    # Apply mappings
    for arg_key, config_key in mapping.items():
        if arg_key in arg_dict and arg_dict[arg_key] is not None:
            overrides[config_key] = arg_dict[arg_key]

    return get_config(overrides)
