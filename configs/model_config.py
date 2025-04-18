# configs/model_config.py

"""
Configuration settings for the GVPermformer model.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration for the GVPermformer model."""

    # Graph VAE architecture
    node_feature_dim: int = 64  # Node feature dimensionality
    edge_feature_dim: int = 32  # Edge feature dimensionality
    hidden_dim: int = 256  # Hidden dimension size
    latent_dim: int = 128  # Latent space dimension
    num_layers: int = 6  # Number of transformer layers
    num_heads: int = 8  # Number of attention heads
    dropout: float = 0.1  # Dropout rate
    use_positional_encoding: bool = True

    # Property prediction
    num_property_features: int = 1  # Number of PAMPA properties to predict
    property_hidden_dim: int = 64  # Hidden dim for property prediction

    # Graph specific
    max_num_nodes: int = 100  # Maximum number of nodes in a graph
    num_node_types: int = 10  # Number of different node types
    num_edge_types: int = 4  # Number of different edge types

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_steps: int = 1000
    kl_weight: float = 0.01  # Weight for KL divergence loss
    property_weight: float = 1.0  # Weight for property prediction loss

    # VAE specific
    beta: float = 1.0  # β-VAE parameter
    use_scheduler: bool = True  # Whether to use KL annealing

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert self.node_feature_dim > 0, "node_feature_dim must be positive"
        assert self.edge_feature_dim > 0, "edge_feature_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert 0 <= self.dropout < 1, "dropout must be between 0 and 1"
        assert self.num_property_features > 0, "num_property_features must be positive"
        assert self.property_hidden_dim > 0, "property_hidden_dim must be positive"
        assert self.max_num_nodes > 0, "max_num_nodes must be positive"
        assert self.num_node_types > 0, "num_node_types must be positive"
        assert self.num_edge_types > 0, "num_edge_types must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.kl_weight >= 0, "kl_weight must be non-negative"
        assert self.property_weight > 0, "property_weight must be positive"
        assert self.beta > 0, "beta must be positive"
