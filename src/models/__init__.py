"""
Models module for GraphVAE Transformer.
This package contains model implementations for molecular property prediction.
"""

from typing import Dict, Any

from .graph_vae import OptimizedGraphVAE
from .graph_vae_transformer import GraphVAETransformer

__all__ = ["OptimizedGraphVAE"]
