# tests/test_model.py

"""
Tests for the Graph Transformer VAE model.
"""

import pytest
import torch

from src.model import GraphTransformerVAE
from configs.model_config import ModelConfig


@pytest.fixture
def model_config() -> ModelConfig:
    """Create a model configuration for testing."""
    return ModelConfig(
        node_feature_dim=32,
        edge_feature_dim=16,
        hidden_dim=64,
        latent_dim=32,
        num_layers=2,
        num_heads=4,
        num_node_types=5,
        num_edge_types=3,
        num_property_features=1,
        property_hidden_dim=32,
        dropout=0.1,
    )


@pytest.fixture
def model(model_config: ModelConfig) -> GraphTransformerVAE:
    """Create a model instance for testing."""
    return GraphTransformerVAE(
        node_feature_dim=model_config.node_feature_dim,
        edge_feature_dim=model_config.edge_feature_dim,
        hidden_dim=model_config.hidden_dim,
        latent_dim=model_config.latent_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        num_node_types=model_config.num_node_types,
        num_edge_types=model_config.num_edge_types,
        num_property_features=model_config.num_property_features,
        property_hidden_dim=model_config.property_hidden_dim,
        dropout=model_config.dropout,
    )


def test_model_initialization(
    model: GraphTransformerVAE, model_config: ModelConfig
) -> None:
    """Test if the model initializes with correct parameters."""
    assert model.node_feature_dim == model_config.node_feature_dim
    assert model.edge_feature_dim == model_config.edge_feature_dim
    assert model.hidden_dim == model_config.hidden_dim
    assert model.latent_dim == model_config.latent_dim
    assert model.num_property_features == model_config.num_property_features


def test_model_forward(model: GraphTransformerVAE, model_config: ModelConfig) -> None:
    """Test the forward pass of the model."""
    batch_size = 4
    num_nodes = 10

    # Create dummy input
    node_features = torch.randn(batch_size, num_nodes, model_config.node_feature_dim)
    edge_features = torch.randn(
        batch_size, num_nodes, num_nodes, model_config.edge_feature_dim
    )
    adj_matrix = torch.ones(batch_size, num_nodes, num_nodes, dtype=torch.bool)

    # Forward pass
    outputs = model(node_features, edge_features, adj_matrix)

    # Check output shapes
    assert outputs["node_logits"].shape == (
        batch_size,
        num_nodes,
        model_config.num_node_types,
    )
    assert outputs["edge_logits"].shape == (
        batch_size,
        num_nodes,
        num_nodes,
        model_config.num_edge_types,
    )
    assert outputs["mu"].shape == (batch_size, model_config.latent_dim)
    assert outputs["logvar"].shape == (batch_size, model_config.latent_dim)
    assert outputs["z"].shape == (batch_size, model_config.latent_dim)
    assert outputs["predicted_properties"].shape == (
        batch_size,
        model_config.num_property_features,
    )


def test_model_with_mask(model: GraphTransformerVAE, model_config: ModelConfig) -> None:
    """Test the model with node mask."""
    batch_size = 4
    num_nodes = 10

    # Create dummy input
    node_features = torch.randn(batch_size, num_nodes, model_config.node_feature_dim)
    edge_features = torch.randn(
        batch_size, num_nodes, num_nodes, model_config.edge_feature_dim
    )
    adj_matrix = torch.ones(batch_size, num_nodes, num_nodes, dtype=torch.bool)

    # Create mask (mask out some nodes)
    mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
    mask[:, num_nodes // 2 :] = False

    # Forward pass with mask
    outputs = model(node_features, edge_features, adj_matrix, mask)

    # Check output shapes
    assert outputs["node_logits"].shape == (
        batch_size,
        num_nodes,
        model_config.num_node_types,
    )
    assert outputs["edge_logits"].shape == (
        batch_size,
        num_nodes,
        num_nodes,
        model_config.num_edge_types,
    )
    assert outputs["mu"].shape == (batch_size, model_config.latent_dim)
    assert outputs["logvar"].shape == (batch_size, model_config.latent_dim)
    assert outputs["z"].shape == (batch_size, model_config.latent_dim)
    assert outputs["predicted_properties"].shape == (
        batch_size,
        model_config.num_property_features,
    )


def test_encode_decode(model: GraphTransformerVAE, model_config: ModelConfig) -> None:
    """Test the encode and decode functions separately."""
    batch_size = 4
    num_nodes = 10

    # Create dummy input
    node_features = torch.randn(batch_size, num_nodes, model_config.node_feature_dim)
    edge_features = torch.randn(
        batch_size, num_nodes, num_nodes, model_config.edge_feature_dim
    )
    adj_matrix = torch.ones(batch_size, num_nodes, num_nodes, dtype=torch.bool)

    # Test encode
    mu, logvar = model.encode(node_features, edge_features, adj_matrix)
    assert mu.shape == (batch_size, model_config.latent_dim)
    assert logvar.shape == (batch_size, model_config.latent_dim)

    # Sample from latent space
    z = model.reparameterize(mu, logvar)
    assert z.shape == (batch_size, model_config.latent_dim)

    # Create dummy property features
    property_features = torch.randn(batch_size, model_config.num_property_features)

    # Test decode
    node_logits, edge_logits = model.decode(z, property_features, num_nodes)
    assert node_logits.shape == (batch_size, num_nodes, model_config.num_node_types)
    assert edge_logits.shape == (
        batch_size,
        num_nodes,
        num_nodes,
        model_config.num_edge_types,
    )
