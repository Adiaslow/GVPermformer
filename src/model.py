# src/model.py

"""
Implementation of the Graph Transformer VAE with Property Conditioning for PAMPA prediction.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


class GraphTransformerVAE(nn.Module):
    """
    Graph Transformer Variational Autoencoder with property conditioning.
    Used for PAMPA prediction and molecule generation.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        num_node_types: int,
        num_edge_types: int,
        num_property_features: int,
        hidden_dim: int = 128,  # Reduced from 256
        latent_dim: int = 64,  # Reduced from 128
        num_layers: int = 4,  # Reduced from 6
        num_heads: int = 8,
        property_hidden_dim: int = 64,
        dropout: float = 0.1,
        *,
        use_positional_encoding: bool = True,
        beta: float = 0.001,  # Reduced from 0.1
        lambda_prop: float = 1.5,  # Increased from 1.0
    ) -> None:
        """Initialize the Graph Transformer VAE."""
        super().__init__()

        # Save parameters
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_property_features = num_property_features
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)
        self.lambda_prop = nn.Parameter(torch.tensor(lambda_prop), requires_grad=False)

        # Encoder components
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim)

        # Graph Transformer Encoder
        self.encoder_layers = nn.ModuleList(
            [
                GraphTransformerLayer(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # VAE components
        self.mu_encoder = nn.Linear(hidden_dim, latent_dim)
        self.logvar_encoder = nn.Linear(hidden_dim, latent_dim)

        # Property predictor
        self.property_predictor = nn.Sequential(
            nn.Linear(latent_dim, property_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(property_hidden_dim, num_property_features),
        )

        # Decoder components
        self.latent_to_hidden = nn.Linear(
            latent_dim + num_property_features, hidden_dim
        )

        # Graph Transformer Decoder
        self.decoder_layers = nn.ModuleList(
            [
                GraphTransformerLayer(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # Output layers - use node_feature_dim instead of num_node_types
        self.node_decoder = nn.Linear(hidden_dim, node_feature_dim)
        self.edge_decoder = nn.Linear(hidden_dim * 2, num_edge_types)

    def encode(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input graph into latent space.

        Args:
            node_features: Node features [num_nodes, node_feature_dim]
            edge_features: Edge features [num_edges, edge_feature_dim]
            edge_index: Edge indices [2, num_edges]
            mask: Optional batch indices for nodes [num_nodes]

        Returns:
            Tuple of (mu, logvar) each of shape [batch_size, latent_dim]
        """
        # Embed node and edge features
        x = self.node_embedding(node_features)
        edge_h = self.edge_embedding(edge_features)

        # Apply encoder transformer layers
        for layer in self.encoder_layers:
            x = layer(x, edge_h, edge_index, mask)

        # Pool graph representation
        if mask is not None:
            # Use scatter_mean to pool node features per graph
            batch_size = mask.max().item() + 1
            graph_repr = scatter_mean(x, mask, dim=0, dim_size=batch_size)
        else:
            # If no batch info, treat as single graph
            graph_repr = x.mean(dim=0, keepdim=True)

        # Get latent parameters
        mu = self.mu_encoder(graph_repr)
        logvar = self.logvar_encoder(graph_repr)

        return mu, logvar

    def decode(
        self,
        z: torch.Tensor,
        property_features: torch.Tensor,
        num_nodes: int,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode latent representation into graph components.

        Args:
            z: Latent vectors [batch_size, latent_dim]
            property_features: Property features [batch_size, num_property_features]
            num_nodes: Number of nodes per graph
            batch: Optional batch indices for nodes [batch_size * num_nodes]

        Returns:
            Tuple of (node_logits, edge_logits)
            node_logits: [batch_size * num_nodes, num_node_types]
            edge_logits: Optional[batch_size * num_nodes, num_nodes, num_edge_types]
        """
        batch_size = z.size(0)

        # Concatenate latent and property features
        z_prop = torch.cat(
            [z, property_features], dim=-1
        )  # [batch_size, latent_dim + num_property_features]

        # Project to hidden dimension
        h = self.latent_to_hidden(z_prop)  # [batch_size, hidden_dim]

        # Create a dummy batch tensor if not provided
        if batch is None:
            # Create batch indices for a single graph
            batch_new = torch.zeros(num_nodes, dtype=torch.long, device=h.device)
        else:
            # Keep the original batch tensor
            batch_new = batch

        # Expand latent to match number of nodes per batch
        h_expanded = []
        for i in range(batch_size):
            # Count nodes for this batch
            if batch is None:
                nodes_in_batch = num_nodes
            else:
                nodes_in_batch = (batch == i).sum().item()

            # Expand latent for this batch
            h_batch = h[i : i + 1].expand(nodes_in_batch, -1)
            h_expanded.append(h_batch)

        # Combine expanded latents
        h = torch.cat(h_expanded, dim=0)  # [total_nodes, hidden_dim]

        # Apply decoder transformer layers
        for layer in self.decoder_layers:
            h = layer(h, None, None, batch_new)

        # Decode nodes
        node_logits = self.node_decoder(h)  # [total_nodes, num_node_types]

        # Decode edges - simplified approach
        edge_logits = None

        return node_logits, edge_logits

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Perform reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass of the model.

        Args:
            node_features: Node feature tensor [num_nodes, node_feature_dim]
            edge_features: Edge feature tensor [num_edges, edge_feature_dim]
            edge_index: Edge index tensor [2, num_edges]
            mask: Optional batch indices for nodes [num_nodes]

        Returns:
            dict containing model outputs and latent variables
        """
        # Encode
        mu, logvar = self.encode(node_features, edge_features, edge_index, mask)

        # Sample latent variable
        z = self.reparameterize(mu, logvar)

        # Predict properties
        predicted_properties = self.property_predictor(z)

        # Get number of nodes per graph for decoding
        if mask is not None:
            batch_size = mask.max().item() + 1
            nodes_per_graph = torch.bincount(mask)[mask.unique()]
            max_nodes = int(nodes_per_graph.max().item())
        else:
            batch_size = 1
            max_nodes = node_features.size(0)

        # Decode
        node_logits, edge_logits = self.decode(z, predicted_properties, max_nodes, mask)

        return {
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "predicted_properties": predicted_properties,
        }


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer Layer with edge features and graph attention.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        """Initialize the graph transformer layer."""
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert (
            self.head_dim * num_heads == hidden_dim
        ), "hidden_dim must be divisible by num_heads"

        # Multi-head attention
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)

        # Edge feature integration
        self.edge_attention = nn.Linear(hidden_dim, num_heads)

        # Output layers
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the graph transformer layer.

        Args:
            x: Node features [batch_size * num_nodes, hidden_dim]
            edge_features: Edge features [num_edges, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for nodes [batch_size * num_nodes]

        Returns:
            Updated node features [batch_size * num_nodes, hidden_dim]
        """
        # Save input for residual connection
        residual = x

        # Apply layer normalization
        x = self.norm1(x)

        # Multi-head attention
        N = x.size(0)  # Total number of nodes across all graphs
        q = self.q_linear(x).view(N, self.num_heads, self.head_dim)  # [N, H, D]
        k = self.k_linear(x).view(N, self.num_heads, self.head_dim)  # [N, H, D]
        v = self.v_linear(x).view(N, self.num_heads, self.head_dim)  # [N, H, D]

        # Compute attention scores
        # [N, H, D] x [N, H, D] -> [N, H, N]
        scores = torch.bmm(
            q.transpose(0, 1),  # [H, N, D]
            k.transpose(0, 1).transpose(-2, -1),  # [H, D, N]
        )  # [H, N, N]
        scores = scores.transpose(0, 1)  # [N, H, N]

        # Scale attention scores
        scores = scores / (self.head_dim**0.5)

        # If batch indices are provided, use them to create attention masks
        if batch is not None:
            # Create mask to only allow attention within same graph
            mask = batch.unsqueeze(-1) == batch.unsqueeze(-2)  # [N, N]
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1)  # [N, H, N]
            scores = scores.masked_fill(~mask, float("-inf"))

        # Apply edge features if available - simplified approach
        if edge_features is not None and edge_index is not None:
            # Skip edge attention for now to simplify the model
            pass

        # Compute attention weights and apply to values
        attn_weights = F.softmax(scores, dim=-1)  # [N, H, N]
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        # [N, H, N] x [N, H, D] -> [N, H, D]
        out = torch.bmm(
            attn_weights.transpose(0, 1), v.transpose(0, 1)  # [H, N, N]  # [H, N, D]
        )  # [H, N, D]
        out = out.transpose(0, 1)  # [N, H, D]

        # Combine heads and apply output transformation
        out = out.reshape(N, self.hidden_dim)
        out = self.output(out)
        out = self.dropout(out)

        # Residual connection
        x = residual + out

        # Feed-forward network with residual connection
        x = x + self.ffn(self.norm2(x))

        return x
