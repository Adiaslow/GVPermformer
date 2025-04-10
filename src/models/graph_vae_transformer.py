"""
Graph VAE Transformer model for peptide permeability prediction.
Combines graph neural networks with transformer architecture in a variational setting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    GCNConv,
    GATConv,
    global_mean_pool,
    global_add_pool,
)
from typing import Dict, List, Tuple, Optional, Union, Any


class GraphEncoder(nn.Module):
    """
    Encodes molecular graphs into latent representations.

    Uses message passing layers to learn node embeddings followed by
    pooling to obtain graph-level embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        edge_dim: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_gat: bool = True,
    ):
        """
        Initialize the graph encoder.

        Args:
            in_channels: Number of input node features
            hidden_channels: Size of hidden representations
            edge_dim: Number of edge features
            num_layers: Number of message passing layers
            dropout: Dropout probability
            use_gat: Whether to use GAT instead of GCN
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Initial node embedding layer
        self.node_encoder = nn.Linear(in_channels, hidden_channels)

        # Message passing layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if use_gat:
                # Graph Attention layer
                conv = GATConv(
                    hidden_channels,
                    hidden_channels,
                    edge_dim=edge_dim,
                    heads=4,
                    concat=False,
                    dropout=dropout,
                )
            else:
                # Graph Convolutional layer
                conv = GCNConv(hidden_channels, hidden_channels, improved=True)
            self.convs.append(conv)

        # Batch normalization after each layer
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_layer = nn.Linear(hidden_channels, hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the graph encoder.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph embeddings [batch_size, hidden_channels]
        """
        # Initial node embedding
        x = self.node_encoder(x)

        # Apply message passing layers
        for i, conv in enumerate(self.convs):
            # Message passing
            if isinstance(conv, GATConv) and edge_attr is not None:
                x_new = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x_new = conv(x, edge_index)

            # Batch normalization and activation
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)

            # Residual connection
            x = x + x_new

            # Dropout
            x = self.dropout(x)

        # Final output projection
        x = self.output_layer(x)

        # Apply global pooling if batch assignments are provided
        if batch is not None:
            # Mean pooling to get graph-level embeddings
            x = global_mean_pool(x, batch)

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing node embeddings.

    Applies self-attention to learn relationships between nodes in the molecular graph.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_nodes: int = 100,
    ):
        """
        Initialize the transformer encoder.

        Args:
            hidden_dim: Size of hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            max_nodes: Maximum number of nodes in a graph
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        # Positional encoding for nodes
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_nodes, hidden_dim))

        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.

        Args:
            x: Node embeddings [batch_size, num_nodes, hidden_dim]
            node_mask: Mask for padding nodes [batch_size, num_nodes]

        Returns:
            Updated node embeddings [batch_size, num_nodes, hidden_dim]
        """
        seq_len = x.size(1)

        # Add positional encodings (truncate if sequence too long)
        pos_emb = self.position_embeddings[:, :seq_len, :]
        x = x + pos_emb

        # Create attention mask from node mask
        if node_mask is not None:
            # Convert mask: True for valid nodes, False for padding
            # Then invert for transformer where True means "mask this position"
            attn_mask = ~node_mask
        else:
            attn_mask = None

        # Pass through transformer layers
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Apply layer normalization
        x = self.norm(x)

        # Output projection
        x = self.output_layer(x)

        return x


class VariationalEncoder(nn.Module):
    """
    Variational encoder for VAE.

    Maps embeddings to parameters of a diagonal Gaussian distribution in latent space.
    """

    def __init__(
        self, input_dim: int, latent_dim: int, hidden_dim: Optional[int] = None
    ):
        """
        Initialize the variational encoder.

        Args:
            input_dim: Dimension of input embeddings
            latent_dim: Dimension of latent space
            hidden_dim: Optional hidden dimension size
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Layers for mean and log variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the variational encoder.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Tuple of (latent_z, mu, logvar)
        """
        # Apply fully connected layers
        h = self.fc(x)

        # Get mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


class Decoder(nn.Module):
    """
    Decoder for reconstructing molecular properties from latent space.
    """

    def __init__(
        self, latent_dim: int, output_dim: int, hidden_dims: List[int] = [128, 256, 128]
    ):
        """
        Initialize the decoder.

        Args:
            latent_dim: Dimension of latent space
            output_dim: Dimension of output
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        # Build fully connected layers
        layers = []

        # Input layer
        layers.append(nn.Linear(latent_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            z: Latent space tensor [batch_size, latent_dim]

        Returns:
            Reconstructed tensor [batch_size, output_dim]
        """
        return self.decoder(z)


class PermeabilityPredictor(nn.Module):
    """
    MLP for predicting permeability from embeddings.
    """

    def __init__(
        self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.2
    ):
        """
        Initialize the permeability predictor.

        Args:
            input_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.predictor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the permeability predictor.

        Args:
            x: Input embeddings [batch_size, input_dim]

        Returns:
            Predicted permeability [batch_size, 1]
        """
        return self.predictor(x)


class GraphVAETransformer(nn.Module):
    """
    Complete Graph VAE Transformer model for peptide permeability prediction.

    Combines graph encoding, transformer processing, variational encoding,
    and permeability prediction.
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        global_input_dim: int = 0,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        encoder_layers: int = 3,
        transformer_layers: int = 2,
        decoder_hidden_dims: List[int] = [64, 128, 64],
        predictor_hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        use_gat: bool = True,
        beta: float = 0.5,  # KL divergence weight
        prediction_weight: float = 1.0,  # Weight of prediction loss
    ):
        """
        Initialize the Graph VAE Transformer model.

        Args:
            node_input_dim: Dimension of node features
            edge_input_dim: Dimension of edge features
            global_input_dim: Dimension of global features (0 if not used)
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
            encoder_layers: Number of graph encoder layers
            transformer_layers: Number of transformer layers
            decoder_hidden_dims: Hidden dimensions of decoder
            predictor_hidden_dims: Hidden dimensions of permeability predictor
            dropout: Dropout probability
            use_gat: Whether to use GAT instead of GCN
            beta: Weight of KL divergence loss term
            prediction_weight: Weight of prediction loss term
        """
        super().__init__()

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.global_input_dim = global_input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.prediction_weight = prediction_weight

        # Graph encoder
        self.graph_encoder = GraphEncoder(
            in_channels=node_input_dim,
            hidden_channels=hidden_dim,
            edge_dim=edge_input_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            use_gat=use_gat,
        )

        # Transformer encoder
        self.transformer = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=transformer_layers,
            dropout=dropout,
        )

        # Combine graph and optional global features
        combined_dim = hidden_dim
        if global_input_dim > 0:
            self.global_projection = nn.Linear(global_input_dim, hidden_dim)
            combined_dim += hidden_dim

        # Variational encoder
        self.variational_encoder = VariationalEncoder(
            input_dim=combined_dim, latent_dim=latent_dim
        )

        # Decoder for reconstruction
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=hidden_dim,
            hidden_dims=decoder_hidden_dims,
        )

        # Permeability predictor
        self.permeability_predictor = PermeabilityPredictor(
            input_dim=latent_dim, hidden_dims=predictor_hidden_dims, dropout=dropout
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        global_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode inputs into latent space.

        Args:
            x: Node features [num_nodes, node_input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_input_dim]
            batch: Batch assignments [num_nodes]
            global_features: Optional global features [batch_size, global_input_dim]

        Returns:
            Tuple of (latent_z, mu, logvar)
        """
        # Encode graph structure
        node_embeddings = self.graph_encoder(x, edge_index, edge_attr, batch)

        # Combine with global features if available
        if global_features is not None and self.global_input_dim > 0:
            global_proj = self.global_projection(global_features)
            combined_embeddings = torch.cat([node_embeddings, global_proj], dim=1)
        else:
            combined_embeddings = node_embeddings

        # Map to latent space
        z, mu, logvar = self.variational_encoder(combined_embeddings)

        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space.

        Args:
            z: Latent space tensor [batch_size, latent_dim]

        Returns:
            Reconstructed embeddings [batch_size, hidden_dim]
        """
        return self.decoder(z)

    def predict_permeability(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict permeability from latent representation.

        Args:
            z: Latent space tensor [batch_size, latent_dim]

        Returns:
            Predicted permeability [batch_size, 1]
        """
        return self.permeability_predictor(z)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full model.

        Args:
            data: Dictionary with graph data
                - x: Node features [num_nodes, node_input_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_input_dim]
                - batch: Batch assignments [num_nodes]
                - global_features: Optional global features [batch_size, global_input_dim]

        Returns:
            Dictionary with model outputs
        """
        # Extract inputs from data
        x = data["x"]
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]
        batch = data["batch"]

        # Get global features if available
        global_features = data.get("global_features", None)

        # Encode to latent space
        z, mu, logvar = self.encode(x, edge_index, edge_attr, batch, global_features)

        # Decode to reconstruct embeddings
        reconstructed = self.decode(z)

        # Predict permeability
        permeability = self.predict_permeability(z)

        # Return all outputs
        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "reconstructed": reconstructed,
            "permeability": permeability,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        original: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE and prediction losses.

        Args:
            outputs: Model outputs from forward pass
            targets: Target permeability values [batch_size, 1]
            original: Original embeddings to compare reconstruction [batch_size, hidden_dim]

        Returns:
            Dictionary with loss components and total loss
        """
        # Extract outputs
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        reconstructed = outputs["reconstructed"]
        predicted_permeability = outputs["permeability"]

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, original)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0)  # Normalize by batch size

        # Prediction loss (MSE)
        pred_loss = F.mse_loss(predicted_permeability, targets)

        # Total loss
        total_loss = (
            recon_loss + self.beta * kl_loss + self.prediction_weight * pred_loss
        )

        # Return all loss components
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "pred_loss": pred_loss,
        }
