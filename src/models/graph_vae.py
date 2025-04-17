"""
Graph Variational Autoencoder model for molecular representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Tuple,
    Union,
    Callable,
    TypeVar,
    Protocol,
    runtime_checkable,
    Type,
)
from typing_extensions import TypeGuard
import torch.jit as jit

try:
    import pytorch_lightning as pl

    LightningModule = pl.LightningModule
except ImportError:
    print("Warning: pytorch_lightning not found. Some functionality may be limited.")

    class LightningModule:
        def save_hyperparameters(self) -> None:
            pass


# Define type aliases and fallbacks for torch_geometric types
try:
    from torch_geometric.nn import (
        GATConv,
        GCNConv,
        TransformerConv,
        global_mean_pool,
        MessagePassing,
    )
    from torch_geometric.data import Data as PyGData, Batch as PyGBatch

    HAS_PYGEOMETRIC = True
    Data = PyGData
    Batch = PyGBatch
except ImportError:
    print("Warning: torch_geometric not found. Some functionality may be limited.")
    HAS_PYGEOMETRIC = False

    # Type aliases for type checking when torch_geometric is not available
    @runtime_checkable
    class Data(Protocol):
        x: Tensor
        edge_index: Tensor
        edge_attr: Optional[Tensor]
        batch: Optional[Tensor]
        y: Optional[Tensor]

    @runtime_checkable
    class Batch(Data, Protocol):
        pass

    class BaseConv(nn.Module):
        def forward(
            self, x: Tensor, edge_index: Tensor, *args: Any, **kwargs: Any
        ) -> Tensor:
            raise NotImplementedError

    GATConv = BaseConv
    GCNConv = BaseConv
    TransformerConv = BaseConv
    MessagePassing = BaseConv

    def global_mean_pool(
        x: Tensor, batch: Optional[Tensor], *args: Any, **kwargs: Any
    ) -> Tensor:
        raise NotImplementedError


T = TypeVar("T", bound=Tensor)
DataT = TypeVar("DataT", bound=Union[Data, Batch])
FixtureFunction = Callable[[T], T]


# JIT-compiled operations
@torch.jit.script
def fused_gelu(x):
    return x * 0.5 * (1.0 + torch.tanh(0.797884560802865 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


@torch.jit.script
def compute_vae_loss(
    x: torch.Tensor,
    recon_x: torch.Tensor,
    z_mean: torch.Tensor,
    z_logvar: torch.Tensor,
    beta: float,
    num_graphs: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / num_graphs
    kl_loss = (
        -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp()) / num_graphs
    )
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


@torch.jit.script
def efficient_mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Efficient MSE loss computation"""
    return F.mse_loss(x, y, reduction="mean")


@torch.jit.script
def efficient_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Efficient KL divergence computation"""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class GraphEncoder(nn.Module):
    """Optimized graph encoder using efficient GNN techniques."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        gnn_type: str = "gat",
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        # Use 16-bit precision for linear layers to reduce memory usage
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim, dtype=torch.float16),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if edge_features > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_features, hidden_dim, dtype=torch.float16),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.edge_encoder = None

        # Efficient GNN layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            if gnn_type == "gat":
                conv = GATConv(
                    hidden_dim,
                    hidden_dim // 4,
                    heads=4,
                    dropout=dropout,
                    edge_dim=hidden_dim if edge_features > 0 else None,
                    add_self_loops=True,
                )
            elif gnn_type == "transformer":
                conv = TransformerConv(
                    hidden_dim,
                    hidden_dim // 4,
                    heads=4,
                    dropout=dropout,
                    edge_dim=hidden_dim if edge_features > 0 else None,
                )
            else:  # GCN
                conv = GCNConv(hidden_dim, hidden_dim)

            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Efficient global pooling
        self.global_pool = global_mean_pool

        # Latent projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim, dtype=torch.float16)
        self.fc_var = nn.Linear(hidden_dim, latent_dim, dtype=torch.float16)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Enable automatic mixed precision
        with torch.cuda.amp.autocast():
            # Initial node embedding
            h = self.node_encoder(x)

            # Process edge features if available
            edge_features = None
            if self.edge_encoder is not None and edge_attr is not None:
                edge_features = self.edge_encoder(edge_attr)

            # Efficient message passing with gradient checkpointing
            for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):

                def custom_forward(h_inner, edge_index_inner, edge_features_inner=None):
                    if isinstance(conv, (GATConv, TransformerConv)):
                        out = conv(h_inner, edge_index_inner, edge_features_inner)
                    else:
                        out = conv(h_inner, edge_index_inner)
                    return out

                h_prev = h
                if self.training:
                    h = torch.utils.checkpoint.checkpoint(
                        custom_forward, h, edge_index, edge_features
                    )
                else:
                    h = custom_forward(h, edge_index, edge_features)

                # Efficient normalization and residual
                h = norm(h)
                h = F.gelu(h)
                if h_prev.shape == h.shape:
                    h = h + h_prev

            # Efficient global pooling
            h = self.global_pool(h, batch)

            # Project to latent space
            z_mean = self.fc_mu(h)
            z_logvar = self.fc_var(h)

        return z_mean, z_logvar


class GraphDecoder(nn.Module):
    """Optimized graph decoder with efficient reconstruction."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        max_nodes: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Efficient latent processing
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, dtype=torch.float16),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Node feature generation
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float16),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_features, dtype=torch.float16),
        )

        # Optional edge feature generation
        if edge_features > 0:
            self.edge_decoder = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim, dtype=torch.float16),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, edge_features, dtype=torch.float16),
            )
        else:
            self.edge_decoder = None

    def forward(
        self, z: torch.Tensor, batch: Optional[Data] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        with torch.cuda.amp.autocast():
            # Project latent to hidden
            h = self.latent_proj(z)

            # Use precomputed structures from batch if available
            if batch is not None and hasattr(batch, "x"):
                num_nodes = batch.x.size(0)
                edge_index = batch.edge_index
                h_expanded = h[batch.batch]
            else:
                num_nodes = self.max_nodes * z.size(0)
                h_expanded = h.repeat_interleave(self.max_nodes, dim=0)
                edge_index = torch.arange(num_nodes, device=z.device)
                edge_index = torch.stack([edge_index, edge_index], dim=0)

            # Generate node features
            if self.training:

                def custom_forward(h_inner):
                    return self.node_decoder(h_inner)

                node_features = torch.utils.checkpoint.checkpoint(
                    custom_forward, h_expanded
                )
            else:
                node_features = self.node_decoder(h_expanded)

            # Generate edge features if needed
            edge_features = None
            if self.edge_decoder is not None and edge_index is not None:
                src, dst = edge_index
                edge_h = torch.cat([h_expanded[src], h_expanded[dst]], dim=-1)

                if self.training:

                    def custom_edge_forward(edge_h_inner):
                        return self.edge_decoder(edge_h_inner)

                    edge_features = torch.utils.checkpoint.checkpoint(
                        custom_edge_forward, edge_h
                    )
                else:
                    edge_features = self.edge_decoder(edge_h)

        return node_features, edge_index, edge_features


class ResidualBlock(nn.Module):
    """Residual block with layer normalization and GELU activation."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return x + self.layers(x)


class EdgeAttention(nn.Module):
    """Edge attention module for improved edge feature generation."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_h, edge_index):
        # Get node pairs for edges
        src, dst = edge_index
        src_h = node_h[src]
        dst_h = node_h[dst]

        # Compute attention scores
        edge_input = torch.cat([src_h, dst_h], dim=-1)
        attention = torch.sigmoid(self.attention(edge_input))

        # Weight the edge features
        edge_h = torch.cat([src_h * attention, dst_h * attention], dim=-1)
        return edge_h


class PropertyPredictor(nn.Module):
    """
    Enhanced property predictor using modern deep learning techniques.
    Uses ensemble of experts and uncertainty estimation.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_tasks: int = 1,
        dropout: float = 0.1,
        num_experts: int = 3,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.num_experts = num_experts

        # Expert networks
        self.experts = nn.ModuleList(
            [ExpertNetwork(latent_dim, hidden_dim, dropout) for _ in range(num_experts)]
        )

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1),
        )

        # Uncertainty estimation
        self.aleatoric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        Returns mean predictions and uncertainties.
        """
        # Get expert weights from gating network
        expert_weights = self.gate(z)  # [batch_size, num_experts]

        # Get predictions from each expert
        expert_preds = []
        expert_features = []
        for expert in self.experts:
            pred, features = expert(z)
            expert_preds.append(pred)
            expert_features.append(features)

        # Stack predictions and features
        expert_preds = torch.stack(
            expert_preds, dim=1
        )  # [batch_size, num_experts, num_tasks]
        expert_features = torch.stack(
            expert_features, dim=1
        )  # [batch_size, num_experts, hidden_dim]

        # Weighted average of predictions
        weighted_preds = torch.sum(expert_preds * expert_weights.unsqueeze(-1), dim=1)

        # Weighted average of features for uncertainty
        weighted_features = torch.sum(
            expert_features * expert_weights.unsqueeze(-1), dim=1
        )

        # Estimate aleatoric uncertainty
        aleatoric_uncertainty = self.aleatoric_head(weighted_features)

        # Estimate epistemic uncertainty from expert disagreement
        mean_pred = expert_preds.mean(dim=1)
        epistemic_uncertainty = torch.var(expert_preds, dim=1)

        # Combine uncertainties
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

        return weighted_preds, total_uncertainty


class ExpertNetwork(nn.Module):
    """Individual expert network with residual connections."""

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(2)]
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial projection
        h = self.input_proj(z)

        # Apply residual blocks
        for block in self.residual_blocks:
            h = block(h)

        # Get prediction
        pred = self.output_head(h)

        return pred, h


class GlobalFeatureProcessor(nn.Module):
    """
    MLP to process global molecular features before combining with graph encoding.
    Uses LayerNorm instead of BatchNorm to avoid issues with small batch sizes.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the global feature processor.

        Args:
            input_dim: Input dimension (number of global features)
            hidden_dim: Hidden dimension
            output_dim: Output dimension (should match latent space dimension)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Using LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),  # Using LayerNorm instead of BatchNorm
            nn.ReLU(),
        )

    def forward(self, x):
        """Forward pass through the MLP."""
        return self.mlp(x)


def is_geometric_data(obj: Any) -> TypeGuard[Union[Data, Batch]]:
    return isinstance(obj, (Data, Batch))


class OptimizedGraphVAE(LightningModule):
    """
    Graph Variational Autoencoder with optimized MPS/GPU performance
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        max_nodes: int = 100,
        dropout: float = 0.2,
        use_edge_features: bool = True,
        use_enhanced_features: bool = False,
        property_prediction: bool = False,
        learning_rate: float = 1e-3,
        beta: float = 0.1,  # KL weight parameter
        gnn_type: str = "gcn",  # Type of GNN to use
    ) -> None:
        """
        Initializes the Graph Variational Autoencoder with optimized performance.

        Args:
            node_features: Dimension of node features
            edge_features: Dimension of edge features
            hidden_dim: Hidden dimension size
            latent_dim: Latent dimension size
            max_nodes: Maximum number of nodes in a graph
            dropout: Dropout rate
            use_edge_features: Whether to use edge features
            use_enhanced_features: Whether enhanced atom features are used
            property_prediction: Whether to include property prediction
            learning_rate: Learning rate for optimization
            beta: Weight for KL divergence loss
            gnn_type: Type of GNN to use (gcn, gat, or transformer)
        """
        super().__init__()
        if not HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required for GraphVAE")
        self.save_hyperparameters()

        # Set device-specific attributes
        self.use_mps = torch.backends.mps.is_available()
        self.use_cuda = torch.cuda.is_available()

        self.learning_rate = learning_rate
        self.beta = beta
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.dropout_rate = dropout
        self.use_edge_features = use_edge_features
        self.use_enhanced_features = use_enhanced_features
        self.property_prediction = property_prediction
        self.gnn_type = gnn_type

        # Enable automatic mixed precision
        self.automatic_optimization = False

        # Initialize model components
        self.encoder = GraphEncoder(
            node_features=node_features,
            edge_features=edge_features if use_edge_features else 0,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            gnn_type=gnn_type,
        )

        self.decoder = GraphDecoder(
            node_features=node_features,
            edge_features=edge_features if use_edge_features else 0,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_nodes=max_nodes,
            dropout=dropout,
        )

        if property_prediction:
            self.property_predictor = PropertyPredictor(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_tasks=1,
                dropout=dropout,
                num_experts=3,
            )
        else:
            self.property_predictor = None

    def _move_to_device(self, data, device):
        """Move data to the specified device with proper error handling"""
        try:
            if hasattr(data, "to"):
                return data.to(device)
            elif isinstance(data, dict):
                return {
                    key: (value.to(device) if hasattr(value, "to") else value)
                    for key, value in data.items()
                }
            return data
        except Exception as e:
            print(f"Error moving data to device {device}: {str(e)}")
            raise

    def forward(self, data: DataT) -> Dict[str, Union[Tensor, Optional[Tensor]]]:
        """Forward pass of the VAE model."""
        if not is_geometric_data(data):
            raise TypeError("Input must be a PyTorch Geometric Data or Batch object")
        try:
            # Move data to the correct device
            device = self.device
            data = self._move_to_device(data, device)

            # Encode input to latent representation
            z_mean, z_logvar = self.encode(data)

            # Sample from the latent distribution
            z = reparameterize(z_mean, z_logvar)

            # Decode latent representation
            node_features, edge_index, edge_features = self.decode(z, data)

            # Property prediction if enabled
            prop_pred = None
            if self.property_prediction and self.property_predictor is not None:
                prop_pred, _ = self.property_predictor(z_mean)

            return {
                "z_mean": z_mean,
                "z_logvar": z_logvar,
                "z": z,
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_features": edge_features if self.use_edge_features else None,
                "prop_pred": prop_pred,
            }
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise

    def _prepare_batch(self, data):
        """
        Extract and validate features from input data.

        Args:
            data: Input data (PyG Data object or dictionary)

        Returns:
            Tuple of (x, edge_index, edge_attr, batch)
        """
        try:
            # Extract features
            x = data.x if hasattr(data, "x") else data.get("x")
            edge_index = (
                data.edge_index
                if hasattr(data, "edge_index")
                else data.get("edge_index")
            )
            edge_attr = (
                data.edge_attr if hasattr(data, "edge_attr") else data.get("edge_attr")
            )
            batch = data.batch if hasattr(data, "batch") else data.get("batch")

            # Validate inputs
            if x is None:
                raise ValueError("Node features (x) are required but missing")
            if edge_index is None:
                raise ValueError("Edge indices are required but missing")

            # Ensure edge_index is correctly formatted
            if edge_index.dim() == 2 and edge_index.size(0) != 2:
                edge_index = edge_index.t()

            # Create batch index if not provided
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

            # Ensure edge attributes match the model's configuration
            if self.use_edge_features and edge_attr is None:
                raise ValueError("Edge features are enabled but missing in input data")
            elif not self.use_edge_features:
                edge_attr = None

            return x, edge_index, edge_attr, batch

        except Exception as e:
            print(f"Error preparing batch: {str(e)}")
            raise

    def encode(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input data to latent space."""
        # Extract features from input data
        x, edge_index, edge_attr, batch = self._prepare_batch(data)

        # Pass through the encoder
        z_mean, z_logvar = self.encoder(x, edge_index, edge_attr, batch)

        return z_mean, z_logvar

    def decode(
        self, z: Tensor, batch: Optional[Data] = None
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Decode latent representation."""
        # Decode the latent representation
        node_features, edge_index, edge_features = self.decoder(z, batch)

        # If edge_index is None, create it
        if edge_index is None:
            # Store original batch size for later use in loss computation
            self._last_batch_size = z.size(0)

            # Create efficient edge index - self loops for compatibility
            batch_size = z.size(0)
            device = z.device

            # MPS optimization: more efficient edge index creation
            if self.use_mps:
                # Optimize edge index creation for MPS
                # Create a simple edge index with self-loops for each node
                # This is a placeholder that will be properly constructed by downstream tasks
                num_nodes = node_features.size(0) // batch_size

                # Efficient edge index creation
                node_indices = torch.arange(num_nodes, device=device)
                edge_index = torch.stack([node_indices, node_indices], dim=0)

                # Repeat for each batch
                edge_indices = []
                for i in range(batch_size):
                    offset = i * num_nodes
                    batch_edge_index = edge_index.clone()
                    batch_edge_index += offset
                    edge_indices.append(batch_edge_index)

                if edge_indices:
                    edge_index = torch.cat(edge_indices, dim=1)
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            else:
                # Standard implementation for other devices
                num_nodes = (
                    node_features.size(0) // batch_size
                    if batch_size > 0
                    else node_features.size(0)
                )

                # Create self-loops for compatibility
                indices = torch.arange(num_nodes, device=device)
                edge_index = torch.stack([indices, indices], dim=0)

                # Repeat for each batch element
                if batch_size > 1:
                    repeated_indices = []
                    for i in range(batch_size):
                        offset = i * num_nodes
                        batch_indices = edge_index.clone() + offset
                        repeated_indices.append(batch_indices)
                    edge_index = torch.cat(repeated_indices, dim=1)

        return node_features, edge_index, edge_features

    def compute_loss(
        self,
        batch: Union[Data, Batch],
        reconstructed: Tuple[Tensor, Optional[Tensor], Optional[Tensor]],
        z_mean: Tensor,
        z_logvar: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute VAE loss components."""
        # Compute reconstruction loss for node features
        # Ensure we only use the actual number of nodes from the input
        recon_loss = efficient_mse_loss(reconstructed[0][: batch.x.size(0)], batch.x)

        # Compute KL divergence loss
        kl_loss = efficient_kl_loss(z_mean, z_logvar)

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def training_step(self, batch: Batch, batch_idx: int):
        opt = self.optimizers()

        # Enable automatic mixed precision
        with torch.cuda.amp.autocast():
            # Forward pass
            z_mean, z_logvar = self.encode(batch)
            z = reparameterize(z_mean, z_logvar)
            reconstructed = self.decode(z, batch)

            # Compute losses efficiently
            loss_dict = self.compute_loss(batch, reconstructed, z_mean, z_logvar)

            # Total loss
            loss = loss_dict["loss"]

        # Backward pass with gradient scaling
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        # Log metrics
        self.log_dict(loss_dict, prog_bar=True)

        return loss

    def validation_step(self, batch: Union[Data, Batch], batch_idx: int) -> Tensor:
        """
        Validation step for Lightning.

        Args:
            batch: Input batch
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        # Forward pass
        outputs = self.forward(batch)

        # Compute loss
        loss_dict = self.compute_loss(
            batch,
            (outputs["node_features"], outputs["edge_index"], outputs["edge_features"]),
            outputs["z_mean"],
            outputs["z_logvar"],
        )

        # Log losses with explicit batch size
        batch_size = batch.num_graphs
        self.log(
            "val_loss",
            loss_dict["loss"],
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_recon_loss",
            loss_dict["recon_loss"],
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_kl_loss",
            loss_dict["kl_loss"],
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )

        return loss_dict["loss"]

    def test_step(self, batch: Union[Data, Batch], batch_idx: int) -> Tensor:
        """
        Test step for Lightning.

        Args:
            batch: Input batch
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        # Forward pass
        outputs = self.forward(batch)

        # Compute loss
        loss_dict = self.compute_loss(
            batch,
            (outputs["node_features"], outputs["edge_index"], outputs["edge_features"]),
            outputs["z_mean"],
            outputs["z_logvar"],
        )

        # Log losses
        self.log("test_loss", loss_dict["loss"], on_step=False, on_epoch=True)
        self.log(
            "test_recon_loss", loss_dict["recon_loss"], on_step=False, on_epoch=True
        )
        self.log("test_kl_loss", loss_dict["kl_loss"], on_step=False, on_epoch=True)

        return loss_dict["loss"]

    def configure_optimizers(self):
        # Use AdamW with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"},
        }

    def predict_pampa(self, smiles):
        """
        Predict PAMPA permeability for a given SMILES string.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Predicted PAMPA value (float)
        """
        from src.utils.smiles_utils import smiles_to_model_input

        # Put model in evaluation mode
        self.eval()

        # Convert SMILES to model input
        try:
            # Convert SMILES to model inputs
            with torch.no_grad():
                # Get molecule graph data from SMILES
                mol_data = smiles_to_model_input(smiles)

                if mol_data is None:
                    print(f"Could not convert SMILES to molecular graph: {smiles}")
                    return None

                # Move data to model device
                device = next(self.parameters()).device
                for key, value in mol_data.items():
                    if isinstance(value, torch.Tensor):
                        mol_data[key] = value.to(device)

                # Forward pass through encoder to get latent representation
                z_mean, _ = self.encode(mol_data)

                # If property predictor is available, predict property
                if self.property_predictor is not None:
                    # Get property prediction from latent space
                    prop_pred, _ = self.property_predictor(z_mean)

                    # Return the predicted value as a float
                    return prop_pred.item()
                else:
                    print("Property predictor not available")
                    return None

        except Exception as e:
            print(f"Error predicting PAMPA for SMILES {smiles}: {str(e)}")
            return None
