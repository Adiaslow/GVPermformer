# train_simple.py
"""
Simplified script for training a Graph VAE model on the cyclic peptide dataset.
Implements basic training loop without PyTorch Lightning.
"""

import os
import time
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from rdkit import Chem
from tqdm import tqdm

# Import from local modules if available
try:
    from src.data.dataset import MoleculeDataset
    from src.models.graph_vae import GraphVAE
except ImportError:
    print("Warning: Unable to import modules from src. Using simplified versions.")
    # Simple dataset implementation
    from torch.utils.data import Dataset

    class MoleculeDataset(Dataset):
        """Simplified molecule dataset for the demo."""

        def __init__(
            self, csv_file, smiles_col="SMILES", property_cols=None, max_atoms=100
        ):
            self.data = pd.read_csv(csv_file)
            self.smiles_col = smiles_col
            self.property_cols = property_cols or []
            self.max_atoms = max_atoms
            self.n_features = 94  # Default feature size

            # Filter invalid molecules
            valid_idx = []
            for i, row in self.data.iterrows():
                smiles = row[self.smiles_col]
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None and mol.GetNumAtoms() <= self.max_atoms:
                        valid_idx.append(i)
                except:
                    pass

            self.data = self.data.iloc[valid_idx].reset_index(drop=True)
            print(f"Loaded {len(self.data)} valid molecules from dataset")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # Simplified item getter (placeholder for real implementation)
            row = self.data.iloc[idx]
            smiles = row[self.smiles_col]
            mol = Chem.MolFromSmiles(smiles)

            # Basic features
            atom_features = torch.rand(mol.GetNumAtoms(), self.n_features)
            edge_index = torch.zeros(2, mol.GetNumBonds() * 2, dtype=torch.long)
            edge_attr = torch.zeros(mol.GetNumBonds() * 2, 4)

            # Handle properties
            properties = None
            if self.property_cols:
                properties = torch.tensor(
                    [row[col] for col in self.property_cols], dtype=torch.float
                )

            sample = {
                "x": atom_features,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "num_nodes": mol.GetNumAtoms(),
                "smiles": smiles,
            }

            if properties is not None:
                sample["properties"] = properties

            return sample

        @staticmethod
        def collate_fn(batch):
            """Basic collate function."""
            batch_dict = {
                "x": [],
                "edge_index": [],
                "edge_attr": [],
                "batch": [],
                "smiles": [],
            }

            if "properties" in batch[0]:
                batch_dict["properties"] = []

            cumulative_nodes = 0
            for i, sample in enumerate(batch):
                num_nodes = sample["num_nodes"]

                batch_dict["x"].append(sample["x"])
                batch_dict["edge_index"].append(sample["edge_index"] + cumulative_nodes)
                batch_dict["edge_attr"].append(sample["edge_attr"])
                batch_dict["batch"].append(
                    torch.full((num_nodes,), i, dtype=torch.long)
                )
                batch_dict["smiles"].append(sample["smiles"])

                if "properties" in sample:
                    batch_dict["properties"].append(sample["properties"])

                cumulative_nodes += num_nodes

            # Concatenate tensors
            batch_dict["x"] = torch.cat(batch_dict["x"], dim=0)
            batch_dict["edge_index"] = torch.cat(batch_dict["edge_index"], dim=1)
            batch_dict["edge_attr"] = torch.cat(batch_dict["edge_attr"], dim=0)
            batch_dict["batch"] = torch.cat(batch_dict["batch"], dim=0)

            if "properties" in batch_dict:
                batch_dict["properties"] = torch.stack(batch_dict["properties"], dim=0)

            return batch_dict

    class GraphVAE(torch.nn.Module):
        """Simplified Graph VAE model for the demo."""

        def __init__(
            self,
            node_features,
            edge_features,
            hidden_dim=64,
            latent_dim=32,
            learning_rate=1e-4,
            property_prediction=True,
            num_properties=1,
            beta=0.5,
            max_atoms=100,
        ):
            super().__init__()
            self.node_features = node_features
            self.edge_features = edge_features
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            self.learning_rate = learning_rate
            self.property_prediction = property_prediction
            self.num_properties = num_properties
            self.beta = beta
            self.max_atoms = max_atoms

            # Encoder
            self.encoder_nn = torch.nn.Sequential(
                torch.nn.Linear(node_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )

            # Latent space
            self.mu = torch.nn.Linear(hidden_dim, latent_dim)
            self.logvar = torch.nn.Linear(hidden_dim, latent_dim)

            # Decoder
            self.decoder_nn = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )

            # Property predictor
            if property_prediction and num_properties > 0:
                self.property_nn = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(hidden_dim, num_properties),
                )

        def forward(self, batch):
            # Extract batch components
            x = batch["x"]
            edge_index = batch["edge_index"]
            edge_attr = batch["edge_attr"]
            batch_idx = batch["batch"]

            # Encode (simplified)
            h = self.encoder_nn(x)

            # Pool to graph level (simplified mean pooling)
            h_graph = torch.zeros(batch_idx.max() + 1, self.hidden_dim, device=x.device)

            # For each graph, get the mean of its node features
            for i in range(batch_idx.max() + 1):
                mask = batch_idx == i
                h_graph[i] = torch.mean(h[mask], dim=0)

            # Get latent parameters
            mu = self.mu(h_graph)
            logvar = self.logvar(h_graph)

            # Sample latent vector
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            # Decode (simplified)
            h_decoded = self.decoder_nn(z)

            # Placeholder for decoded outputs
            node_features = torch.rand(
                batch_idx.max() + 1, self.max_atoms, self.node_features
            )
            edge_features = torch.rand(batch_idx.max() + 1, 4)

            # Predict properties if needed
            output = {
                "z": z,
                "mu": mu,
                "logvar": logvar,
                "node_features": node_features,
                "edge_features": edge_features,
            }

            if self.property_prediction and self.num_properties > 0:
                properties = self.property_nn(z)
                output["properties"] = properties

            return output

        def training_step(self, batch, batch_idx):
            # Forward pass
            outputs = self(batch)

            # Calculate loss
            kl_loss = -0.5 * torch.sum(
                1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()
            )
            recon_loss = torch.nn.functional.mse_loss(
                outputs["node_features"].reshape(-1, self.node_features),
                batch["x"],
                reduction="mean",
            )

            loss = recon_loss + self.beta * kl_loss

            # Add property prediction loss if needed
            if (
                self.property_prediction
                and self.num_properties > 0
                and "properties" in batch
            ):
                property_loss = torch.nn.functional.mse_loss(
                    outputs["properties"], batch["properties"]
                )
                loss = loss + property_loss

            return loss


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(args):
    """Train the Graph VAE model."""
    # Set random seed
    set_seed(args.seed)

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = MoleculeDataset(
        csv_file=args.data_path,
        smiles_col=args.smiles_col,
        property_cols=[args.prop_col],
        max_atoms=args.max_atoms,
    )

    # Split dataset
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    print(f"Training set: {len(train_dataset)} molecules")
    print(f"Validation set: {len(val_dataset)} molecules")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    # Create model
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Using device: {device}")

    model = GraphVAE(
        node_features=dataset.n_features,
        edge_features=4,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        property_prediction=True,
        num_properties=1,
        beta=args.beta,
        max_atoms=args.max_atoms,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(args.max_epochs):
        # Training
        model.train()
        epoch_loss = 0

        # Progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                optimizer.zero_grad()
                loss = model.training_step(batch, batch_idx)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Update progress bar
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move batch to device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                loss = model.training_step(batch, batch_idx)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{args.max_epochs} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}"
        )

        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(
                args.output_dir,
                "checkpoints",
                f"model_epoch{epoch+1}_valloss{avg_val_loss:.4f}.pt",
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save final model
    final_model_path = os.path.join(args.output_dir, "checkpoints", "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "plots", "training_curve.png"))
    print(
        f"Training curve saved to {os.path.join(args.output_dir, 'plots', 'training_curve.png')}"
    )

    return final_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Graph VAE on cyclic peptide data"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="training_data/CycPeptMPDB_Peptide_All.csv",
        help="Path to CSV file with SMILES and properties",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="SMILES",
        help="Name of column containing SMILES strings",
    )
    parser.add_argument(
        "--prop_col",
        type=str,
        default="Permeability",
        help="Name of column containing permeability values",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="simple_training_output",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension size in model"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=32, help="Latent space dimension size"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training vs validation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_atoms",
        type=int,
        default=100,
        help="Maximum number of atoms in molecules",
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="Weight for KL divergence loss"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for training even if GPU is available",
    )

    args = parser.parse_args()

    train_model(args)
