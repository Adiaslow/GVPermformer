#!/usr/bin/env python3
# train_cycpept.py

import torch
import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.graph_vae import OptimizedGraphVAE
from torch_geometric.loader import DataLoader
import glob
import os
from torch_geometric.data import Data, Batch
from src.utils.device_utils import optimize_for_device
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch.multiprocessing as mp
import concurrent.futures


# Set optimal sharing strategy for multiprocessing
mp.set_sharing_strategy("file_system")

# Enable tensor cores if available
torch.set_float32_matmul_precision("high")


# JIT compile common operations
@torch.jit.script
def compute_loss(x1, x2, z_mean, z_logvar, beta: float, num_graphs: int):
    recon_loss = torch.nn.functional.mse_loss(x1, x2)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    kl_loss = kl_loss / num_graphs
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Graph VAE model on cyclic peptide data"
    )

    # Add arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="processed_cycpept_data",
        help="Directory containing preprocessed data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_outputs",
        help="Directory to save model outputs",
    )
    parser.add_argument(
        "--node_features", type=int, default=9, help="Number of node features"
    )
    parser.add_argument(
        "--edge_features", type=int, default=5, help="Number of edge features"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension size"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=64, help="Latent dimension size"
    )
    parser.add_argument(
        "--max_nodes", type=int, default=100, help="Maximum number of nodes"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--beta", type=float, default=0.1, help="KL weight parameter")
    parser.add_argument(
        "--gnn_type", type=str, default="gcn", help="Type of GNN to use"
    )
    parser.add_argument(
        "--use_edge_features", action="store_true", help="Whether to use edge features"
    )
    parser.add_argument(
        "--use_enhanced_features",
        action="store_true",
        help="Whether to use enhanced features",
    )
    parser.add_argument(
        "--property_prediction",
        action="store_true",
        help="Whether to include property prediction",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient clipping value",
    )

    return parser.parse_args()


def load_split_info(data_dir: str) -> Dict:
    """Load split configuration from json file."""
    split_info_path = os.path.join(data_dir, "split_info.json")
    if not os.path.exists(split_info_path):
        raise FileNotFoundError(f"Split info file not found: {split_info_path}")

    with open(split_info_path, "r") as f:
        return json.load(f)


class OptimizedDataset(torch.utils.data.Dataset):
    """Memory-optimized dataset with efficient data loading."""

    def __init__(self, data_dir: str):
        super().__init__()

        # Get sorted file paths
        self.data_files = sorted(
            [
                f
                for f in glob.glob(os.path.join(data_dir, "mol_*.pt"))
                if self._valid_file_index(f)
            ]
        )

        if not self.data_files:
            raise ValueError(f"No valid molecular data files found in {data_dir}")

        # Pre-load and process all data
        print(f"Loading and preprocessing {len(self.data_files)} molecules...")
        self.cached_data = self._preload_data()
        print("Dataset loading complete!")

    @staticmethod
    def _valid_file_index(filepath: str) -> bool:
        try:
            int(os.path.basename(filepath).split("_")[1].split(".")[0])
            return True
        except (IndexError, ValueError):
            return False

    def _preload_data(self) -> List[Data]:
        # Add safe globals for PyTorch 2.6
        from torch_geometric.data import Data
        import torch.serialization

        torch.serialization.add_safe_globals([Data])

        cached_data = []
        batch_size = 1000  # Process in larger batches for efficiency

        for i in range(0, len(self.data_files), batch_size):
            batch_files = self.data_files[i : i + batch_size]
            batch_data = []

            # Process files in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(self._process_file, file_path)
                    for file_path in batch_files
                ]

                for future in concurrent.futures.as_completed(futures):
                    data = future.result()
                    if data is not None:
                        batch_data.append(data)

            cached_data.extend(batch_data)
            print(
                f"Processed {min(i + batch_size, len(self.data_files))}/{len(self.data_files)} molecules"
            )

        return cached_data

    def _process_file(self, file_path: str) -> Optional[Data]:
        try:
            data = torch.load(file_path, weights_only=False)
            if not isinstance(data, Data):
                return None

            # Efficient preprocessing
            if data.x.dim() == 1:
                data.x = data.x.unsqueeze(-1)
            data.x = data.x.to(torch.float32)

            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                if data.edge_attr.dim() == 1:
                    data.edge_attr = data.edge_attr.unsqueeze(-1)
                data.edge_attr = data.edge_attr.to(torch.float32)

            if not hasattr(data, "edge_index") or data.edge_index is None:
                num_nodes = data.x.size(0)
                data.edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()

            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            return data
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, idx: int) -> Data:
        return self.cached_data[idx]


def optimized_collate(data_list: List[Data]) -> Batch:
    """Optimized collate function for faster batching."""
    return Batch.from_data_list(data_list)


def setup_data_loaders(
    data_dir: str,
    fold_idx: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Setup optimized data loaders."""

    # Load split configuration
    split_info = load_split_info(data_dir)

    # Determine fold
    if fold_idx is None:
        fold_idx = 0
    elif fold_idx >= split_info["n_splits"]:
        raise ValueError(
            f"Invalid fold index {fold_idx}, max is {split_info['n_splits']-1}"
        )

    fold_dir = os.path.join(data_dir, f"fold_{fold_idx}")

    # Optimize device parameters
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    if batch_size is None or num_workers is None:
        device_params = optimize_for_device(device)
        batch_size = batch_size or device_params["batch_size"]
        num_workers = num_workers or min(os.cpu_count() or 1, 8)

    print(f"Using {num_workers} workers for data loading")

    # Create optimized datasets
    train_dataset = OptimizedDataset(os.path.join(fold_dir, "train"))
    val_dataset = OptimizedDataset(os.path.join(fold_dir, "val"))
    test_dataset = OptimizedDataset(os.path.join(fold_dir, "test"))

    # Optimized DataLoader settings
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device != "cpu",
        "prefetch_factor": 4,
        "persistent_workers": True,
        "collate_fn": optimized_collate,
    }

    # Create data loaders with optimized settings
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    print(
        f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


def setup_training(args):
    """Setup training callbacks and logger"""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="model-{epoch:02d}-{train_loss:.4f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="train_loss", patience=10, mode="min", min_delta=0.001
    )

    # Setup logger
    logger = TensorBoardLogger(save_dir=str(output_dir), name="training_logs")

    return checkpoint_callback, early_stopping, logger


def get_device():
    """Get the appropriate device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_fold(args, fold_idx: int, output_dir: str):
    """Train model on a specific fold."""
    # Load split info
    split_info = load_split_info(args.data_dir)

    # Update args with correct feature dimensions from split info
    args.node_features = 8  # Number of node features from create_graph_data
    args.edge_features = 4  # Number of edge features from create_graph_data
    args.max_nodes = split_info["max_nodes"]

    # Setup data loaders
    train_loader, val_loader, test_loader = setup_data_loaders(
        args.data_dir, fold_idx=fold_idx, batch_size=args.batch_size
    )

    print(
        f"Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}"
    )

    # Initialize model
    model = OptimizedGraphVAE(
        node_features=args.node_features,
        edge_features=args.edge_features,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_nodes=args.max_nodes,
        dropout=args.dropout,
        use_edge_features=args.use_edge_features,
        use_enhanced_features=args.use_enhanced_features,
        property_prediction=args.property_prediction,
        learning_rate=args.learning_rate,
        beta=args.beta,
        gnn_type=args.gnn_type,
    )

    # Setup training
    fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=fold_dir,
        filename="model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    # Logger
    logger = TensorBoardLogger(fold_dir, name="training_logs")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test
    test_results = trainer.test(model, test_loader)

    return test_results


def main(args):
    """Main training function with k-fold cross-validation."""
    # Load split configuration
    split_info = load_split_info(args.data_dir)
    n_splits = split_info["n_splits"]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train on each fold
    all_results = []
    for fold_idx in range(n_splits):
        print(f"\nTraining fold {fold_idx}/{n_splits-1}")
        fold_results = train_fold(args, fold_idx, args.output_dir)
        all_results.append(fold_results)

    # Compute and save average results
    avg_results = {}
    std_results = {}
    for metric in all_results[0].keys():
        values = [r[metric] for r in all_results]
        avg_results[metric] = float(np.mean(values))
        std_results[metric] = float(np.std(values))

    summary = {"average": avg_results, "std": std_results, "folds": all_results}

    summary_path = os.path.join(args.output_dir, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCross-validation complete!")
    print(f"Average test results:")
    for metric, value in avg_results.items():
        print(f"{metric}: {value:.4f} ± {std_results[metric]:.4f}")


if __name__ == "__main__":
    main(parse_args())
