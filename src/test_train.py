# src/test_train.py
"""
Test script for running a small training and evaluation of the Graph VAE model.
"""

import os
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt

from src.config import Config
from src.models.graph_vae import GraphVAE
from src.data.data_module import MoleculeDataModule
from src.utils.training_utils import set_seed
from src.utils.evaluation_utils import visualize_latent_space, calculate_metrics

# Set random seed for reproducibility
set_seed(42)

# Create directory for test outputs
os.makedirs("test_outputs", exist_ok=True)


def create_test_dataset(n_samples=100):
    """
    Create a small test dataset with SMILES and random properties.

    Args:
        n_samples: Number of molecules to include

    Returns:
        Path to the created dataset
    """
    # Some example molecules (common drug-like molecules)
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "C1=CC=C2C(=C1)C=CC=C2C3=CC=CC=C3C(=O)O",  # Naproxen
        "CC(=O)NC1=CC=C(C=C1)O",  # Acetaminophen
        "COC1=CC2=C(C=C1)C(=O)C=CC2=O",  # Warfarin
        "CC12CCC(CC1)C(C)(C)O2",  # Eucalyptol
        "C1=CC=C(C=C1)CN2C=NC3=C2NC=NC=3",  # Tetramisole
        "CC1=C(C=C(C=C1)S(=O)(=O)NC(=O)NN2CCCCCC2)Cl",  # Chlorthalidone
        "O=C(NCCNC(=O)NC1=CC=CC=C1)NC2=CC=CC=C2",  # Sevelamer
    ]

    # Generate dataset with random samples from the list
    indices = np.random.choice(len(smiles_list), n_samples, replace=True)
    df_data = {
        "smiles": [smiles_list[i] for i in indices],
        "logP": np.random.normal(3.0, 1.0, n_samples),
        "QED": np.random.uniform(0.1, 0.9, n_samples),
        "SA": np.random.uniform(1.0, 5.0, n_samples),
    }

    # Create DataFrame
    df = pd.DataFrame(df_data)

    # Save to CSV
    output_path = "test_outputs/test_molecules.csv"
    df.to_csv(output_path, index=False)
    print(f"Created test dataset with {n_samples} molecules at {output_path}")

    return output_path


def run_quick_training(data_path):
    """
    Run a quick training on the test dataset.

    Args:
        data_path: Path to the test dataset

    Returns:
        Trained model and test dataloader
    """
    # Create a minimal configuration
    config = Config()
    config.project_name = "GraphVAE_Test"
    config.data.data_path = data_path
    config.data.batch_size = 8
    config.training.max_epochs = 3
    config.training.checkpoint_dir = "test_outputs/checkpoints"
    config.training.log_dir = "test_outputs/logs"

    # Create data module
    data_module = MoleculeDataModule(
        data_path=config.data.data_path,
        smiles_col=config.data.smiles_col,
        property_cols=config.data.property_cols,
        batch_size=config.data.batch_size,
        train_val_test_split=config.data.train_val_test_split,
        num_workers=0,  # Use 0 for test
        max_atoms=config.data.max_atoms,
        seed=config.data.random_seed,
    )

    # Setup data module to get feature dimensions
    data_module.prepare_data()
    data_module.setup()

    # Create model with smaller dimensions for quick testing
    model = GraphVAE(
        node_features=data_module.get_node_features(),
        edge_features=data_module.get_edge_features(),
        hidden_dim=64,  # Smaller hidden dimension
        latent_dim=8,  # Smaller latent dimension
        learning_rate=config.training.learning_rate,
        property_prediction=config.model.property_prediction,
        num_properties=data_module.get_num_properties(),
        beta=0.01,  # Lower beta for quicker convergence
        max_atoms=config.data.max_atoms,
    )

    # Create directory for checkpoints
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Run manual training loop for a few batches
    print("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for 2 epochs
    for epoch in range(2):
        model.train()
        train_loss = 0.0
        batch_count = 0

        # Limit to 5 batches per epoch for quick testing
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 5:
                break

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

            # Accumulate loss
            train_loss += loss.item()
            batch_count += 1

            print(
                f"Epoch {epoch+1}/{2}, Batch {batch_idx+1}/5, Loss: {loss.item():.4f}"
            )

        # Calculate average loss
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{2} completed, Avg Loss: {avg_train_loss:.4f}")

    print("Training completed.")

    # Save the model
    model_path = os.path.join(config.training.checkpoint_dir, "test_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model, data_module.test_dataloader()


def evaluate_model(model, test_loader):
    """
    Run evaluation on the trained model.

    Args:
        model: Trained GraphVAE model
        test_loader: Test data loader
    """
    print("Evaluating model...")
    device = next(model.parameters()).device
    model.eval()

    # Storage for evaluation
    all_z = []
    all_properties = []
    all_pred_properties = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get model outputs
            outputs = model(batch)

            # Store latent vectors and properties
            all_z.append(outputs["z"].cpu().numpy())

            if model.property_prediction:
                true_props = batch["properties"].cpu().numpy()
                pred_props = outputs["properties"].cpu().numpy()

                all_properties.append(true_props)
                all_pred_properties.append(pred_props)

    # Concatenate results
    z_array = np.concatenate(all_z, axis=0)

    # Visualize latent space
    if len(all_properties) > 0:
        prop_array = np.concatenate(all_properties, axis=0)
        pred_prop_array = np.concatenate(all_pred_properties, axis=0)

        # Compute metrics for each property
        for i, prop_name in enumerate(["logP", "QED", "SA"]):
            true_vals = prop_array[:, i]
            pred_vals = pred_prop_array[:, i]

            metrics = calculate_metrics(pred_vals, true_vals)
            print(f"\nProperty: {prop_name}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"R²: {metrics['r2']:.4f}")

        # Visualize latent space colored by first property
        plt.figure(figsize=(10, 8))
        fig = visualize_latent_space(
            z_array, prop_array[:, 0], save_path="test_outputs/latent_space.png"
        )
        plt.close()

    print("Evaluation completed.")


def main():
    """Main function to run the test."""
    # Create test dataset
    data_path = create_test_dataset(n_samples=50)

    # Run quick training
    model, test_loader = run_quick_training(data_path)

    # Evaluate model
    evaluate_model(model, test_loader)

    print("\nTest completed successfully!")
    print("Check the 'test_outputs' directory for results.")


if __name__ == "__main__":
    main()
