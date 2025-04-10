"""
Comprehensive script for training and evaluating the Graph VAE model.

This script handles:
1. Data loading and preparation
2. Model initialization and training
3. Model evaluation with metrics
4. Visualization of results and latent space
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from src.config import get_config, config_from_args
from src.models.graph_vae import GraphVAE
from src.data.data_module import MoleculeDataModule
from src.utils.training_utils import set_seed
from src.utils.model_utils import (
    get_best_checkpoint,
    load_model_from_checkpoint,
    batch_predict,
    count_trainable_parameters,
)
from src.utils.metrics_utils import (
    calculate_regression_metrics,
    calculate_molecular_metrics,
    MetricsTracker,
)
from src.utils.visualization_utils import (
    visualize_latent_space,
    visualize_molecules_grid,
    plot_training_history,
    plot_property_correlations,
)
from src.utils.augmentation_utils import augment_smiles


def train_model(config, output_dir):
    """
    Train the Graph VAE model.

    Args:
        config: Configuration object with data, model, and training settings
        output_dir: Directory for outputs

    Returns:
        Tuple of (trained_model, data_module, best_model_path)
    """
    print("\n" + "=" * 50)
    print("TRAINING MODEL")
    print("=" * 50)

    # Initialize data module
    data_module = MoleculeDataModule(
        data_path=config.data.data_path,
        smiles_col=config.data.smiles_col,
        property_cols=config.data.property_cols,
        batch_size=config.data.batch_size,
        train_val_test_split=config.data.train_val_test_split,
        num_workers=config.data.num_workers,
        max_atoms=config.data.max_atoms,
        seed=config.data.random_seed,
        pin_memory=(
            config.data.pin_memory if hasattr(config.data, "pin_memory") else True
        ),
        prefetch_factor=(
            config.data.prefetch_factor
            if hasattr(config.data, "prefetch_factor")
            else 2
        ),
        filter_pampa=(
            config.data.filter_pampa if hasattr(config.data, "filter_pampa") else False
        ),
        pampa_threshold=(
            config.data.pampa_threshold
            if hasattr(config.data, "pampa_threshold")
            else -9.0
        ),
    )

    # Prepare data
    data_module.prepare_data()
    data_module.setup()

    # Calculate number of node and edge features
    node_features = data_module.get_node_features()
    edge_features = data_module.get_edge_features()
    global_features = data_module.get_global_features()

    print(f"Node features: {node_features}")
    print(f"Edge features: {edge_features}")
    print(f"Global features: {global_features}")

    # Create model
    model = GraphVAE(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        learning_rate=config.training.learning_rate,
        weight_decay=config.model.weight_decay,
        property_prediction=config.model.property_prediction,
        num_properties=(
            len(config.data.property_cols) if config.data.property_cols else 0
        ),
        beta=config.model.beta,
        max_nodes=config.data.max_atoms,
        global_features=global_features,
        use_huber_loss=(
            config.model.use_huber_loss
            if hasattr(config.model, "use_huber_loss")
            else False
        ),
        use_feature_attention=(
            config.model.use_feature_attention
            if hasattr(config.model, "use_feature_attention")
            else False
        ),
        batch_size=config.data.batch_size,
    )

    print(
        f"Created model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )

    # Create model checkpoint callback
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=config.training.patience,
        verbose=True,
        mode="min",
    )

    callbacks = [checkpoint_callback, early_stop_callback]

    # Create trainer
    trainer_kwargs = {
        "max_epochs": config.training.max_epochs,
        "accelerator": "mps" if torch.backends.mps.is_available() else "auto",
        "devices": 1,
        "callbacks": callbacks,
        "logger": False,  # No TensorBoard logger
    }

    # Add gradient clipping if specified
    if hasattr(config.training, "gradient_clip_val"):
        trainer_kwargs["gradient_clip_val"] = config.training.gradient_clip_val

    # Add accumulate_grad_batches if specified
    if hasattr(config.training, "accumulate_grad_batches"):
        trainer_kwargs["accumulate_grad_batches"] = (
            config.training.accumulate_grad_batches
        )

    # Add val_check_interval if specified
    if hasattr(config.training, "val_check_interval"):
        trainer_kwargs["val_check_interval"] = config.training.val_check_interval

    trainer = pl.Trainer(**trainer_kwargs)

    # Train model
    print(f"Starting training for {config.training.max_epochs} epochs")
    trainer.fit(model, data_module)

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"Saved final model to {final_path}")

    # Get path to best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Best model path: {best_model_path}")
    else:
        best_model_path = final_path
        print(f"No best model found, using final model: {best_model_path}")

    return model, data_module, best_model_path


def evaluate_model(model, data_module, output_dir):
    """
    Evaluate the trained model on test data with model ensembling.

    Args:
        model: Trained Graph VAE model
        data_module: Data module containing test data
        output_dir: Directory for outputs

    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 50)
    print("EVALUATING MODEL WITH ENSEMBLING")
    print("=" * 50)

    # Create evaluation directory
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Get test dataloader
    test_loader = data_module.test_dataloader()
    print(f"Evaluating on {len(test_loader.dataset)} test samples")

    # Find checkpoint files (for ensembling)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    checkpoint_files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".ckpt") and "val_loss" in f
    ]

    # Sort by loss value (best models first)
    checkpoint_files.sort(
        key=lambda x: float(x.split("val_loss=")[1].split(".ckpt")[0])
    )

    # Use up to 5 best models for ensembling
    checkpoint_files = checkpoint_files[: min(5, len(checkpoint_files))]

    if len(checkpoint_files) == 0:
        print("No checkpoint files found for ensembling. Using current model only.")
        models = [model]
    else:
        print(f"Using ensemble of {len(checkpoint_files)} models for evaluation:")
        for i, cp in enumerate(checkpoint_files):
            print(f"  {i+1}. {os.path.basename(cp)}")

        # Load models from checkpoints
        models = []
        for cp in checkpoint_files:
            # Create a new model instance with the same hyperparameters
            loaded_model = type(model)(
                node_features=data_module.get_node_features(),
                edge_features=data_module.get_edge_features(),
                hidden_dim=model.hidden_dim,
                latent_dim=model.latent_dim,
                learning_rate=model.learning_rate,
                weight_decay=model.weight_decay,
                property_prediction=model.property_prediction,
                num_properties=model.num_properties,
                beta=model.beta,
                max_nodes=model.max_nodes,
                global_features=data_module.get_global_features(),
                batch_size=model.batch_size,
            )

            # Load checkpoint weights
            checkpoint = torch.load(cp, map_location=torch.device("cpu"))
            loaded_model.load_state_dict(checkpoint["state_dict"])

            # Move to appropriate device
            device = next(model.parameters()).device
            loaded_model = loaded_model.to(device)
            loaded_model.eval()

            models.append(loaded_model)

    # Create a trainer for testing
    trainer = pl.Trainer(
        accelerator="mps" if torch.backends.mps.is_available() else "auto",
        devices=1,
        logger=False,
    )

    # Evaluate each model in the ensemble
    all_property_preds = []
    all_property_targets = []

    # Collect predictions from all models
    for i, ensemble_model in enumerate(models):
        print(f"Evaluating model {i+1}/{len(models)}")

        all_preds = []
        all_targets = []

        # Get predictions
        for batch in test_loader:
            # Move batch to appropriate device
            device = next(ensemble_model.parameters()).device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get predictions
            with torch.no_grad():
                outputs = ensemble_model(batch)

            if outputs["property_pred"] is not None and "properties" in batch:
                pred = outputs["property_pred"].cpu()
                target = batch["properties"].cpu()

                all_preds.append(pred)
                all_targets.append(target)

        if all_preds:
            all_property_preds.append(torch.cat(all_preds))
            if i == 0:  # Only need to save targets once
                all_property_targets = torch.cat(all_targets)

    # Average predictions across models for ensemble prediction
    if all_property_preds:
        # Stack along a new dimension [num_models, num_samples, num_properties]
        stacked_preds = torch.stack(all_property_preds)

        # Average across models dimension
        ensemble_predictions = torch.mean(stacked_preds, dim=0)

        # Get valid entries for evaluation
        mask = ~torch.isnan(all_property_targets)
        if torch.any(mask):
            valid_pred = ensemble_predictions[mask]
            valid_target = all_property_targets[mask]

            # Calculate metrics
            mse = F.mse_loss(valid_pred, valid_target).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()
            mae = F.l1_loss(valid_pred, valid_target).item()

            # Calculate R²
            target_mean = torch.mean(valid_target)
            total_var = torch.sum((valid_target - target_mean) ** 2)
            unexplained_var = torch.sum((valid_target - valid_pred) ** 2)
            r2 = 1 - (unexplained_var / total_var).item()

            # Create evaluation dictionary
            evaluation_metrics = {
                "test_mse": mse,
                "test_rmse": rmse,
                "test_mae": mae,
                "test_r2": r2,
            }

            # Print metrics
            print(f"\nEnsemble Evaluation Metrics:")
            print(f"• MSE: {mse:.4f}")
            print(f"• RMSE: {rmse:.4f}")
            print(f"• MAE: {mae:.4f}")
            print(f"• R²: {r2:.4f}")

            # Plot predicted vs actual values
            plt.figure(figsize=(8, 8))
            plt.scatter(valid_target.numpy(), valid_pred.numpy(), alpha=0.6)
            plt.plot(
                [min(valid_target).item(), max(valid_target).item()],
                [min(valid_target).item(), max(valid_target).item()],
                "r--",
            )
            plt.xlabel("True PAMPA")
            plt.ylabel("Predicted PAMPA")
            plt.title(f"Ensemble Predictions (R² = {r2:.4f}, RMSE = {rmse:.4f})")
            plt.tight_layout()
            plt.savefig(os.path.join(eval_dir, "ensemble_predictions.png"))
            plt.close()

            # Save predictions to CSV
            df = pd.DataFrame(
                {
                    "True_PAMPA": valid_target.numpy().flatten(),
                    "Predicted_PAMPA": valid_pred.numpy().flatten(),
                }
            )
            df.to_csv(os.path.join(eval_dir, "ensemble_predictions.csv"), index=False)

            return evaluation_metrics
        else:
            print("No valid property targets found for evaluation")
    else:
        print("No property predictions available for evaluation")

    return {}


def generate_samples(model, data_module, num_samples=10, output_dir=None):
    """
    Generate new molecule samples from the latent space.

    Args:
        model: Trained Graph VAE model
        data_module: Data module
        num_samples: Number of samples to generate
        output_dir: Directory for outputs

    Returns:
        List of generated SMILES strings
    """
    print("\n" + "=" * 50)
    print("GENERATING SAMPLES")
    print("=" * 50)

    # Create generation directory
    if output_dir:
        gen_dir = os.path.join(output_dir, "generated")
        os.makedirs(gen_dir, exist_ok=True)

    # Sample from the latent space
    device = next(model.parameters()).device
    z = torch.randn(num_samples, model.latent_dim, device=device)
    print(f"Sampling {num_samples} vectors from the latent space")

    # Generate molecules from latent vectors
    model.eval()
    with torch.no_grad():
        # Forward pass through the decoder
        try:
            generated_data = model.decode(z)

            if isinstance(generated_data, dict) and "smiles" in generated_data:
                generated_smiles = generated_data["smiles"]
                print(f"Generated {len(generated_smiles)} SMILES strings")

                # Calculate molecular metrics
                mol_metrics = calculate_molecular_metrics(generated_smiles)
                print("Generation metrics:")
                print(f"  Validity: {mol_metrics['validity']:.4f}")
                print(f"  Uniqueness: {mol_metrics['uniqueness']:.4f}")

                # Visualize generated molecules
                if output_dir:
                    try:
                        from rdkit import Chem

                        # Convert to molecules
                        valid_mols = [Chem.MolFromSmiles(s) for s in generated_smiles]
                        valid_mols = [m for m in valid_mols if m is not None]

                        if valid_mols:
                            # Visualize generated molecules
                            plt.figure(figsize=(12, 8))
                            visualize_molecules_grid(
                                valid_mols[: min(20, len(valid_mols))],
                                labels=[
                                    f"Gen {i}" for i in range(min(20, len(valid_mols)))
                                ],
                                title="Generated Molecules from Random Sampling",
                                save_path=os.path.join(
                                    gen_dir, "sampled_molecules.png"
                                ),
                            )

                            # Save generated SMILES to file
                            with open(
                                os.path.join(gen_dir, "generated_smiles.txt"), "w"
                            ) as f:
                                for i, smiles in enumerate(generated_smiles):
                                    f.write(f"{i+1}\t{smiles}\n")
                    except Exception as e:
                        print(f"Error visualizing molecules: {e}")

                return generated_smiles
            else:
                print("Generation output does not contain SMILES strings")
                return []
        except Exception as e:
            print(f"Error generating molecules: {e}")
            return []


def main(args):
    """
    Main function to train and evaluate the model.

    Args:
        args: Command-line arguments
    """
    # Get configuration
    if args.config:
        # Load from config file
        config = get_config(args.config)
    else:
        # Create from command-line args
        config = config_from_args(args)

    # Create output directory
    output_dir = args.output_dir or "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Train model if requested
    if not args.skip_training:
        model, data_module, best_checkpoint_path = train_model(config, output_dir)
    else:
        # Load existing model
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            # Try to find best checkpoint
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            checkpoint_path = get_best_checkpoint(checkpoint_dir, mode="min")

            if not checkpoint_path:
                raise ValueError("No checkpoint found and training is skipped")

        print(f"Loading model from checkpoint: {checkpoint_path}")

        # Create data module
        data_module = MoleculeDataModule(
            data_path=config.data.data_path,
            smiles_col=config.data.smiles_col,
            property_cols=config.data.property_cols,
            batch_size=config.data.batch_size,
            train_val_test_split=config.data.train_val_test_split,
            num_workers=config.data.num_workers,
            max_atoms=config.data.max_atoms,
            seed=config.data.random_seed,
        )

        # Setup data module
        data_module.prepare_data()
        data_module.setup()

        # Create and load model
        model = GraphVAE.load_from_checkpoint(checkpoint_path)

    # Evaluate model
    if not args.skip_evaluation:
        evaluation_metrics = evaluate_model(model, data_module, output_dir)

    # Generate samples
    if args.generate_samples:
        generated_smiles = generate_samples(
            model, data_module, num_samples=args.num_samples, output_dir=output_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Graph VAE model")

    # Config options
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )

    # Control flow
    parser.add_argument(
        "--skip_training", action="store_true", help="Skip training phase"
    )
    parser.add_argument(
        "--skip_evaluation", action="store_true", help="Skip evaluation phase"
    )
    parser.add_argument(
        "--generate_samples", action="store_true", help="Generate new molecules"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to model checkpoint to load"
    )
    parser.add_argument(
        "--num_samples", type=int, default=20, help="Number of samples to generate"
    )

    # Model hyperparameters (optional, can be specified in config file)
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension")
    parser.add_argument("--latent_dim", type=int, help="Latent dimension")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")

    args = parser.parse_args()
    main(args)
