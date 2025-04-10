"""
Utility functions for model handling, checkpointing, and inference.

This module provides functions for loading and saving models,
extracting model representations, and running inference.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.config import Config


def get_best_checkpoint(checkpoint_dir: str, mode: str = "min") -> Optional[str]:
    """
    Get the path to the best checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        mode: 'min' or 'max' for determining the best checkpoint

    Returns:
        Path to the best checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not checkpoints:
        return None

    # Extract values from checkpoint names
    values = []
    for ckpt in checkpoints:
        try:
            # Checkpoints are named like: epoch=10-val_loss=0.123.ckpt
            val_str = ckpt.split("=")[-1].split(".")[0]
            values.append(float(val_str))
        except (ValueError, IndexError):
            # Skip checkpoints with unexpected naming
            values.append(float("inf") if mode == "min" else float("-inf"))

    # Find best checkpoint
    if mode == "min":
        best_idx = np.argmin(values)
    else:
        best_idx = np.argmax(values)

    return os.path.join(checkpoint_dir, checkpoints[best_idx])


def load_model_from_checkpoint(
    model_class: Any, checkpoint_path: str, **kwargs: Any
) -> Any:
    """
    Load a model from a checkpoint.

    Args:
        model_class: Model class to instantiate
        checkpoint_path: Path to the checkpoint file
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        Loaded model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Get hyperparameters from checkpoint
    hparams = checkpoint.get("hyper_parameters", {})

    # Update with provided kwargs
    hparams.update(kwargs)

    # Create model
    model = model_class(**hparams)

    # Load state dict
    model.load_state_dict(checkpoint["state_dict"])

    return model


def save_model_checkpoint(
    model: torch.nn.Module, path: str, metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a model checkpoint.

    Args:
        model: Model to save
        path: Path to save the checkpoint
        metadata: Additional metadata to save with the checkpoint
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Prepare checkpoint
    checkpoint = {"state_dict": model.state_dict(), "metadata": metadata or {}}

    # Save checkpoint
    torch.save(checkpoint, path)


def create_checkpoint_callback(
    checkpoint_dir: str,
    monitor: str = "val_loss",
    mode: str = "min",
    save_top_k: int = 3,
    filename: Optional[str] = None,
) -> ModelCheckpoint:
    """
    Create a checkpoint callback for PyTorch Lightning.

    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor: Metric to monitor
        mode: 'min' or 'max' for determining best checkpoint
        save_top_k: Number of best models to save
        filename: Filename format for checkpoints

    Returns:
        ModelCheckpoint callback
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    filename = filename or "epoch={epoch:02d}-{" + monitor + ":.4f}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        verbose=True,
    )

    return checkpoint_callback


def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get the device of a model.

    Args:
        model: Model to check

    Returns:
        Device of the model
    """
    return next(model.parameters()).device


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move a batch of data to a device.

    Args:
        batch: Batch data
        device: Target device

    Returns:
        Batch on the target device
    """
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


def get_model_activation(
    model: torch.nn.Module, layer_name: str, batch: Dict[str, Any]
) -> torch.Tensor:
    """
    Get the activation of a specific layer in a model.

    Args:
        model: Model to check
        layer_name: Name of the layer to get activation from
        batch: Input batch data

    Returns:
        Activation tensor
    """
    # Dictionary to store activations
    activations = {}

    # Function to get activation
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    # Get layer from model
    layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            layer = module
            break

    if layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")

    # Register hook
    handle = layer.register_forward_hook(get_activation(layer_name))

    # Run forward pass
    device = get_model_device(model)
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        model(batch)

    # Remove hook
    handle.remove()

    return activations[layer_name]


def get_model_latent_representation(
    model: torch.nn.Module, batch: Dict[str, Any], latent_name: str = "z"
) -> torch.Tensor:
    """
    Get the latent representation from a model.

    Args:
        model: Model to use
        batch: Input batch data
        latent_name: Name of the latent variable in the model output

    Returns:
        Latent representation tensor
    """
    device = get_model_device(model)
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        outputs = model(batch)

    if isinstance(outputs, dict) and latent_name in outputs:
        return outputs[latent_name]
    elif isinstance(outputs, torch.Tensor):
        return outputs
    else:
        raise ValueError(
            f"Could not find latent representation '{latent_name}' in model output"
        )


def batch_predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    process_batch_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    output_keys: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Run batch predictions with a model.

    Args:
        model: Model to use for predictions
        dataloader: DataLoader for batch data
        process_batch_fn: Optional function to process batch data
        output_keys: List of keys to extract from model output
        max_batches: Maximum number of batches to process

    Returns:
        Dictionary of prediction results
    """
    device = get_model_device(model)
    model.eval()

    # Initialize results
    results = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Move batch to device
            batch = move_batch_to_device(batch, device)

            # Process batch if needed
            if process_batch_fn is not None:
                batch = process_batch_fn(batch)

            # Forward pass
            outputs = model(batch)

            # Extract results
            if isinstance(outputs, dict):
                # If output_keys provided, only extract those keys
                if output_keys is not None:
                    for key in output_keys:
                        if key in outputs:
                            if key not in results:
                                results[key] = []
                            results[key].append(outputs[key].cpu().numpy())
                else:
                    # Extract all keys
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            if key not in results:
                                results[key] = []
                            results[key].append(value.cpu().numpy())
            elif isinstance(outputs, torch.Tensor):
                # If output is a tensor, store as 'output'
                if "output" not in results:
                    results["output"] = []
                results["output"].append(outputs.cpu().numpy())

    # Concatenate results
    for key in results:
        results[key] = np.concatenate(results[key], axis=0)

    return results


def get_model_summary(model: torch.nn.Module, input_shape: Tuple) -> str:
    """
    Get a summary of a model.

    Args:
        model: Model to summarize
        input_shape: Shape of input tensor

    Returns:
        String representation of model summary
    """
    try:
        # Try to use torchinfo if available
        import torchinfo

        summary = torchinfo.summary(model, input_shape)
        return str(summary)
    except ImportError:
        # Fallback to basic summary
        return str(model)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: Model to count parameters for

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer, scheduler_type: str, **kwargs: Any
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get a learning rate scheduler.

    Args:
        optimizer: Optimizer to use with the scheduler
        scheduler_type: Type of scheduler
        **kwargs: Additional arguments for the scheduler

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 10),
            verbose=kwargs.get("verbose", True),
        )
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 100),
            eta_min=kwargs.get("eta_min", 0),
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_model_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Extract model configuration from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary of model configuration
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Extract hyper parameters
    hparams = checkpoint.get("hyper_parameters", {})

    return hparams
