"""
Utility functions for device management across different hardware configurations.
"""

import os
from typing import Dict, Union, Any, Optional
import torch
import warnings


def get_device() -> torch.device:
    """
    Determine the optimal device for training with performance optimizations.

    Returns:
        torch.device: The most appropriate device (mps, cuda, or cpu)
    """
    # First check for Apple Metal (MPS)
    if torch.backends.mps.is_available():
        # Configure MPS memory allocation
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = (
            "0.0"  # Disable watermark limiting
        )
        os.environ["PYTORCH_MPS_ALLOCATOR_MEM_FRACTION"] = (
            "0.95"  # Use 95% of available GPU memory
        )
        os.environ["PYTORCH_ENABLE_GRAD_SCALERS"] = "1"  # Enable gradient scaling

        # Suppress common MPS warnings
        warnings.filterwarnings("ignore", message=".*MPS.*")
        warnings.filterwarnings(
            "ignore",
            message=".*Tensor for argument weight is on cpu but expected on mps.*",
        )

        # Try to enable MPS optimizations if available
        try:
            # Check for MPS backend attributes dynamically
            mps_backend = torch.backends.mps
            if hasattr(mps_backend, "enable_graph_mode"):
                # Use getattr to avoid static type checking
                enable_graph_mode = getattr(mps_backend, "enable_graph_mode")
                if callable(enable_graph_mode):
                    enable_graph_mode(True)
        except Exception as e:
            warnings.warn(f"Failed to enable MPS optimizations: {e}")

        # Return MPS device
        device = torch.device("mps")

        # Add small warmup tensor operation to initialize MPS
        try:
            x = torch.randn(10, 10, device=device)
            y = x @ x.t()  # Matrix multiplication as warmup
        except Exception as e:
            warnings.warn(f"MPS warmup failed (non-critical): {e}")

        return device

    # Then check for CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Finally, fall back to CPU
    return torch.device("cpu")


def optimize_for_device(device: torch.device) -> Dict[str, Union[int, float, bool]]:
    """
    Apply device-specific optimizations for training.

    Args:
        device: The device to optimize for

    Returns:
        Dictionary of optimized parameters
    """
    base_params: Dict[str, Union[int, float, bool]] = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "pin_memory": True,
        "num_workers": 2,
        "use_amp": False,
        "gradient_clip": 1.0,
        "prefetch_factor": 2,
    }

    # MPS-specific optimizations
    if device.type == "mps":
        base_params["batch_size"] = 64  # Larger batches work well on MPS
        base_params["learning_rate"] = 3e-4  # Slightly higher learning rate
        base_params["pin_memory"] = False  # Pin memory not beneficial for MPS
        base_params["num_workers"] = 0  # MPS works best with main thread data loading
        base_params["use_amp"] = True  # Use automatic mixed precision when available
        base_params["gradient_clip"] = 1.0  # Keep gradient clipping
        base_params["prefetch_factor"] = 1  # No prefetching needed for MPS

    # CUDA-specific optimizations
    elif device.type == "cuda":
        cpu_count = os.cpu_count() or 8  # Default to 8 if CPU count is unknown
        base_params["batch_size"] = 128  # Much larger batches for CUDA
        base_params["learning_rate"] = 1e-3  # Higher learning rate
        base_params["pin_memory"] = True  # Use pinned memory for faster transfers
        base_params["num_workers"] = min(8, cpu_count)  # More workers for CUDA
        base_params["use_amp"] = True  # Use automatic mixed precision
        base_params["gradient_clip"] = 1.0  # Keep gradient clipping
        base_params["prefetch_factor"] = 3  # Prefetch more batches

    # CPU-specific optimizations
    else:
        cpu_count = os.cpu_count() or 2  # Default to 2 if CPU count is unknown
        base_params["batch_size"] = 16  # Smaller batches for CPU
        base_params["learning_rate"] = 5e-5  # Lower learning rate
        base_params["pin_memory"] = False
        base_params["num_workers"] = max(1, cpu_count // 2)  # Use half available cores
        base_params["use_amp"] = False  # No mixed precision on CPU
        base_params["gradient_clip"] = 0.5  # More conservative gradient clipping
        base_params["prefetch_factor"] = 2

    return base_params


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """
    Move a batch of data to the specified device.

    Args:
        batch: Dictionary of tensors or PyTorch Geometric data
        device: Target device

    Returns:
        Batch data on the target device
    """
    if hasattr(batch, "to"):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    elif isinstance(batch, (list, tuple)):
        return [move_batch_to_device(x, device) for x in batch]
    else:
        return batch
