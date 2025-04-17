#!/usr/bin/env python3
"""
Checkpoint analyzer for model inspection.
Extracts information from model checkpoints without loading the actual model.
"""

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def analyze_checkpoint(checkpoint_path):
    """
    Analyze a model checkpoint without loading the model.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        dict: Analysis results
    """
    print(f"Analyzing checkpoint: {checkpoint_path}")

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

    # Analysis results
    results = {}

    # Check checkpoint type
    if isinstance(checkpoint, dict):
        print("Checkpoint is a dictionary")
        results["type"] = "dictionary"
        results["keys"] = list(checkpoint.keys())
        print(f"Keys: {results['keys']}")

        # Check for model state dict
        if "model_state_dict" in checkpoint:
            analyze_state_dict(checkpoint["model_state_dict"], results)

        # Check for optimizer state
        if "optimizer_state_dict" in checkpoint:
            print("Optimizer state found")
            results["has_optimizer"] = True

        # Check for other metadata
        for key in checkpoint:
            if key not in ["model_state_dict", "optimizer_state_dict"]:
                print(f"Additional key: {key}, Value: {checkpoint[key]}")
                results[key] = checkpoint[key]
    else:
        print(f"Checkpoint is not a dictionary, type: {type(checkpoint)}")
        results["type"] = str(type(checkpoint))

    return results


def analyze_state_dict(state_dict, results):
    """
    Analyze the model's state dictionary.

    Args:
        state_dict: Model state dictionary
        results: Results dictionary to update
    """
    print("Analyzing model state dictionary...")

    # Parameter statistics
    param_count = 0
    layer_groups = defaultdict(list)

    # Collect parameter shapes and group them
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            shape = tuple(param.shape)
            param_count += param.numel()

            # Group parameters by their prefix
            prefix = name.split(".")[0]
            layer_groups[prefix].append((name, shape, param.numel()))

    # Add to results
    results["total_parameters"] = param_count
    print(f"Total parameters: {param_count:,}")

    # Print parameter distribution
    print("\nParameter distribution by layer group:")
    for group_name, params in layer_groups.items():
        group_params = sum(p[2] for p in params)
        percentage = (group_params / param_count) * 100
        print(f"  {group_name}: {group_params:,} parameters ({percentage:.2f}%)")

        # Print some example parameters from each group
        print("  Example parameters:")
        for i, (name, shape, numel) in enumerate(params[:3]):
            print(f"    {name}: shape={shape}, params={numel:,}")
        if len(params) > 3:
            print(f"    ... and {len(params) - 3} more")

    # Add to results
    results["layer_groups"] = {
        group: sum(p[2] for p in params) for group, params in layer_groups.items()
    }

    # Look for embedding dimensions
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor) and "embedding" in name.lower():
            print(f"\nEmbedding found: {name} with shape {param.shape}")
            if len(param.shape) == 2:
                results["embedding_dim"] = param.shape[1]

    # Find latent dimension (if applicable)
    latent_indicators = ["mean", "mu", "z_mean", "fc_mean"]
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor) and any(
            indicator in name.lower() for indicator in latent_indicators
        ):
            if len(param.shape) == 2:
                results["latent_dim"] = param.shape[0]
                print(f"\nLatent dimension found: {param.shape[0]} (from {name})")
            elif len(param.shape) >= 1:
                results["latent_dim"] = param.shape[-1]
                print(f"\nLatent dimension found: {param.shape[-1]} (from {name})")


def visualize_results(results, output_dir):
    """
    Visualize the analysis results.

    Args:
        results: Analysis results
        output_dir: Directory to save visualizations
    """
    if not results:
        print("No results to visualize")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Create parameter distribution chart
    if "layer_groups" in results:
        plt.figure(figsize=(10, 6))
        groups = list(results["layer_groups"].keys())
        counts = list(results["layer_groups"].values())

        # Sort by parameter count
        sorted_idx = np.argsort(counts)
        groups = [groups[i] for i in sorted_idx]
        counts = [counts[i] for i in sorted_idx]

        plt.barh(groups, counts)
        plt.xscale("log")
        plt.xlabel("Parameter Count")
        plt.ylabel("Layer Group")
        plt.title("Parameter Distribution by Layer Group")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "parameter_distribution.png"))
        plt.close()

    # Save results to JSON
    import json

    with open(os.path.join(output_dir, "checkpoint_analysis.json"), "w") as f:
        json.dump(
            {
                k: v
                for k, v in results.items()
                if isinstance(v, (dict, list, str, int, float, bool))
            },
            f,
            indent=2,
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze a model checkpoint")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoint_analysis", help="Output directory"
    )
    args = parser.parse_args()

    results = analyze_checkpoint(args.checkpoint)
    if results:
        visualize_results(results, args.output_dir)

    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
