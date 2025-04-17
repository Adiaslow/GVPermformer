# /Users/Adam/GVPermformer/debug_script.py
"""
Debugging script to check the edge_index format in PyTorch Geometric Data objects.
"""

import os
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try importing from src
try:
    from src.data.cycpept_dataset import CycPeptDataset
    from torch_geometric.data import Data, Batch

    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Create a minimal dataset
    dataset = CycPeptDataset(
        csv_file="training_data/CycPeptMPDB_Peptide_All.csv",
        property_prediction=True,
        pampa_threshold=-9.0,
        max_atoms=500,
        use_enhanced_atom_features=True,
    )

    print(f"Dataset loaded with {len(dataset)} items")

    # Get a single item
    item = dataset[0]
    print("\nSingle item keys:", list(item.keys()))

    # Check edge_index format
    if "edge_index" in item:
        edge_index = item["edge_index"]
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Edge index dtype: {edge_index.dtype}")
        print(f"Edge index first few values:\n{edge_index[:, :5]}")

        # Ensure edge_index is in the correct format
        if edge_index.dtype != torch.long:
            edge_index = edge_index.to(torch.long)
            print(f"Converted edge_index to dtype: {edge_index.dtype}")

        if edge_index.dim() == 2 and edge_index.size(0) != 2:
            edge_index = edge_index.t()
            print(f"Transposed edge_index to shape: {edge_index.shape}")

    # Create a PyG Data object
    data_obj = Data(
        x=item["node_features"],
        edge_index=edge_index,
        edge_attr=item["edge_attr"] if "edge_attr" in item else None,
    )

    print("\nPyG Data object attributes:")
    for attr_name in ["x", "edge_index", "edge_attr"]:
        if hasattr(data_obj, attr_name):
            attr = getattr(data_obj, attr_name)
            if attr is not None:
                print(f"{attr_name} shape: {attr.shape}, dtype: {attr.dtype}")

    # Create a batch with multiple items
    batch_size = 4
    batch_items = [dataset[i] for i in range(batch_size)]

    # Convert to PyG Data objects
    data_objects = []
    for item in batch_items:
        edge_index = item["edge_index"]
        if edge_index.dtype != torch.long:
            edge_index = edge_index.to(torch.long)
        if edge_index.dim() == 2 and edge_index.size(0) != 2:
            edge_index = edge_index.t()

        data_obj = Data(
            x=item["node_features"],
            edge_index=edge_index,
            edge_attr=item["edge_attr"] if "edge_attr" in item else None,
        )
        data_objects.append(data_obj)

    # Create a batched Data object
    batch = Batch.from_data_list(data_objects)

    print("\nBatched Data object attributes:")
    for attr_name in ["x", "edge_index", "edge_attr", "batch"]:
        if hasattr(batch, attr_name):
            attr = getattr(batch, attr_name)
            if attr is not None:
                print(f"{attr_name} shape: {attr.shape}, dtype: {attr.dtype}")

    # Move to device
    batch = batch.to(device)
    print(f"\nBatch moved to device: {device}")
    print(f"edge_index device: {batch.edge_index.device}")

    # Print edge_index values to confirm it's valid
    print(
        f"\nEdge index first few values after batch creation:\n{batch.edge_index[:, :5]}"
    )

except ImportError as e:
    print(f"Error importing required modules: {e}")
except Exception as e:
    print(f"Error during execution: {e}")
