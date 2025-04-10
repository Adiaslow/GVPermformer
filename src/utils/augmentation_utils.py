"""
Data augmentation utilities for molecular graphs.

This module provides functions for augmenting molecular data through
various graph transformations, including node/edge masking, feature perturbation,
and subgraph sampling.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from rdkit import Chem
import torch


def mask_random_nodes(
    node_features: torch.Tensor, mask_ratio: float = 0.15, mask_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask node features.

    Args:
        node_features: Node feature tensor of shape (num_nodes, feat_dim)
        mask_ratio: Ratio of nodes to mask
        mask_value: Value to use for masking

    Returns:
        Tuple of (masked_features, mask) where mask is a boolean tensor
        indicating which nodes were masked
    """
    num_nodes = node_features.size(0)
    num_to_mask = int(num_nodes * mask_ratio)

    # Create mask indices
    mask_indices = torch.randperm(num_nodes)[:num_to_mask]
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=node_features.device)
    mask[mask_indices] = True

    # Create masked features
    masked_features = node_features.clone()
    masked_features[mask] = mask_value

    return masked_features, mask


def drop_random_edges(
    edge_index: torch.Tensor,
    edge_features: Optional[torch.Tensor] = None,
    drop_ratio: float = 0.15,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Randomly drop edges from a graph.

    Args:
        edge_index: Edge index tensor of shape (2, num_edges)
        edge_features: Edge feature tensor of shape (num_edges, feat_dim),
            or None if no edge features are available
        drop_ratio: Ratio of edges to drop

    Returns:
        Tuple of (new_edge_index, new_edge_features, mask) where mask is a boolean
        tensor indicating which edges were kept
    """
    num_edges = edge_index.size(1)
    num_to_keep = int(num_edges * (1 - drop_ratio))

    # Create mask indices
    perm = torch.randperm(num_edges, device=edge_index.device)
    kept_indices = perm[:num_to_keep]
    mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)
    mask[kept_indices] = True

    # Create new edge index and features
    new_edge_index = edge_index[:, kept_indices]

    if edge_features is not None:
        new_edge_features = edge_features[kept_indices]
    else:
        new_edge_features = None

    return new_edge_index, new_edge_features, mask


def perturb_features(features: torch.Tensor, noise_scale: float = 0.1) -> torch.Tensor:
    """
    Add Gaussian noise to features.

    Args:
        features: Feature tensor
        noise_scale: Scale of the Gaussian noise

    Returns:
        Perturbed features
    """
    noise = torch.randn_like(features) * noise_scale
    perturbed_features = features + noise
    return perturbed_features


def random_subgraph(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_features: Optional[torch.Tensor] = None,
    node_features: Optional[torch.Tensor] = None,
    ratio: float = 0.8,
) -> Dict[str, torch.Tensor]:
    """
    Extract a random connected subgraph from a graph.

    Args:
        edge_index: Edge index tensor of shape (2, num_edges)
        num_nodes: Number of nodes in the graph
        edge_features: Edge feature tensor (optional)
        node_features: Node feature tensor (optional)
        ratio: Ratio of nodes to keep in the subgraph

    Returns:
        Dictionary containing the subgraph data
    """
    # Choose a random starting node
    start_node = random.randint(0, num_nodes - 1)

    # Initialize sets to track visited nodes and frontier
    visited = set([start_node])
    frontier = set([start_node])
    target_size = int(num_nodes * ratio)

    # Perform BFS until we reach the target size
    while len(visited) < target_size and frontier:
        # Take a node from the frontier
        current = frontier.pop()

        # Find neighbors (both incoming and outgoing edges)
        neighbors = []
        for i in range(edge_index.size(1)):
            if (
                edge_index[0, i].item() == current
                and edge_index[1, i].item() not in visited
            ):
                neighbors.append(edge_index[1, i].item())
            elif (
                edge_index[1, i].item() == current
                and edge_index[0, i].item() not in visited
            ):
                neighbors.append(edge_index[0, i].item())

        # Add neighbors to visited and frontier
        for neighbor in neighbors:
            if len(visited) >= target_size:
                break
            visited.add(neighbor)
            frontier.add(neighbor)

    # Create node mapping for the subgraph
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(visited)}

    # Filter the edge index to only include edges between visited nodes
    mask = []
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src in visited and dst in visited:
            mask.append(i)

    mask = torch.tensor(mask, device=edge_index.device)
    subgraph_edge_index = edge_index[:, mask].clone()

    # Remap node indices
    for i in range(subgraph_edge_index.size(1)):
        subgraph_edge_index[0, i] = node_mapping[subgraph_edge_index[0, i].item()]
        subgraph_edge_index[1, i] = node_mapping[subgraph_edge_index[1, i].item()]

    # Prepare the output
    result = {"edge_index": subgraph_edge_index}

    # Add edge features if available
    if edge_features is not None:
        result["edge_features"] = edge_features[mask]

    # Add node features if available
    if node_features is not None:
        indices = torch.tensor(list(visited), device=node_features.device)
        indices = torch.sort(indices)[0]  # Sort the indices
        result["node_features"] = node_features[indices]

        # Create a mapping for the original indices
        result["original_indices"] = indices

    return result


def augment_smiles(
    smiles: str,
    num_augmentations: int = 1,
    augmentation_types: List[str] = ["atom_order"],
) -> List[str]:
    """
    Augment a SMILES string by applying various transformations.

    Args:
        smiles: Input SMILES string
        num_augmentations: Number of augmented SMILES to generate
        augmentation_types: List of augmentation types to apply
            Options: "atom_order", "stereochemistry", "tautomer"

    Returns:
        List of augmented SMILES strings
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles] * num_augmentations

    augmented_smiles = []

    for _ in range(num_augmentations):
        aug_mol = Chem.Mol(mol)

        # Apply augmentations
        if "atom_order" in augmentation_types:
            # Randomize atom order by doing a random renumbering
            atoms = list(range(mol.GetNumAtoms()))
            random.shuffle(atoms)
            aug_mol = Chem.RenumberAtoms(aug_mol, atoms)

        if "stereochemistry" in augmentation_types:
            # Randomly flip some stereochemistry
            for atom in aug_mol.GetAtoms():
                if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                    if random.random() < 0.5:
                        atom.SetChiralTag(atom.GetChiralTag())

        # Convert back to SMILES
        augmented = Chem.MolToSmiles(aug_mol)
        augmented_smiles.append(augmented)

    return augmented_smiles


def get_augmentation_fn(augmentation_type: str, **kwargs: Any):
    """
    Get an augmentation function based on the specified type.

    Args:
        augmentation_type: Type of augmentation
            Options: "node_mask", "edge_drop", "feature_noise", "subgraph"
        **kwargs: Additional arguments to pass to the augmentation function

    Returns:
        Augmentation function
    """
    if augmentation_type == "node_mask":
        return lambda x: mask_random_nodes(x, **kwargs)
    elif augmentation_type == "edge_drop":
        return lambda x, y: drop_random_edges(x, y, **kwargs)
    elif augmentation_type == "feature_noise":
        return lambda x: perturb_features(x, **kwargs)
    elif augmentation_type == "subgraph":
        return lambda x, n, y=None, z=None: random_subgraph(x, n, y, z, **kwargs)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


def apply_multiple_augmentations(
    graph_data: Dict[str, torch.Tensor], augmentation_types: List[str], **kwargs: Any
) -> Dict[str, torch.Tensor]:
    """
    Apply multiple augmentations to graph data.

    Args:
        graph_data: Dictionary containing graph data
        augmentation_types: List of augmentation types to apply
        **kwargs: Additional arguments for augmentation functions

    Returns:
        Augmented graph data
    """
    result = graph_data.copy()

    for aug_type in augmentation_types:
        if aug_type == "node_mask" and "node_features" in result:
            result["node_features"], result["node_mask"] = mask_random_nodes(
                result["node_features"],
                **{
                    k.replace("node_mask_", ""): v
                    for k, v in kwargs.items()
                    if k.startswith("node_mask_")
                },
            )

        elif aug_type == "edge_drop" and "edge_index" in result:
            edge_features = result.get("edge_features", None)
            result["edge_index"], result["edge_features"], result["edge_mask"] = (
                drop_random_edges(
                    result["edge_index"],
                    edge_features,
                    **{
                        k.replace("edge_drop_", ""): v
                        for k, v in kwargs.items()
                        if k.startswith("edge_drop_")
                    },
                )
            )

        elif aug_type == "feature_noise" and "node_features" in result:
            result["node_features"] = perturb_features(
                result["node_features"],
                **{
                    k.replace("feature_noise_", ""): v
                    for k, v in kwargs.items()
                    if k.startswith("feature_noise_")
                },
            )

        elif (
            aug_type == "subgraph" and "edge_index" in result and "num_nodes" in result
        ):
            subgraph = random_subgraph(
                result["edge_index"],
                result["num_nodes"],
                result.get("edge_features", None),
                result.get("node_features", None),
                **{
                    k.replace("subgraph_", ""): v
                    for k, v in kwargs.items()
                    if k.startswith("subgraph_")
                },
            )
            result.update(subgraph)

    return result
