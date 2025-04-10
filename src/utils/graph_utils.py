"""
Utility functions for working with molecular graphs.
Includes conversions between different representations and helper functions.
"""

import torch
import numpy as np
import math
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any


def mol_to_nx(mol) -> nx.Graph:
    """
    Convert RDKit molecule to NetworkX graph.

    Args:
        mol: RDKit molecule object

    Returns:
        NetworkX graph representation
    """
    # Create graph
    G = nx.Graph()

    # Add atoms (nodes)
    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
            chiral_tag=atom.GetChiralTag(),
            hybridization=atom.GetHybridization(),
            num_explicit_hs=atom.GetNumExplicitHs(),
            is_aromatic=atom.GetIsAromatic(),
            in_ring=atom.IsInRing(),
        )

    # Add bonds (edges)
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=bond.GetBondType(),
            is_conjugated=bond.GetIsConjugated(),
            is_in_ring=bond.IsInRing(),
            stereo=bond.GetStereo(),
        )

    return G


def nx_to_pyg_data(G, y=None) -> Dict:
    """
    Convert NetworkX graph to PyTorch Geometric data format.

    Args:
        G: NetworkX graph
        y: Optional target value

    Returns:
        Dictionary in PyTorch Geometric format
    """
    # Get node features
    node_features = []
    for _, node_data in G.nodes(data=True):
        # Extract node features in a consistent order
        features = [
            node_data.get("atomic_num", 0),
            node_data.get("formal_charge", 0),
            int(node_data.get("is_aromatic", False)),
            int(node_data.get("in_ring", False)),
            # Add more features as needed
        ]
        node_features.append(features)

    # Convert to tensor
    x = torch.tensor(node_features, dtype=torch.float)

    # Get edges
    edge_index = []
    edge_features = []

    for u, v, edge_data in G.edges(data=True):
        # Add edges in both directions for undirected graph
        edge_index.extend([[u, v], [v, u]])

        # Extract edge features
        features = [
            int(edge_data.get("bond_type", 0)),
            int(edge_data.get("is_conjugated", False)),
            int(edge_data.get("is_in_ring", False)),
            # Add more features as needed
        ]

        # Add edge features for both directions
        edge_features.extend([features, features])

    # Convert to tensors
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros(
            (0, len(edge_features[0]) if edge_features else 0), dtype=torch.float
        )

    # Create data dictionary
    data = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": len(G.nodes),
    }

    # Add target if provided
    if y is not None:
        data["y"] = torch.tensor([y], dtype=torch.float)

    return data


def smiles_to_pyg_data(smiles: str, y: Optional[float] = None) -> Dict:
    """
    Convert SMILES string to PyTorch Geometric data format.

    Args:
        smiles: SMILES string
        y: Optional target value

    Returns:
        Dictionary in PyTorch Geometric format
    """
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Convert to NetworkX
    G = mol_to_nx(mol)

    # Convert to PyG data
    return nx_to_pyg_data(G, y)


def mol_to_pyg_data(mol, y: Optional[float] = None) -> Dict:
    """
    Convert RDKit molecule to PyTorch Geometric data format.

    Args:
        mol: RDKit molecule object
        y: Optional target value

    Returns:
        Dictionary in PyTorch Geometric format
    """
    # Convert to NetworkX
    G = mol_to_nx(mol)

    # Convert to PyG data
    return nx_to_pyg_data(G, y)


def get_atom_features(atom) -> List[Union[int, float]]:
    """
    Get extended atom features.

    Args:
        atom: RDKit atom object

    Returns:
        List of atom features
    """
    return [
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetDegree(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetHybridization(),
        atom.GetChiralTag(),
        atom.GetImplicitValence(),
        atom.GetExplicitValence(),
        # Add more features as needed
    ]


def get_bond_features(bond) -> List[Union[int, float]]:
    """
    Get extended bond features.

    Args:
        bond: RDKit bond object

    Returns:
        List of bond features
    """
    return [
        int(bond.GetBondType()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        int(bond.GetStereo()),
        # Add more features as needed
    ]


def mol_to_complete_graph(mol) -> nx.Graph:
    """
    Convert molecule to a complete graph where each atom is connected to every other atom.
    Useful for transformer-based processing.

    Args:
        mol: RDKit molecule object

    Returns:
        NetworkX complete graph
    """
    # Create graph
    G = nx.Graph()

    # Add atoms (nodes)
    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
            chiral_tag=atom.GetChiralTag(),
            hybridization=atom.GetHybridization(),
            num_explicit_hs=atom.GetNumExplicitHs(),
            is_aromatic=atom.GetIsAromatic(),
            in_ring=atom.IsInRing(),
        )

    # Add edges for all pairs of atoms
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            G.add_edge(
                i, j, bond_type=0, is_conjugated=False, is_in_ring=False, stereo=0
            )

    return G


def calculate_3d_distance_matrix(mol) -> np.ndarray:
    """
    Calculate 3D distance matrix between atoms.

    Args:
        mol: RDKit molecule with 3D conformer

    Returns:
        Matrix of pairwise distances between atoms
    """
    # Ensure molecule has 3D coordinates
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        # Note: These functions may not be available in all RDKit installations
        # Depending on the environment, you might need to import specific modules
        try:
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
        except AttributeError:
            # Fallback to using basic distance calculation if 3D embedding fails
            print("Warning: 3D embedding failed, using 2D coordinates")
        mol = Chem.RemoveHs(mol)

    # Get conformer
    conf = mol.GetConformer()

    # Get number of atoms
    num_atoms = mol.GetNumAtoms()

    # Initialize distance matrix
    distance_matrix = np.zeros((num_atoms, num_atoms))

    # Calculate pairwise distances
    for i in range(num_atoms):
        pos_i = conf.GetAtomPosition(i)
        for j in range(i + 1, num_atoms):
            pos_j = conf.GetAtomPosition(j)
            dist = math.sqrt(
                (pos_i.x - pos_j.x) ** 2
                + (pos_i.y - pos_j.y) ** 2
                + (pos_i.z - pos_j.z) ** 2
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def visualize_molecular_graph(G, save_path: Optional[str] = None):
    """
    Visualize a molecular graph.

    Args:
        G: NetworkX graph
        save_path: Optional path to save the visualization

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    # Get node positions using spring layout
    pos = nx.spring_layout(G)

    # Get node colors based on atomic number
    atomic_nums = nx.get_node_attributes(G, "atomic_num")
    atom_colors = []
    for node in G.nodes():
        atomic_num = atomic_nums.get(node, 0)
        if atomic_num == 6:  # Carbon
            atom_colors.append("black")
        elif atomic_num == 7:  # Nitrogen
            atom_colors.append("blue")
        elif atomic_num == 8:  # Oxygen
            atom_colors.append("red")
        elif atomic_num == 16:  # Sulfur
            atom_colors.append("yellow")
        else:
            atom_colors.append("green")

    # Get edge colors based on bond type
    bond_types = nx.get_edge_attributes(G, "bond_type")
    edge_colors = []
    for edge in G.edges():
        bond_type = bond_types.get(edge, 0)
        if bond_type == 1:  # Single bond
            edge_colors.append("black")
        elif bond_type == 2:  # Double bond
            edge_colors.append("blue")
        elif bond_type == 3:  # Triple bond
            edge_colors.append("red")
        elif bond_type == 4:  # Aromatic bond
            edge_colors.append("purple")
        else:
            edge_colors.append("gray")

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=atom_colors, node_size=300)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)

    # Draw labels
    labels = {}
    for node in G.nodes():
        atomic_num = atomic_nums.get(node, 0)
        if atomic_num == 6:  # Carbon (don't label)
            labels[node] = ""
        else:
            element = Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
            labels[node] = element

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color="white")

    plt.title("Molecular Graph Visualization")
    plt.axis("off")

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.gcf()
