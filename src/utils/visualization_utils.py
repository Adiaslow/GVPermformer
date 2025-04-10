"""
Visualization utility functions for molecular graphs and model results.

This module provides functions for visualizing molecules, latent spaces,
and training metrics for the Graph VAE model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


def visualize_molecule(
    mol: Chem.Mol,
    highlight_atoms: Optional[List[int]] = None,
    highlight_bonds: Optional[List[int]] = None,
    atom_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
    bond_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
    size: Tuple[int, int] = (300, 300),
    save_path: Optional[str] = None,
    show: bool = True,
) -> np.ndarray:
    """
    Visualize a molecule with optional highlighting.

    Args:
        mol: RDKit molecule object
        highlight_atoms: List of atom indices to highlight
        highlight_bonds: List of bond indices to highlight
        atom_colors: Dictionary mapping atom indices to RGB color tuples
        bond_colors: Dictionary mapping bond indices to RGB color tuples
        size: Image size (width, height) in pixels
        save_path: Path to save the image (if None, image is not saved)
        show: Whether to display the image

    Returns:
        NumPy array containing the image data
    """
    if mol is None:
        raise ValueError("Molecule cannot be None")

    # Default colors if not provided
    if highlight_atoms and atom_colors is None:
        atom_colors = {i: (0.7, 0.2, 0.2) for i in highlight_atoms}
    if highlight_bonds and bond_colors is None:
        bond_colors = {i: (0.7, 0.2, 0.2) for i in highlight_bonds}

    # Create drawer object
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().prepareMolsBeforeDrawing = True

    # Draw molecule
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()

    # Get image data
    png_data = drawer.GetDrawingText()

    # Convert to numpy array
    import io
    from PIL import Image

    img = Image.open(io.BytesIO(png_data))
    img_array = np.array(img)

    # Save image if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        img.save(save_path)

    # Display image if requested
    if show:
        plt.figure(figsize=(size[0] / 100, size[1] / 100))
        plt.imshow(img_array)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return img_array


def visualize_molecules_grid(
    mols: List[Chem.Mol],
    labels: Optional[List[str]] = None,
    mols_per_row: int = 4,
    mol_size: Tuple[int, int] = (200, 200),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True,
) -> np.ndarray:
    """
    Visualize a grid of molecules.

    Args:
        mols: List of RDKit molecule objects
        labels: List of labels for each molecule
        mols_per_row: Number of molecules per row
        mol_size: Size of each molecule image
        save_path: Path to save the combined image
        title: Title for the figure
        show: Whether to display the image

    Returns:
        NumPy array containing the combined image data
    """
    if not mols:
        raise ValueError("No molecules provided")

    # Add empty labels if needed
    if labels is None:
        labels = [""] * len(mols)

    # Generate molecule images
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=mol_size,
        legends=labels,
        useSVG=False,
    )

    # Convert to numpy array
    img_array = np.array(img)

    # Save image if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        img.save(save_path)

    # Display image if requested
    if show:
        plt.figure(figsize=(12, max(2, len(mols) // mols_per_row * 2)))
        plt.imshow(img_array)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return img_array


def visualize_latent_space(
    z: np.ndarray,
    colors: Optional[np.ndarray] = None,
    method: str = "pca",
    n_components: int = 2,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
    cmap: str = "viridis",
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize the latent space using dimensionality reduction.

    Args:
        z: Latent vectors as numpy array of shape (n_samples, latent_dim)
        colors: Values to use for coloring points
        method: Dimensionality reduction method ('pca' or 'tsne')
        n_components: Number of components for visualization (2 or 3)
        figsize: Figure size
        alpha: Alpha value for scatter points
        cmap: Colormap for coloring points
        title: Plot title
        colorbar_label: Label for the colorbar if colors are provided
        save_path: Path to save the plot
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")

    # Apply dimensionality reduction
    if method.lower() == "pca":
        reducer = PCA(n_components=n_components)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Reduce dimensionality
    z_reduced = reducer.fit_transform(z)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # 2D or 3D visualization
    if n_components == 2:
        if colors is not None:
            scatter = plt.scatter(
                z_reduced[:, 0], z_reduced[:, 1], c=colors, cmap=cmap, alpha=alpha
            )
            if colorbar_label:
                plt.colorbar(scatter, label=colorbar_label)
        else:
            plt.scatter(z_reduced[:, 0], z_reduced[:, 1], alpha=alpha)

        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")

    else:  # 3D plot
        ax = fig.add_subplot(111, projection="3d")

        if colors is not None:
            scatter = ax.scatter(
                z_reduced[:, 0],
                z_reduced[:, 1],
                z_reduced[:, 2],
                c=colors,
                cmap=cmap,
                alpha=alpha,
            )
            if colorbar_label:
                plt.colorbar(scatter, label=colorbar_label)
        else:
            ax.scatter(z_reduced[:, 0], z_reduced[:, 1], z_reduced[:, 2], alpha=alpha)

        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_zlabel(f"{method.upper()} Component 3")

    # Add title if provided
    if title:
        plt.title(title)

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show figure if requested
    if show:
        plt.show()

    return fig


def plot_training_history(
    metrics: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot training metrics history.

    Args:
        metrics: Dictionary mapping metric names to lists of values
        figsize: Figure size
        save_path: Path to save the plot
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    if not metrics:
        raise ValueError("No metrics provided")

    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(values)
        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show figure if requested
    if show:
        plt.show()

    return fig


def plot_property_correlations(
    true_values: Dict[str, np.ndarray],
    predicted_values: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot correlations between true and predicted property values.

    Args:
        true_values: Dictionary mapping property names to arrays of true values
        predicted_values: Dictionary mapping property names to arrays of predicted values
        figsize: Figure size
        save_path: Path to save the plot
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    if not true_values or not predicted_values:
        raise ValueError("No values provided")

    # Check that properties match
    assert set(true_values.keys()) == set(
        predicted_values.keys()
    ), "Property names in true_values and predicted_values must match"

    properties = list(true_values.keys())
    n_props = len(properties)

    # Create figure
    fig, axes = plt.subplots(1, n_props, figsize=figsize)
    if n_props == 1:
        axes = [axes]

    # Plot each property
    for i, prop in enumerate(properties):
        ax = axes[i]
        true = true_values[prop]
        pred = predicted_values[prop]

        # Calculate correlation
        corr = np.corrcoef(true, pred)[0, 1]

        # Scatter plot
        ax.scatter(true, pred, alpha=0.6)

        # Determine axis limits
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())

        # Add diagonal line
        ax.plot([min_val, max_val], [min_val, max_val], "r--")

        # Labels and title
        ax.set_xlabel(f"True {prop}")
        ax.set_ylabel(f"Predicted {prop}")
        ax.set_title(f"{prop} (r = {corr:.3f})")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show figure if requested
    if show:
        plt.show()

    return fig
