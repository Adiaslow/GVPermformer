"""
Dataset module for cyclic peptide data with enhanced features.
"""

import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset
import logging
from typing import List, Dict, Optional, Union, Tuple

from src.data.dataset import MoleculeDataset


class CycPeptDataset(MoleculeDataset):
    """
    Dataset for cyclic peptide data with enhanced molecular features.

    Extends the base MoleculeDataset with specific handling for cyclic peptides
    and enhanced atom features.
    """

    def __init__(
        self,
        csv_file: str,
        smiles_col: str = "smiles",
        target_col: str = "permeability",
        max_atoms: int = 100,
        filter_pampa: bool = True,
        pampa_threshold: float = -9.0,
        use_edge_features: bool = True,
        use_enhanced_features: bool = True,
        property_prediction: bool = True,
    ) -> None:
        """
        Initialize the cyclic peptide dataset.

        Args:
            csv_file: Path to the CSV file containing SMILES and target values
            smiles_col: Name of the column containing SMILES strings
            target_col: Name of the column containing target values
            max_atoms: Maximum number of atoms allowed in a molecule
            filter_pampa: Whether to filter out entries with PAMPA values below threshold
            pampa_threshold: Threshold for filtering PAMPA values
            use_edge_features: Whether to use edge features in the graph
            use_enhanced_features: Whether to use enhanced node/edge features
            property_prediction: Whether this is a property prediction task
        """
        super().__init__(
            csv_file=csv_file,
            smiles_col=smiles_col,
            target_col=target_col,
            max_atoms=max_atoms,
            filter_pampa=filter_pampa,
            pampa_threshold=pampa_threshold,
            use_edge_features=use_edge_features,
            use_enhanced_features=use_enhanced_features,
            property_prediction=property_prediction,
        )

    def _process_smiles(self, smiles: str) -> Dict[str, torch.Tensor]:
        """
        Process SMILES strings to create graph data for the model.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Dictionary with graph features
        """
        from src.utils.smiles_to_features import SmilesConverter, smiles_to_graph_data

        # If using enhanced features, create a custom converter with our settings
        if self.use_enhanced_features:
            converter = SmilesConverter(max_atoms=self.max_atoms)
            return converter.convert(smiles)
        else:
            # Otherwise use the default implementation
            return smiles_to_graph_data(smiles)
