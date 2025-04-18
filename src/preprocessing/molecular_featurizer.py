"""
Module for molecular featurization using DeepChem and custom features.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


class MolecularFeaturizer:
    """Class for generating molecular features using DeepChem and custom methods."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the featurizer with configuration.

        Args:
            config: Dictionary containing featurization configuration
        """
        self.config = config
        self.graph_featurizer = dc.feat.MolGraphConvFeaturizer(
            use_edges=config["featurization"]["graph_conv"]["use_edges"],
            use_chirality=config["featurization"]["graph_conv"]["use_chirality"],
            use_partial_charge=config["featurization"]["graph_conv"][
                "use_partial_charge"
            ],
        )
        self.descriptor_featurizer = dc.feat.RDKitDescriptors(
            use_fragment=config["featurization"]["rdkit_descriptors"]["use_fragments"],
            ipc_avg=True,
        )

    def _compute_amide_bonds(self, mol: Chem.Mol) -> int:
        """
        Count the number of amide bonds in the molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            Number of amide bonds
        """
        pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
        return len(mol.GetSubstructMatches(pattern))

    def _compute_ring_size(self, mol: Chem.Mol) -> int:
        """
        Calculate the size of the largest ring in the molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            Size of the largest ring
        """
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() == 0:
            return 0
        return max(len(ring) for ring in ring_info.AtomRings())

    def _compute_amino_acid_composition(self, mol: Chem.Mol) -> Dict[str, int]:
        """
        Calculate the amino acid composition of the peptide.

        Args:
            mol: RDKit molecule object

        Returns:
            Dictionary with amino acid counts
        """
        # This is a simplified version - expand based on your needs
        aa_patterns = {
            "Ala": "[NH2][CH](C)C(=O)",
            "Gly": "[NH2][CH2]C(=O)",
            # Add more amino acid patterns as needed
        }

        composition = {}
        for aa_name, pattern in aa_patterns.items():
            pattern_mol = Chem.MolFromSmarts(pattern)
            composition[aa_name] = len(mol.GetSubstructMatches(pattern_mol))
        return composition

    def _compute_charge_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate charge-related features.

        Args:
            mol: RDKit molecule object

        Returns:
            Dictionary with charge-related features
        """
        return {
            "total_charge": Chem.GetFormalCharge(mol),
            "num_hbd": rdMolDescriptors.CalcNumHBD(mol),
            "num_hba": rdMolDescriptors.CalcNumHBA(mol),
        }

    def _compute_custom_features(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        Compute all custom molecular features.

        Args:
            mol: RDKit molecule object

        Returns:
            Dictionary containing all custom features
        """
        custom_features = {}

        if self.config["custom_features"]["compute_amide_bonds"]:
            custom_features["n_amide_bonds"] = self._compute_amide_bonds(mol)

        if self.config["custom_features"]["compute_ring_size"]:
            custom_features["largest_ring_size"] = self._compute_ring_size(mol)

        if self.config["custom_features"]["compute_amino_acid_composition"]:
            custom_features.update(self._compute_amino_acid_composition(mol))

        if self.config["custom_features"]["compute_charge_features"]:
            custom_features.update(self._compute_charge_features(mol))

        return custom_features

    def featurize(
        self, smiles_list: List[str]
    ) -> Tuple[List, List[Dict[str, Any]], np.ndarray]:
        """
        Generate all features for a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings to featurize

        Returns:
            Tuple containing (graph features, custom features, descriptor features)
        """
        # Convert SMILES to RDKit molecules
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        valid_mols = [mol for mol in mols if mol is not None]

        if len(valid_mols) != len(smiles_list):
            raise ValueError("Some SMILES strings could not be converted to molecules")

        # Generate graph features
        graph_features = self.graph_featurizer.featurize(smiles_list)

        # Generate custom features
        custom_features = [self._compute_custom_features(mol) for mol in valid_mols]

        # Generate RDKit descriptor features
        descriptor_features = self.descriptor_featurizer.featurize(smiles_list)

        return graph_features, custom_features, descriptor_features
