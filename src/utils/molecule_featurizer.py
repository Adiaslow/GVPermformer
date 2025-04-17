"""
Comprehensive molecular featurization module using PyTorch Geometric.
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import Descriptors as rdChemDescriptors  # type: ignore
from typing import Dict, List, Optional, Tuple, Union, Any
from torch_geometric.data import Data  # type: ignore
import deepchem as dc  # type: ignore
from deepchem.feat import CircularFingerprint, RDKitDescriptors, MolGraphConvFeaturizer  # type: ignore


class MoleculeFeaturizer:
    """A class to featurize molecules using DeepChem's featurizers."""

    def __init__(self) -> None:
        """Initialize featurizers."""
        self.descriptor_featurizer = RDKitDescriptors()
        self.fingerprint_featurizer = CircularFingerprint()
        self.graph_featurizer = MolGraphConvFeaturizer()

    def compute_molecular_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """Compute molecular descriptors using DeepChem's RDKitDescriptors.

        Args:
            mol: RDKit molecule object

        Returns:
            Array of molecular descriptors
        """
        return self.descriptor_featurizer.featurize([mol])[0]

    def compute_fingerprints(self, mol: Chem.Mol) -> np.ndarray:
        """Compute molecular fingerprints using DeepChem's CircularFingerprint.

        Args:
            mol: RDKit molecule object

        Returns:
            Array of fingerprint features
        """
        return self.fingerprint_featurizer.featurize([mol])[0]

    def create_graph_data(self, mol: Chem.Mol) -> Data:
        """Create a PyTorch Geometric graph from a molecule using DeepChem's graph featurizer.

        Args:
            mol: RDKit molecule object

        Returns:
            PyTorch Geometric Data object containing the molecular graph
        """
        graph_features = self.graph_featurizer.featurize([mol])[0]

        # Convert DeepChem graph features to PyTorch Geometric format
        node_features = torch.FloatTensor(graph_features.node_features)
        edge_index = torch.LongTensor(graph_features.edge_index).t().contiguous()
        edge_features = torch.FloatTensor(graph_features.edge_features)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    def featurize_molecule(self, mol: Chem.Mol) -> Tuple[Data, Dict[str, float]]:
        """Featurize a molecule using all available featurizers.

        Args:
            mol: RDKit molecule object

        Returns:
            Tuple containing:
                - PyTorch Geometric Data object with graph features
                - Dictionary of molecular descriptors
        """
        graph_data = self.create_graph_data(mol)
        descriptors = self.compute_molecular_descriptors(mol)

        # Convert descriptors to a dictionary using DeepChem's feature names
        descriptor_dict = {
            name: float(value)
            for name, value in zip(self.descriptor_featurizer.descriptors, descriptors)
        }

        return graph_data, descriptor_dict

    def _get_atom_features(self, atom: Chem.Atom) -> List[float]:
        """Get comprehensive atom features."""
        features = []

        # One-hot encoding of atom type
        features += [
            int(atom.GetAtomicNum() == x) for x in self.atom_features["atomic_num"]
        ]

        # One-hot encoding of degree
        features += [int(atom.GetDegree() == x) for x in self.atom_features["degree"]]

        # One-hot encoding of formal charge
        features += [
            int(atom.GetFormalCharge() == x)
            for x in self.atom_features["formal_charge"]
        ]

        # One-hot encoding of hybridization
        features += [
            int(atom.GetHybridization() == x)
            for x in self.atom_features["hybridization"]
        ]

        # Binary aromatic feature
        features.append(int(atom.GetIsAromatic()))

        # One-hot encoding of number of hydrogens
        features += [
            int(atom.GetTotalNumHs() == x) for x in self.atom_features["num_hs"]
        ]

        # One-hot encoding of chirality
        features += [
            int(atom.GetChiralTag() == x) for x in self.atom_features["chirality"]
        ]

        # Additional features
        features.extend(
            [
                atom.GetMass(),
                atom.GetExplicitValence(),
                atom.GetImplicitValence(),
                atom.GetNumRadicalElectrons(),
                0.0,  # Placeholder for charge (removing Gasteiger charge calculation)
            ]
        )

        # Ring membership (size 3-8)
        ring_info = atom.GetOwningMol().GetRingInfo()
        features.extend(
            [
                int(ring_info.IsAtomInRingOfSize(atom.GetIdx(), size))
                for size in range(3, 9)
            ]
        )

        return features

    def _get_bond_features(self, bond: Chem.Bond) -> List[float]:
        """Get comprehensive bond features."""
        features = []

        # One-hot encoding of bond type
        features += [
            int(bond.GetBondType() == x) for x in self.bond_features["bond_type"]
        ]

        # Binary conjugated feature
        features.append(int(bond.GetIsConjugated()))

        # Binary ring feature
        features.append(int(bond.IsInRing()))

        # One-hot encoding of stereo
        features += [int(bond.GetStereo() == x) for x in self.bond_features["stereo"]]

        # Additional features
        features.extend(
            [
                bond.GetValenceContrib(bond.GetBeginAtom()),
                bond.GetValenceContrib(bond.GetEndAtom()),
                float(bond.GetBondTypeAsDouble()),
            ]
        )

        # Ring membership (size 3-8)
        ring_info = bond.GetOwningMol().GetRingInfo()
        features.extend(
            [
                int(ring_info.IsBondInRingOfSize(bond.GetIdx(), size))
                for size in range(3, 9)
            ]
        )

        return features

    def _get_global_features(self, mol: Chem.Mol) -> List[float]:
        """Get global molecular features."""
        features = []

        # Basic molecular properties
        features.extend(
            [
                Descriptors.ExactMolWt(mol),  # type: ignore
                Descriptors.MolLogP(mol),  # type: ignore
                Descriptors.TPSA(mol),  # type: ignore
                rdMolDescriptors.CalcNumHBA(mol),  # Number of H-bond acceptors
                rdMolDescriptors.CalcNumHBD(mol),  # Number of H-bond donors
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol),
                rdMolDescriptors.CalcNumSaturatedRings(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                len(Chem.GetSymmSSSR(mol)),  # Number of SSSR rings
            ]
        )

        return features

    def featurize(self, smiles: str) -> Optional[Data]:
        """
        Create comprehensive molecular features for PyTorch Geometric.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            PyTorch Geometric Data object with rich molecular features
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Get basic molecular structure
            num_atoms = mol.GetNumAtoms()

            # Get atom features
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append(self._get_atom_features(atom))
            x = torch.tensor(atom_features, dtype=torch.float)

            # Get bond features and edge indices
            edge_indices = []
            edge_features = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices += [[i, j], [j, i]]  # Add both directions

                bond_feats = self._get_bond_features(bond)
                edge_features += [bond_feats, bond_feats]  # Add for both directions

            if len(edge_indices) > 0:
                edge_index = (
                    torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                )
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros(
                    (
                        0,
                        (
                            len(self._get_bond_features(mol.GetBonds()[0]))
                            if mol.GetNumBonds() > 0
                            else 0
                        ),
                    ),
                    dtype=torch.float,
                )

            # Get global molecular features
            global_features = torch.tensor(
                self._get_global_features(mol), dtype=torch.float
            ).unsqueeze(0)

            # Create PyTorch Geometric Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_features=global_features,
                num_nodes=num_atoms,
            )

            # Add 3D conformer information if possible
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                conf = mol.GetConformer()
                positions = torch.tensor(
                    [
                        [
                            conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z,
                        ]
                        for i in range(num_atoms)
                    ],
                    dtype=torch.float,
                )
                data.pos = positions
            except:
                # If 3D conformer generation fails, add zero positions
                data.pos = torch.zeros((num_atoms, 3), dtype=torch.float)

            return data

        except Exception as e:
            print(f"Error featurizing molecule: {e}")
            return None

    def _optimize_3d_conformer(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Generate and optimize 3D conformer for a molecule."""
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMoleculeConfs(mol)
            return mol
        except Exception as e:
            print(f"Failed to generate 3D conformer: {e}")
            return None
