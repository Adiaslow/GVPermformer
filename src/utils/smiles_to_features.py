"""
Utility module for converting SMILES strings to graph features that can be used
with the GraphVAE model for predictions.
"""

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    rdMolDescriptors,
    Lipinski,
    AllChem,
    rdPartialCharges,
)
from rdkit.Chem import GraphDescriptors, Crippen, MolSurf, EState
import networkx as nx
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.MolStandardize import rdMolStandardize
from typing import Dict, Tuple, List, Optional, Union, Any, cast
import multiprocessing as mp
from functools import partial
import time
from tqdm import tqdm


class SmilesConverter:
    """
    Convert SMILES strings to graph features compatible with the GraphVAE model.
    Enhanced with comprehensive graph-based molecular features.
    """

    def __init__(self, max_atoms: int = 50, optimize_speed: bool = True):
        """
        Initialize the SMILES converter.

        Args:
            max_atoms: Maximum number of atoms to consider in a molecule
            optimize_speed: Whether to optimize for speed over feature completeness
        """
        self.max_atoms = max_atoms
        self.optimize_speed = optimize_speed

        # Hybridization types for atom feature extraction
        self.hybridization_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]

        # Define atom feature dimensions - enhanced for better performance
        self.atom_features_dim = 142  # Enhanced from 126

        # Define edge feature dimensions - enhanced for better performance
        self.edge_features_dim = 16  # Enhanced from 9

        # Define global feature dimensions - enhanced for better performance
        self.global_features_dim = 32  # Enhanced from 17

    def convert(self, smiles: str) -> Dict[str, torch.Tensor]:
        """
        Convert a SMILES string to graph features.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Dict containing node features, edge index, edge attributes, and global features
        """
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            # Return empty tensors if the molecule is invalid
            return {
                "node_features": torch.zeros(
                    0, self.atom_features_dim, dtype=torch.float
                ),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "edge_attr": torch.zeros(0, self.edge_features_dim, dtype=torch.float),
                "global_features": torch.zeros(
                    1, self.global_features_dim, dtype=torch.float
                ),
            }

        # Preprocess molecule (simplified if optimize_speed is True)
        mol = self._preprocess_molecule(mol)

        # Calculate descriptors directly from molecule (optimized)
        descriptors = self._calculate_descriptors(mol)

        # Extract graph-based features (optimized)
        if not self.optimize_speed:
            graph_features = self._calculate_graph_features(mol)
            # Merge descriptors with graph features
            descriptors.update(graph_features)
        else:
            # Skip expensive graph feature calculations for speed
            graph_features = self._calculate_basic_graph_features(mol)
            descriptors.update(graph_features)

        # Extract features
        node_features = self._get_node_features(mol)
        edge_index, edge_attr = self._get_edge_features(mol)
        global_features = self._get_global_features(descriptors)

        # Return dictionary with all features
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "global_features": global_features,
        }

    def _preprocess_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Preprocess molecule by adding hydrogen atoms, generating 3D coordinates,
        computing charges, and calculating other important properties.

        Args:
            mol: RDKit molecule object

        Returns:
            Preprocessed RDKit molecule
        """
        try:
            # Skip expensive operations if optimize_speed is True
            if self.optimize_speed:
                return mol

            # Remove salts, normalize structure
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)

            # Add hydrogen atoms (if not already present)
            if (
                mol.GetNumHeavyAtoms()
                + Chem.AddHs(mol).GetNumAtoms()
                - mol.GetNumAtoms()
                != mol.GetNumAtoms()
            ):
                mol = Chem.AddHs(mol)

            # Generate 3D coordinates if needed for certain descriptors
            if not mol.GetNumConformers():
                try:
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)  # Energy minimization
                except:
                    # Continue without 3D coordinates if generation fails
                    pass

            # Compute Gasteiger charges
            try:
                rdPartialCharges.ComputeGasteigerCharges(mol)
            except:
                # Continue without charges if computation fails
                pass

            return mol
        except Exception as e:
            # If any errors occur, return the original molecule
            print(f"Warning: Error during molecule preprocessing: {e}")
            return mol

    def _calculate_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate molecular descriptors from a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            Dictionary containing molecular descriptors
        """
        # Create a dictionary to store descriptors
        descriptors = {}

        try:
            # Basic molecular properties (keep these as they're fast to compute)
            descriptors["MolWt"] = float(Descriptors.MolWt(mol))
            descriptors["HeavyAtomCount"] = float(mol.GetNumHeavyAtoms())
            descriptors["NumRotatableBonds"] = float(Lipinski.NumRotatableBonds(mol))
            descriptors["NumHAcceptors"] = float(Lipinski.NumHAcceptors(mol))
            descriptors["NumHDonors"] = float(Lipinski.NumHDonors(mol))

            # Skip some descriptors if optimize_speed is True
            if not self.optimize_speed:
                descriptors["NumRings"] = float(rdMolDescriptors.CalcNumRings(mol))
                descriptors["NumAromaticRings"] = float(
                    rdMolDescriptors.CalcNumAromaticRings(mol)
                )
                descriptors["NumAliphaticRings"] = float(
                    rdMolDescriptors.CalcNumAliphaticRings(mol)
                )

                # Electronic properties
                descriptors["LogP"] = float(Descriptors.MolLogP(mol))
                descriptors["TPSA"] = float(Descriptors.TPSA(mol))

                # Charge-related properties
                descriptors["FractionCSP3"] = float(
                    rdMolDescriptors.CalcFractionCSP3(mol)
                )
                descriptors["NumHeteroatoms"] = float(Lipinski.NumHeteroatoms(mol))

                # Physical properties relevant for membrane permeability
                descriptors["MolMR"] = float(Crippen.MolMR(mol))
                descriptors["LabuteASA"] = float(MolSurf.LabuteASA(mol))

                # Fragment-based properties
                descriptors["MaxAbsPartialCharge"] = float(
                    Descriptors.MaxAbsPartialCharge(mol)
                )
                descriptors["MinAbsPartialCharge"] = float(
                    Descriptors.MinAbsPartialCharge(mol)
                )

                # Bertz complexity - measure of molecular complexity
                descriptors["BertzCT"] = float(GraphDescriptors.BertzCT(mol))
            else:
                # Add default values for skipped descriptors
                default_descriptors = {
                    "NumRings": 0.0,
                    "NumAromaticRings": 0.0,
                    "NumAliphaticRings": 0.0,
                    "LogP": 0.0,
                    "TPSA": 0.0,
                    "FractionCSP3": 0.0,
                    "NumHeteroatoms": 0.0,
                    "MolMR": 0.0,
                    "LabuteASA": 0.0,
                    "MaxAbsPartialCharge": 0.0,
                    "MinAbsPartialCharge": 0.0,
                    "BertzCT": 0.0,
                }
                descriptors.update(default_descriptors)
        except Exception as e:
            # Set default values if descriptor calculations fail
            default_descriptors = {
                "MolWt": float(
                    mol.GetNumHeavyAtoms() * 12
                ),  # Approximate by carbon weight
                "HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
                "NumRotatableBonds": 0.0,
                "NumHAcceptors": 0.0,
                "NumHDonors": 0.0,
                "NumRings": 0.0,
                "NumAromaticRings": 0.0,
                "NumAliphaticRings": 0.0,
                "LogP": 0.0,
                "TPSA": 0.0,
                "FractionCSP3": 0.0,
                "NumHeteroatoms": 0.0,
                "MolMR": 0.0,
                "LabuteASA": 0.0,
                "MaxAbsPartialCharge": 0.0,
                "MinAbsPartialCharge": 0.0,
                "BertzCT": 0.0,
            }
            descriptors.update(default_descriptors)

        return descriptors

    def _calculate_basic_graph_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate simplified graph-based features from the molecular structure.
        Optimized for speed.

        Args:
            mol: RDKit molecule object

        Returns:
            Dictionary containing basic graph-based descriptors
        """
        graph_features = {
            "AvgDegree": 0.0,
            "MaxDegree": 0.0,
            "Diameter": 0.0,
            "AvgBetweenness": 0.0,
            "MaxBetweenness": 0.0,
            "AvgCloseness": 0.0,
            "MaxCloseness": 0.0,
            "AvgClustering": 0.0,
            "Kappa1": 0.0,
            "Kappa2": 0.0,
            "Kappa3": 0.0,
            "Chi0": 0.0,
            "Chi1": 0.0,
            "MaxEStateIndex": 0.0,
            "MinEStateIndex": 0.0,
            "AvgEStateIndex": 0.0,
        }

        try:
            # Only compute the essential and fast-to-calculate graph features
            # Basic graph degree properties
            num_atoms = mol.GetNumAtoms()
            if num_atoms > 0:
                degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
                graph_features["AvgDegree"] = float(sum(degrees) / num_atoms)
                graph_features["MaxDegree"] = float(max(degrees))

                # Essential topological indices (fast to compute)
                try:
                    graph_features["Kappa1"] = float(GraphDescriptors.Kappa1(mol))
                    graph_features["Chi0"] = float(GraphDescriptors.Chi0(mol))
                except:
                    pass
        except Exception as e:
            # Keep the default values on error
            pass

        return graph_features

    def _calculate_graph_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate graph-based features from the molecular structure.

        Args:
            mol: RDKit molecule object

        Returns:
            Dictionary containing graph-based descriptors
        """
        graph_features = {}

        try:
            # Convert molecule to NetworkX graph
            adjacency = GetAdjacencyMatrix(mol)
            G = nx.from_numpy_array(adjacency)

            # Basic graph properties
            if G.number_of_nodes() > 0:
                graph_features["AvgDegree"] = float(
                    sum(dict(G.degree()).values()) / G.number_of_nodes()
                )
                graph_features["MaxDegree"] = float(
                    max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
                )

                # Connectivity measures
                try:
                    graph_features["Diameter"] = float(
                        nx.diameter(G) if nx.is_connected(G) else 0
                    )
                except nx.NetworkXError:
                    graph_features["Diameter"] = 0.0

                # Centrality measures
                if G.number_of_nodes() > 1:
                    # Betweenness centrality - important for identifying bridge atoms
                    try:
                        betweenness = nx.betweenness_centrality(G)
                        graph_features["AvgBetweenness"] = float(
                            sum(betweenness.values()) / len(betweenness)
                        )
                        graph_features["MaxBetweenness"] = float(
                            max(betweenness.values())
                        )
                    except:
                        graph_features["AvgBetweenness"] = 0.0
                        graph_features["MaxBetweenness"] = 0.0

                    # Closeness centrality - relates to the efficiency of transport through the molecule
                    try:
                        closeness = nx.closeness_centrality(G)
                        graph_features["AvgCloseness"] = float(
                            sum(closeness.values()) / len(closeness)
                        )
                        graph_features["MaxCloseness"] = float(max(closeness.values()))
                    except:
                        graph_features["AvgCloseness"] = 0.0
                        graph_features["MaxCloseness"] = 0.0
                else:
                    graph_features["AvgBetweenness"] = 0.0
                    graph_features["MaxBetweenness"] = 0.0
                    graph_features["AvgCloseness"] = 0.0
                    graph_features["MaxCloseness"] = 0.0

                # Clustering coefficient - relates to rigidity/flexibility
                try:
                    clustering = nx.clustering(G)
                    graph_features["AvgClustering"] = float(
                        sum(clustering.values()) / len(clustering)
                    )
                except:
                    graph_features["AvgClustering"] = 0.0
            else:
                # Default values for empty graphs
                graph_features["AvgDegree"] = 0.0
                graph_features["MaxDegree"] = 0.0
                graph_features["Diameter"] = 0.0
                graph_features["AvgBetweenness"] = 0.0
                graph_features["MaxBetweenness"] = 0.0
                graph_features["AvgCloseness"] = 0.0
                graph_features["MaxCloseness"] = 0.0
                graph_features["AvgClustering"] = 0.0

            # RDKit graph descriptors
            graph_features["Kappa1"] = float(GraphDescriptors.Kappa1(mol))
            graph_features["Kappa2"] = float(GraphDescriptors.Kappa2(mol))
            graph_features["Kappa3"] = float(GraphDescriptors.Kappa3(mol))
            graph_features["Chi0"] = float(GraphDescriptors.Chi0(mol))
            graph_features["Chi1"] = float(GraphDescriptors.Chi1(mol))

            # Electrotopological state indices
            try:
                estate_indices = EState.EStateIndices(mol)
                if estate_indices is not None and len(estate_indices) > 0:
                    # Handle numpy arrays properly
                    max_estate = float(max(estate_indices))
                    min_estate = float(min(estate_indices))
                    avg_estate = float(sum(estate_indices) / len(estate_indices))

                    graph_features["MaxEStateIndex"] = max_estate
                    graph_features["MinEStateIndex"] = min_estate
                    graph_features["AvgEStateIndex"] = avg_estate
                else:
                    graph_features["MaxEStateIndex"] = 0.0
                    graph_features["MinEStateIndex"] = 0.0
                    graph_features["AvgEStateIndex"] = 0.0
            except Exception as e:
                # Fall back to default values if there's an error
                graph_features["MaxEStateIndex"] = 0.0
                graph_features["MinEStateIndex"] = 0.0
                graph_features["AvgEStateIndex"] = 0.0

        except Exception as e:
            # Set default values if graph calculations fail
            print(f"Warning: Error during graph feature calculation: {e}")
            graph_features = {
                "AvgDegree": 0.0,
                "MaxDegree": 0.0,
                "Diameter": 0.0,
                "AvgBetweenness": 0.0,
                "MaxBetweenness": 0.0,
                "AvgCloseness": 0.0,
                "MaxCloseness": 0.0,
                "AvgClustering": 0.0,
                "Kappa1": 0.0,
                "Kappa2": 0.0,
                "Kappa3": 0.0,
                "Chi0": 0.0,
                "Chi1": 0.0,
                "MaxEStateIndex": 0.0,
                "MinEStateIndex": 0.0,
                "AvgEStateIndex": 0.0,
            }

        return graph_features

    def _get_node_features(self, mol: Chem.Mol) -> torch.Tensor:
        """
        Extract node features from a molecule with enhanced atom properties.

        Args:
            mol: RDKit molecule object

        Returns:
            torch.Tensor: Node features tensor of shape [num_atoms, atom_features_dim]
        """
        if mol is None:
            return torch.zeros(0, self.atom_features_dim, dtype=torch.float)

        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, self.atom_features_dim, dtype=torch.float)

        # Compute Gasteiger charges if not already done
        try:
            has_charge = False
            for atom in mol.GetAtoms():
                if atom.HasProp("_GasteigerCharge"):
                    has_charge = True
                    break
            if not has_charge:
                rdPartialCharges.ComputeGasteigerCharges(mol)
        except:
            pass  # Continue without charges if computation fails

        for atom_idx, atom in enumerate(mol.GetAtoms()):
            feature_idx = 0

            # One-hot encoding of atom type (94 features)
            atom_type = atom.GetAtomicNum() - 1  # Hydrogen = 0
            if 0 <= atom_type < 94:
                features[atom_idx, atom_type] = 1
            feature_idx += 94

            # One-hot encoding of atom degree (11 features)
            degree = min(atom.GetDegree(), 10)
            features[atom_idx, feature_idx + degree] = 1
            feature_idx += 11

            # One-hot encoding of formal charge (11 features)
            formal_charge = atom.GetFormalCharge()
            # Shift to range [0, 10] from [-5, 5]
            charge_idx = min(max(formal_charge + 5, 0), 10)
            features[atom_idx, feature_idx + charge_idx] = 1
            feature_idx += 11

            # One-hot encoding of hybridization (6 features)
            hybridization = atom.GetHybridization()
            hyb_idx = (
                self.hybridization_types.index(hybridization)
                if hybridization in self.hybridization_types
                else 5
            )
            features[atom_idx, feature_idx + hyb_idx] = 1
            feature_idx += 6

            # Basic features (5 features)
            # Aromaticity
            features[atom_idx, feature_idx] = 1 if atom.GetIsAromatic() else 0
            feature_idx += 1

            # Chirality
            features[atom_idx, feature_idx] = (
                1
                if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
                else 0
            )
            feature_idx += 1

            # Number of H atoms (normalized)
            features[atom_idx, feature_idx] = (
                atom.GetTotalNumHs() / 4.0
            )  # Normalize by typical max
            feature_idx += 1

            # Atom is in ring
            features[atom_idx, feature_idx] = 1 if atom.IsInRing() else 0
            feature_idx += 1

            # Explicit valence (normalized)
            features[atom_idx, feature_idx] = atom.GetExplicitValence() / 8.0
            feature_idx += 1

            # NEW ATOM FEATURES

            # Gasteiger partial charge (normalized)
            try:
                charge = atom.GetDoubleProp("_GasteigerCharge")
                # Normalize to range [0, 1] based on typical charge range [-1, 1]
                features[atom_idx, feature_idx] = (charge + 1.0) / 2.0
            except:
                features[atom_idx, feature_idx] = 0.5  # Default to neutral
            feature_idx += 1

            # Crippen contribution to logP
            try:
                logp_contrib = Crippen.MolLogP_Contribution(mol, atom_idx)
                # Normalize: typical range from -2 to 6
                features[atom_idx, feature_idx] = (logp_contrib + 2.0) / 8.0
            except:
                features[atom_idx, feature_idx] = 0.25  # Default
            feature_idx += 1

            # Van der Waals volume contribution (normalized)
            try:
                vdw_contrib = Crippen.MolMR_Contribution(mol, atom_idx)
                # Normalize: typical range is 0 to 2.5
                features[atom_idx, feature_idx] = min(vdw_contrib / 2.5, 1.0)
            except:
                features[atom_idx, feature_idx] = 0.2  # Default
            feature_idx += 1

            # ElectroTopological State (E-State) - measures electron accessibility
            try:
                estate_indices = EState.EStateIndices(mol)
                if len(estate_indices) > atom_idx:
                    # Normalize to [0, 1] based on typical range [-10, 15]
                    estate = estate_indices[atom_idx]
                    features[atom_idx, feature_idx] = (estate + 10.0) / 25.0
                else:
                    features[atom_idx, feature_idx] = 0.4  # Default
            except:
                features[atom_idx, feature_idx] = 0.4  # Default
            feature_idx += 1

            # Atom is in aromatic ring - size features (3 features)
            is_in_ring_size = {5: False, 6: False, 7: False}
            for ring_size in is_in_ring_size:
                is_in_ring_size[ring_size] = atom.IsInRingSize(ring_size)

            features[atom_idx, feature_idx] = (
                1 if is_in_ring_size[5] else 0
            )  # 5-membered ring
            feature_idx += 1
            features[atom_idx, feature_idx] = (
                1 if is_in_ring_size[6] else 0
            )  # 6-membered ring
            feature_idx += 1
            features[atom_idx, feature_idx] = (
                1 if is_in_ring_size[7] else 0
            )  # 7-membered ring
            feature_idx += 1

            # MMFF94 partial charges (normalized)
            try:
                # First check if we have 3D coordinates needed for MMFF94
                if mol.GetNumConformers() > 0:
                    mp = AllChem.MMFFGetMoleculeProperties(mol)
                    if mp:
                        mmff_charge = mp.GetMMFFPartialCharge(atom_idx)
                        # Normalize to range [0, 1] based on typical range [-1, 1]
                        features[atom_idx, feature_idx] = (mmff_charge + 1.0) / 2.0
                    else:
                        features[atom_idx, feature_idx] = 0.5  # Default to neutral
                else:
                    features[atom_idx, feature_idx] = 0.5  # Default to neutral
            except:
                features[atom_idx, feature_idx] = 0.5  # Default to neutral
            feature_idx += 1

        assert (
            features.shape[1] == self.atom_features_dim
        ), f"Expected {self.atom_features_dim} atom features, got {features.shape[1]}"
        return features

    def _get_edge_features(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract enhanced edge features from a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            tuple: (edge_index, edge_attr) where edge_index is a tensor of shape [2, num_edges]
                  and edge_attr is a tensor of shape [num_edges, edge_features_dim]
        """
        if mol is None:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(
                0, self.edge_features_dim, dtype=torch.float
            )

        num_atoms = mol.GetNumAtoms()
        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Enhanced bond features (increased from 9 to 16 features)
            edge_feature = [0] * self.edge_features_dim

            # Bond type (first 4 positions)
            bond_type = bond.GetBondTypeAsDouble()
            if bond_type == 1:
                edge_feature[0] = 1  # Single
            elif bond_type == 2:
                edge_feature[1] = 1  # Double
            elif bond_type == 3:
                edge_feature[2] = 1  # Triple
            elif bond_type == 1.5:
                edge_feature[3] = 1  # Aromatic

            # Bond is conjugated
            edge_feature[4] = int(bond.GetIsConjugated())

            # Bond is in ring
            edge_feature[5] = int(bond.IsInRing())

            # Bond stereochemistry
            stereo = bond.GetStereo()
            if (
                stereo == Chem.rdchem.BondStereo.STEREOZ
                or stereo == Chem.rdchem.BondStereo.STEREOCIS
            ):
                edge_feature[6] = 1  # Z/cis
            elif (
                stereo == Chem.rdchem.BondStereo.STEREOE
                or stereo == Chem.rdchem.BondStereo.STEREOTRANS
            ):
                edge_feature[7] = 1  # E/trans
            else:
                edge_feature[8] = 1  # None or unspecified

            # Bond is rotatable - using Lipinski's RotatableBondCount function to check
            is_rotatable = (
                not bond.IsInRing()
                and bond.GetBondTypeAsDouble() == 1.0
                and not bond.GetIsConjugated()
            )
            edge_feature[9] = int(is_rotatable)

            # Bond ring membership with size encoding
            # Check if the bond is in rings of specific sizes (useful for assessing rigidity)
            bond_in_ring_size = {5: 0, 6: 0, 7: 0}
            if bond.IsInRing():
                # Get the rings containing this bond
                ring_info = (
                    mol.GetSSSR() if hasattr(mol, "GetSSSR") else Chem.GetSSSR(mol)
                )
                for ring_size in bond_in_ring_size.keys():
                    if bond.IsInRingSize(ring_size):
                        bond_in_ring_size[ring_size] = 1

            edge_feature[10] = bond_in_ring_size[5]  # In 5-membered ring
            edge_feature[11] = bond_in_ring_size[6]  # In 6-membered ring
            edge_feature[12] = bond_in_ring_size[7]  # In 7-membered ring

            # Calculate bond length if 3D coordinates are available
            # Bond length can affect molecular flexibility and permeability
            if mol.GetNumConformers() > 0:
                begin_atom = mol.GetAtomWithIdx(i)
                end_atom = mol.GetAtomWithIdx(j)
                begin_pos = mol.GetConformer().GetAtomPosition(i)
                end_pos = mol.GetConformer().GetAtomPosition(j)

                # Calculate Euclidean distance
                bond_length = (
                    (begin_pos.x - end_pos.x) ** 2
                    + (begin_pos.y - end_pos.y) ** 2
                    + (begin_pos.z - end_pos.z) ** 2
                ) ** 0.5

                # Normalize based on typical bond length range (0.9 - 2.1 Å)
                norm_bond_length = min(max((bond_length - 0.9) / 1.2, 0.0), 1.0)
                edge_feature[13] = norm_bond_length
            else:
                # Default value if no 3D coords
                edge_feature[13] = 0.5

            # Add topological distance feature
            # This will be 1 for direct bonds
            edge_feature[14] = 1.0

            # Add feature for bonds connecting different elements
            # This can impact charge distribution across the bond
            begin_atom = mol.GetAtomWithIdx(i)
            end_atom = mol.GetAtomWithIdx(j)
            edge_feature[15] = int(begin_atom.GetAtomicNum() != end_atom.GetAtomicNum())

            # Add edges in both directions
            edge_indices.append([i, j])
            edge_indices.append([j, i])

            edge_attrs.append(edge_feature)
            edge_attrs.append(edge_feature)

        if not edge_indices:
            # If no bonds, create self-loops
            edge_indices = [[i, i] for i in range(num_atoms)]
            # Create default edge features for self-loops
            default_self_loop = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            edge_attrs = [default_self_loop for _ in range(num_atoms)]

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        return edge_index, edge_attr

    def _get_global_features(self, descriptors: Dict[str, float]) -> torch.Tensor:
        """
        Create global molecular features from descriptors.

        Args:
            descriptors: Dictionary of molecular descriptors

        Returns:
            torch.Tensor: Global features tensor of shape [1, global_features_dim]
        """
        # Expanded list of global features for better PAMPA prediction
        feature_list = [
            # Basic molecular properties
            "MolWt",  # Molecular weight
            "HeavyAtomCount",  # Number of non-hydrogen atoms
            "NumRotatableBonds",  # Flexibility marker
            "NumHAcceptors",  # H-bond acceptors
            "NumHDonors",  # H-bond donors
            "NumRings",  # Total rings
            "NumAromaticRings",  # Aromatic rings
            "NumAliphaticRings",  # Aliphatic rings
            # Electronic properties - critical for PAMPA
            "LogP",  # Lipophilicity - crucial for membrane permeability
            "TPSA",  # Topological polar surface area - important for permeability
            "FractionCSP3",  # Fraction of sp3 carbon atoms - flexibility
            "NumHeteroatoms",  # Heteroatoms - polar interactions
            # Physical properties
            "MolMR",  # Molar refractivity - dispersion forces
            "LabuteASA",  # Approximate surface area
            "MaxAbsPartialCharge",  # Max absolute partial charge
            "MinAbsPartialCharge",  # Min absolute partial charge
            # Complexity
            "BertzCT",  # Molecular complexity
            # Graph-based properties
            "AvgDegree",  # Average atom degree
            "MaxDegree",  # Maximum atom degree
            "Diameter",  # Graph diameter
            "AvgBetweenness",  # Average betweenness centrality
            "MaxBetweenness",  # Maximum betweenness centrality
            "AvgCloseness",  # Average closeness centrality
            "MaxCloseness",  # Maximum closeness centrality
            "AvgClustering",  # Average clustering coefficient
            # Topological indices - encode molecular shape/connectivity
            "Kappa1",  # Kappa shape index 1
            "Kappa2",  # Kappa shape index 2
            "Kappa3",  # Kappa shape index 3
            "Chi0",  # Connectivity index chi0
            "Chi1",  # Connectivity index chi1
            # Electrotopological indices
            "MaxEStateIndex",  # Maximum E-State index
            "MinEStateIndex",  # Minimum E-State index
            "AvgEStateIndex",  # Average E-State index
        ]

        # Create feature vector and fill with values
        features = torch.zeros(1, self.global_features_dim, dtype=torch.float)
        for i, feature_name in enumerate(feature_list):
            if i < self.global_features_dim:  # Ensure we don't exceed dimensions
                # Get the feature value, default to 0.0 if not present
                value = descriptors.get(feature_name, 0.0)

                # Normalize based on feature type
                if feature_name == "MolWt":
                    # Normalize molecular weight to [0,1] - typical range 0-1000
                    features[0, i] = min(value / 1000.0, 1.0)
                elif feature_name == "LogP":
                    # Normalize LogP to [0,1] - typical range -5 to 10
                    features[0, i] = (value + 5.0) / 15.0
                elif feature_name == "TPSA":
                    # Normalize TPSA to [0,1] - typical range 0-250
                    features[0, i] = min(value / 250.0, 1.0)
                elif feature_name == "HeavyAtomCount":
                    # Normalize heavy atom count - typical range 0-100
                    features[0, i] = min(value / 100.0, 1.0)
                elif feature_name == "BertzCT":
                    # Normalize complexity - typical range 0-1500
                    features[0, i] = min(value / 1500.0, 1.0)
                elif feature_name == "LabuteASA":
                    # Normalize ASA - typical range 0-400
                    features[0, i] = min(value / 400.0, 1.0)
                elif "EStateIndex" in feature_name:
                    # Normalize E-State indices - typical range -10 to 15
                    features[0, i] = (value + 10.0) / 25.0
                elif feature_name in [
                    "MaxDegree",
                    "NumRings",
                    "NumRotatableBonds",
                    "NumHAcceptors",
                    "NumHDonors",
                ]:
                    # Normalize count features - typically small integers (0-20)
                    features[0, i] = min(value / 20.0, 1.0)
                else:
                    # Default normalization - many descriptors fall in [0,5] range
                    features[0, i] = min(value / 5.0, 1.0)

        return features


def _process_smiles_chunk(chunk, max_atoms=150, optimize_speed=True):
    """
    Process a chunk of SMILES strings in parallel.

    Args:
        chunk: List of SMILES strings to process
        max_atoms: Maximum number of atoms to consider
        optimize_speed: Whether to optimize for speed

    Returns:
        List of processed graph data dictionaries
    """
    converter = SmilesConverter(max_atoms=max_atoms, optimize_speed=optimize_speed)
    results = []

    for smiles in chunk:
        try:
            graph_data = converter.convert(smiles)
            results.append(graph_data)
        except Exception as e:
            # On error, add empty data
            empty_data = {
                "node_features": torch.zeros(
                    0, converter.atom_features_dim, dtype=torch.float
                ),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "edge_attr": torch.zeros(
                    0, converter.edge_features_dim, dtype=torch.float
                ),
                "global_features": torch.zeros(
                    1, converter.global_features_dim, dtype=torch.float
                ),
            }
            results.append(empty_data)

    return results


def smiles_to_graph_data(smiles: str) -> Dict[str, torch.Tensor]:
    """
    Convert a SMILES string to graph data.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        Dict containing graph data tensors
    """
    # Use enhanced SmilesConverter for feature extraction
    converter = SmilesConverter(max_atoms=150, optimize_speed=True)
    return converter.convert(smiles)


def batch_smiles_to_features_parallel(
    smiles_list: List[str],
    n_jobs: int = None,
    chunk_size: int = 100,
    optimize_speed: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Convert a batch of SMILES strings to batched features using parallel processing.

    Args:
        smiles_list: List of SMILES strings
        n_jobs: Number of parallel jobs (defaults to number of CPU cores - 1)
        chunk_size: Size of SMILES chunks to process in each job
        optimize_speed: Whether to optimize for speed over feature completeness

    Returns:
        Dictionary with batched features or None if all molecules are invalid
    """
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    start_time = time.time()
    print(f"Processing {len(smiles_list)} molecules with {n_jobs} workers...")

    # Split SMILES list into chunks
    chunks = [
        smiles_list[i : i + chunk_size] for i in range(0, len(smiles_list), chunk_size)
    ]

    # Create a pool of workers
    pool = mp.Pool(processes=n_jobs)

    # Process chunks in parallel
    process_func = partial(
        _process_smiles_chunk, max_atoms=150, optimize_speed=optimize_speed
    )
    results = []

    try:
        # Process all chunks with a progress bar
        for chunk_result in tqdm(pool.imap(process_func, chunks), total=len(chunks)):
            results.extend(chunk_result)
    finally:
        pool.close()
        pool.join()

    # Filter valid results
    valid_data = [data for data in results if data["node_features"].shape[0] > 0]

    end_time = time.time()
    print(
        f"Processed {len(valid_data)} valid molecules in {end_time - start_time:.2f}s"
    )

    # Return None if no valid molecules
    if not valid_data:
        return None

    # Batch the data
    return batch_graph_data(valid_data)


def batch_smiles_to_features(
    smiles_list: List[str], parallel: bool = True, optimize_speed: bool = True
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Convert a batch of SMILES strings to batched features.

    Args:
        smiles_list: List of SMILES strings
        parallel: Whether to use parallel processing
        optimize_speed: Whether to optimize for speed

    Returns:
        Dictionary with batched features or None if all molecules are invalid
    """
    if parallel and len(smiles_list) > 10:
        return batch_smiles_to_features_parallel(
            smiles_list, optimize_speed=optimize_speed
        )

    # Process each SMILES string
    valid_data = []
    converter = SmilesConverter(max_atoms=150, optimize_speed=optimize_speed)

    for smiles in smiles_list:
        try:
            data = converter.convert(smiles)
            # Skip empty/invalid molecules
            if data["node_features"].shape[0] > 0:
                valid_data.append(data)
        except Exception as e:
            continue

    # Return None if no valid molecules
    if not valid_data:
        return None

    # Batch the data
    return batch_graph_data(valid_data)


def batch_graph_data(
    data_list: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Batch a list of graph data dictionaries.

    Args:
        data_list: List of data dictionaries from smiles_to_graph_data

    Returns:
        Dictionary with batched features
    """
    # Initialize with empty lists for each feature
    batched_data: Dict[str, List[torch.Tensor]] = {
        "node_features": [],
        "edge_index": [],
        "edge_attr": [],
        "global_features": [],
        "batch": [],  # Node batch indices
        "edge_batch": [],  # Edge batch indices
    }

    # Track cumulative number of nodes for edge indices
    cumulative_nodes = 0

    # Process each sample
    for i, data in enumerate(data_list):
        num_nodes = data["node_features"].shape[0]

        # Skip if there are no nodes (invalid molecule)
        if num_nodes == 0:
            continue

        # Add node features
        batched_data["node_features"].append(data["node_features"])

        # Add edge indices with offset
        if data["edge_index"].shape[1] > 0:
            offset_edge_index = data["edge_index"].clone()
            offset_edge_index += cumulative_nodes
            batched_data["edge_index"].append(offset_edge_index)

            # Add edge attributes
            batched_data["edge_attr"].append(data["edge_attr"])

            # Add edge batch indices
            edge_batch = torch.full((data["edge_attr"].shape[0],), i, dtype=torch.long)
            batched_data["edge_batch"].append(edge_batch)

        # Add global features
        batched_data["global_features"].append(data["global_features"])

        # Add node batch indices
        batch = torch.full((num_nodes,), i, dtype=torch.long)
        batched_data["batch"].append(batch)

        # Update cumulative node count
        cumulative_nodes += num_nodes

    # Create batched result dictionary with concatenated tensors
    result = {}

    # Concatenate tensors if lists are non-empty
    if batched_data["node_features"]:
        result["node_features"] = torch.cat(batched_data["node_features"], dim=0)
        result["batch"] = torch.cat(batched_data["batch"], dim=0)
        result["global_features"] = torch.cat(batched_data["global_features"], dim=0)
    else:
        # Empty tensors if no valid data
        result["node_features"] = torch.zeros((0, 142), dtype=torch.float)
        result["batch"] = torch.zeros((0,), dtype=torch.long)
        result["global_features"] = torch.zeros((0, 32), dtype=torch.float)

    # Concatenate edge data if present
    if batched_data["edge_index"]:
        result["edge_index"] = torch.cat(batched_data["edge_index"], dim=1)
        result["edge_attr"] = torch.cat(batched_data["edge_attr"], dim=0)
        result["edge_batch"] = torch.cat(batched_data["edge_batch"], dim=0)
    else:
        # Create empty tensors if no edges
        result["edge_index"] = torch.zeros((2, 0), dtype=torch.long)
        result["edge_attr"] = torch.zeros((0, 16), dtype=torch.float)
        result["edge_batch"] = torch.zeros((0,), dtype=torch.long)

    return result
