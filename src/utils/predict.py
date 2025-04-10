"""
Utility module for making predictions using the trained GraphVAE model.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem

from src.models.graph_vae import GraphVAE
from src.utils.smiles_to_features import SmilesConverter


class GraphPredictor:
    """
    Utility class to make predictions using the trained GraphVAE model.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run the model on (None for auto-detection)
        """
        # Determine device
        if device is None:
            self.device = torch.device(
                "mps"
                if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Initialize SMILES converter
        self.converter = SmilesConverter()

        print(f"Model loaded on {self.device}")
        print(f"Model has {self._count_parameters(self.model)} parameters")

    def _load_model(self, model_path: str) -> GraphVAE:
        """
        Load a trained model from a checkpoint path.

        Args:
            model_path: Path to the model checkpoint

        Returns:
            Loaded model in evaluation mode
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model hyperparameters from checkpoint
        hparams = checkpoint.get("hyper_parameters", {})

        # Create model with the same architecture
        model = GraphVAE(
            node_features=hparams.get("node_features", 126),
            edge_features=hparams.get("edge_features", 9),
            hidden_dim=hparams.get("hidden_dim", 256),
            latent_dim=hparams.get("latent_dim", 64),
            num_layers=hparams.get("num_layers", 3),
            dropout=hparams.get("dropout", 0.1),
            use_huber_loss=hparams.get("use_huber_loss", True),
        )

        # Load weights
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return model

    def _count_parameters(self, model: torch.nn.Module) -> int:
        """Count the number of trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def predict_property(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Predict properties for one or more SMILES strings.

        Args:
            smiles: A single SMILES string or a list of SMILES strings

        Returns:
            Numpy array of property predictions
        """
        is_single = isinstance(smiles, str)
        smiles_list = [smiles] if is_single else smiles

        results = []

        for smile in smiles_list:
            try:
                # Convert SMILES to graph features
                graph_data = self.converter.convert(smile)

                # Skip invalid molecules
                if graph_data["node_features"].shape[0] == 0:
                    results.append(np.nan)
                    continue

                # Move data to device
                graph_data = {k: v.to(self.device) for k, v in graph_data.items()}

                # Make prediction
                with torch.no_grad():
                    outputs = self.model(
                        x=graph_data["node_features"].unsqueeze(0),
                        edge_index=graph_data["edge_index"].unsqueeze(0),
                        edge_attr=graph_data["edge_attr"].unsqueeze(0),
                        batch=torch.zeros(
                            graph_data["node_features"].size(0),
                            dtype=torch.long,
                            device=self.device,
                        ),
                    )

                    # Get property prediction
                    prediction = outputs["property_pred"].item()
                    results.append(prediction)

            except Exception as e:
                print(f"Error processing SMILES {smile}: {e}")
                results.append(np.nan)

        # Return single value or array based on input
        if is_single:
            return results[0]
        return np.array(results)

    def get_latent_representation(self, smiles: str) -> np.ndarray:
        """
        Get the latent representation of a molecule.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Latent vector as numpy array
        """
        # Convert SMILES to graph features
        graph_data = self.converter.convert(smiles)

        # Skip invalid molecules
        if graph_data["node_features"].shape[0] == 0:
            return np.zeros(self.model.latent_dim)

        # Move data to device
        graph_data = {k: v.to(self.device) for k, v in graph_data.items()}

        # Get latent representation
        with torch.no_grad():
            outputs = self.model.encode(
                x=graph_data["node_features"].unsqueeze(0),
                edge_index=graph_data["edge_index"].unsqueeze(0),
                edge_attr=graph_data["edge_attr"].unsqueeze(0),
                batch=torch.zeros(
                    graph_data["node_features"].size(0),
                    dtype=torch.long,
                    device=self.device,
                ),
            )

            # Return z_mean as the latent representation
            return outputs["z_mean"].cpu().numpy().squeeze()

    def predict_batch(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Make predictions for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary with predictions and statistics
        """
        predictions = []
        valid_indices = []
        latent_vectors = []

        for i, smile in enumerate(smiles_list):
            try:
                # Convert SMILES to graph features
                graph_data = self.converter.convert(smile)

                # Skip invalid molecules
                if graph_data["node_features"].shape[0] == 0:
                    predictions.append(np.nan)
                    continue

                # Move data to device
                graph_data = {k: v.to(self.device) for k, v in graph_data.items()}

                # Make prediction
                with torch.no_grad():
                    outputs = self.model(
                        x=graph_data["node_features"].unsqueeze(0),
                        edge_index=graph_data["edge_index"].unsqueeze(0),
                        edge_attr=graph_data["edge_attr"].unsqueeze(0),
                        batch=torch.zeros(
                            graph_data["node_features"].size(0),
                            dtype=torch.long,
                            device=self.device,
                        ),
                    )

                    # Get property prediction and latent vector
                    prediction = outputs["property_pred"].item()
                    latent_vector = outputs["z_mean"].cpu().numpy().squeeze()

                    predictions.append(prediction)
                    latent_vectors.append(latent_vector)
                    valid_indices.append(i)

            except Exception as e:
                print(f"Error processing SMILES {i}: {e}")
                predictions.append(np.nan)

        # Create results dataframe
        results_df = pd.DataFrame({"SMILES": smiles_list, "Prediction": predictions})

        # Calculate statistics for valid predictions
        valid_predictions = [p for p in predictions if not np.isnan(p)]
        stats = {
            "mean": np.mean(valid_predictions) if valid_predictions else np.nan,
            "std": np.std(valid_predictions) if valid_predictions else np.nan,
            "min": np.min(valid_predictions) if valid_predictions else np.nan,
            "max": np.max(valid_predictions) if valid_predictions else np.nan,
            "count": len(valid_predictions),
            "total": len(smiles_list),
        }

        # Create latent matrix for valid molecules
        latent_matrix = np.vstack(latent_vectors) if latent_vectors else np.array([])

        return {
            "predictions": results_df,
            "stats": stats,
            "latent_vectors": latent_matrix,
            "valid_indices": valid_indices,
        }
