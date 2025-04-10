"""
Utility functions for calculating and tracking model metrics.

This module provides functions for computing common evaluation metrics
for molecular property prediction and generation tasks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
import torch


def calculate_regression_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics (RMSE, MAE, R²)
    """
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def calculate_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    y_score: Optional[Union[np.ndarray, List[float]]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate classification metrics.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        y_score: Predicted class probabilities (for ROC-AUC)
        class_names: List of class names for better result labeling

    Returns:
        Dictionary of metrics (accuracy, precision, recall, F1, etc.)
    """
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(y_true, y_pred, average=None)
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # ROC-AUC if probabilities provided
    auc = None
    if y_score is not None:
        try:
            if y_score.ndim > 1 and y_score.shape[1] > 1:
                # Multi-class
                auc = roc_auc_score(y_true, y_score, multi_class="ovr")
            else:
                # Binary
                auc = roc_auc_score(y_true, y_score)
        except ValueError:
            # ROC-AUC calculation may fail in certain cases
            pass

    # Create results dictionary
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix,
    }

    if auc is not None:
        results["auc"] = auc

    # Add per-class metrics with class names if provided
    per_class_metrics = {
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
    }

    if class_names is not None:
        named_per_class_metrics = {}
        for metric_name, values in per_class_metrics.items():
            named_per_class_metrics[metric_name] = {
                class_names[i]: value for i, value in enumerate(values)
            }
        results.update(named_per_class_metrics)
    else:
        results.update(per_class_metrics)

    return results


def calculate_molecular_metrics(
    gen_smiles: List[str],
    ref_smiles: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate molecular generation metrics.

    Args:
        gen_smiles: List of generated SMILES strings
        ref_smiles: List of reference SMILES strings (for comparison)

    Returns:
        Dictionary of metrics (validity, uniqueness, etc.)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import QED, Crippen, Descriptors
    except ImportError:
        raise ImportError("RDKit is required for molecular metrics calculation")

    # Check validity
    valid_mols = []
    for smi in gen_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mols.append(mol)

    validity = len(valid_mols) / len(gen_smiles) if gen_smiles else 0

    # Check uniqueness
    unique_smiles = set()
    for mol in valid_mols:
        unique_smiles.add(Chem.MolToSmiles(mol, isomericSmiles=True))

    uniqueness = len(unique_smiles) / len(valid_mols) if valid_mols else 0

    # Calculate properties (for valid molecules)
    properties = {
        "logP": [],
        "QED": [],
        "MW": [],
    }

    for mol in valid_mols:
        properties["logP"].append(Crippen.MolLogP(mol))
        properties["QED"].append(QED.qed(mol))
        properties["MW"].append(Descriptors.MolWt(mol))

    # Calculate novelty if reference SMILES provided
    novelty = None
    if ref_smiles:
        ref_smiles_set = set()
        for smi in ref_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                ref_smiles_set.add(Chem.MolToSmiles(mol, isomericSmiles=True))

        # Count novel molecules (not in reference set)
        novel_count = sum(1 for smi in unique_smiles if smi not in ref_smiles_set)
        novelty = novel_count / len(unique_smiles) if unique_smiles else 0

    # Prepare results
    results = {
        "validity": validity,
        "uniqueness": uniqueness,
    }

    if novelty is not None:
        results["novelty"] = novelty

    # Add property statistics
    for prop_name, values in properties.items():
        if values:
            results[f"{prop_name}_mean"] = np.mean(values)
            results[f"{prop_name}_std"] = np.std(values)
            results[f"{prop_name}_min"] = np.min(values)
            results[f"{prop_name}_max"] = np.max(values)

    return results


class MetricsTracker:
    """
    Class for tracking metrics during training or evaluation.
    """

    def __init__(self):
        """
        Initialize the metrics tracker.
        """
        self.metrics = {}
        self.epoch_metrics = {}
        self.best_metrics = {}
        self.current_epoch = 0

    def update(
        self, metrics_dict: Dict[str, float], epoch: Optional[int] = None
    ) -> None:
        """
        Update metrics with new values.

        Args:
            metrics_dict: Dictionary of metric values
            epoch: Epoch number (if None, use current_epoch)
        """
        epoch = epoch if epoch is not None else self.current_epoch

        # Initialize epoch metrics if not exists
        if epoch not in self.epoch_metrics:
            self.epoch_metrics[epoch] = {}

        # Update epoch metrics
        self.epoch_metrics[epoch].update(metrics_dict)

        # Update best metrics
        for key, value in metrics_dict.items():
            # Check if this is a new best
            if key not in self.best_metrics or self._is_better(
                key, value, self.best_metrics[key]
            ):
                self.best_metrics[key] = value

    def _is_better(self, metric_name: str, new_value: float, old_value: float) -> bool:
        """
        Check if a new metric value is better than the old one.

        Args:
            metric_name: Name of the metric
            new_value: New metric value
            old_value: Old metric value

        Returns:
            True if new value is better, False otherwise
        """
        # Metrics where lower is better
        lower_is_better = {
            "loss",
            "val_loss",
            "test_loss",
            "rmse",
            "mae",
            "error",
        }

        # Check if metric name contains any of the lower_is_better keywords
        is_lower_better = any(
            keyword in metric_name.lower() for keyword in lower_is_better
        )

        if is_lower_better:
            return new_value < old_value
        else:
            return new_value > old_value

    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get metrics for the current epoch.

        Returns:
            Dictionary of current metrics
        """
        return self.epoch_metrics.get(self.current_epoch, {})

    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get best metrics across all epochs.

        Returns:
            Dictionary of best metrics
        """
        return self.best_metrics

    def get_metrics_history(self) -> Dict[int, Dict[str, float]]:
        """
        Get metrics history across all epochs.

        Returns:
            Dictionary mapping epoch numbers to metric dictionaries
        """
        return self.epoch_metrics

    def next_epoch(self) -> None:
        """
        Move to the next epoch.
        """
        self.current_epoch += 1

    def summarize(self) -> Dict[str, Any]:
        """
        Generate a summary of tracked metrics.

        Returns:
            Dictionary with metric summaries
        """
        # Get all metric keys across all epochs
        all_keys = set()
        for epoch_metrics in self.epoch_metrics.values():
            all_keys.update(epoch_metrics.keys())

        # Create summary
        summary = {
            "epochs": len(self.epoch_metrics),
            "best_metrics": self.best_metrics,
            "metrics_by_key": {},
        }

        # Collect values for each metric
        for key in all_keys:
            values = [
                metrics.get(key)
                for epoch, metrics in sorted(self.epoch_metrics.items())
                if key in metrics
            ]

            # Only compute statistics for numeric values
            numeric_values = [
                v for v in values if isinstance(v, (int, float)) and not np.isnan(v)
            ]

            if numeric_values:
                summary["metrics_by_key"][key] = {
                    "mean": np.mean(numeric_values),
                    "std": np.std(numeric_values),
                    "min": np.min(numeric_values),
                    "max": np.max(numeric_values),
                    "last": numeric_values[-1],
                }

        return summary


def tensor_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    metric_type: str = "regression",
) -> Dict[str, torch.Tensor]:
    """
    Calculate metrics for torch tensors directly (without conversion to numpy).

    Args:
        y_true: True values tensor
        y_pred: Predicted values tensor
        metric_type: Type of metrics to calculate ('regression' or 'classification')

    Returns:
        Dictionary of metric tensors
    """
    if metric_type == "regression":
        # MSE
        mse = torch.nn.functional.mse_loss(y_pred, y_true)

        # RMSE
        rmse = torch.sqrt(mse)

        # MAE
        mae = torch.nn.functional.l1_loss(y_pred, y_true)

        # R²
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    elif metric_type == "classification":
        # Binary or multi-class
        if y_pred.dim() > 1 and y_pred.shape[1] > 1:
            # Multi-class: y_pred is [batch_size, num_classes]
            _, predicted_classes = torch.max(y_pred, dim=1)

            # Accuracy
            accuracy = (predicted_classes == y_true).float().mean()

            return {"accuracy": accuracy}
        else:
            # Binary: y_pred is [batch_size] or [batch_size, 1]
            if y_pred.dim() > 1:
                y_pred = y_pred.squeeze(-1)

            # Apply sigmoid if needed
            if torch.min(y_pred) < 0 or torch.max(y_pred) > 1:
                y_pred = torch.sigmoid(y_pred)

            # Binary accuracy
            binary_preds = (y_pred > 0.5).float()
            accuracy = (binary_preds == y_true).float().mean()

            # Binary cross-entropy
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                y_pred, y_true.float()
            )

            return {
                "accuracy": accuracy,
                "bce": bce,
            }

    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")
