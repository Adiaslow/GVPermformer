"""
Module for loading and preprocessing molecular data from CSV files.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any


class DataLoader:
    """Class responsible for loading and initial preprocessing of molecular data."""

    def __init__(self, config_path: str):
        """
        Initialize the DataLoader with configuration.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.data = None

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary containing configuration parameters
        """
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file specified in config.

        Returns:
            DataFrame containing the loaded data
        """
        data_config = self.config["data"]
        file_path = Path(data_config["input_file"])

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        self.data = pd.read_csv(
            file_path, delimiter=data_config["delimiter"], encoding="utf-8"
        )

        # Validate required columns
        required_columns = [data_config["smiles_column"], data_config["target_column"]]
        missing_columns = [
            col for col in required_columns if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return self.data

    def get_features_and_targets(self) -> Tuple[pd.Series, pd.Series]:
        """
        Extract features (SMILES) and targets from the loaded data.

        Returns:
            Tuple containing (SMILES series, target series)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        data_config = self.config["data"]
        return (
            self.data[data_config["smiles_column"]],
            self.data[data_config["target_column"]],
        )
