# src/config/config_cycpept.py
"""
Configuration for the cyclic peptide Graph VAE model training.
"""

from src.config import Config, DataConfig, ModelConfig, TrainingConfig


def get_config():
    """Create and configure the training configuration."""
    # Create configuration
    config = Config()

    # Project name
    config.project_name = "GraphVAE_CycPept_Enhanced"

    # Data configuration
    config.data = DataConfig()
    config.data.data_path = "training_data/CycPeptMPDB_Peptide_All.csv"
    config.data.smiles_col = "SMILES"
    config.data.property_cols = ["PAMPA"]  # Predicting PAMPA permeability
    config.data.batch_size = 32  # Optimized batch size for Metal
    config.data.train_val_test_split = (0.8, 0.1, 0.1)
    config.data.max_atoms = (
        500  # Much higher limit to prevent atom limit warnings/errors
    )
    config.data.num_workers = 0  # Set to 0 to avoid multiprocessing issues
    config.data.pin_memory = True  # Enable pinned memory for faster transfers
    config.data.prefetch_factor = 2  # Prefetch batches
    config.data.random_seed = 42

    # Add dataset-specific configurations
    config.data.train_csv = config.data.data_path
    config.data.pampa_threshold = -9.0  # Threshold for filtering PAMPA values
    config.data.use_enhanced_atom_features = True

    # Model configuration
    config.model = ModelConfig()
    # Increased hidden dimensions to handle the enhanced feature set
    config.model.hidden_dim = 256  # Increased from 128 for enhanced features
    config.model.latent_dim = 64  # Increased from 32 for better representation capacity
    config.model.dropout = 0.1  # Reduced dropout for better MPS performance
    config.model.beta = 0.5
    config.model.property_prediction = True
    config.model.node_features = 126  # Will be updated based on dataset
    config.model.edge_features = 9  # Will be updated based on dataset
    config.model.num_properties = 1  # Default for PAMPA prediction
    config.model.weight_decay = 1e-5  # Better regularization

    # Training configuration
    config.training = TrainingConfig()
    config.training.num_epochs = 20  # Increased from 10 for more training cycles
    config.training.learning_rate = 1e-3  # Higher learning rate for initial training
    config.training.batch_size = 128
    config.training.num_workers = 0  # Set to 0 to avoid multiprocessing issues

    # Output configuration
    config.output = type("OutputConfig", (), {})()
    config.output.output_dir = "outputs/cycpept_model_enhanced"

    return config
