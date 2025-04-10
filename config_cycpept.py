"""
Configuration file for training the Graph VAE model on cyclic peptide data.
"""

from src.config import Config, DataConfig, ModelConfig, TrainingConfig

# Create configuration
config = Config()

# Project name
config.project_name = "GraphVAE_CycPept"

# Data configuration
config.data = DataConfig()
config.data.data_path = "training_data/CycPeptMPDB_Peptide_All.csv"
config.data.smiles_col = "SMILES"
config.data.property_cols = ["MolLogP", "PSA", "MolWt"]
config.data.batch_size = 16
config.data.train_val_test_split = (0.8, 0.1, 0.1)
config.data.max_atoms = 150  # Increased for larger peptides
config.data.num_workers = 4
config.data.random_seed = 42

# Model configuration
config.model = ModelConfig()
config.model.hidden_dim = 128
config.model.latent_dim = 32
config.model.num_layers = 4
config.model.num_heads = 8
config.model.dropout = 0.2
config.model.beta = 0.5
config.model.property_prediction = True

# Training configuration
config.training = TrainingConfig()
config.training.max_epochs = 5
config.training.learning_rate = 1e-4
config.training.weight_decay = 1e-5
config.training.early_stopping = True
config.training.patience = 5
config.training.checkpoint_dir = "outputs/cycpept_model/checkpoints"
config.training.log_dir = "outputs/cycpept_model/logs"


# Define function to get the config
def get_cycpept_config():
    return config
