"""
Configuration classes for the Graph VAE model.
"""


class Config:
    """Base configuration class."""

    def __init__(self):
        self.project_name = "GraphVAE"
        self.output = None


class DataConfig:
    """Data configuration settings."""

    def __init__(self):
        self.data_path = None
        self.smiles_col = "SMILES"
        self.property_cols = []
        self.batch_size = 32
        self.train_val_test_split = (0.8, 0.1, 0.1)
        self.max_atoms = 100
        self.num_workers = 2
        self.pin_memory = True
        self.prefetch_factor = 2
        self.random_seed = 42
        # Additional attributes
        self.train_csv = None
        self.pampa_threshold = -9.0
        self.use_enhanced_atom_features = False


class ModelConfig:
    """Model configuration settings."""

    def __init__(self):
        self.hidden_dim = 128
        self.latent_dim = 32
        self.dropout = 0.2
        self.beta = 0.5
        self.property_prediction = False
        # Additional attributes
        self.node_features = 21
        self.edge_features = 8
        self.num_properties = 1
        self.weight_decay = 1e-5


class TrainingConfig:
    """Training configuration settings."""

    def __init__(self):
        self.num_epochs = 100
        self.learning_rate = 1e-3
        self.batch_size = 32
        self.num_workers = 2
