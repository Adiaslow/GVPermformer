# GraphPermT: Graph VAE Transformer for Peptide Permeability Prediction

This project focuses on predicting the permeability of cyclic peptides using graph-based deep learning approaches. The model combines a Graph Variational Autoencoder (VAE) with transformer architecture to capture complex relationships between peptide structures and their permeability properties.

## Project Structure

```
├── notebooks/            # Jupyter notebooks for exploration and visualization
├── src/
│   ├── data/             # Data loading and preprocessing modules
│   ├── features/         # Feature engineering and molecular representation
│   ├── models/           # Graph VAE transformer model implementation
│   ├── utils/            # Utility functions and helper modules
│   └── visualization/    # Visualization tools for model performance
└── training_data/        # Raw training data
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Data preprocessing and feature engineering:

```bash
python src/data/preprocess.py
```

2. Train the model:

```bash
python src/models/train.py
```

3. Evaluate model performance:

```bash
python src/models/evaluate.py
```

## Apple Metal Support

This project leverages Apple Metal for GPU acceleration on Mac devices. Make sure you have the appropriate PyTorch version installed with MPS support.

## License

MIT
