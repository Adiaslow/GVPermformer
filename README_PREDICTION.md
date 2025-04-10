# GraphVAE Molecular Property Prediction

This guide explains how to use the trained Graph Variational Autoencoder (GraphVAE) model for predicting molecular properties from SMILES strings.

## Requirements

- Python 3.8+
- PyTorch
- RDKit
- Pandas
- A trained GraphVAE model checkpoint

Make sure to activate your environment before running any commands:

```bash
source ./graph_vae_env/bin/activate
```

## Prediction Script

The `predict_properties.py` script provides a convenient interface for making predictions. The script can be used in two ways:

1. **Single molecule prediction**: Provide a single SMILES string directly.
2. **Batch prediction**: Provide a file containing multiple SMILES strings (one per line).

### Basic Usage

For a single molecule:

```bash
python predict_properties.py --model_path checkpoints/latest.ckpt --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --device auto
```

For multiple molecules:

```bash
python predict_properties.py --model_path checkpoints/latest.ckpt --smiles_file molecules.txt --output predictions.csv --device auto
```

### Command-Line Arguments

- `--model_path`: Path to the trained model checkpoint (required)
- `--smiles`: A single SMILES string for prediction
- `--smiles_file`: Path to a file containing SMILES strings (one per line)
- `--output`: Path to save the prediction results as CSV (optional for single prediction)
- `--device`: Device to run inference on. Options are "auto", "cpu", "cuda", or "mps" (default: "auto")
- `--extract_latent`: Include latent representations in the output (optional)

Either `--smiles` or `--smiles_file` must be specified.

## Example Script

The repository includes an example script `example_predict.py` that demonstrates how to use the prediction functionality:

```bash
python example_predict.py
```

The example script:

1. Creates a file with example SMILES strings
2. Demonstrates single molecule prediction
3. Demonstrates batch prediction from a file
4. Loads and displays the prediction results

## Programmatic Usage

You can also use the prediction functionality in your own Python code:

```python
import torch
from predict_properties import load_model, predict_from_smiles

# Load the model
model = load_model("checkpoints/latest.ckpt", device="auto")
device = next(model.parameters()).device

# Define SMILES strings
smiles_list = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC(C)(C)NCC(O)C1=CC(=C(C=C1)O)CO"  # Salbutamol
]

# Make predictions
results = predict_from_smiles(
    model=model,
    smiles_list=smiles_list,
    device=device,
    extract_latent=True
)

# Print results
print(results)
```

## Interpreting Results

The prediction output includes:

- `SMILES`: The input SMILES string
- `Predicted_PAMPA`: The predicted PAMPA permeability value
- `latent_X` (optional): Components of the latent representation

Higher PAMPA values generally indicate better membrane permeability, which is an important factor in drug absorption.

## Troubleshooting

If you encounter errors:

1. **Invalid SMILES strings**: Make sure the SMILES strings are valid and properly formatted.
2. **Model loading errors**: Check that the model checkpoint exists and is accessible.
3. **Out of memory errors**: Try using a smaller batch size by processing SMILES strings in smaller groups.
4. **Device errors**: If you encounter issues with GPU (CUDA/MPS), try using `--device cpu`.

For advanced users, you can extend the `MoleculeFeaturizer` class in `src/utils/molecule_features.py` to customize how molecules are processed.
