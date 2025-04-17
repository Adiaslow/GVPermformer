# GraphVAE Model Evaluation

This directory contains the `quick_evaluate.py` script for evaluating a trained GraphVAE model on cyclic peptide data.

## Purpose

The `quick_evaluate.py` script provides a simplified way to:

1. Load a saved GraphVAE model checkpoint
2. Create a dataloader with the same parameters used during training
3. Generate a summary of the dataset and model configuration
4. Visualize the distribution of PAMPA values in the dataset

This script is primarily intended for inspection and verification of model checkpoints and data, rather than for conducting a full evaluation with predictions.

## Usage

```bash
python quick_evaluate.py --model_path <path_to_checkpoint> --data_csv <path_to_data> --output_dir <output_directory>
```

### Arguments

- `--model_path`: Path to the model checkpoint file (`.pt`)
- `--data_csv`: Path to the CSV file containing the evaluation data
- `--batch_size`: Batch size for evaluation (default: 32)
- `--output_dir`: Directory to save evaluation results (default: `evaluation_results`)

### Example

```bash
python quick_evaluate.py \
  --model_path outputs/cycpept_model_enhanced/models/final_model.pt \
  --data_csv training_data/CycPeptMPDB_Peptide_All.csv \
  --output_dir quick_evaluation
```

## Output

The script generates the following output:

1. **Console Output**:

   - Model configuration summary (node features, edge features, hidden dim, latent dim)
   - Dataset statistics (total size, PAMPA value range, mean PAMPA value)
   - Sample batch information (shapes of tensors)

2. **Output Files**:
   - `evaluation_summary.json`: JSON file containing dataset statistics
   - `dataset_sample.csv`: CSV file with a sample of molecules from the dataset
   - `pampa_distribution.png`: Histogram showing the distribution of PAMPA values

## Limitations

This script has the following limitations:

1. It does not perform actual forward passes through the model to generate predictions
2. It loads only a simplified version of the model architecture
3. It cannot fully reconstruct the original model's behavior

For a complete evaluation with predictions, you would need to:

1. Implement the complete model architecture
2. Load weights with the proper state_dict structure
3. Ensure the forward pass handles all tensor operations correctly

## Next Steps

After using this script to verify your model and data, you may want to:

1. Implement a full evaluation script with the complete model architecture
2. Generate predictions on a test set
3. Calculate performance metrics (RMSE, MAE, R²)
4. Visualize the relationship between predicted and actual values

## Requirements

- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn (for metrics)
- RDKit (for molecular processing)

## Model Architecture

The GraphVAE model is designed to:

1. Encode molecular graphs into a latent space
2. Decode the latent representations back to molecular graphs
3. Predict molecular properties (such as PAMPA) from the latent space

The model consists of:

- A graph encoder that processes node and edge features
- A variational autoencoder middle layer (mu and logvar)
- A decoder that reconstructs the molecular graph
- A property predictor that estimates PAMPA values
