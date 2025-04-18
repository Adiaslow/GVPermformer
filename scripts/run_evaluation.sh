#!/bin/bash
# scripts/run_evaluation.sh
# Script to run model evaluation with configurable parameters

# Default parameters
DATA_DIR="data/processed"
MODEL_PATH="models/best_model.pt"
OUTPUT_DIR="evaluation_results"
BATCH_SIZE=32

# Automatically detect device
if [ -x "$(command -v nvidia-smi)" ]; then
  DEVICE="cuda"
else
  DEVICE="cpu"
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --data_dir)
      DATA_DIR="$2"
      shift
      shift
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift
      shift
      ;;
    --device)
      DEVICE="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: ./run_evaluation.sh [--data_dir DIR] [--model_path PATH] [--output_dir DIR] [--batch_size SIZE] [--device DEVICE]"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Display the parameters
echo "Running evaluation with the following parameters:"
echo "  Data directory: $DATA_DIR"
echo "  Model path: $MODEL_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Device: $DEVICE"

# Run the evaluation
echo "Starting evaluation..."
python scripts/evaluate_model.py \
  --data_dir "$DATA_DIR" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE"

# Check if the evaluation was successful
if [ $? -eq 0 ]; then
  echo "Evaluation completed successfully!"
  echo "Results can be found in: $OUTPUT_DIR"
  echo "View the evaluation report at: $OUTPUT_DIR/evaluation_report.md"
else
  echo "Evaluation failed. Please check the logs for details."
fi 