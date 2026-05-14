#!/bin/bash

# Extract features for Llama-3.2-3B models across all datasets
# Each model is paired with its matching dataset

SCRIPT="extract.py"
POOL="attn"
OUTPUT_BASE="/home/miria/COALA/extracted_features2803/batchsize32"

set -e
mkdir -p "$OUTPUT_BASE"

# Model-dataset pairs (model|dataset)
PAIRS=(
  "/home/miria/COALA/from_downloads/SFT_meta-llama_Llama-3.2-3B_helpsteer|/home/miria/COALA/datasets/helpsteer/"
  "/home/miria/COALA/from_downloads/SFT_meta-llama_Llama-3.2-3B_imdb|/home/miria/COALA/datasets/aclImdb/all/"
  "/home/miria/COALA/from_downloads/SFT_meta-llama_Llama-3.2-3B_ultra|/home/miria/COALA/datasets/ultra/"
  "/home/miria/COALA/from_downloads/SFT_meta-llama_Llama-3.2-3B_edu|/home/miria/COALA/datasets/edu/"
)

for PAIR in "${PAIRS[@]}"; do
  MODEL_PATH="${PAIR%%|*}"
  DATA_PATH="${PAIR##*|}"
  echo "Running extraction for model: $MODEL_PATH and dataset: $DATA_PATH"
  python "$SCRIPT" --model_path "$MODEL_PATH" --data_path "$DATA_PATH" --pool "$POOL" --output_base "$OUTPUT_BASE"
  echo "Finished: $MODEL_PATH x $DATA_PATH"
  echo "----------------------------------------"
done

echo "All extractions completed successfully!"
