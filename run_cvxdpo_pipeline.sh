#!/bin/bash

# List of model names to process
MODEL_NAMES=("gpt2_imdb" "gpt2_attn_ultra" "gpt2_attn_edu")

# Function to estimate TFLOPS usage (simplified approach)
estimate_tflops() {
    # Get GPU utilization percentage and memory usage during the run
    # This is a simplified estimation - would need more precise metrics for accurate measurement
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/NR}')
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/NR}')
    
    # RTX 4090 has around 82 TFLOPS for FP32
    # Rough estimation based on utilization percentage and runtime
    max_tflops=82
    elapsed=$1  # Time in seconds
    
    # Very rough estimation: tflops = max_tflops * (utilization/100) * (time in seconds)
    estimated_tflops=$(echo "$max_tflops * $gpu_util * $elapsed / 100" | bc -l)
    
    echo "$estimated_tflops"
}

# Create results file
RESULTS_FILE="cvxdpo_pipeline_results.csv"
echo "model_name,cronos_time,finetune_time,total_time,estimated_tflops,output_model" > $RESULTS_FILE

# For each model in the list
for model_name in "${MODEL_NAMES[@]}"; do
    echo "============================================================"
    echo "Processing model: $model_name"
    echo "============================================================"
    
    # Start monitoring GPU usage
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 1 > "gpu_usage_${model_name}.csv" &
    MONITOR_PID=$!
    
    # Start timing
    START_TIME=$(date +%s)
    
    # Step 1: Run cronos_trainer.py
    echo "Running cronos_trainer.py for $model_name..."
    
    # Create a temporary Python script to run cronos_trainer with the specific model
    cat > temp_cronos_script.py << EOF
import os
import jax
import numpy as np
import jax.numpy as jnp
from solve.optimizers.admm import admm
import pickle
from defrun import run, RunResults
import random
import pandas as pd

model_names = '$model_name'
OUTPUT_DIR = f'/home/miria/CVXDPO/cvxNN/cvxNN_trained_{os.path.basename(os.path.normpath(model_names))}'

cronos_params = dict(
    rank=20, beta=0.001, rho=0.0001,
    gamma_ratio=1, admm_iters=6, pcg_iters=32,
    check_opt=False
)

adamW_params = dict(optimizer='AdamW', gamma=10**-4, n_epoch=30, batch_size=1024)

opt_seed = 1024
data_seed = random.randint(1, 10)

results = run(model_names, cronos_params, adamW_params, opt_seed, data_seed, OUTPUT_DIR)

data = {
    "global_max_test_peak": [results.global_max_test_peak],
    "global_best_params": [results.global_best_params],
    "global_delta_test_peak": [results.global_delta_test_peak],
    "global_best_delta_params": [results.global_best_delta_params],
    "model_path": [results.model_path]
}

df = pd.DataFrame(data)
print(df)
print(f"Trained convex 2 layer model saved at: {results.model_path}")

# Save model_path to a file for the next step
with open('model_path.txt', 'w') as f:
    f.write(results.model_path)
EOF

    # Run the temporary script
    python temp_cronos_script.py
    
    # Get the end time for cronos
    CRONOS_END_TIME=$(date +%s)
    CRONOS_TIME=$((CRONOS_END_TIME - START_TIME))
    
    # Get the model path from the file
    MODEL_PATH=$(cat model_path.txt)
    
    # Step 2: Run finetune_cvxdpo.py
    echo "Running finetune_cvxdpo.py with model from $MODEL_PATH..."
    
    # Create output directory for fine-tuned model
    OUTPUT_DIR="/home/miria/CVXDPO/Finetuned_cvx_${model_name}_inference_ready"
    
    # Run finetune_cvxdpo.py
    python -c "
import os
from finetune_cvxdpo import finetune_cvxdpo

model_path = '$MODEL_PATH'
output_dir = '$OUTPUT_DIR'

print(f'Using model name: ${model_name}')
print(f'Output directory: {output_dir}')

results = finetune_cvxdpo(
    model_path=model_path,
    output_dir=output_dir,
    model_name='${model_name}',
    learning_rate=1e-4,
    num_epochs=300,
    beta=2.5,  # Higher beta value
    gamma=0.5,
    batch_size=128
)

print('output directory is: ', output_dir)
"
    
    # Get the end time for the entire pipeline
    END_TIME=$(date +%s)
    FINETUNE_TIME=$((END_TIME - CRONOS_END_TIME))
    TOTAL_TIME=$((END_TIME - START_TIME))
    
    # Stop monitoring GPU usage
    kill $MONITOR_PID
    
    # Estimate TFLOPS
    ESTIMATED_TFLOPS=$(estimate_tflops $TOTAL_TIME)
    
    # Record results
    echo "$model_name,$CRONOS_TIME,$FINETUNE_TIME,$TOTAL_TIME,$ESTIMATED_TFLOPS,$OUTPUT_DIR/$(basename $MODEL_PATH | sed 's/trained/finetuned/')" >> $RESULTS_FILE
    
    echo "Completed pipeline for $model_name in $TOTAL_TIME seconds"
    echo "CRONOS training: $CRONOS_TIME seconds"
    echo "Fine-tuning: $FINETUNE_TIME seconds"
    echo "Estimated TFLOPS: $ESTIMATED_TFLOPS"
    echo "============================================================"
done

# Clean up temporary files
rm -f temp_cronos_script.py model_path.txt

echo "All models processed. Results saved to $RESULTS_FILE"
cat $RESULTS_FILE