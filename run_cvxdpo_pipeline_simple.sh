#!/bin/bash

# This version does not contain full logging to wandb 
# there exists two output directories for each model: "cvxNN_trained_${model_name}" and "finetuned_cvx_${model_name}_inference_ready"
# output log file is "cvxdpo_pipeline_sft_log_datasplit.txt"

export WANDB_API_KEY="14eb70e2cb32fb3e6701239a44ae3ccbbfd6b8bf"


# list of all SFT trained first models 
# List of model_names to process
model_names=(
    "dolphin_imdb"
    # "dolphin_edu"
    # "dolphin_ultra"
    # "llama_edu"
    # "llama_imdb"
    # "llama_ultra"
    # "mistral_edu"
    # "mistral_imdb"
    # "mistral_ultra"
    # "distilgpt2_edu"
    # "distilgpt2_imdb"
    # "distilgpt2_ultra"
    # "gpt2_edu"
    # "gpt2_imdb"
    # "gpt2_ultra"
)

# Paths
CRONOS_SCRIPT="cronos_trainer_.py"
FINETUNE_SCRIPT="finetune_cvxdpo_.py"
LOG_FILE="cvxdpo_pipeline_sft_log_datasplit.txt"




# TFLOPs estimation helper (for NVIDIA RTX 4090 @ 70% bf16 Tensor Core efficiency)
estimate_tflops() {
    duration=$1  # in seconds
    gflops_per_sec=231000  # 231 TFLOPs = 70% of 330 peak bf16 performance
    tflops_used=$(echo "scale=2; $gflops_per_sec * $duration / 1000" | bc)
    echo "$tflops_used"
}

# Clean log file
echo "--- CVX-DPO FULL PIPELINE RUN LOG ---" > $LOG_FILE
echo "Start time: $(date)" >> $LOG_FILE
echo "" >> $LOG_FILE

for model_name in "${model_names[@]}"; do
    echo 
    echo
    echo "----------------------------------------"
    echo "Processing model: $model_name"
    echo "----------------------------------------"

    echo "[$model_name] Start time: $(date)" >> $LOG_FILE
    start_time=$(date +%s)

    # Set output directory from cronos trainer
    output_dir="/home/miria/CVXDPO/cvxNN/cvxNN_trained_${model_name}"

    # Run CRONOS trainer
    echo "Running CRONOS trainer for $model_name..."
    python $CRONOS_SCRIPT --model_name $model_name

    # Append CRONOS results to log
    cronos_results_file="${output_dir}/cronos_results.txt"
    if [ -f "$cronos_results_file" ]; then
        echo "[$model_name] CRONOS Results:" >> $LOG_FILE
        cat "$cronos_results_file" >> $LOG_FILE
    else
        echo "[$model_name] CRONOS Results: Not Found" >> $LOG_FILE
    fi

    # Construct the model path (model_name.pkl inside output dir)
    model_path="${output_dir}/${model_name}_trained_cvx_mlp.pkl"

    # Run fine-tuning and log full output to both console and log file
    echo "Running CVX-DPO fine-tuning for $model_name..."
    {
        echo "========== [FINE-TUNE LOG: $model_name] =========="
        python $FINETUNE_SCRIPT --model_path "$model_path"
        echo "========== [END FINE-TUNE LOG: $model_name] =========="
    } | tee -a "$LOG_FILE"

    # Extract the fine-tuned model directory from the log
    finetuned_output_dir=$(grep '\[FINETUNE_OUTPUT_DIR\]' "$LOG_FILE" | tail -n1 | sed 's/\[FINETUNE_OUTPUT_DIR\]//g' | xargs)

    # Timing and TFLOPs usage
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    duration_min=$(echo "$duration / 60" | bc)
    tflops=$(estimate_tflops $duration)

    echo "[$model_name] End time: $(date)" >> $LOG_FILE
    echo "[$model_name] Duration: $duration seconds (~${duration_min} min)" >> $LOG_FILE
    echo "[$model_name] Approx TFLOPs used: $tflops" >> $LOG_FILE
    echo "[$model_name] Trained model: $model_path" >> $LOG_FILE
    echo "[$model_name] Finetuned model dir: $finetuned_output_dir" >> $LOG_FILE
    echo "----------------------------------------" >> $LOG_FILE
    echo "" >> $LOG_FILE
done

echo "Pipeline completed at: $(date)" >> $LOG_FILE
