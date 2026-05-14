#!/bin/bash

# Modified script to process SFT and non-SFT models separately

export WANDB_API_KEY="14eb70e2cb32fb3e6701239a44ae3ccbbfd6b8bf"

# Separate arrays for SFT and non-SFT models
sft_models=(
    "dolphin_imdb_sft"
    "dolphin_edu_sft"
    "dolphin_ultra_sft"
    "llama_edu_sft"
    "llama_imdb_sft"
    "llama_ultra_sft"
    "mistral_edu_sft"
    "mistral_imdb_sft"
    "mistral_ultra_sft"
    "distilgpt2_edu_sft" #
    "distilgpt2_imdb_sft" #
    "distilgpt2_ultra_sft"
    "gpt2_edu_sft"
    "gpt2_imdb_sft" #
    "gpt2_ultra_sft"
)

non_sft_models=(
    "dolphin_imdb"
    "dolphin_edu"
    "dolphin_ultra"
    "llama_edu"
    "llama_imdb"
    "llama_ultra"
    "mistral_edu"
    "mistral_imdb"
    "mistral_ultra"
    "distilgpt2_edu" #
    "distilgpt2_imdb"
    "distilgpt2_ultra"
    "gpt2_edu" #
    "gpt2_imdb"
    "gpt2_ultra" #
)

# Paths to scripts
CRONOS_SCRIPT="cronos_trainer.py"  
FINETUNE_SCRIPT="finetune_cvxdpo.py"  
TODAY=$(date +%Y%m%d)

# TFLOPs estimation helper
estimate_tflops() {
    duration=$1
    gflops_per_sec=231000
    tflops_used=$(echo "scale=2; $gflops_per_sec * $duration / 1000" | bc)
    echo "$tflops_used"
}

# Function to process a list of models
process_models() {
    local model_type=$1
    shift # Remove first argument (model_type) from the arguments
    local models=("$@")  # Now models contains the actual model names
    
    local log_file="Cvxdpo_pipeline_${model_type}_log_datasplit_${TODAY}.txt"
    
    {
        echo "--- CVX-DPO ${model_type} PIPELINE RUN LOG ---"
        echo "Start time: $(date)"
        echo "Processing ${#models[@]} ${model_type} models"
        echo ""
    } | tee "$log_file"
    
    for model_name in "${models[@]}"; do
        {
            echo 
            echo "----------------------------------------"
            echo "Processing ${model_type} model: $model_name"
            echo "----------------------------------------"
            echo "[$model_name] Start time: $(date)"
        } | tee -a "$log_file"

        pipeline_start_time=$(date +%s)

        # Set output directory
        output_dir="/home/miria/CVXDPO/cvxNN/cvxNN_trained_${model_name}"

        # Run CRONOS trainer
        {
            echo "Running CRONOS trainer for $model_name..."
            echo "========== [CRONOS LOG: $model_name] =========="
        } | tee -a "$log_file"
        
        cronos_start_time=$(date +%s)
        
        {
            if ! python $CRONOS_SCRIPT --model_name $model_name; then
                echo "[$model_name] ERROR: CRONOS training failed!"
                exit 1
            fi
            echo "========== [END CRONOS LOG: $model_name] =========="
        } 2>&1 | tee -a "$log_file"
        
        # Check if CRONOS failed
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "[$model_name] ERROR: CRONOS training failed!" | tee -a "$log_file"
            continue
        fi
        
        cronos_end_time=$(date +%s)
        cronos_duration=$((cronos_end_time - cronos_start_time))
        cronos_tflops=$(estimate_tflops $cronos_duration)

        # Check CRONOS results
        cronos_results_file="${output_dir}/cronos_results.txt"
        if [ -f "$cronos_results_file" ]; then
            {
                echo "[$model_name] CRONOS Results:"
                cat "$cronos_results_file"
                echo "[$model_name] CRONOS Training time: $cronos_duration seconds"
                echo "[$model_name] CRONOS Estimated TFLOPS: $cronos_tflops"
            } | tee -a "$log_file"
        else
            echo "[$model_name] ERROR: CRONOS results file not found!" | tee -a "$log_file"
            continue
        fi

        # Check model file
        model_path="${output_dir}/${model_name}_trained_cvx_mlp.pkl"
        if [ ! -f "$model_path" ]; then
            echo "[$model_name] ERROR: Trained model not found at $model_path" | tee -a "$log_file"
            continue
        fi

        # Run fine-tuning
        {
            echo "Running CVX-DPO fine-tuning for $model_name..."
            echo "========== [FINE-TUNE LOG: $model_name] =========="
        } | tee -a "$log_file"
        
        finetune_start_time=$(date +%s)
        
        {
            if ! python $FINETUNE_SCRIPT --model_path "$model_path" --cronos_training_time $cronos_duration --cronos_tflops $cronos_tflops; then
                echo "[$model_name] ERROR: Fine-tuning failed!"
                exit 1
            fi
            echo "========== [END FINE-TUNE LOG: $model_name] =========="
        } 2>&1 | tee -a "$log_file"
        
        # Check if fine-tuning failed
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "[$model_name] ERROR: Fine-tuning failed!" | tee -a "$log_file"
            continue
        fi
        
        finetune_end_time=$(date +%s)
        finetune_duration=$((finetune_end_time - finetune_start_time))
        finetune_tflops=$(estimate_tflops $finetune_duration)

        # Extract results and log timing
        finetuned_output_dir=$(grep '\[FINETUNE_OUTPUT_DIR\]' "$log_file" | tail -n1 | sed 's/\[FINETUNE_OUTPUT_DIR\]//g' | xargs)
        pipeline_end_time=$(date +%s)
        pipeline_duration=$((pipeline_end_time - pipeline_start_time))
        total_duration=$((cronos_duration + finetune_duration))
        total_tflops=$(echo "scale=2; $cronos_tflops + $finetune_tflops" | bc)

        {
            echo "[$model_name] Pipeline duration: $pipeline_duration seconds"
            echo "[$model_name] Total TFLOPS used: $total_tflops"
            echo "[$model_name] Trained model: $model_path"
            echo "[$model_name] Finetuned model dir: $finetuned_output_dir"
            echo ""
            echo "🎉 MODEL $model_name has finished CRONOS training and finetuning, ready for inference! 🎉"
            echo "----------------------------------------"
            echo ""
        } | tee -a "$log_file"
    done
    
    {
        echo "${model_type} pipeline completed at: $(date)"
        echo "Processed ${#models[@]} ${model_type} models"
    } | tee -a "$log_file"
}

# Log the user interaction
{
    echo "Which models would you like to process?"
    echo "1) SFT models only (15 models)"
    echo "2) Non-SFT models only (15 models)" 
    echo "3) Both (30 models)"
} | tee pipeline_user_interaction_${TODAY}.log

read -p "Enter your choice (1, 2, or 3): " choice

echo "User selected option: $choice" | tee -a pipeline_user_interaction_${TODAY}.log

case $choice in
    1)
        echo "Processing SFT models only..." | tee -a pipeline_user_interaction_${TODAY}.log
        process_models "sftbase" "${sft_models[@]}"
        ;;
    2)
        echo "Processing non-SFT models only..." | tee -a pipeline_user_interaction_${TODAY}.log
        process_models "base" "${non_sft_models[@]}"
        ;;
    3)
        echo "Processing SFT models first..." | tee -a pipeline_user_interaction_${TODAY}.log
        process_models "sftbase" "${sft_models[@]}"
        echo "Processing non-SFT models..." | tee -a pipeline_user_interaction_${TODAY}.log
        process_models "base" "${non_sft_models[@]}"
        ;;
    *)
        echo "Invalid choice. Exiting." | tee -a pipeline_user_interaction_${TODAY}.log
        exit 1
        ;;
esac

echo "All processing completed!" | tee -a pipeline_user_interaction_${TODAY}.log