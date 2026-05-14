#!/bin/bash

# to do: change this to 4090 machine
DATASET="/home/ubuntu/arizonafiles/cvxdpo/train_dataset_ultra.json" # 
MODEL="openai-community/gpt2"
CHECKPOINT_DIR="checkpoints/sft_model"
FEATURE_DIR="features"
CVX_OUTPUT_DIR="cvx_model"

# ----------------------------------------------

echo "==== Step 1: Train LM body with SFTTrainer ===="
python train_LM_body.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$CHECKPOINT_DIR"

echo "==== Step 2: Extract features using checkpoint ===="
python extract.py \
    --model_checkpoint "$CHECKPOINT_DIR" \
    --dataset "$DATASET" \
    --output_dir "$FEATURE_DIR"

echo "==== Step 3: Train CVX-DPO binary classifier ===="
python cronos_trainer.py \
    --feature_dir "$FEATURE_DIR" \
    --output_dir "$CVX_OUTPUT_DIR"

echo "==== (Optional) Check if model saved correctly ===="
python test_model_weights.py \
    --model_path "$CVX_OUTPUT_DIR/custom_trained_cvx_mlp.pkl"

echo "==== Step 4: Fine-tune with CVX-DPO loss ===="
python finetune_cvxdpo.py \
    --sft_checkpoint "$CHECKPOINT_DIR" \
    --theta2_path "$CVX_OUTPUT_DIR/custom_trained_cvx_mlp.pkl" \
    --dataset "$DATASET" \
    --output_dir "cvxdpo_finetune"