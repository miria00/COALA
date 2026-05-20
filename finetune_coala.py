'''
this file loads the data, and runs the COALA finetuning stage
this file contains TO DOs
TO DO: change features loading to allow new dynamic features, instead of just on disc

example output: 
Fine-tuning completed in 2.21 seconds
Fine-tuned model saved to /home/miria/CVXDPO/Finetuned_cvxmlp_custom_inference_ready/custom_finetuned_cvx_mlp.pkl
Fine-tuning results saved to /home/miria/CVXDPO/Finetuned_cvxmlp_custom_inference_ready/finetuning_results.pkl

Comparison of original and fine-tuned theta2:
Original theta2 min/max/mean: -0.137282/0.137303/-0.000006
Fine-tuned theta2 min/max/mean: -0.311286/0.308702/0.003083
Delta magnitude: 0.628745
output directory is:  /home/miria/CVXDPO/Finetuned_cvxmlp_custom_inference_ready

beta=2.5, margin=1.0, gamma=0.5
'''
'''
finetune_coala.py needs to accept model_path and output_dir arguments for use with run_coala_pipeline_simple.sh

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()
model_path = args.model_path

'''


import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
import optax
import wandb  
from typing import NamedTuple
from jax.nn import relu
from functools import partial
import time
import argparse

# Import the load_data function from gpt2_dataloader.py
from solve.utils.gpt2_dataloader import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--cronos_training_time', type=float, default=0.0, help='Training time from cronos phase')
parser.add_argument('--cronos_tflops', type=float, default=0.0, help='Estimated TFLOPS from cronos phase')
parser.add_argument('--model_name', type=str, default=None, help='Model name key for gpt2_dataloader (overrides auto-detection from pkl filename)')
#parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()
model_path = args.model_path
cronos_training_time = args.cronos_training_time
cronos_tflops = args.cronos_tflops
#output_dir = args.output_dir


# Class for storing fine-tuning results
class FineTunedResults(NamedTuple):
    original_model_path: str
    finetuned_model_path: str
    final_loss: float
    original_theta2: jnp.ndarray
    finetuned_theta2: jnp.ndarray

def finetune_coala(model_path, output_dir, model_name="gpt2_imdb", data_seed=1024, learning_rate=1e-4, num_epochs=350, beta=1.0, gamma=0.5, batch_size=128):
    """
    - Loads the trained convex model
    - Freezes the first layer (theta1)
    - Fine-tunes only the second layer (theta2)
    - Uses the CVX-DPO loss from equation 7 in overleaf
    
    Args:
        model_path: Path to the trained model
        output_dir: Directory to save fine-tuned model
        model_name: Name of the model to load data for (used by load_data function)
        data_seed: Random seed for data loading
        learning_rate: Learning rate for AdamW optimizer
        num_epochs: Number of training epochs
        beta: Scaling parameter for the log-ratio term
        gamma: Offset parameter
        batch_size: Batch size for training
    """
    # Initialize wandb
    run = wandb.init(
        project="Neurips_coalaJuly",
        name=f"finetune_{os.path.basename(model_path)}",
        config={
            "model_name": model_name,
            "data_seed": data_seed,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "beta": beta,
            "gamma": gamma,
            "batch_size": batch_size,
            "model_path": model_path,
            "output_dir": output_dir,
            "cronos_training_time": cronos_training_time,
            "cronos_tflops": cronos_tflops
        },
        resume="allow"
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trained model
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Verify theta1 and theta2 are available
    if not hasattr(model, 'theta1') or not hasattr(model, 'theta2'):
        raise ValueError("Model does not have theta1 and theta2 attributes")
    
    # Get the dimensions
    feature_dim = model.theta1.shape[0]
    num_neurons = model.theta1.shape[1] # for some reaon this gives 20 when it should be 10
    
    print(f"Model has feature dimension {feature_dim} and {num_neurons} neurons")
    wandb.log({"feature_dim": feature_dim, "num_neurons": num_neurons})
    
    # Log model structure info to wandb
    wandb.log({
        "theta1_shape": model.theta1.shape,
        "theta2_shape": model.theta2.shape,
        "theta1_mean": float(np.mean(model.theta1)),
        "theta1_std": float(np.std(model.theta1)),
        "theta2_mean": float(np.mean(model.theta2)),
        "theta2_std": float(np.std(model.theta2)),
    })
    
    # Load data using the load_data function from gpt2_dataloader.py
    print(f"Loading data for model {model_name} with seed {data_seed}")
    #Atr, ytr, Atst, ytst, ntr, ntst = load_data(model_name, data_seed)
    Atr, ytr, Atst, ytst, ntr, ntst = load_data(model_name, data_seed, caller_script="finetune")

    # Log data loading info
    wandb.log({
        "train_samples": int(ntr),
        "test_samples": int(ntst),
        "train_data_shape": Atr.shape,
        "test_data_shape": Atst.shape
    })
    
    # Convert data to chosen/rejected pairs format for DPO
    print("Converting data to preference pairs...")
    
    # Get positive and negative samples from the training data
    pos_indices = jnp.where(ytr == 1)[0]
    neg_indices = jnp.where(ytr == -1)[0]
    
    # Determine number of pairs (minimum of positive and negative samples)
    num_pairs = min(len(pos_indices), len(neg_indices))
    print(f"Creating {num_pairs} preference pairs")
    wandb.log({"num_preference_pairs": num_pairs})
    
    # Select random samples if we have more than we need
    if num_pairs > 2000:  # Optional: limit to a specific number of pairs
        pos_indices = jnp.array(np.random.choice(pos_indices, 2000, replace=False))
        neg_indices = jnp.array(np.random.choice(neg_indices, 2000, replace=False))
        num_pairs = 2000
    
    # Create the chosen (positive) and rejected (negative) datasets
    X_chosen = Atr[pos_indices[:num_pairs]]
    X_rejected = Atr[neg_indices[:num_pairs]]
    
    # Also create test sets for reward margin evaluation
    test_pos_indices = jnp.where(ytst == 1)[0]
    test_neg_indices = jnp.where(ytst == -1)[0]
    
    # Limit test pairs to a reasonable number for evaluation
    test_num_pairs = min(len(test_pos_indices), len(test_neg_indices), 500)
    test_pos_indices = jnp.array(np.random.choice(test_pos_indices, test_num_pairs, replace=False))
    test_neg_indices = jnp.array(np.random.choice(test_neg_indices, test_num_pairs, replace=False))
    
    X_test_chosen = Atst[test_pos_indices]
    X_test_rejected = Atst[test_neg_indices]
    
    print(f"X_chosen shape: {X_chosen.shape}, X_rejected shape: {X_rejected.shape}")
    print(f"X_test_chosen shape: {X_test_chosen.shape}, X_test_rejected shape: {X_test_rejected.shape}")
    
    # Get initial weights (freeze theta1, but make a copy of theta2 for fine-tuning)
    theta1 = model.theta1  # This will be frozen
    original_theta2 = model.theta2.copy()  # Make a copy for comparison
    theta2 = jnp.array(model.theta2)  # Convert to JAX array for optimization
    
    # compute features using the first layer (frozen theta1)
    def compute_features(X):
        """Compute features using the frozen first layer (theta1)"""
        return relu(X @ theta1)
    
    # Function to compute reward margin (log ratio between chosen and rejected) #### synonymous with dpo
    def compute_reward_margins(theta2, chosen_features, rejected_features, beta=2.5):
        """
        Compute reward margins in the same scale as DPO/ORPO (values between 0 and 1).
        
        This computes the probability that the chosen example is preferred over the rejected example,
        which is the same scale used by DPO and ORPO reward margins.
        
        Args:
            theta2: Current theta2 weights
            chosen_features: Features from chosen examples
            rejected_features: Features from rejected examples
            beta: Scaling parameter (not used directly for the probability margin)
            
        Returns:
            reward_margins: Reward margins as probabilities (0-1 scale)
            mean_reward_margin: Mean reward margin
            min_reward_margin: Minimum reward margin
            max_reward_margin: Maximum reward margin
        """
        chosen_logits = chosen_features @ theta2
        rejected_logits = rejected_features @ theta2
        
        # Compute raw logit differences
        logit_diff = chosen_logits - rejected_logits
        
        # Convert to probability scale (0-1), same as DPO/ORPO reward margins
        # This represents the probability that chosen > rejected according to the preference model
        reward_margins = 1 / (1 + jnp.exp(-logit_diff))
        
        return reward_margins, jnp.mean(reward_margins), jnp.min(reward_margins), jnp.max(reward_margins)
    
    # Precompute features since theta1 is frozen
    print("Precomputing features for chosen and rejected examples...")
    chosen_features = compute_features(X_chosen)
    rejected_features = compute_features(X_rejected)
    
    # Precompute test features for reward margin evaluation
    test_chosen_features = compute_features(X_test_chosen)
    test_rejected_features = compute_features(X_test_rejected)
    
     # Compute initial reward margins on train and test data
    train_margins, train_mean_margin, train_min_margin, train_max_margin = compute_reward_margins(
        theta2, chosen_features, rejected_features)
    
    test_margins, test_mean_margin, test_min_margin, test_max_margin = compute_reward_margins(
        theta2, test_chosen_features, test_rejected_features)
    
    print(f"Initial train reward margin: Mean={train_mean_margin:.4f}, Min={train_min_margin:.4f}, Max={train_max_margin:.4f}")
    print(f"Initial test reward margin: Mean={test_mean_margin:.4f}, Min={test_min_margin:.4f}, Max={test_max_margin:.4f}")
    
    # Log initial reward margins to wandb - Fixed to use epoch=0
    wandb.log({
        "epoch": 0,  # Fixed: using explicit epoch=0 instead of undefined 'epoch'
        "train_mean_reward_margin": float(train_mean_margin),
        "train_min_reward_margin": float(train_min_margin),
        "train_max_reward_margin": float(train_max_margin),
        "test_mean_reward_margin": float(test_mean_margin),
        "test_min_reward_margin": float(test_min_margin),
        "test_max_reward_margin": float(test_max_margin),
    })

    # Create dataset with shuffled indices
    num_samples = len(X_chosen)
    indices = np.arange(num_samples)
    
    def loss_fn(theta2, chosen_batch, rejected_batch):
        """
        Implements the enhanced CVX-DPO loss:
        - Increased beta for stronger preference signal
        - Added margin term to create larger separation between chosen and rejected examples
        """
        chosen_logits = chosen_batch @ theta2
        rejected_logits = rejected_batch @ theta2
        
        # Increased beta (try 2.0-5.0 instead of the default 1.0)
        #beta = 2.5  # Adjust this value based on your experiments
        
        # Add a margin term to create more separation
        margin = 1.0  # This pushes the model to have a larger gap between chosen and rejected
        
        # The enhanced log ratio term
        log_ratio = beta * (chosen_logits - rejected_logits) - gamma
        
        # The modified CVX-DPO loss with margin
        # This encourages log_ratio to be at least 'margin' larger than 0
        loss = jnp.mean(jnp.log(1 + jnp.exp(margin - log_ratio)))
        
        return loss
    
    # Create optimizer (AdamW as mentioned in the paper)
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(theta2)
    
    # Compile the update function with jit (this can be much faster)
    @jax.jit
    def update_step(theta2, opt_state, chosen_batch, rejected_batch):
        """Single optimization step"""
        loss_val, grads = jax.value_and_grad(loss_fn)(theta2, chosen_batch, rejected_batch)
        # Add theta2 parameter here:
        updates, opt_state = optimizer.update(grads, opt_state, theta2)  # Pass theta2 as the 3rd argument
        theta2 = optax.apply_updates(theta2, updates)
        return theta2, opt_state, loss_val
    
    # Training loop
    print(f"Starting fine-tuning for {num_epochs} epochs...")
    losses = []
    best_loss = float('inf')
    best_theta2 = theta2
    
    start_time = time.time()
    
    # For reward margin plotting
    train_margins_history = []
    test_margins_history = []
    train_mean_margins = []
    test_mean_margins = []
    
    for epoch in range(num_epochs):
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Mini-batch training
        epoch_losses = []
        for i in range(0, num_samples, batch_size):
            # Get batch indices
            batch_idx = indices[i:min(i+batch_size, num_samples)]
            
            # Create batches
            chosen_batch = chosen_features[batch_idx]
            rejected_batch = rejected_features[batch_idx]
            
            # Update step
            theta2, opt_state, loss_val = update_step(theta2, opt_state, chosen_batch, rejected_batch)
            epoch_losses.append(loss_val)
        
        # Compute average loss for the epoch
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_theta2 = theta2
        
        # Compute reward margins on train and test data every few epochs
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            # Compute reward margins on train data
            train_margins, train_mean_margin, train_min_margin, train_max_margin = compute_reward_margins(
                theta2, chosen_features, rejected_features)
            
            # Compute reward margins on test data
            test_margins, test_mean_margin, test_min_margin, test_max_margin = compute_reward_margins(
                theta2, test_chosen_features, test_rejected_features)
            
            # Store for history
            train_margins_history.append(np.array(train_margins))
            test_margins_history.append(np.array(test_margins))
            train_mean_margins.append(float(train_mean_margin))
            test_mean_margins.append(float(test_mean_margin))
            
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")
            print(f"  Train reward margin: Mean={train_mean_margin:.4f}, Min={train_min_margin:.4f}, Max={train_max_margin:.4f}")
            print(f"  Test reward margin: Mean={test_mean_margin:.4f}, Min={test_min_margin:.4f}, Max={test_max_margin:.4f}")
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "loss": float(avg_loss),
                "train_mean_reward_margin": float(train_mean_margin),
                "train_min_reward_margin": float(train_min_margin),
                "train_max_reward_margin": float(train_max_margin),
                "test_mean_reward_margin": float(test_mean_margin),
                "test_min_reward_margin": float(test_min_margin),
                "test_max_reward_margin": float(test_max_margin),
                "theta2_mean": float(np.mean(np.array(theta2))),
                "theta2_std": float(np.std(np.array(theta2))),
                "theta2_max": float(np.max(np.array(theta2))),
                "theta2_min": float(np.min(np.array(theta2))),
            })
            
            # Create histograms of reward margins for visualization
            if epoch % 50 == 0 or epoch == num_epochs - 1:
                # Plot train reward margin distribution
                train_margin_hist = wandb.Histogram(np.array(train_margins))
                test_margin_hist = wandb.Histogram(np.array(test_margins))
                
                wandb.log({
                    "train_reward_margin_hist": train_margin_hist,
                    "test_reward_margin_hist": test_margin_hist,
                    "epoch_hist": epoch
                })
    
    # Final reward margin evaluation
    final_train_margins, final_train_mean_margin, final_train_min_margin, final_train_max_margin = compute_reward_margins(
        best_theta2, chosen_features, rejected_features)
    
    final_test_margins, final_test_mean_margin, final_test_min_margin, final_test_max_margin = compute_reward_margins(
        best_theta2, test_chosen_features, test_rejected_features)
    
    # Log final train and test margin distributions
    wandb.log({
        "final_train_reward_margin_hist": wandb.Histogram(np.array(final_train_margins)),
        "final_test_reward_margin_hist": wandb.Histogram(np.array(final_test_margins)),
    })
    
    elapsed_time = time.time() - start_time
    print(f"Fine-tuning completed in {elapsed_time:.2f} seconds")
    
    # Update the model with the best fine-tuned theta2
    model.theta2 = np.array(best_theta2)
    
    # Save the fine-tuned model
    finetuned_model_path = os.path.join(output_dir, os.path.basename(model_path).replace("trained", "finetuned"))
    with open(finetuned_model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Fine-tuned model saved to {finetuned_model_path}")
    
    # Calculate estimated TFLOPS for finetuning
    # This is a rough estimate based on the number of operations in the model
    # You might want to adjust this calculation based on your specific hardware
    def estimate_finetuning_tflops(num_neurons, feature_dim, num_samples, batch_size, num_epochs, elapsed_time):
        """
        Estimate TFLOPs used during finetuning
        
        Args:
            num_neurons: Number of neurons in the model
            feature_dim: Feature dimension
            num_samples: Number of training samples
            batch_size: Batch size used during training
            num_epochs: Number of training epochs
            elapsed_time: Total training time in seconds
            
        Returns:
            Estimated TFLOPs
        """
        operations_per_sample = 2 * (feature_dim * num_neurons + num_neurons) # Forward + backward pass
        total_operations = operations_per_sample * num_samples * num_epochs
        tflops = total_operations / (elapsed_time * 1e12)
        return tflops
    
    finetune_tflops = estimate_finetuning_tflops(
        num_neurons, 
        feature_dim, 
        num_samples, 
        batch_size, 
        num_epochs, 
        elapsed_time
    )
    
    # Total TFLOPS and time across both phases
    total_tflops = cronos_tflops + finetune_tflops
    total_time = cronos_training_time + elapsed_time
    
    # save results
    results = FineTunedResults(
        original_model_path=model_path,
        finetuned_model_path=finetuned_model_path,
        final_loss=best_loss,
        original_theta2=original_theta2,
        finetuned_theta2=np.array(best_theta2)
    )
    
    results_path = os.path.join(output_dir, "finetuning_complete.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Fine-tuning results saved to {results_path}")
    
    # OG versus fine-tuned theta2
    print("\nComparison of original and fine-tuned theta2:")
    print(f"Original theta2 min/max/mean: {np.min(original_theta2):.6f}/{np.max(original_theta2):.6f}/{np.mean(original_theta2):.6f}")
    print(f"Fine-tuned theta2 min/max/mean: {np.min(best_theta2):.6f}/{np.max(best_theta2):.6f}/{np.mean(best_theta2):.6f}")
    delta_magnitude = np.linalg.norm(best_theta2 - original_theta2)
    print(f"Delta magnitude: {delta_magnitude:.6f}")
    
    # Print reward margin improvement
    print("\nReward margin improvement:")
    print(f"Train: Initial={train_mean_margins[0]:.4f} → Final={final_train_mean_margin:.4f} (Δ={final_train_mean_margin-train_mean_margins[0]:.4f})")
    print(f"Test: Initial={test_mean_margins[0]:.4f} → Final={final_test_mean_margin:.4f} (Δ={final_test_mean_margin-test_mean_margins[0]:.4f})")
    
    # Create a final reward margin progress chart
    margin_data = []
    for i, (train_margin, test_margin) in enumerate(zip(train_mean_margins, test_mean_margins)):
        margin_data.append({
            "epoch": i * 10,  # We computed margins every 10 epochs
            "train_margin": train_margin,
            "test_margin": test_margin
        })
    
    margin_table = wandb.Table(data=margin_data, columns=["epoch", "train_margin", "test_margin"])
    wandb.log({"reward_margin_progress": margin_table})
    
    # Log final metrics to wandb
    wandb.log({
        "finetune_time": elapsed_time,
        "finetune_tflops": finetune_tflops,
        "total_time": total_time,
        "total_tflops": total_tflops,
        "final_loss": float(best_loss),
        "theta2_delta_magnitude": float(delta_magnitude),
        "original_theta2_mean": float(np.mean(original_theta2)),
        "original_theta2_min": float(np.min(original_theta2)),
        "original_theta2_max": float(np.max(original_theta2)),
        "finetuned_theta2_mean": float(np.mean(best_theta2)),
        "finetuned_theta2_min": float(np.min(best_theta2)),
        "finetuned_theta2_max": float(np.max(best_theta2)),
        "initial_train_reward_margin": float(train_mean_margins[0]),
        "final_train_reward_margin": float(final_train_mean_margin),
        "train_reward_margin_improvement": float(final_train_mean_margin - train_mean_margins[0]),
        "initial_test_reward_margin": float(test_mean_margins[0]),
        "final_test_reward_margin": float(final_test_mean_margin),
        "test_reward_margin_improvement": float(final_test_mean_margin - test_mean_margins[0]),
    })
    
    # Create a reward margin improvement summary
    wandb.log({
        "reward_margin_summary": wandb.Table(
            data=[
                ["Train", float(train_mean_margins[0]), float(final_train_mean_margin), float(final_train_mean_margin - train_mean_margins[0])],
                ["Test", float(test_mean_margins[0]), float(final_test_mean_margin), float(final_test_mean_margin - test_mean_margins[0])]
            ],
            columns=["Dataset", "Initial Margin", "Final Margin", "Improvement"]
        )
    })
    
    # Finish wandb run
    wandb.finish()
    
    return results

if __name__ == "__main__":
    # To do: pass these in from shell
    # INPUT: path to trained model

    #model_path = "/home/miria/CVXDPO/cvxNN_trained_gpt2_imdb/gpt2_imdb_trained_cvx_mlp.pkl"
    #model_path = "/home/miria/CVXDPO/cvxNN_trained_gpt2_attn_edu/gpt2_attn_edu_trained_cvx_mlp.pkl"
    #model_path = "/home/miria/CVXDPO/cvxNN/cvxNN_trained_mistral_imdb/mistral_imdb_trained_cvx_mlp.pkl"

    # Use --model_name if provided, otherwise auto-detect from pkl filename
    if args.model_name:
        model_name = args.model_name
    else:
        # Extract model name from the filename (first 2 segments)
        model_filename = os.path.basename(model_path)  # e.g., "gpt2_attn_ultra_trained_cvx_mlp.pkl"
        model_parts = model_filename.split('_')  # ['gpt2', 'attn', 'ultra', 'trained', 'cvx', 'mlp.pkl']

        # Get the first 2 word segments if available
        if len(model_parts) >= 3:
            model_name = '_'.join(model_parts[:2])  # e.g., "gpt2_attn_ultra"
        else:
            # Fallback if there are fewer than 3 segments
            model_name = '_'.join(model_parts).split('.')[0]  # Use all parts without extension

    # Dynamically create output directory
    output_dir = f"/home/miria/CVXDPO/finetuned/Finetuned_cvx_{model_name}_inference_ready"
    
    print(f"Using model name: {model_name}")
    print(f"Output directory: {output_dir}")
    
    # finetune designed to be in same spirit as HuggingFace DPOtrainer
    results = finetune_coala(
        model_path=model_path,
        output_dir=output_dir,
        model_name=model_name,  # Add this parameter
        learning_rate=1e-4,
        num_epochs=500, # 300 
        beta=2.5, # default beta = 1.0, added margin = 1.0
        gamma=0.5,
        batch_size=128
    )

    #print("output directory is: ", output_dir)
    print(f"[FINETUNE_OUTPUT_DIR]{output_dir}") # easier to grep in batch, brute force way