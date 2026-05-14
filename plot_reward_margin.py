'''
Plot Reward Margin over Training Steps for COALA
This script is not used as of Nov 16, 2025
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---

# 1. Set the style to look like the ICLR paper
# (Your example plot uses LaTeX fonts, so this is a close approximation)
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# 2. Define the paths to your metric files
#    Replace these with the actual paths from your finetune script output
base_output_dir = "/home/miria/CVXDPO/finetuned64split"

# Define models and their corresponding CSV files
models_to_plot = {
    "mistral": os.path.join(base_output_dir, "Finetuned_cvx_mistral_imdb_inference_ready", "training_metrics.csv"),
    "dolphin": os.path.join(base_output_dir, "Finetuned_cvx_dolphin_imdb_inference_ready", "training_metrics.csv"),
    "llama": os.path.join(base_output_dir, "Finetuned_cvx_llama_imdb_inference_ready", "training_metrics.csv"),
}

# Define colors (optional, to match your example)
colors = {
    "mistral": "green",
    "dolphin": "C0", # Default blue
    "llama": "C1"  # Default orange
}

# --- Plotting ---

plt.figure(figsize=(10, 6))

for model_name, csv_path in models_to_plot.items():
    if not os.path.exists(csv_path):
        print(f"Warning: File not found, skipping: {csv_path}")
        continue
        
    # Load the metrics data
    df = pd.read_csv(csv_path)
    
    # Plot 'step' vs. 'val_reward_margin'
    plt.plot(
        df["step"], 
        df["val_reward_margin"], 
        label=model_name,
        color=colors.get(model_name)
    )

# --- Finalize Plot ---
plt.title("COALA Stage 2: Reward Margin Over Training", fontsize=16)
plt.xlabel("Step", fontsize=12)
plt.ylabel("Reward margin", fontsize=12)
plt.legend(fontsize=11)

# Save the plot
output_plot_path = "coala_reward_margin_plot.pdf"
plt.savefig(output_plot_path, bbox_inches='tight')
print(f"Plot saved to {output_plot_path}")

# Display the plot
plt.show()