'''
plots given names of models and datasets only those matching criteria
'''

import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
# Base directory containing all your 'wandb_data_csv_*' folders
BASE_DIR = "." 

# The prefix of the folders to process
FOLDER_PREFIX = "wandb_data_csv_" 

# The column to use as the x-axis (from your W&B data)
STEP_COLUMN = "_step"

# --- NEW: Add a string to ignore ---
NOPLOT_TAG = "NOPLOT"
# --- End Configuration ---

def should_plot_run(run_name):
    """
    Returns True if the run name matches our filter criteria,
    False otherwise.
    """
    run_name_lower = run_name.lower()
    
    if "llama" in run_name_lower and "edu" in run_name_lower:
        return True
    if "mistral" in run_name_lower and "edu" in run_name_lower:
        return True
    if "dolphin" in run_name_lower and "edu" in run_name_lower:
        return True
        
    return False

def filter_and_plot_runs(folder_path):
    """
    Processes a single folder, finds runs matching our filter,
    and saves 'test' and 'train' plots for them.
    """
    folder_name = os.path.basename(folder_path)
    parent_folder_name = os.path.basename(os.path.dirname(folder_path))
    
    # Create a clean title prefix (e.g., "wandb_data_csv_coala1 / DPO")
    if folder_name == parent_folder_name or parent_folder_name == ".":
         title_prefix = f"{folder_name}"
    else:
         title_prefix = f"{parent_folder_name} / {folder_name}"
         
    print(f"\n--- Processing: {title_prefix} ---")

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print("  No CSV files found. Skipping.")
        return

    # --- Part 1: Load and Melt Data for ONLY the Filtered Runs ---
    all_test_dfs = []
    all_train_dfs = []
    test_metric_names = set()
    train_metric_names = set()

    runs_plotted_count = 0

    for csv_path in csv_files:
        try:
            run_name = os.path.basename(csv_path).replace(".csv", "")
            
            # --- NEW: Check for "NOPLOT" string ---
            if NOPLOT_TAG in run_name:
                print(f"    Skipping {run_name}: Contains 'NOPLOT' tag.")
                continue
            
            # --- Check if this run matches our filter ---
            if not should_plot_run(run_name):
                continue
            
            runs_plotted_count += 1
            df = pd.read_csv(csv_path)
            
            # Find all 'test' or 'eval' columns
            test_cols = [col for col in df.columns if "test" in col.lower() or "eval" in col.lower()]
            # Find all 'train' columns
            train_cols = [col for col in df.columns if "train" in col.lower()]

            if STEP_COLUMN not in df.columns:
                print(f"    Skipping {run_name}: Missing '{STEP_COLUMN}' column.")
                continue
                
            # Melt and store 'test' data
            if test_cols:
                df_test_melted = df.melt(
                    id_vars=[STEP_COLUMN], value_vars=test_cols, 
                    var_name="Metric", value_name="Value"
                )
                df_test_melted["run_name"] = run_name
                all_test_dfs.append(df_test_melted)
                test_metric_names.update(test_cols)

            # Melt and store 'train' data
            if train_cols:
                df_train_melted = df.melt(
                    id_vars=[STEP_COLUMN], value_vars=train_cols, 
                    var_name="Metric", value_name="Value"
                )
                df_train_melted["run_name"] = run_name
                all_train_dfs.append(df_train_melted)
                train_metric_names.update(train_cols)
        
        except Exception as e:
            print(f"  [Error] Could not load data for plotting {run_name}: {e}")

    if runs_plotted_count == 0:
        print("  No runs matched the filter criteria. Skipping plots.")
        return

    # --- Part 2: Generate Test Plot ---
    if all_test_dfs:
        combined_test_df = pd.concat(all_test_dfs, ignore_index=True)
        
        # Determine Y-axis label
        test_ylabel = "Metric Value"
        if len(test_metric_names) == 1:
            test_ylabel = test_metric_names.pop()

        plt.figure(figsize=(16, 8))
        sns.set_theme(style="darkgrid")
        
        sns.lineplot(
            data=combined_test_df, 
            x=STEP_COLUMN, y="Value", 
            hue="run_name", style="Metric"
        )
        
        # Updated Title
        plt.title(f"{title_prefix}: Test/Eval Metrics (Filtered)", fontsize=16)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel(test_ylabel, fontsize=12) # Use dynamic label
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plot_filename = os.path.join(folder_path, f"{folder_name}_FILTERED_test_metrics.pdf")
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        print(f"  Saved filtered test plot to: {plot_filename}")
    else:
        print("  No 'test' or 'eval' data found to plot for filtered runs.")

    # --- Part 3: Generate Train Plot ---
    if all_train_dfs:
        combined_train_df = pd.concat(all_train_dfs, ignore_index=True)

        # Determine Y-axis label
        train_ylabel = "Metric Value"
        if len(train_metric_names) == 1:
            train_ylabel = train_metric_names.pop()

        plt.figure(figsize=(16, 8))
        sns.set_theme(style="darkgrid")

        sns.lineplot(
            data=combined_train_df, 
            x=STEP_COLUMN, y="Value", 
            hue="run_name", style="Metric"
        )

        # Updated Title
        plt.title(f"{title_prefix}: Train Metrics (Filtered)", fontsize=16)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel(train_ylabel, fontsize=12) # Use dynamic label
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plot_filename = os.path.join(folder_path, f"{folder_name}_FILTERED_train_metrics.pdf")
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        print(f"  Saved filtered train plot to: {plot_filename}")
    else:
        print("  No 'train' data found to plot for filtered runs.")

# --- Main script execution ---
if __name__ == "__main__":
    folders_to_process = glob.glob(os.path.join(BASE_DIR, f"{FOLDER_PREFIX}*"))
    folders_to_process = [f for f in folders_to_process if os.path.isdir(f)]

    if not folders_to_process:
        print(f"No folders found in '{BASE_DIR}' with prefix '{FOLDER_PREFIX}'.")
    else:
        print(f"Found {len(folders_to_process)} folders to process.")
        for folder_path in sorted(folders_to_process):
            
            dpo_path = os.path.join(folder_path, "DPO")
            orpo_path = os.path.join(folder_path, "ORPO")
            
            subfolders_found = []
            if os.path.isdir(dpo_path):
                subfolders_found.append(dpo_path)
            if os.path.isdir(orpo_path):
                subfolders_found.append(orpo_path)

            if subfolders_found:
                for subfolder_path in subfolders_found:
                    filter_and_plot_runs(subfolder_path)
            else:
                filter_and_plot_runs(folder_path)
    
    print("\nAll processing complete.")