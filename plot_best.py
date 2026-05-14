'''
plots and returns a print out of the top 12 best runs in a given wandb_data_csv_* folder
used for sanity check 
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

# A prioritized list of metric columns to look for.
METRIC_CANDIDATES = [
    "eval/rewards/margins",
    "test_mean_reward_margin",
    "train/rewards/margins",
    "val_accuracy",
    "val_scaled_margin",
    "test/rewards/margin" # Singular
]
# --- End Configuration ---

def find_best_metric(df):
    """
    Finds the first available metric column from our priority list.
    """
    for metric in METRIC_CANDIDATES:
        if metric in df.columns:
            return metric
            
    # Fallback: if no priority metrics found, try any column with 'test' or 'eval' or 'train'
    for col in df.columns:
        if any(kw in col.lower() for kw in ["test", "eval", "train"]):
            return col
            
    return None

def rank_and_plot_runs(folder_path):
    """
    Processes a single folder, finds its top 12 runs,
    PRINTS them to the console, and saves 'test' and 'train' plots.
    """
    folder_name = os.path.basename(folder_path)
    parent_folder_name = os.path.basename(os.path.dirname(folder_path))
    
    # Create a clean title prefix
    if folder_name == parent_folder_name or parent_folder_name == ".":
         title_prefix = f"{folder_name}"
    else:
         title_prefix = f"{parent_folder_name} / {folder_name}"
         
    print(f"\n--- Processing: {title_prefix} ---")

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print("  No CSV files found. Skipping.")
        return

    # --- Part 1: Rank all runs to find the Top 12 ---
    run_scores = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            run_name = os.path.basename(csv_path).replace(".csv", "")
            
            metric_col = find_best_metric(df)
            if metric_col is None:
                continue

            final_value = df[metric_col].dropna().iloc[-1]
            
            # Store run_name, score, and the path to its CSV
            run_scores.append((run_name, final_value, csv_path, metric_col))
            
        except pd.errors.EmptyDataError:
            pass # Skip empty files
        except IndexError:
             pass # Skip files where metric column has no data
        except Exception as e:
            print(f"  [Error] Could not rank {os.path.basename(csv_path)}: {e}")

    # Sort by score (highest first) and get the top 12
    run_scores.sort(key=lambda x: x[1], reverse=True)
    top_12_runs = run_scores[:12]
    
    if not top_12_runs:
        print("  No valid runs found to plot. Skipping.")
        return
        
    # --- Part 2: Print the Top 12 List to Console (RE-ADDED) ---
    print("\n" + "="*50)
    print(f" 🏆 Top 12 Runs for: {title_prefix}")
    print("="*50)
    for i, (run_name, score, _, metric) in enumerate(top_12_runs):
        print(f"  {i+1:>2}. {run_name:<70} (Score: {score:.6f} from '{metric}')")
    print("="*50)

    # --- Part 3: Load and Melt Data for ONLY the Top 12 ---
    all_test_dfs = []
    all_train_dfs = []
    test_metric_names = set()
    train_metric_names = set()

    # We use the 'csv_path' from the top_12_runs list
    for run_name, score, csv_path, rank_metric in top_12_runs:
        try:
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

    # --- Part 4: Generate Test Plot ---
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
        
        plt.title(f"Top 12 Test/Eval Metrics: {title_prefix}", fontsize=16)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel(test_ylabel, fontsize=12) # Use dynamic label
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plot_filename = os.path.join(folder_path, f"{folder_name}_TOP12_test_metrics.pdf")
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        print(f"  Saved test plot to: {plot_filename}")
    else:
        print("  No 'test' or 'eval' data found to plot.")

    # --- Part 5: Generate Train Plot ---
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

        plt.title(f"Top 12 Train Metrics: {title_prefix}", fontsize=16)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel(train_ylabel, fontsize=12) # Use dynamic label
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plot_filename = os.path.join(folder_path, f"{folder_name}_TOP12_train_metrics.pdf")
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        print(f"  Saved train plot to: {plot_filename}")
    else:
        print("  No 'train' data found to plot.")

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
                    rank_and_plot_runs(subfolder_path)
            else:
                rank_and_plot_runs(folder_path)
    
    print("\nAll processing complete.")