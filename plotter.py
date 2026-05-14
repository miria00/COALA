'''
plots all CSVs in 6 folders starting with "wandb_data_csv" for quick sanity check
'''

import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
# Set this to the directory containing your folders
BASE_DIR = "." 

# The prefix of the folders to process
FOLDER_PREFIX = "wandb_data_csv_" 

# The column to use as the x-axis (from your W&B data)
STEP_COLUMN = "_step"
# --- End Configuration ---

def generate_plots_for_csvs(csv_search_path, output_save_path):
    """
    Finds all CSVs in a specific folder, melts them, and generates
    a 'test' and 'train' plot, saving them to the output_save_path.
    """
    # Use the name of the folder we are searching as the plot title/filename prefix
    folder_name = os.path.basename(csv_search_path)
    print(f"\n  --- Processing Subfolder/Folder: {folder_name} ---")
    
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(csv_search_path, "*.csv"))
    if not csv_files:
        print("    No CSV files found. Skipping.")
        return

    all_test_dfs = []
    all_train_dfs = []

    # Loop through all CSVs, read and "melt" them for plotting
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            # Use filename as the 'run_name'
            run_name = os.path.basename(csv_path).replace(".csv", "")
            
            # Check if the step column exists
            if STEP_COLUMN not in df.columns:
                print(f"    Skipping {run_name}: Missing '{STEP_COLUMN}' column.")
                continue

            # Find all 'test' or 'eval' columns
            test_cols = [col for col in df.columns if "test" in col.lower() or "eval" in col.lower()]
            # Find all 'train' columns
            train_cols = [col for col in df.columns if "train" in col.lower()]

            # Melt and store 'test' data
            if test_cols:
                df_test_melted = df.melt(
                    id_vars=[STEP_COLUMN], 
                    value_vars=test_cols, 
                    var_name="Metric", 
                    value_name="Value"
                )
                df_test_melted["run_name"] = run_name
                all_test_dfs.append(df_test_melted)

            # Melt and store 'train' data
            if train_cols:
                df_train_melted = df.melt(
                    id_vars=[STEP_COLUMN], 
                    value_vars=train_cols, 
                    var_name="Metric", 
                    value_name="Value"
                )
                df_train_melted["run_name"] = run_name
                all_train_dfs.append(df_train_melted)
        
        except pd.errors.EmptyDataError:
            print(f"    Skipping {csv_path}: File is empty.")
        except Exception as e:
            print(f"    Error processing {csv_path}: {e}")

    # --- Generate Test Plot ---
    if all_test_dfs:
        combined_test_df = pd.concat(all_test_dfs, ignore_index=True)
        print(f"    Generating 'test/eval' plot for {len(all_test_dfs)} runs...")
        
        plt.figure(figsize=(16, 8)) # Make a wide figure for readability
        sns.set_theme(style="darkgrid")
        
        # Plot with run_name as color and Metric as line style
        sns.lineplot(
            data=combined_test_df, 
            x=STEP_COLUMN, 
            y="Value", 
            hue="run_name",
            style="Metric"
        )
        
        plt.title(f"{folder_name} - Test/Eval Metrics", fontsize=16)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        
        # Move legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save the plot as a PDF
        plot_filename = os.path.join(output_save_path, f"{folder_name}_test_metrics.pdf")
        plt.savefig(plot_filename, bbox_inches='tight') # use bbox_inches to fit legend
        plt.close() # Close figure to free memory
        print(f"    Saved test plot to: {plot_filename}")
    else:
        print("    No 'test' or 'eval' data found to plot.")

    # --- Generate Train Plot ---
    if all_train_dfs:
        combined_train_df = pd.concat(all_train_dfs, ignore_index=True)
        print(f"    Generating 'train' plot for {len(all_train_dfs)} runs...")

        plt.figure(figsize=(16, 8)) # Make a wide figure
        sns.set_theme(style="darkgrid")

        sns.lineplot(
            data=combined_train_df, 
            x=STEP_COLUMN, 
            y="Value", 
            hue="run_name",
            style="Metric"
        )

        plt.title(f"{folder_name} - Train Metrics", fontsize=16)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)

        # Move legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save the plot as a PDF
        plot_filename = os.path.join(output_save_path, f"{folder_name}_train_metrics.pdf")
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close() # Close figure to free memory
        print(f"    Saved train plot to: {plot_filename}")
    else:
        print("    No 'train' data found to plot.")

def process_base_folder(folder_path):
    """
    Checks a base folder (e.g., wandb_data_csv_competition1) for
    DPO/ORPO subfolders. If they exist, process them.
    If not, process the base folder itself.
    """
    folder_name = os.path.basename(folder_path)
    print(f"\n--- Checking Folder: {folder_name} ---")

    dpo_path = os.path.join(folder_path, "DPO")
    orpo_path = os.path.join(folder_path, "ORPO")
    
    subfolders_to_process = []
    if os.path.isdir(dpo_path):
        subfolders_to_process.append(dpo_path)
    if os.path.isdir(orpo_path):
        subfolders_to_process.append(orpo_path)

    if subfolders_to_process:
        # If we found DPO or ORPO, process them individually
        print(f"  Found subfolders in {folder_name}, processing them...")
        for subfolder_path in subfolders_to_process:
            # Pass the subfolder path for both searching and saving
            generate_plots_for_csvs(subfolder_path, subfolder_path)
    else:
        # No DPO/ORPO subfolders, process the main folder as before
        print(f"  No DPO/ORPO subfolders found, processing main folder...")
        # Pass the main folder path for both searching and saving
        generate_plots_for_csvs(folder_path, folder_path)


# --- Main script execution ---
if __name__ == "__main__":
    # Find all directories in BASE_DIR that start with FOLDER_PREFIX
    folders_to_process = glob.glob(os.path.join(BASE_DIR, f"{FOLDER_PREFIX}*"))
    folders_to_process = [f for f in folders_to_process if os.path.isdir(f)]

    if not folders_to_process:
        print(f"No folders found in '{BASE_DIR}' with prefix '{FOLDER_PREFIX}'.")
    else:
        print(f"Found {len(folders_to_process)} folders to process.")
        for folder_path in sorted(folders_to_process):
            process_base_folder(folder_path)

    print("\nAll processing complete.")