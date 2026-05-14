'''final plot generation with FULL RUN NAMES in legend'''

import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# --- Configuration ---
# Specific folders to process
FOLDERS_TO_PROCESS = [
    "/home/miria/COALA/wandb_data_csv_competition2_train/DPO",
    "/home/miria/COALA/wandb_data_csv_competition2_eval/DPO",
    "/home/miria/COALA/wandb_data_csv_coala4"
]

# Output directory for plots (current directory)
OUTPUT_DIR = "/home/miria/COALA/plots/"

# The column to use as the x-axis (from your W&B data)
STEP_COLUMN = "_step"

# String to ignore
NOPLOT_TAG = "NOPLOT"

# --- Color mapping for consistent colors across plots ---
# Define a color palette with distinct colors (no red)
# Will create 6 combinations based on model + SFT status:
# 1. mistral_edu (no SFT)
# 2. mistral_edu_sft
# 3. llama_edu (no SFT)
# 4. llama_edu_sft
# 5. dolphin_edu (no SFT)
# 6. dolphin_edu_sft
COLOR_PALETTE = [
    '#0066CC',  # strong blue
    '#00AA44',  # forest green  
    '#FF8C00',  # dark orange
    '#9933CC',  # purple
    '#00B8B8',  # teal/turquoise
    '#CCAA00',  # gold/amber
    '#FF00FF',  # magenta (backup)
    '#8B4513',  # saddle brown (backup)
    '#4B0082',  # indigo (backup)
    '#32CD32',  # lime green (backup)
    '#000080',  # navy (backup)
    '#FF69B4',  # hot pink (backup)
]

# Global dictionary to store run-to-color mapping
RUN_COLOR_MAP = {}
NEXT_COLOR_INDEX = 0

def get_run_signature(run_name):
    """
    Create a signature for a run based on its key characteristics.
    This helps identify the same run across different folders.
    Now includes SFT status for 6 different color combinations.
    """
    run_lower = run_name.lower()
    
    # Extract model type
    model = ""
    if "llama" in run_lower:
        model = "llama"
    elif "mistral" in run_lower:
        model = "mistral"
    elif "dolphin" in run_lower:
        model = "dolphin"
    
    # Extract dataset
    dataset = ""
    if "edu" in run_lower:
        dataset = "edu"
    elif "ultrafeedback" in run_lower:
        dataset = "ultrafeedback"
    
    # Check for SFT
    has_sft = "sft" in run_lower
    sft_tag = "_sft" if has_sft else "_nosft"
    
    # Create signature including SFT status
    signature = f"{model}_{dataset}{sft_tag}"
    
    # If signature is empty or incomplete, use the full run name
    if model == "" or dataset == "":
        signature = run_name
        
    return signature

def get_color_for_run(run_name):
    """
    Get a consistent color for a run based on its signature.
    """
    global NEXT_COLOR_INDEX
    
    signature = get_run_signature(run_name)
    
    if signature not in RUN_COLOR_MAP:
        # Assign a new color
        RUN_COLOR_MAP[signature] = COLOR_PALETTE[NEXT_COLOR_INDEX % len(COLOR_PALETTE)]
        NEXT_COLOR_INDEX += 1
    
    return RUN_COLOR_MAP[signature]

def get_display_label(run_name):
    """
    Return the full run name for the legend (no simplification).
    """
    # Return the full run name as-is
    return run_name

def get_legend_sort_order(label):
    """
    Return sort order for legend labels based on model type and SFT status.
    Using the full run name, we need to extract the model type.
    Order: LLaMA runs, then Mistral runs, then Dolphin runs
    """
    label_lower = label.lower()
    
    # Determine primary sort by model type
    if "llama" in label_lower:
        primary = 0
    elif "mistral" in label_lower:
        primary = 1
    elif "dolphin" in label_lower:
        primary = 2
    else:
        primary = 999
    
    # Secondary sort by SFT status (non-SFT first, then SFT)
    if "sft" in label_lower:
        secondary = 1
    else:
        secondary = 0
    
    # Combine for final sort order
    return primary * 10 + secondary

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
    
    # Create identifier for the folder
    if "DPO" in folder_path:
        folder_identifier = "DPO"
    elif "coala4" in folder_path.lower():
        folder_identifier = "COALA4"
    else:
        folder_identifier = folder_name
    
    # Extract additional context for title
    if "competition1" in folder_path:
        title_context = "Competition1"
    elif "competition2" in folder_path:
        title_context = "Competition2"
    else:
        title_context = folder_identifier
         
    print(f"\n--- Processing: {folder_path} ---")
    print(f"    Folder identifier: {folder_identifier}")

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print("  No CSV files found. Skipping.")
        return

    # --- Part 1: Load and Melt Data for ONLY the Filtered Runs ---
    all_test_dfs = []
    all_train_dfs = []
    test_metric_names = set()
    train_metric_names = set()
    run_colors = {}  # Store colors for this plot

    runs_plotted_count = 0

    for csv_path in csv_files:
        try:
            run_name = os.path.basename(csv_path).replace(".csv", "")
            
            # Check for "NOPLOT" string
            if NOPLOT_TAG in run_name:
                print(f"    Skipping {run_name}: Contains 'NOPLOT' tag.")
                continue
            
            # Check if this run matches our filter
            if not should_plot_run(run_name):
                continue
            
            runs_plotted_count += 1
            df = pd.read_csv(csv_path)
            
            # Get consistent color for this run (strip _raw suffix for color consistency)
            base_run_name = run_name.replace("_raw", "")
            run_colors[run_name] = get_color_for_run(base_run_name)
            # Also add color for the _raw version if we're applying smoothing
            if "competition2_train/DPO" in folder_path or "_eval" in folder_path:
                run_colors[run_name + "_raw"] = get_color_for_run(base_run_name)
            
            # Find all 'test' or 'eval' columns
            test_cols = [col for col in df.columns if "test" in col.lower() or "eval" in col.lower()]
            # Find all 'train' columns
            train_cols = [col for col in df.columns if "train" in col.lower()]

            if STEP_COLUMN not in df.columns:
                print(f"    Skipping {run_name}: Missing '{STEP_COLUMN}' column.")
                continue
            
            # Check if this is a DPO folder and divide values by 100 if needed
            if "DPO" in folder_path:
                # Divide all metric columns by 100 for DPO folders
                for col in test_cols:
                    if col in df.columns:
                        df[col] = df[col] / 100.0
                for col in train_cols:
                    if col in df.columns:
                        df[col] = df[col] / 100.0
                print(f"    Note: Divided values by 100 for DPO folder")
                
            # For competition2_train/DPO OR _eval folders, keep both raw and smoothed data
            if "competition2_train/DPO" in folder_path or "competition1_train/ORPO" in folder_path:
                print(f"    Processing with smoothing overlay (Competition2/DPO or eval folder)")
                
                # Store raw data before smoothing
                df_raw = df.copy()
                
                # Sort dataframe by step first
                df = df.sort_values(STEP_COLUMN).reset_index(drop=True)
                
                # Apply smoothing to each test column
                for col in test_cols:
                    if col in df.columns:
                        # Get non-NaN values
                        values = df[col].values
                        smoothed_values = []
                        
                        for i, value in enumerate(values):
                            if i == 0:
                                ema = value if not pd.isna(value) else 0
                            else:
                                if pd.isna(value):
                                    # Keep previous EMA if current value is NaN
                                    ema = smoothed_values[-1] if smoothed_values else 0
                                else:
                                    # Wandb formula: ema = alpha * previous_ema + (1-alpha) * current
                                    prev_ema = smoothed_values[-1] if smoothed_values else value
                                    ema = 0.95 * prev_ema + 0.05 * value
                            smoothed_values.append(ema)
                        
                        df[col] = smoothed_values
                
                # Apply smoothing to each train column
                for col in train_cols:
                    if col in df.columns:
                        values = df[col].values
                        smoothed_values = []
                        
                        for i, value in enumerate(values):
                            if i == 0:
                                ema = value if not pd.isna(value) else 0
                            else:
                                if pd.isna(value):
                                    # Keep previous EMA if current value is NaN
                                    ema = smoothed_values[-1] if smoothed_values else 0
                                else:
                                    # Wandb formula: ema = alpha * previous_ema + (1-alpha) * current
                                    prev_ema = smoothed_values[-1] if smoothed_values else value
                                    ema = 0.95 * prev_ema + 0.05 * value
                            smoothed_values.append(ema)
                        
                        df[col] = smoothed_values
                
                # Melt both raw and smoothed data for test
                if test_cols:
                    # Raw data (will be plotted pale)
                    df_test_raw = df_raw.melt(
                        id_vars=[STEP_COLUMN], value_vars=test_cols, 
                        var_name="Metric", value_name="Value"
                    )
                    df_test_raw["run_name"] = run_name + "_raw"
                    df_test_raw["is_raw"] = True
                    
                    # Smoothed data (will be plotted bold)
                    df_test_smoothed = df.melt(
                        id_vars=[STEP_COLUMN], value_vars=test_cols, 
                        var_name="Metric", value_name="Value"
                    )
                    df_test_smoothed["run_name"] = run_name
                    df_test_smoothed["is_raw"] = False
                    
                    all_test_dfs.append(df_test_raw)
                    all_test_dfs.append(df_test_smoothed)
                    test_metric_names.update(test_cols)

                # Melt both raw and smoothed data for train
                if train_cols:
                    # Raw data (will be plotted pale)
                    df_train_raw = df_raw.melt(
                        id_vars=[STEP_COLUMN], value_vars=train_cols, 
                        var_name="Metric", value_name="Value"
                    )
                    df_train_raw["run_name"] = run_name + "_raw"
                    df_train_raw["is_raw"] = True
                    
                    # Smoothed data (will be plotted bold)
                    df_train_smoothed = df.melt(
                        id_vars=[STEP_COLUMN], value_vars=train_cols, 
                        var_name="Metric", value_name="Value"
                    )
                    df_train_smoothed["run_name"] = run_name
                    df_train_smoothed["is_raw"] = False
                    
                    all_train_dfs.append(df_train_raw)
                    all_train_dfs.append(df_train_smoothed)
                    train_metric_names.update(train_cols)
                    
            else:
                # For non-smoothing folders, process normally
                # Melt and store 'test' data
                if test_cols:
                    df_test_melted = df.melt(
                        id_vars=[STEP_COLUMN], value_vars=test_cols, 
                        var_name="Metric", value_name="Value"
                    )
                    df_test_melted["run_name"] = run_name
                    df_test_melted["is_raw"] = False
                    
                    all_test_dfs.append(df_test_melted)
                    test_metric_names.update(test_cols)

                # Melt and store 'train' data
                if train_cols:
                    df_train_melted = df.melt(
                        id_vars=[STEP_COLUMN], value_vars=train_cols, 
                        var_name="Metric", value_name="Value"
                    )
                    df_train_melted["run_name"] = run_name
                    df_train_melted["is_raw"] = False
                    
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
        
        # Ensure is_raw column exists with default False
        if 'is_raw' not in combined_test_df.columns:
            combined_test_df['is_raw'] = False
        
        # Determine Y-axis label
        test_ylabel = "Metric Value"
        if len(test_metric_names) == 1:
            test_ylabel = list(test_metric_names)[0]

        plt.figure(figsize=(12, 7))  # Reduced width since legend is inside
        sns.set_theme(style="darkgrid")
        
        # Create the plot axes
        ax = plt.gca()
        
        # Special handling for folders with smoothing overlay effect
        if "competition2_train/DPO" in folder_path or "_eval" in folder_path:
            # First, plot the raw data (pale/faded)
            raw_data = combined_test_df[combined_test_df.get('is_raw', False) == True]
            for (run_name, metric), group in raw_data.groupby(['run_name', 'Metric']):
                base_name = run_name.replace('_raw', '')
                # Get color, with fallback to base name color
                color = run_colors.get(run_name, run_colors.get(base_name, get_color_for_run(base_name)))
                group_sorted = group.sort_values(STEP_COLUMN)
                ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                       color=color, alpha=0.15, linewidth=0.8, label='_nolegend_')
            
            # Then, plot the smoothed data (bold)
            smooth_data = combined_test_df[combined_test_df.get('is_raw', False) == False]
            
            # Track which labels we've already added to avoid duplicates
            added_labels = set()
            
            for (run_name, metric), group in smooth_data.groupby(['run_name', 'Metric']):
                color = run_colors.get(run_name, get_color_for_run(run_name))
                group_sorted = group.sort_values(STEP_COLUMN)
                
                # Get display label (full run name)
                display_label = get_display_label(run_name)
                
                # Only add label once per run (not per metric)
                if display_label not in added_labels:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, alpha=1.0, linewidth=2, label=display_label)
                    added_labels.add(display_label)
                else:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, alpha=1.0, linewidth=2, label='_nolegend_')
        else:
            # For other folders, plot manually to control labels
            added_labels = set()
            
            for (run_name, metric), group in combined_test_df.groupby(['run_name', 'Metric']):
                color = run_colors.get(run_name, get_color_for_run(run_name))
                group_sorted = group.sort_values(STEP_COLUMN)
                
                # Get display label (full run name)
                display_label = get_display_label(run_name)
                
                # Only add label once per run (not per metric)
                if display_label not in added_labels:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, linewidth=1.5, label=display_label)
                    added_labels.add(display_label)
                else:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, linewidth=1.5, label='_nolegend_')
        
        # Format y-axis based on folder type
        from matplotlib.ticker import FormatStrFormatter
        if "DPO" in folder_path:
            # DPO folders: use 2 decimal places for consistency (0.10, 0.70, 0.90)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # Set y-axis maximum to 1.00 for DPO folders
            ax.set_ylim(top=1.00)
        else:
            # COALA4 folder: use 2 decimal places for precision (0.75, 0.80, 0.85, 0.90)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # No title - removed for cleaner look
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Reward Margin", fontsize=12)  # Changed from test_ylabel
        
        # Customize legend to show circles instead of lines
        handles, labels = ax.get_legend_handles_labels()
        # Create circle markers for legend
        new_handles = []
        new_labels = []
        for handle, label in zip(handles, labels):
            if hasattr(handle, 'get_color'):
                color = handle.get_color()
                # Create a circle marker
                new_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                   markersize=10, markeredgecolor=color, linewidth=0)
                new_handles.append(new_handle)
                new_labels.append(label)
        
        # Sort legend entries in the specified order
        if new_handles and new_labels:
            # Create pairs of (handle, label) and sort by label order
            legend_pairs = list(zip(new_handles, new_labels))
            legend_pairs.sort(key=lambda x: get_legend_sort_order(x[1]))
            
            # Unzip back to separate lists
            sorted_handles = [pair[0] for pair in legend_pairs]
            sorted_labels = [pair[1] for pair in legend_pairs]
            
            # Add smoothing info to legend title for folders with smoothing
            if "competition2_train/DPO" in folder_path or "_eval" in folder_path:
                legend_title = "Smoothing EMA α=0.95"
            else:
                legend_title = None
            
            ax.legend(sorted_handles, sorted_labels, loc='lower right', 
                     frameon=True, fancybox=True, shadow=False,
                     title=legend_title, title_fontsize=10, fontsize=8)  # Smaller font for long names
        
        # Save in current directory with descriptive name
        plot_filename = os.path.join(OUTPUT_DIR, f"{folder_identifier}_{title_context}_FILTERED_test_metrics.pdf")
        plt.tight_layout()
        plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1)  # Minimal padding for self-contained PDF
        plt.close()
        print(f"  Saved filtered test plot to: {plot_filename}")
    else:
        print("  No 'test' or 'eval' data found to plot for filtered runs.")

    # --- Part 3: Generate Train Plot ---
    if all_train_dfs:
        combined_train_df = pd.concat(all_train_dfs, ignore_index=True)

        # Ensure is_raw column exists with default False
        if 'is_raw' not in combined_train_df.columns:
            combined_train_df['is_raw'] = False

        # Determine Y-axis label
        train_ylabel = "Metric Value"
        if len(train_metric_names) == 1:
            train_ylabel = list(train_metric_names)[0]

        plt.figure(figsize=(12, 7))  # Reduced width since legend is inside
        sns.set_theme(style="darkgrid")

        # Create the plot axes
        ax = plt.gca()
        
        # Special handling for folders with smoothing overlay effect
        if "competition2_train/DPO" in folder_path or "_eval" in folder_path:
            # First, plot the raw data (pale/faded)
            raw_data = combined_train_df[combined_train_df.get('is_raw', False) == True]
            for (run_name, metric), group in raw_data.groupby(['run_name', 'Metric']):
                base_name = run_name.replace('_raw', '')
                # Get color, with fallback to base name color
                color = run_colors.get(run_name, run_colors.get(base_name, get_color_for_run(base_name)))
                group_sorted = group.sort_values(STEP_COLUMN)
                ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                       color=color, alpha=0.20, linewidth=0.8, label='_nolegend_')
            
            # Then, plot the smoothed data (bold)
            smooth_data = combined_train_df[combined_train_df.get('is_raw', False) == False]
            
            # Track which labels we've already added to avoid duplicates
            added_labels = set()
            
            for (run_name, metric), group in smooth_data.groupby(['run_name', 'Metric']):
                color = run_colors.get(run_name, get_color_for_run(run_name))
                group_sorted = group.sort_values(STEP_COLUMN)
                
                # Get display label (full run name)
                display_label = get_display_label(run_name)
                
                # Only add label once per run (not per metric)
                if display_label not in added_labels:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, alpha=1.0, linewidth=2, label=display_label)
                    added_labels.add(display_label)
                else:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, alpha=1.0, linewidth=2, label='_nolegend_')
        else:
            # For other folders, plot manually to control labels
            added_labels = set()
            
            for (run_name, metric), group in combined_train_df.groupby(['run_name', 'Metric']):
                color = run_colors.get(run_name, get_color_for_run(run_name))
                group_sorted = group.sort_values(STEP_COLUMN)
                
                # Get display label (full run name)
                display_label = get_display_label(run_name)
                
                # Only add label once per run (not per metric)
                if display_label not in added_labels:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, linewidth=1.5, label=display_label)
                    added_labels.add(display_label)
                else:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, linewidth=1.5, label='_nolegend_')
        
        # Format y-axis based on folder type
        from matplotlib.ticker import FormatStrFormatter
        if "DPO" in folder_path:
            # DPO folders: use 2 decimal places for consistency (0.10, 0.70, 0.90)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # Set y-axis maximum to 1.00 for DPO folders
            ax.set_ylim(top=1.00)
        else:
            # COALA4 folder: use 2 decimal places for precision (0.75, 0.80, 0.85, 0.90)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # No title - removed for cleaner look
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Reward Margin", fontsize=12)  # Changed from train_ylabel
        
        # Customize legend to show circles instead of lines
        handles, labels = ax.get_legend_handles_labels()
        # Create circle markers for legend
        new_handles = []
        new_labels = []
        for handle, label in zip(handles, labels):
            if hasattr(handle, 'get_color'):
                color = handle.get_color()
                # Create a circle marker
                new_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                   markersize=10, markeredgecolor=color, linewidth=0)
                new_handles.append(new_handle)
                new_labels.append(label)
        
        # Sort legend entries in the specified order
        if new_handles and new_labels:
            # Create pairs of (handle, label) and sort by label order
            legend_pairs = list(zip(new_handles, new_labels))
            legend_pairs.sort(key=lambda x: get_legend_sort_order(x[1]))
            
            # Unzip back to separate lists
            sorted_handles = [pair[0] for pair in legend_pairs]
            sorted_labels = [pair[1] for pair in legend_pairs]
            
            # Add smoothing info to legend title for folders with smoothing
            if "competition2_train/DPO" in folder_path or "competition1_train/ORPO" in folder_path:
                legend_title = "Smoothing EMA α=0.95"
            else:
                legend_title = None
            
            ax.legend(sorted_handles, sorted_labels, loc='lower right', 
                     frameon=True, fancybox=True, shadow=False,
                     title=legend_title, title_fontsize=10, fontsize=8)  # Smaller font for long names

        # Save in current directory with descriptive name
        plot_filename = os.path.join(OUTPUT_DIR, f"{folder_identifier}_{title_context}_FILTERED_train_metrics.pdf")
        plt.tight_layout()
        plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1)  # Minimal padding for self-contained PDF
        plt.close()
        print(f"  Saved filtered train plot to: {plot_filename}")
    else:
        print("  No 'train' data found to plot for filtered runs.")

# --- Main script execution ---
if __name__ == "__main__":
    print(f"Processing {len(FOLDERS_TO_PROCESS)} specified folders...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    for folder_path in FOLDERS_TO_PROCESS:
        if os.path.isdir(folder_path):
            filter_and_plot_runs(folder_path)
        else:
            print(f"\nWarning: Folder not found: {folder_path}")
    
    print("\n--- Color Mapping Summary ---")
    print("Run signatures assigned colors (6 combinations based on model + SFT status):")
    
    # Sort signatures for better readability
    sorted_signatures = sorted(RUN_COLOR_MAP.items())
    for signature, color in sorted_signatures:
        # Make the output clearer
        if "_nosft" in signature:
            display_name = signature.replace("_nosft", " (no SFT)")
        elif "_sft" in signature:
            display_name = signature.replace("_sft", " (with SFT)")
        else:
            display_name = signature
        print(f"  {display_name}: {color}")
    
    print(f"\nTotal unique color assignments: {len(RUN_COLOR_MAP)}")
    print("\nAll processing complete.")
    print(f"Plots saved in: {OUTPUT_DIR}")