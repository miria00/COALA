'''
final plot generation for IMDB with selective smoothing and x-axis limits
this is for imdb dataset: 
FOLDERS_TO_PROCESS = [
    "/home/miria/COALA/wandb_data_csv_competition2_train/DPO",
    "/home/miria/COALA/wandb_data_csv_competition1_eval/DPO",
    "/home/miria/COALA/wandb_data_csv_coala4"
]
'''

import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter, LinearLocator # Keep LinearLocator for fallbacks
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# --- Configuration ---
# Specific folders to process for imdb dataset
FOLDERS_TO_PROCESS = [
    "/home/miria/COALA/wandb_data_csv_competition2_train/DPO",
    "/home/miria/COALA/wandb_data_csv_competition1_eval/DPO",
    "/home/miria/COALA/wandb_data_csv_coala4"
]
# Output directory for plots (current directory)
OUTPUT_DIR = "/home/miria/COALA/plots_imdb/"

# Font size configuration
AXIS_LABEL_FONT_SIZE = 22  # Font size for x and y axis labels
LEGEND_FONT_SIZE = 20      # Font size for all legend items (entries and title)

# The column to use as the x-axis (from your W&B data)
STEP_COLUMN = "_step"

# String to ignore
NOPLOT_TAG = "NOPLOT"

# --- Color mapping for consistent colors across plots ---
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
    if "imdb" in run_lower:  # <-- CORRECTED
        dataset = "imdb"
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
    Convert run name to a clean display label for the legend.
    """
    run_lower = run_name.lower()
    
    # Check for model and SFT status
    if "mistral" in run_lower:
        if "sft" in run_lower:
            return "Mistral-7B-SFT"
        else:
            return "Mistral7B"
    elif "dolphin" in run_lower:
        if "sft" in run_lower:
            return "Dolphin2.6-7B-SFT"
        else:
            return "Dolphin2.6-7B"
    elif "llama" in run_lower:
        if "sft" in run_lower:
            return "LLaMA-8B-SFT"
        else:
            return "LLaMA-8B"
    
    # Fallback to original name if no match
    return run_name

def get_legend_sort_order(label):
    """
    Return sort order for legend labels to maintain consistent ordering.
    """
    order_map = {
        'LLaMA-8B': 0,
        'LLaMA-8B-SFT': 1,
        'Mistral7B': 2,
        'Mistral-7B-SFT': 3,
        'Dolphin2.6-7B': 4,
        'Dolphin2.6-7B-SFT': 5
    }
    return order_map.get(label, 999)  # Unknown labels go to the end

def should_plot_run(run_name):
    """
    Returns True if the run name matches our filter criteria,
    False otherwise.
    """
    run_name_lower = run_name.lower()
    
    if "llama" in run_name_lower and "imdb" in run_name_lower: # <-- CORRECTED
        return True
    if "mistral" in run_name_lower and "imdb" in run_name_lower: # <-- CORRECTED
        return True
    if "dolphin" in run_name_lower and "imdb" in run_name_lower: # <-- CORRECTED
        return True
        
    return False

def should_apply_smoothing(folder_path):
    """
    Returns True if smoothing should be applied to this folder.
    """
    # This logic might need to be updated if you add more train folders
    return (folder_path == "/home/miria/COALA/wandb_data_csv_competition1_train/ORPO" or 
            folder_path == "/home/miria/COALA/wandb_data_csv_competition2_train/DPO")

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
    elif "ORPO" in folder_path:
        folder_identifier = "ORPO"
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
            
            # Get consistent color for this run
            base_run_name = run_name.replace("_raw", "")
            run_colors[run_name] = get_color_for_run(base_run_name)
            if should_apply_smoothing(folder_path):
                run_colors[run_name + "_raw"] = get_color_for_run(base_run_name)
            
            # Find all 'test' or 'eval' columns
            test_cols = [col for col in df.columns if "test" in col.lower() or "eval" in col.lower()]
            # Find all 'train' columns
            train_cols = [col for col in df.columns if "train" in col.lower()]

            if STEP_COLUMN not in df.columns:
                print(f"    Skipping {run_name}: Missing '{STEP_COLUMN}' column.")
                continue
            
            # Check if this is a DPO or ORPO folder and divide values by 100 if needed
            if "DPO" in folder_path or "ORPO" in folder_path:
                for col in test_cols:
                    if col in df.columns:
                        df[col] = df[col] / 100.0
                for col in train_cols:
                    if col in df.columns:
                        df[col] = df[col] / 100.0
                print(f"    Note: Divided values by 100 for DPO/ORPO folder")
                
            # Check if smoothing should be applied
            if should_apply_smoothing(folder_path):
                print(f"    Processing with smoothing overlay (specified training folder)")
                
                # Store raw data before smoothing
                df_raw = df.copy()
                
                # Sort dataframe by step first
                df = df.sort_values(STEP_COLUMN).reset_index(drop=True)
                
                # Apply smoothing (EMA)
                for col in test_cols + train_cols:
                    if col in df.columns:
                        values = df[col].values
                        smoothed_values = []
                        
                        for i, value in enumerate(values):
                            if i == 0:
                                ema = value if not pd.isna(value) else 0
                            else:
                                if pd.isna(value):
                                    ema = smoothed_values[-1] if smoothed_values else 0
                                else:
                                    prev_ema = smoothed_values[-1] if smoothed_values else value
                                    ema = 0.95 * prev_ema + 0.05 * value
                            smoothed_values.append(ema)
                        
                        df[col] = smoothed_values
                
                # Melt both raw and smoothed data for test
                if test_cols:
                    df_test_raw = df_raw.melt(
                        id_vars=[STEP_COLUMN], value_vars=test_cols, 
                        var_name="Metric", value_name="Value"
                    )
                    df_test_raw["run_name"] = run_name + "_raw"
                    df_test_raw["is_raw"] = True
                    
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
                    df_train_raw = df_raw.melt(
                        id_vars=[STEP_COLUMN], value_vars=train_cols, 
                        var_name="Metric", value_name="Value"
                    )
                    df_train_raw["run_name"] = run_name + "_raw"
                    df_train_raw["is_raw"] = True
                    
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
                if test_cols:
                    df_test_melted = df.melt(
                        id_vars=[STEP_COLUMN], value_vars=test_cols, 
                        var_name="Metric", value_name="Value"
                    )
                    df_test_melted["run_name"] = run_name
                    df_test_melted["is_raw"] = False
                    all_test_dfs.append(df_test_melted)
                    test_metric_names.update(test_cols)

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
        
        if 'is_raw' not in combined_test_df.columns:
            combined_test_df['is_raw'] = False
        
        test_ylabel = "Metric Value"
        if len(test_metric_names) == 1:
            test_ylabel = list(test_metric_names)[0]

        plt.figure(figsize=(12, 7))
        sns.set_theme(style="darkgrid")
        ax = plt.gca()
        
        if should_apply_smoothing(folder_path):
            # Plot raw data (faded)
            raw_data = combined_test_df[combined_test_df.get('is_raw', False) == True]
            for (run_name, metric), group in raw_data.groupby(['run_name', 'Metric']):
                base_name = run_name.replace('_raw', '')
                color = run_colors.get(run_name, run_colors.get(base_name, get_color_for_run(base_name)))
                group_sorted = group.sort_values(STEP_COLUMN)
                ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                       color=color, alpha=0.15, linewidth=0.8, label='_nolegend_')
            
            # Plot smoothed data (bold)
            smooth_data = combined_test_df[combined_test_df.get('is_raw', False) == False]
            added_labels = set()
            for (run_name, metric), group in smooth_data.groupby(['run_name', 'Metric']):
                color = run_colors.get(run_name, get_color_for_run(run_name))
                group_sorted = group.sort_values(STEP_COLUMN)
                display_label = get_display_label(run_name)
                if display_label not in added_labels:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, alpha=1.0, linewidth=2, label=display_label)
                    added_labels.add(display_label)
                else:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, alpha=1.0, linewidth=2, label='_nolegend_')
        else:
            # For other folders, plot normally
            added_labels = set()
            for (run_name, metric), group in combined_test_df.groupby(['run_name', 'Metric']):
                color = run_colors.get(run_name, get_color_for_run(run_name))
                group_sorted = group.sort_values(STEP_COLUMN)
                display_label = get_display_label(run_name)
                if display_label not in added_labels:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, linewidth=1.5, label=display_label)
                    added_labels.add(display_label)
                else:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, linewidth=1.5, label='_nolegend_')
        
        # --- MODIFIED Y-AXIS TICK HANDLING ---
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) # Always format as 2 decimal
        
        if "coala4" in folder_path.lower():
            # COALA4: Specific, non-even ticks
            yticks = [0.75, 0.78, 0.80, 0.83, 0.85, 0.88, 0.90]
            ax.set_yticks(yticks)
        elif "DPO" in folder_path or "ORPO" in folder_path:
            # DPO/ORPO Test: Even ticks from 0.20 to 1.00
            yticks = [0.20, 0.40, 0.60, 0.80, 1.00]
            ax.set_yticks(yticks)
            ax.set_ylim(top=1.0) # Set top to 1.0
        else:
            # Fallback for other folders
            ax.yaxis.set_major_locator(LinearLocator(numticks=5)) 
        # --- END MODIFIED Y-AXIS HANDLING ---

        # Set x-axis limits
        if "coala4" in folder_path.lower():
            ax.set_xlim(right=60)
            print("    Note: X-axis limited to 55 for COALA4 plot")
        
        plt.xlabel("Step", fontsize=AXIS_LABEL_FONT_SIZE)
        plt.ylabel("Reward Margin", fontsize=AXIS_LABEL_FONT_SIZE)
        
        # Customize legend - MODIFIED FOR 3 COLUMNS
        handles, labels = ax.get_legend_handles_labels()
        new_handles = []
        new_labels = []
        for handle, label in zip(handles, labels):
            if hasattr(handle, 'get_color'):
                color = handle.get_color()
                new_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                   markersize=10, markeredgecolor=color, linewidth=0)
                new_handles.append(new_handle)
                new_labels.append(label)
        
        if new_handles and new_labels:
            legend_pairs = list(zip(new_handles, new_labels))
            legend_pairs.sort(key=lambda x: get_legend_sort_order(x[1]))
            sorted_handles = [pair[0] for pair in legend_pairs]
            sorted_labels = [pair[1] for pair in legend_pairs]
            
            legend_title = "Smoothing EMA α=0.95" if should_apply_smoothing(folder_path) else None
            
            # MODIFIED LEGEND - 3 columns in lower right corner
            ax.legend(sorted_handles, sorted_labels, 
                    loc='lower right',  # Keep in lower right corner
                    ncol=3,  # 3 columns for the legend entries
                    frameon=True, fancybox=True, shadow=False,
                    title=legend_title, title_fontsize=LEGEND_FONT_SIZE, 
                    fontsize=LEGEND_FONT_SIZE,
                    columnspacing=0.5,  # Reduced spacing to fit better
                    handletextpad=0.3)  # Reduce space between marker and text
        
        # Save plot
        plot_filename = os.path.join(OUTPUT_DIR, f"{folder_identifier}_{title_context}_FILTERED_test_metrics.pdf")
        plt.tight_layout()
        plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"  Saved filtered test plot to: {plot_filename}")
    else:
        print("  No 'test' or 'eval' data found to plot for filtered runs.")

    # --- Part 3: Generate Train Plot ---
    if all_train_dfs:
        combined_train_df = pd.concat(all_train_dfs, ignore_index=True)

        if 'is_raw' not in combined_train_df.columns:
            combined_train_df['is_raw'] = False

        train_ylabel = "Metric Value"
        if len(train_metric_names) == 1:
            train_ylabel = list(train_metric_names)[0]

        plt.figure(figsize=(12, 7))
        sns.set_theme(style="darkgrid")
        ax = plt.gca()
        
        if should_apply_smoothing(folder_path):
            # Plot raw data (faded)
            raw_data = combined_train_df[combined_train_df.get('is_raw', False) == True]
            for (run_name, metric), group in raw_data.groupby(['run_name', 'Metric']):
                base_name = run_name.replace('_raw', '')
                color = run_colors.get(run_name, run_colors.get(base_name, get_color_for_run(base_name)))
                group_sorted = group.sort_values(STEP_COLUMN)
                ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                       color=color, alpha=0.20, linewidth=0.8, label='_nolegend_')
            
            # Plot smoothed data (bold)
            smooth_data = combined_train_df[combined_train_df.get('is_raw', False) == False]
            added_labels = set()
            for (run_name, metric), group in smooth_data.groupby(['run_name', 'Metric']):
                color = run_colors.get(run_name, get_color_for_run(run_name))
                group_sorted = group.sort_values(STEP_COLUMN)
                display_label = get_display_label(run_name)
                if display_label not in added_labels:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, alpha=1.0, linewidth=2, label=display_label)
                    added_labels.add(display_label)
                else:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, alpha=1.0, linewidth=2, label='_nolegend_')
        else:
            # For other folders, plot normally
            added_labels = set()
            for (run_name, metric), group in combined_train_df.groupby(['run_name', 'Metric']):
                color = run_colors.get(run_name, get_color_for_run(run_name))
                group_sorted = group.sort_values(STEP_COLUMN)
                display_label = get_display_label(run_name)
                if display_label not in added_labels:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, linewidth=1.5, label=display_label)
                    added_labels.add(display_label)
                else:
                    ax.plot(group_sorted[STEP_COLUMN], group_sorted['Value'], 
                           color=color, linewidth=1.5, label='_nolegend_')
        
        # --- MODIFIED Y-AXIS TICK HANDLING ---
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) # Always format as 2 decimal
        
        if "coala4" in folder_path.lower():
            # COALA4: Specific, non-even ticks
            yticks = [0.75, 0.78, 0.80, 0.83, 0.85, 0.88, 0.90]
            ax.set_yticks(yticks)
        elif "DPO" in folder_path or "ORPO" in folder_path:
            # DPO/ORPO Train: Even ticks from 0.00 to 1.00
            yticks = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
            ax.set_yticks(yticks)
            ax.set_ylim(bottom=0.0, top=1.0) # Set both bottom and top
        else:
            # Fallback for other folders
            ax.yaxis.set_major_locator(LinearLocator(numticks=5))
        # --- END MODIFIED Y-AXIS HANDLING ---

        # Set x-axis limits
        if "coala4" in folder_path.lower():
            ax.set_xlim(right=60)
            print("    Note: X-axis limited to 55 for COALA4 plot")
        elif "competition1_train/ORPO" in folder_path:
            ax.set_xlim(right=2300)
            print("    Note: X-axis limited to 2300 for ORPO train plot")
            
        plt.xlabel("Step", fontsize=AXIS_LABEL_FONT_SIZE)
        plt.ylabel("Reward Margin", fontsize=AXIS_LABEL_FONT_SIZE)
        
        # Customize legend - MODIFIED FOR 3 COLUMNS
        handles, labels = ax.get_legend_handles_labels()
        new_handles = []
        new_labels = []
        for handle, label in zip(handles, labels):
            if hasattr(handle, 'get_color'):
                color = handle.get_color()
                new_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                   markersize=10, markeredgecolor=color, linewidth=0)
                new_handles.append(new_handle)
                new_labels.append(label)
        
        if new_handles and new_labels:
            legend_pairs = list(zip(new_handles, new_labels))
            legend_pairs.sort(key=lambda x: get_legend_sort_order(x[1]))
            
            sorted_handles = [pair[0] for pair in legend_pairs]
            sorted_labels = [pair[1] for pair in legend_pairs]
            
            legend_title = "Smoothing EMA α=0.95" if should_apply_smoothing(folder_path) else None
            
            # MODIFIED LEGEND - 3 columns in lower right corner
            ax.legend(sorted_handles, sorted_labels, 
                    loc='lower right',  # Keep in lower right corner
                    ncol=3,  # 3 columns for the legend entries
                    frameon=True, fancybox=True, shadow=False,
                    title=legend_title, title_fontsize=LEGEND_FONT_SIZE, 
                    fontsize=LEGEND_FONT_SIZE,
                    columnspacing=0.5,  # Reduced spacing to fit better
                    handletextpad=0.3)  # Reduce space between marker and text

        # Save plot
        plot_filename = os.path.join(OUTPUT_DIR, f"{folder_identifier}_{title_context}_FILTERED_train_metrics.pdf")
        plt.tight_layout()
        plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"  Saved filtered train plot to: {plot_filename}")
    else:
        print("  No 'train' data found to plot for filtered runs.")

# --- Main script execution ---
if __name__ == "__main__":
    print(f"Processing {len(FOLDERS_TO_PROCESS)} specified folders...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # First, build the consistent color map
    unique_folders = sorted(list(set(FOLDERS_TO_PROCESS)))
    global_color_map = {}
    next_color_index = 0
    
    print("--- Scanning for all run names to build consistent color map... ---")
    all_matching_run_names = set()
    
    for folder_path in unique_folders:
        if not os.path.isdir(folder_path):
            print(f"\nWarning: Folder not found during scan: {folder_path}")
            continue
            
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        for csv_path in csv_files:
            run_name = os.path.basename(csv_path).replace(".csv", "")
            if NOPLOT_TAG in run_name:
                continue
            if should_plot_run(run_name):
                base_run_name = run_name.replace("_raw", "")
                signature = get_run_signature(base_run_name)
                all_matching_run_names.add(signature)
    
    # Assign colors to the unique signatures
    sorted_signatures = sorted(list(all_matching_run_names))
    for signature in sorted_signatures:
        if signature not in RUN_COLOR_MAP:
            RUN_COLOR_MAP[signature] = COLOR_PALETTE[NEXT_COLOR_INDEX % len(COLOR_PALETTE)]
            NEXT_COLOR_INDEX += 1
            
    print(f"Found {len(RUN_COLOR_MAP)} unique run signatures to plot.")
    
    # Now, process the folders for plotting
    for folder_path in FOLDERS_TO_PROCESS:
        if os.path.isdir(folder_path):
            filter_and_plot_runs(folder_path)
        else:
            print(f"\nWarning: Folder not found: {folder_path}")
    
    print("\n--- Color Mapping Summary ---")
    print("Run signatures assigned colors (6 combinations based on model + SFT status):")
    
    # Sort for better readability
    sorted_signatures_map = sorted(RUN_COLOR_MAP.items())
    for signature, color in sorted_signatures_map:
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