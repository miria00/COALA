"""
add_error_bars.py
─────────────────
Generates reward margin plots with bootstrapped 95% CI bands.

Sources:
  COALA: wandb_data_csv_coala1-4 (cols: _step, train_mean_reward_margin, test_mean_reward_margin)
  DPO/ORPO: wandb_data_csv_competition1-2_{train,eval}/DPO/ and /ORPO/ (cols: _step, train/rewards/margins)

Only includes: Dolphin, LLaMA, Mistral models.
Style matches plot_last_final_imdb.py.

Usage:
    python add_error_bars.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots_rebuttal")

# ── Style (matching plot_last_final_imdb.py) ─────────────────────────────────
COLOR_PALETTE = [
    '#0066CC',  # strong blue
    '#00AA44',  # forest green
    '#FF8C00',  # dark orange
    '#9933CC',  # purple
    '#00B8B8',  # teal/turquoise
    '#CCAA00',  # gold/amber
]
AXIS_LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 20

DATASETS = ["edu", "imdb", "ultra"]
MODELS_FILTER = ["dolphin", "llama", "mistral"]  # only these


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    if len(arr) == 1:
        return arr[0], arr[0], arr[0]
    boot_means = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.mean(arr)), float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def detect_model(name):
    """Return clean model name or None if not in filter."""
    s = name.lower()
    if "dolphin" in s:
        return "Dolphin2.6-7B"
    if "llama" in s and "8b" in s:
        return "LLaMA-8B"
    if "llama" in s and "3" in s:
        return "LLaMA-8B"  # llama-3.1 is 8B
    if "mistral" in s:
        return "Mistral-7B"
    return None


def detect_dataset(name):
    s = name.lower()
    for d in DATASETS:
        if d in s:
            return d
    return None


def has_sft(name):
    s = name.lower()
    return "sft" in s and "nosft" not in s


# ── Load COALA data (coala1-4) ───────────────────────────────────────────────

def load_coala():
    """Load COALA finetune CSVs. Each has: _step, test_mean_reward_margin, train_mean_reward_margin."""
    rows = []
    for i in range(1, 5):
        dirpath = os.path.join(BASE_DIR, f"wandb_data_csv_coala{i}")
        if not os.path.isdir(dirpath):
            continue
        for csv_path in glob.glob(os.path.join(dirpath, "*.csv")):
            fname = os.path.basename(csv_path)
            if "NOPLOT" in fname or "seaborn" in fname:
                continue

            model = detect_model(fname)
            if model is None:
                continue
            dataset = detect_dataset(fname)
            if dataset is None:
                continue
            sft = has_sft(fname)
            label = f"{model}-SFT" if sft else model

            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            for split, col in [("train", "train_mean_reward_margin"), ("eval", "test_mean_reward_margin")]:
                if col not in df.columns:
                    continue
                sub = df[["_step", col]].dropna().copy()
                sub.columns = ["step", "reward_margin"]
                sub["method"] = "COALA"
                sub["model"] = model
                sub["label"] = label
                sub["dataset"] = dataset
                sub["split"] = split
                sub["sft"] = sft
                rows.append(sub)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── Load DPO/ORPO data (competition1-2) ─────────────────────────────────────

def load_competition():
    """Load DPO and ORPO CSVs. Each has: _step, train/rewards/margins or eval/rewards/margins."""
    rows = []

    for comp in ["competition1", "competition2"]:
        for split_dir in ["train", "eval"]:
            base = os.path.join(BASE_DIR, f"wandb_data_csv_{comp}_{split_dir}")
            if not os.path.isdir(base):
                continue

            for method_dir in ["DPO", "ORPO"]:
                method_path = os.path.join(base, method_dir)
                if not os.path.isdir(method_path):
                    continue

                for csv_path in glob.glob(os.path.join(method_path, "*.csv")):
                    fname = os.path.basename(csv_path)
                    if "NOPLOT" in fname or "seaborn" in fname:
                        continue

                    model = detect_model(fname)
                    if model is None:
                        continue
                    dataset = detect_dataset(fname)
                    if dataset is None:
                        continue
                    sft = has_sft(fname)
                    label = f"{model}-SFT" if sft else model

                    try:
                        df = pd.read_csv(csv_path)
                    except Exception:
                        continue

                    # Find margin column
                    margin_col = None
                    for c in df.columns:
                        if "margins" in c.lower() or "reward_margin" in c.lower():
                            margin_col = c
                            break
                    if margin_col is None:
                        continue

                    step_col = "_step" if "_step" in df.columns else "step"
                    if step_col not in df.columns:
                        continue

                    sub = df[[step_col, margin_col]].dropna().copy()
                    sub.columns = ["step", "reward_margin"]
                    sub["method"] = method_dir
                    sub["model"] = model
                    sub["label"] = label
                    sub["dataset"] = dataset
                    sub["split"] = split_dir
                    sub["sft"] = sft
                    rows.append(sub)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── Load wandb_metrics_rebuttal CSVs ─────────────────────────────────────────

def load_wandb_metrics_rebuttal():
    """Load wandb_metrics_rebuttal/*.csv. Has columns: run_name, step, train_rewards_margins, eval_rewards_margins."""
    metrics_dir = os.path.join(BASE_DIR, "wandb_metrics_rebuttal")
    if not os.path.isdir(metrics_dir):
        return pd.DataFrame()

    rows = []
    for csv_path in glob.glob(os.path.join(metrics_dir, "*.csv")):
        fname = os.path.basename(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if df.empty or "run_name" not in df.columns:
            continue

        for _, run_group in df.groupby("run_name"):
            run_name = run_group["run_name"].iloc[0]

            # Parse method from run_name
            rn_lower = run_name.lower()
            method = None
            if "coala" in rn_lower or "finetune" in rn_lower or "cronos" in rn_lower:
                method = "COALA"
            elif "simpo" in rn_lower:
                method = "SimPO"
            elif "orpo" in rn_lower:
                method = "ORPO"
            elif "dpo" in rn_lower:
                method = "DPO"
            elif "sft" in rn_lower:
                method = "SFT"
            if method is None:
                continue

            model = detect_model(run_name)
            if model is None:
                continue
            dataset = detect_dataset(run_name)
            if dataset is None:
                continue
            sft = has_sft(run_name)
            label = f"{model}-SFT" if sft else model

            if "step" not in run_group.columns:
                continue

            # Train margins
            for margin_col, split in [("train_rewards_margins", "train"), ("eval_rewards_margins", "eval")]:
                if margin_col not in run_group.columns:
                    continue
                sub = run_group[["step", margin_col]].dropna().copy()
                if sub.empty:
                    continue
                sub.columns = ["step", "reward_margin"]
                sub["reward_margin"] = pd.to_numeric(sub["reward_margin"], errors="coerce")
                sub = sub.dropna()
                if sub.empty:
                    continue
                sub["method"] = method
                sub["model"] = model
                sub["label"] = label
                sub["dataset"] = dataset
                sub["split"] = split
                sub["sft"] = sft
                rows.append(sub)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── Plot function ────────────────────────────────────────────────────────────

def make_plot(plot_df, title, filename, group_col="label", caption=None):
    """One plot with bootstrapped 95% CI bands per group.
    Legend placed to the right of the plot. Caption below."""
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = sorted(plot_df[group_col].unique())
    color_map = {}

    for i, group_val in enumerate(groups):
        gdf = plot_df[plot_df[group_col] == group_val]

        # Aggregate per step across all runs with same label
        agg = gdf.groupby("step")["reward_margin"].agg(list).reset_index()
        agg = agg.sort_values("step")

        steps = agg["step"].values
        means, lows, highs = [], [], []
        for vals in agg["reward_margin"]:
            m, lo, hi = bootstrap_ci(vals)
            means.append(m)
            lows.append(lo)
            highs.append(hi)

        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        color_map[group_val] = color

        ax.plot(steps, means, color=color, linewidth=2, label=group_val)
        ax.fill_between(steps, lows, highs, color=color, alpha=0.15)

    ax.set_xlabel("Step", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Reward Margin", fontsize=AXIS_LABEL_FONT_SIZE)

    # Circle legend markers, placed to the right of the plot
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map.get(l, '#000'), markersize=10,
               markeredgecolor=color_map.get(l, '#000'), linewidth=0)
        for l in labels
    ]
    if new_handles:
        ax.legend(new_handles, labels,
                  bbox_to_anchor=(1.02, 1), loc='upper left',
                  frameon=True, fancybox=True, shadow=False,
                  fontsize=14, handletextpad=0.3, borderaxespad=0)

    # Caption below the plot
    if caption:
        fig.text(0.5, -0.02, caption, ha='center', fontsize=11,
                 fontstyle='italic', wrap=True)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"  Saved: {filename}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Reward Margin Plots with Bootstrapped 95% CI")
    print(f"  Models: {MODELS_FILTER}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    print("\n>>> Loading COALA data (coala1-4)...")
    coala_df = load_coala()
    print(f"  COALA: {len(coala_df)} rows")

    print("\n>>> Loading DPO/ORPO data (competition1-2)...")
    comp_df = load_competition()
    print(f"  DPO/ORPO: {len(comp_df)} rows")

    print("\n>>> Loading wandb_metrics_rebuttal...")
    rebuttal_df = load_wandb_metrics_rebuttal()
    print(f"  Rebuttal metrics: {len(rebuttal_df)} rows")

    frames = [f for f in [coala_df, comp_df, rebuttal_df] if not f.empty]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if df.empty:
        print("  No data found!")
        return

    print(f"\n  Total: {len(df)} rows")
    print(f"  Methods: {sorted(df['method'].unique())}")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Labels: {sorted(df['label'].unique())}")

    # ── Plot A: Per method, per dataset — models compared (Figure 1 style) ─
    print("\n>>> Plot A: Per method+dataset, models compared...")
    for method in sorted(df["method"].unique()):
        for dataset in DATASETS:
            for split in ["train", "eval"]:
                sub = df[(df["method"] == method) & (df["dataset"] == dataset) & (df["split"] == split)]
                if sub.empty or sub["label"].nunique() < 1:
                    continue
                split_label = "test set" if split == "eval" else "training set"
                caption = f"Figure: {method} reward margins on the {dataset.upper()} dataset ({split_label}). Shaded regions show bootstrapped 95% CI (2000 resamples)."
                fname = os.path.join(OUTPUT_DIR, f"{method}_{dataset}_{split}.png")
                make_plot(sub, f"{method} — {dataset.upper()} ({split})", fname, group_col="label", caption=caption)

    # ── Plot B: Per dataset — COALA vs DPO vs ORPO (all models aggregated) ─
    print("\n>>> Plot B: Per dataset, methods compared...")
    for dataset in DATASETS:
        for split in ["train", "eval"]:
            sub = df[(df["dataset"] == dataset) & (df["split"] == split)]
            if sub.empty or sub["method"].nunique() < 2:
                continue
            split_label = "test set" if split == "eval" else "training set"
            caption = f"Figure: Comparison of alignment methods on {dataset.upper()} ({split_label}). Bootstrapped 95% CI over per-step reward margins."
            fname = os.path.join(OUTPUT_DIR, f"all_methods_{dataset}_{split}.png")
            make_plot(sub, f"{dataset.upper()} ({split})", fname, group_col="method", caption=caption)

    # ── Plot C: COALA vs DPO side-by-side per dataset (Figure 1a vs 1b) ──
    print("\n>>> Plot C: COALA vs DPO per dataset, per model...")
    for dataset in DATASETS:
        for split in ["train", "eval"]:
            coala = df[(df["method"] == "COALA") & (df["dataset"] == dataset) & (df["split"] == split)].copy()
            dpo = df[(df["method"] == "DPO") & (df["dataset"] == dataset) & (df["split"] == split)].copy()
            if coala.empty and dpo.empty:
                continue
            coala["label"] = coala["label"].apply(lambda x: f"COALA {x}")
            dpo["label"] = dpo["label"].apply(lambda x: f"DPO {x}")
            combined = pd.concat([coala, dpo], ignore_index=True)
            if combined["label"].nunique() < 2:
                continue
            split_label = "test set" if split == "eval" else "training set"
            caption = f"Figure: COALA vs DPO reward margin comparison on {dataset.upper()} ({split_label}). Bootstrapped 95% CI (2000 resamples) per model."
            fname = os.path.join(OUTPUT_DIR, f"COALA_vs_DPO_{dataset}_{split}.png")
            make_plot(combined, f"COALA vs DPO — {dataset.upper()} ({split})", fname, group_col="label", caption=caption)

    # ── Summary CSV ──────────────────────────────────────────────────────
    print("\n>>> Summary with CIs...")
    summary_rows = []
    for (method, model, dataset, split), grp in df.groupby(["method", "model", "dataset", "split"]):
        vals = grp["reward_margin"].values
        m, lo, hi = bootstrap_ci(vals)
        hw = (hi - lo) / 2 if not (np.isnan(lo) or np.isnan(hi)) else np.nan
        summary_rows.append({
            "method": method, "model": model, "dataset": dataset, "split": split,
            "mean": round(m, 4), "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            "ci_half_width": round(hw, 4), "formatted": f"{m:.4f} +/-{hw:.4f}",
            "n_steps": len(vals),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["method", "dataset", "model"])
    summary_path = os.path.join(OUTPUT_DIR, "reward_margin_summary_with_ci.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary: {summary_path}")

    n_plots = len(glob.glob(os.path.join(OUTPUT_DIR, "*.png")))
    print(f"\n{'=' * 60}")
    print(f"  DONE — {n_plots} plots in {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
