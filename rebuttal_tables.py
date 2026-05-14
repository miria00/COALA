"""
rebuttal_tables.py
──────────────────
Reads all reward margin data from the COALA project and outputs markdown tables:
  1. Mean ± Standard Error on the test (eval) set
  2. Mean with Bootstrapped 95% CI

Sources (auto-discovered):
  - wandb_metrics_rebuttal/*.csv (all_methods_combined, DPO_rebuttal, ORPO_rebuttal, SIMPO, rebuttal)
  - orpo_rebuttal_results.csv, dpo_rebuttal_results.csv
  - csv_results_rebuttal/*.csv
  - DPO/ORPO run dirs with reward_margins.csv

Usage:
  python rebuttal_tables.py
  python rebuttal_tables.py --split eval
  python rebuttal_tables.py --split train
  python rebuttal_tables.py --split both
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "eval_results", "tables")

DATASETS = ["edu", "imdb", "ultra", "helpsteer"]


# ── Parsing helpers ──────────────────────────────────────────────────────────

def detect_method(name):
    s = name.lower()
    if "coala" in s or "finetune" in s or "cronos" in s:
        return "COALA"
    if "simpo" in s:
        return "SimPO"
    if "orpo" in s:
        return "ORPO"
    if "dpo" in s:
        return "DPO"
    if "sft" in s:
        return "SFT"
    return None


def detect_model(name):
    s = name.lower()
    if "dolphin" in s:
        return "Dolphin-7B"
    if "llama" in s and ("8b" in s or "3.1" in s):
        return "LLaMA-8B"
    if "llama" in s and ("3b" in s or "3.2" in s):
        return "LLaMA-3B"
    if "mistral" in s:
        return "Mistral-7B"
    if "distilgpt2" in s or "distilgpt" in s:
        return "DistilGPT2"
    if "gpt2" in s:
        return "GPT-2"
    return None


def detect_dataset(name):
    s = name.lower()
    for d in DATASETS:
        if d in s:
            return d
    return None


def detect_sft(name):
    s = name.lower()
    return "sft" in s and "nosft" not in s and "no_sft" not in s


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    if len(arr) == 1:
        return arr[0], arr[0], arr[0]
    rng = np.random.RandomState(seed)
    boot = np.array([np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 - (1 - ci) / 2) * 100)
    return float(np.mean(arr)), float(lo), float(hi)


# ── Load all data sources ────────────────────────────────────────────────────

def load_all():
    all_frames = []

    # Source 1: wandb_metrics_rebuttal/*.csv (richest source)
    metrics_dir = os.path.join(BASE_DIR, "wandb_metrics_rebuttal")
    if os.path.isdir(metrics_dir):
        for csv_path in glob.glob(os.path.join(metrics_dir, "*.csv")):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if "run_name" not in df.columns or "step" not in df.columns:
                continue

            for run_name, grp in df.groupby("run_name"):
                method = detect_method(run_name)
                model = detect_model(run_name)
                dataset = detect_dataset(run_name)
                if not all([method, model, dataset]):
                    continue

                for margin_col, split in [("train_rewards_margins", "train"), ("eval_rewards_margins", "eval")]:
                    if margin_col not in grp.columns:
                        continue
                    sub = grp[["step", margin_col]].dropna().copy()
                    if sub.empty:
                        continue
                    sub.columns = ["step", "reward_margin"]
                    sub["method"] = method
                    sub["model"] = model
                    sub["dataset"] = dataset
                    sub["split"] = split
                    sub["sft"] = detect_sft(run_name)
                    all_frames.append(sub)

        print(f"  wandb_metrics_rebuttal: loaded")

    # Source 2: Top-level CSVs (orpo_rebuttal_results.csv, etc.)
    for csv_name in ["orpo_rebuttal_results.csv", "dpo_rebuttal_results.csv"]:
        csv_path = os.path.join(BASE_DIR, csv_name)
        if not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.lower().strip() for c in df.columns]
        except Exception:
            continue

        if "reward_margin" not in df.columns or "step" not in df.columns:
            continue

        # These have: run_name, model, dataset, split, step, reward_margin
        for _, row in df.iterrows():
            rn = str(row.get("run_name", ""))
            method = detect_method(rn)
            model_val = detect_model(rn) or str(row.get("model", ""))
            if model_val:
                model_val = detect_model(model_val) or model_val
            dataset_val = detect_dataset(rn) or str(row.get("dataset", ""))
            if not method or not model_val or not dataset_val:
                continue

        # Bulk load instead of row-by-row
        needed = ["step", "reward_margin"]
        if "split" in df.columns:
            needed.append("split")
        sub = df[needed].dropna(subset=["reward_margin"]).copy()
        if "split" not in sub.columns:
            sub["split"] = "train"

        # Parse metadata from run_name
        if "run_name" in df.columns:
            sub["method"] = df["run_name"].apply(lambda x: detect_method(str(x)))
            sub["model"] = df["run_name"].apply(lambda x: detect_model(str(x)))
            sub["dataset"] = df["run_name"].apply(lambda x: detect_dataset(str(x)))
            sub["sft"] = df["run_name"].apply(lambda x: detect_sft(str(x)))
        elif "model" in df.columns:
            sub["method"] = df.get("method", "unknown")
            sub["model"] = df["model"].apply(lambda x: detect_model(str(x)) or x)
            sub["dataset"] = df.get("dataset", "unknown")
            sub["sft"] = False

        sub = sub.dropna(subset=["method", "model", "dataset"])
        if not sub.empty:
            all_frames.append(sub)
            print(f"  {csv_name}: {len(sub)} rows")

    # Source 3: Per-run reward_margins.csv from DPO/ORPO output dirs
    for run_dir_base in [
        os.path.join(BASE_DIR, "DPO_rebuttal_llama3B_alldatasets"),
        os.path.join(BASE_DIR, "ORPO_rebuttal"),
    ]:
        if not os.path.isdir(run_dir_base):
            continue
        for run_dir in glob.glob(os.path.join(run_dir_base, "*")):
            margin_csv = os.path.join(run_dir, "reward_margins.csv")
            if not os.path.isfile(margin_csv):
                continue
            dirname = os.path.basename(run_dir)
            method = detect_method(dirname)
            model = detect_model(dirname)
            dataset = detect_dataset(dirname)
            if not all([method, model, dataset]):
                continue

            try:
                df = pd.read_csv(margin_csv)
            except Exception:
                continue

            step_col = "step" if "step" in df.columns else "_step"
            margin_col = "reward_margin" if "reward_margin" in df.columns else None
            if margin_col is None:
                for c in df.columns:
                    if "margin" in c.lower():
                        margin_col = c
                        break
            if margin_col is None or step_col not in df.columns:
                continue

            sub = df[[step_col, margin_col]].dropna().copy()
            sub.columns = ["step", "reward_margin"]
            split_col = "split" if "split" in df.columns else None
            if split_col and split_col in df.columns:
                sub["split"] = df[split_col]
            else:
                sub["split"] = "train"
            sub["method"] = method
            sub["model"] = model
            sub["dataset"] = dataset
            sub["sft"] = detect_sft(dirname)
            all_frames.append(sub)

        print(f"  {os.path.basename(run_dir_base)}: loaded")

    # Source 4: csv_results_rebuttal
    csv_results_dir = os.path.join(BASE_DIR, "csv_results_rebuttal")
    if os.path.isdir(csv_results_dir):
        for csv_path in glob.glob(os.path.join(csv_results_dir, "*.csv")):
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.lower().strip() for c in df.columns]
            except Exception:
                continue
            if "reward_margin" not in df.columns:
                continue
            if "step" not in df.columns:
                continue

            sub = df[["step", "reward_margin"]].dropna().copy()
            if "split" in df.columns:
                sub["split"] = df["split"]
            else:
                sub["split"] = "train"

            fname = os.path.basename(csv_path)
            if "run_name" in df.columns:
                sub["method"] = df["run_name"].apply(lambda x: detect_method(str(x)))
                sub["model"] = df["run_name"].apply(lambda x: detect_model(str(x)))
                sub["dataset"] = df["run_name"].apply(lambda x: detect_dataset(str(x)))
            else:
                sub["method"] = detect_method(fname)
                sub["model"] = detect_model(fname)
                sub["dataset"] = detect_dataset(fname)
            sub["sft"] = detect_sft(fname)
            sub = sub.dropna(subset=["method", "model", "dataset"])
            if not sub.empty:
                all_frames.append(sub)

        print(f"  csv_results_rebuttal: loaded")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined["reward_margin"] = pd.to_numeric(combined["reward_margin"], errors="coerce")
    combined = combined.dropna(subset=["reward_margin"])
    return combined


# ── Summarise ────────────────────────────────────────────────────────────────

def summarise(df, group_cols):
    rows = []
    for keys, grp in df.groupby(group_cols):
        vals = grp["reward_margin"].dropna().values
        if len(vals) == 0:
            continue
        mean = np.mean(vals)
        std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        n = len(vals)
        se = std / np.sqrt(n)
        _, lo, hi = bootstrap_ci(vals)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        row.update({"mean": mean, "std": std, "se": se, "ci_lo": lo, "ci_hi": hi, "n": n})
        rows.append(row)
    return pd.DataFrame(rows)


# ── Markdown tables ──────────────────────────────────────────────────────────

def table_se(summary, group_cols):
    header = group_cols + ["Mean RM", "Std Dev", "SE", "n"]
    sep = "|" + "|".join(["---"] * len(header)) + "|"
    lines = [
        "## Table A — Mean ± Standard Error",
        "", "*SE = std / sqrt(n), where n = number of logged steps.*", "",
        "|" + "|".join(header) + "|", sep,
    ]
    for _, r in summary.iterrows():
        cells = [str(r[c]) for c in group_cols]
        cells += [f"{r['mean']:.4f}", f"{r['std']:.4f}", f"**±{r['se']:.4f}**", str(int(r['n']))]
        lines.append("|" + "|".join(cells) + "|")
    return "\n".join(lines)


def table_ci(summary, group_cols):
    header = group_cols + ["Mean", "CI Lower", "CI Upper", "±Half-width", "n"]
    sep = "|" + "|".join(["---"] * len(header)) + "|"
    lines = [
        "## Table B — Bootstrapped 95% Confidence Interval",
        "", "*2000-sample bootstrap resampling over per-step reward margins.*", "",
        "|" + "|".join(header) + "|", sep,
    ]
    for _, r in summary.iterrows():
        hw = (r['ci_hi'] - r['ci_lo']) / 2
        cells = [str(r[c]) for c in group_cols]
        cells += [f"{r['mean']:.4f}", f"{r['ci_lo']:.4f}", f"{r['ci_hi']:.4f}", f"**±{hw:.4f}**", str(int(r['n']))]
        lines.append("|" + "|".join(cells) + "|")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="both", choices=["train", "eval", "both"])
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "=" * 55)
    print("  Generating CI Tables for Rebuttal")
    print("=" * 55 + "\n")

    df = load_all()
    if df.empty:
        print("  No data found!")
        return

    print(f"\n  Total: {len(df):,} rows")
    print(f"  Methods: {sorted(df['method'].unique())}")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")

    if args.split != "both" and "split" in df.columns:
        df = df[df["split"] == args.split]
        print(f"  Filtered to split='{args.split}': {len(df):,} rows")

    group_cols = ["method", "model", "dataset"]
    if args.split == "both" and "split" in df.columns:
        group_cols.append("split")

    summary = summarise(df, group_cols)
    if summary.empty:
        print("  No summarizable data found.")
        return

    summary = summary.sort_values(group_cols).reset_index(drop=True)

    # Build markdown
    sections = [
        "# Reward Margin Error Bar Tables — COALA Rebuttal\n",
        f"*Split: {args.split} | Bootstrap: {args.n_boot} resamples*\n",
        "---\n",
        table_se(summary, group_cols),
        "\n---\n",
        table_ci(summary, group_cols),
    ]
    md = "\n".join(sections)

    # Save
    md_path = os.path.join(OUT_DIR, "rebuttal_ci_tables.md")
    with open(md_path, "w") as f:
        f.write(md)

    csv_path = os.path.join(OUT_DIR, "rebuttal_ci_summary.csv")
    summary.to_csv(csv_path, index=False)

    print(f"\n  Markdown -> {md_path}")
    print(f"  CSV      -> {csv_path}")

    # Print preview
    print("\n" + "-" * 55)
    print(md[:3000])
    if len(md) > 3000:
        print("  [truncated — see file for full table]")

    print(f"\n{'=' * 55}")
    print(f"  DONE")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
