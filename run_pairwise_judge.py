"""
GPT-4 pairwise judge for COALA rebuttal — produces Table 7 (ArenaHard-style WR%).
Compares all method pairs on the same prompts using GPT-4 as judge.

For each (model, dataset) combination, compares every method pair:
  COALA vs DPO, COALA vs ORPO, COALA vs SFT, DPO vs ORPO, etc.

Outputs:
  eval_results/pairwise/judgments.csv       — raw per-question judgments
  eval_results/pairwise/win_rates.csv       — aggregated win rates
  eval_results/tables/table7_pairwise.md    — markdown table matching paper Table 7

Usage:
  python run_pairwise_judge.py
  python run_pairwise_judge.py --judge_model gpt-4o
  python run_pairwise_judge.py --dry_run   # preview matchups without API calls
"""

import os
import json
import csv
import time
import random
import argparse
from collections import defaultdict
from pathlib import Path

# Load API key from .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["OPENAI_API_KEY"] = key
                break

from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIRS = {
    "COALA": os.path.join(BASE_DIR, "generated_output_rebuttal_COALA"),
    "DPO":   os.path.join(BASE_DIR, "generated_output_rebuttal_DPO"),
    "ORPO":  os.path.join(BASE_DIR, "generated_output_rebuttal_ORPO"),
    "SFT":   os.path.join(BASE_DIR, "generated_output_rebuttal_SFT"),
    "SimPO": os.path.join(BASE_DIR, "generated_output_rebuttal_SIMPO"),
}

RESULTS_DIR = os.path.join(BASE_DIR, "eval_results", "pairwise")
TABLES_DIR = os.path.join(BASE_DIR, "eval_results", "tables")

# Known dataset/model keywords for parsing filenames
DATASETS = ["edu", "imdb", "ultra", "helpsteer"]
MODEL_KEYWORDS = {
    "distilgpt2": "distilgpt2", "distilgpt": "distilgpt2",
    "gpt2": "gpt2", "gpt2-124m": "gpt2",
    "dolphin": "dolphin7b", "dolphin7b": "dolphin7b", "dolphin-2.1-7b": "dolphin7b",
    "mistral": "mistral7b", "mistral7b": "mistral7b", "mistral-7b": "mistral7b",
    "llama8b": "llama8b", "llama-8b": "llama8b", "llama-3.1": "llama8b",
    "llama3b": "llama3b", "llama-3.2-3b": "llama3b", "llama-3.2": "llama3b",
}

MODEL_DISPLAY = {
    "distilgpt2": "DistilGPT", "gpt2": "GPT-2", "mistral7b": "Mistral-7B",
    "dolphin7b": "Dolphin-7B", "llama8b": "LLaMA-8B", "llama3b": "LLaMA-3B",
}

JUDGE_PROMPT = """You are an impartial judge evaluating two AI assistant responses to the same question.

**Question:** {question}

**Response A:**
{response_a}

**Response B:**
{response_b}

Compare the two responses on helpfulness, accuracy, relevance, and overall quality.
Which response is better? Reply with EXACTLY one of:
- "A" if Response A is better
- "B" if Response B is better
- "tie" if they are equally good

Your verdict:"""


def parse_filename(filename, method_label):
    """Extract (model, dataset, sft_status) from a JSON filename."""
    s = filename.lower().replace(".json", "").replace("-", "_").replace(".", "_")

    dataset = None
    for d in DATASETS:
        if d in s:
            dataset = d
            break

    model = None
    for kw, canon in sorted(MODEL_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if kw.replace("-", "_").replace(".", "_") in s:
            model = canon
            break

    sft = "sft" in s and "nosft" not in s

    return model, dataset, sft


def load_all_outputs():
    """Load all generated outputs, indexed by (method, model, dataset) -> list of QA pairs."""
    all_outputs = {}

    for method, dirpath in OUTPUT_DIRS.items():
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dirpath, fname)
            model, dataset, sft = parse_filename(fname, method)
            if model is None or dataset is None:
                continue

            with open(fpath) as f:
                data = json.load(f)

            key = (method, model, dataset)
            if key not in all_outputs:
                all_outputs[key] = []
            all_outputs[key].extend(data)

    return all_outputs


def find_matchups(all_outputs):
    """Find all valid method pairs that share the same (model, dataset) and prompts."""
    # Group by (model, dataset)
    by_model_dataset = defaultdict(dict)
    for (method, model, dataset), qa_list in all_outputs.items():
        by_model_dataset[(model, dataset)][method] = qa_list

    matchups = []
    for (model, dataset), method_outputs in by_model_dataset.items():
        methods = sorted(method_outputs.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                m1, m2 = methods[i], methods[j]
                # Find shared questions
                q1 = {item["question"]: item for item in method_outputs[m1]}
                q2 = {item["question"]: item for item in method_outputs[m2]}
                shared = set(q1.keys()) & set(q2.keys())
                if shared:
                    matchups.append({
                        "model": model, "dataset": dataset,
                        "method_a": m1, "method_b": m2,
                        "pairs": [(q1[q], q2[q]) for q in shared],
                    })
    return matchups


def judge_pair(client, question, response_a, response_b, judge_model, swap=False):
    """Call GPT-4 to judge which response is better. Optionally swap to reduce position bias."""
    if swap:
        response_a, response_b = response_b, response_a

    prompt = JUDGE_PROMPT.format(
        question=question, response_a=response_a, response_b=response_b
    )

    try:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        verdict = resp.choices[0].message.content.strip().lower()

        if "tie" in verdict:
            return "tie"
        elif "a" in verdict and "b" not in verdict:
            return "B" if swap else "A"
        elif "b" in verdict and "a" not in verdict:
            return "A" if swap else "B"
        else:
            return "tie"
    except Exception as e:
        print(f"    API error: {e}")
        return "error"


def run_judgments(matchups, judge_model, dry_run=False):
    """Run all pairwise judgments."""
    client = None if dry_run else OpenAI()
    all_judgments = []

    total_pairs = sum(len(m["pairs"]) for m in matchups)
    print(f"\n  Total matchups: {len(matchups)}")
    print(f"  Total question pairs to judge: {total_pairs}")
    if dry_run:
        print("  (DRY RUN — no API calls)")
        return []

    for mi, matchup in enumerate(matchups):
        model = matchup["model"]
        dataset = matchup["dataset"]
        m_a = matchup["method_a"]
        m_b = matchup["method_b"]

        print(f"\n  [{mi+1}/{len(matchups)}] {MODEL_DISPLAY.get(model, model)} / {dataset}: {m_a} vs {m_b} ({len(matchup['pairs'])} questions)")

        for qi, (item_a, item_b) in enumerate(matchup["pairs"]):
            question = item_a["question"]
            ans_a = item_a.get("answer", "")
            ans_b = item_b.get("answer", "")

            # Judge twice (swapped) to reduce position bias
            v1 = judge_pair(client, question, ans_a, ans_b, judge_model, swap=False)
            v2 = judge_pair(client, question, ans_a, ans_b, judge_model, swap=True)

            # Resolve: if both agree, use that; if disagree, tie
            if v1 == v2:
                final = v1
            else:
                final = "tie"

            all_judgments.append({
                "model": model, "dataset": dataset,
                "method_a": m_a, "method_b": m_b,
                "question": question[:100],
                "verdict_normal": v1, "verdict_swapped": v2,
                "final_verdict": final,
            })

            print(f"    Q{qi+1}: {final} (raw: {v1}/{v2})")
            time.sleep(0.5)  # Rate limiting

    return all_judgments


def compute_win_rates(judgments):
    """Compute win rates excluding ties, per (model, dataset, method_a, method_b)."""
    groups = defaultdict(lambda: {"A": 0, "B": 0, "tie": 0, "error": 0})

    for j in judgments:
        key = (j["model"], j["dataset"], j["method_a"], j["method_b"])
        groups[key][j["final_verdict"]] += 1

    # Also compute per-method overall win rate (like Table 7)
    # For each (model, dataset), compute each method's WR against all others
    method_wins = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})

    for (model, dataset, m_a, m_b), counts in groups.items():
        key_a = (model, dataset, m_a)
        key_b = (model, dataset, m_b)
        method_wins[key_a]["wins"] += counts["A"]
        method_wins[key_a]["losses"] += counts["B"]
        method_wins[key_a]["ties"] += counts["tie"]
        method_wins[key_b]["wins"] += counts["B"]
        method_wins[key_b]["losses"] += counts["A"]
        method_wins[key_b]["ties"] += counts["tie"]

    win_rates = {}
    for (model, dataset, method), stats in method_wins.items():
        total_ex_ties = stats["wins"] + stats["losses"]
        if total_ex_ties > 0:
            wr = (stats["wins"] / total_ex_ties) * 100
        else:
            wr = 50.0
        win_rates[(method, model, dataset)] = {
            "wr_ex_ties": round(wr, 2),
            "wins": stats["wins"],
            "losses": stats["losses"],
            "ties": stats["ties"],
        }

    return groups, win_rates


def build_table7_md(win_rates, out_dir):
    """Build Table 7 in markdown — ArenaHard-style pairwise WR excluding ties."""
    datasets = ["edu", "imdb", "ultra", "helpsteer"]
    methods_order = ["COALA", "ORPO", "DPO", "SFT", "SimPO"]

    # Group by model
    models_with_data = set()
    for (method, model, dataset) in win_rates:
        models_with_data.add(model)

    lines = []
    lines.append("## Table 7. Custom Pairwise Win Rate Excluding Ties (GPT-4 Judge)")
    lines.append("")

    csv_rows = []

    for model in ["distilgpt2", "gpt2", "mistral7b", "dolphin7b", "llama8b", "llama3b"]:
        if model not in models_with_data:
            continue

        display = MODEL_DISPLAY.get(model, model)
        lines.append(f"### {display}")
        lines.append("")
        lines.append("| METHOD | " + " | ".join(d.upper() for d in datasets) + " |")
        lines.append("|" + "---|" * (1 + len(datasets)))

        for method in methods_order:
            cells = [f"**{method}**"]
            for dataset in datasets:
                wr_data = win_rates.get((method, model, dataset))
                if wr_data:
                    val = wr_data["wr_ex_ties"]
                    cells.append(f"{val:.2f}")
                else:
                    cells.append("—")
                csv_rows.append({
                    "model": model, "method": method, "dataset": dataset,
                    "wr_ex_ties": wr_data["wr_ex_ties"] if wr_data else None,
                    "wins": wr_data["wins"] if wr_data else None,
                    "losses": wr_data["losses"] if wr_data else None,
                    "ties": wr_data["ties"] if wr_data else None,
                })
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    os.makedirs(out_dir, exist_ok=True)

    md_path = os.path.join(out_dir, "table7_pairwise.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    csv_path = os.path.join(out_dir, "table7_pairwise.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "method", "dataset", "wr_ex_ties", "wins", "losses", "ties"])
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\n  Table 7 -> {md_path}")
    print(f"  Table 7 -> {csv_path}")
    return md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", default="gpt-4o", help="OpenAI model for judging")
    parser.add_argument("--dry_run", action="store_true", help="Preview matchups without API calls")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    print("="*60)
    print("  COALA Pairwise Judge (Table 7)")
    print(f"  Judge model: {args.judge_model}")
    print("="*60)

    print("\n>>> Loading all generated outputs...")
    all_outputs = load_all_outputs()
    print(f"  Loaded {len(all_outputs)} (method, model, dataset) combinations")
    for key in sorted(all_outputs.keys()):
        print(f"    {key[0]:8s} | {key[1]:12s} | {key[2]:10s} | {len(all_outputs[key])} questions")

    print("\n>>> Finding matchups...")
    matchups = find_matchups(all_outputs)
    print(f"  Found {len(matchups)} matchups")

    if args.dry_run:
        for m in matchups:
            print(f"    {m['model']:12s} / {m['dataset']:10s}: {m['method_a']} vs {m['method_b']} ({len(m['pairs'])} Qs)")
        return

    print("\n>>> Running GPT-4 judgments...")
    judgments = run_judgments(matchups, args.judge_model, dry_run=args.dry_run)

    # Save raw judgments
    judgments_csv = os.path.join(RESULTS_DIR, "judgments.csv")
    if judgments:
        with open(judgments_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=judgments[0].keys())
            w.writeheader()
            w.writerows(judgments)
        print(f"\n  Raw judgments -> {judgments_csv}")

    # Compute win rates
    print("\n>>> Computing win rates...")
    pairwise_groups, win_rates = compute_win_rates(judgments)

    # Save win rates
    wr_csv = os.path.join(RESULTS_DIR, "win_rates.csv")
    with open(wr_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "model", "dataset", "wr_ex_ties", "wins", "losses", "ties"])
        w.writeheader()
        for (method, model, dataset), stats in sorted(win_rates.items()):
            w.writerow({"method": method, "model": model, "dataset": dataset, **stats})
    print(f"  Win rates -> {wr_csv}")

    # Build Table 7
    print("\n>>> Building Table 7...")
    md_path = build_table7_md(win_rates, TABLES_DIR)

    # Print table to stdout
    print("\n")
    with open(md_path) as f:
        print(f.read())

    print("="*60)
    print("  COMPLETE")
    print(f"  Judgments : {judgments_csv}")
    print(f"  Win rates : {wr_csv}")
    print(f"  Table 7   : {md_path}")
    print("="*60)


if __name__ == "__main__":
    main()
