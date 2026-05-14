"""
Downloads nvidia/HelpSteer from HuggingFace and processes it into
pos/neg preference pairs matching the format in datasets/ultra/{pos,neg}.

Output: datasets/helpsteer/pos/ and datasets/helpsteer/neg/
Each file is a ChatML-formatted conversation (same prompt, different responses).
"""

import os
from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "helpsteer")
POS_DIR = os.path.join(OUTPUT_DIR, "pos")
NEG_DIR = os.path.join(OUTPUT_DIR, "neg")

SYSTEM_MSG = "You are a helpful AI assistant."


def format_chatml(prompt, response):
    """Format a prompt-response pair into ChatML template."""
    return (
        f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def main():
    print("Downloading nvidia/HelpSteer...")
    ds = load_dataset("nvidia/HelpSteer", split="train")
    print(f"Loaded {len(ds)} rows")

    os.makedirs(POS_DIR, exist_ok=True)
    os.makedirs(NEG_DIR, exist_ok=True)

    # Group consecutive rows that share the same prompt
    # HelpSteer docs: consecutive samples share prompts (up to 4 responses per prompt)
    groups = {}
    for row in ds:
        prompt = row["prompt"]
        if prompt not in groups:
            groups[prompt] = []
        groups[prompt].append(row)

    pair_idx = 0
    skipped = 0

    for prompt, responses in groups.items():
        if len(responses) < 2:
            skipped += 1
            continue

        # Sort by helpfulness (descending) to pick best and worst
        responses.sort(key=lambda r: r["helpfulness"], reverse=True)
        best = responses[0]
        worst = responses[-1]

        # Skip if same helpfulness score (no clear preference)
        if best["helpfulness"] == worst["helpfulness"]:
            skipped += 1
            continue

        pos_text = format_chatml(prompt, best["response"])
        neg_text = format_chatml(prompt, worst["response"])

        pos_file = os.path.join(POS_DIR, f"example_{pair_idx:06d}.txt")
        neg_file = os.path.join(NEG_DIR, f"example_{pair_idx:06d}.txt")

        with open(pos_file, "w", encoding="utf-8") as f:
            f.write(pos_text)
        with open(neg_file, "w", encoding="utf-8") as f:
            f.write(neg_text)

        pair_idx += 1
        if pair_idx % 1000 == 0:
            print(f"  Written {pair_idx} pairs...")

    print(f"\nDone! {pair_idx} preference pairs written to {OUTPUT_DIR}")
    print(f"Skipped {skipped} prompts (single response or tied helpfulness)")
    print(f"  pos: {POS_DIR}")
    print(f"  neg: {NEG_DIR}")


if __name__ == "__main__":
    main()
