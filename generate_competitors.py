"""
Generate text from DPO, ORPO, and SFT finetuned models (competitor baselines).
Uses the same prompts as guidance_sampling_pool2.py for fair comparison.
Output in FastChat/MT-Bench compatible JSON format.

Each method x model x prompt is run N_RUNS times (default 3) with different
seeds so that mean +/- std and 95% CIs can be reported (Tables 1 & 7).
Results are saved under run1/, run2/, run3/ sub-folders.

Usage:
    python generate_competitors.py                    # run all methods, 3 runs
    python generate_competitors.py --method dpo       # DPO only
    python generate_competitors.py --method orpo      # ORPO only
    python generate_competitors.py --method sft       # SFT only
    python generate_competitors.py --runs 5           # 5 runs instead of 3
"""

import os
import json
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel, PeftConfig

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DPO_BASE = os.path.join(BASE_DIR, "DPO_rebuttal_llama3B_alldatasets")
ORPO_BASE = os.path.join(BASE_DIR, "ORPO_rebuttal")
SFT_BASE = os.path.join(BASE_DIR, "SFT_models")
DPO_OUTPUT_DIR = os.path.join(BASE_DIR, "generated_output_rebuttal2_DPO")
ORPO_OUTPUT_DIR = os.path.join(BASE_DIR, "generated_output_rebuttal2_ORPO")
SFT_OUTPUT_DIR = os.path.join(BASE_DIR, "generated_output_rebuttal2_SFT")

N_RUNS = 3
BASE_SEED = 42
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_K = 15
TOP_P = 0.9

# Same prompts as guidance_sampling_pool2.py
DATASET_PROMPTS = {
    "edu": [
        "Explain the concept of photosynthesis to a high school student.",
        "What is the difference between mitosis and meiosis?",
        "Describe how the water cycle works in simple terms.",
        "What are the main causes of climate change?",
        "Explain Newton's three laws of motion.",
    ],
    "helpsteer": [
        "How do I write a professional email to request time off?",
        "What are some effective strategies for managing stress?",
        "Explain the pros and cons of remote work.",
        "How can I improve my public speaking skills?",
        "What steps should I take to start learning a new programming language?",
    ],
    "imdb": [
        "This movie was absolutely",
        "I went to see the movie and thought it was",
        "The acting in this film was",
        "The plot of this movie was",
        "As a film enthusiast, I found this movie to be",
    ],
    "ultra": [
        "What are the key differences between Python and JavaScript?",
        "Explain quantum computing in simple terms.",
        "What is the significance of the Turing test?",
        "How does blockchain technology work?",
        "Describe the process of natural selection.",
    ],
}

# ── Model name mapping for display ───────────────────────────────────────────
MODEL_SIZE_MAP = {
    "meta-llama/Llama-3.2-3B": ("Llama", "3.2-3B"),
    "meta-llama/Llama-3.1-8B": ("Llama", "3.1-8B"),
    "distilgpt2": ("DistilGPT2", "82M"),
    "distilbert/distilgpt2": ("DistilGPT2", "82M"),
    "gpt2": ("GPT2", "124M"),
    "openai-community/gpt2": ("GPT2", "124M"),
    "cognitivecomputations/dolphin-2.1-mistral-7b": ("Dolphin", "2.1-7B"),
    "mistralai/Mistral-7B-v0.1": ("Mistral", "7B"),
}

# ── DPO experiments (auto-discovered) ───────────────────────────────────────
def discover_dpo_experiments():
    """Auto-discover DPO checkpoints that have adapter_config.json."""
    experiments = []
    if not os.path.isdir(DPO_BASE):
        return experiments

    for subdir in sorted(os.listdir(DPO_BASE)):
        full_path = os.path.join(DPO_BASE, subdir)
        adapter_config = os.path.join(full_path, "adapter_config.json")
        if not os.path.isfile(adapter_config):
            continue

        with open(adapter_config) as f:
            base_model = json.load(f)["base_model_name_or_path"]

        # Detect dataset: last known dataset token in the directory name
        dataset = None
        for ds in DATASET_PROMPTS.keys():
            # Match _dataset at end, or _dataset_ somewhere in the name
            if subdir.endswith("_" + ds) or ("_" + ds + "_") in subdir:
                dataset = ds
                break
        if dataset is None:
            print(f"  Skipping {subdir}: could not detect dataset")
            continue

        # SFT if dir name contains "SFT" or ends with _sft
        sft = "SFT" in subdir or subdir.endswith("_sft")

        experiments.append({
            "checkpoint": full_path,
            "base_model": base_model,
            "dataset": dataset,
            "sft": sft,
            "dir_name": subdir,
        })

    return experiments


# ── ORPO experiments (auto-discovered) ───────────────────────────────────────
def discover_orpo_experiments():
    """Auto-discover ORPO checkpoints that have adapter_config.json.

    Some ORPO dirs only have checkpoint-N/ subdirs without a top-level
    adapter_config.json.  In that case, use the highest-numbered checkpoint.
    """
    experiments = []
    if not os.path.isdir(ORPO_BASE):
        return experiments

    for subdir in sorted(os.listdir(ORPO_BASE)):
        full_path = os.path.join(ORPO_BASE, subdir)
        if not os.path.isdir(full_path):
            continue

        # Try top-level adapter_config first
        adapter_config = os.path.join(full_path, "adapter_config.json")
        checkpoint_path = full_path

        if not os.path.isfile(adapter_config):
            # Fall back to the highest-numbered checkpoint-N/ subdir
            ckpt_dirs = sorted(
                [d for d in os.listdir(full_path)
                 if d.startswith("checkpoint-") and os.path.isfile(
                     os.path.join(full_path, d, "adapter_config.json"))],
                key=lambda x: int(x.split("-")[1]),
            )
            if not ckpt_dirs:
                continue
            checkpoint_path = os.path.join(full_path, ckpt_dirs[-1])
            adapter_config = os.path.join(checkpoint_path, "adapter_config.json")

        with open(adapter_config) as f:
            base_model = json.load(f)["base_model_name_or_path"]

        # Detect dataset
        dataset = None
        for ds in DATASET_PROMPTS.keys():
            if subdir.endswith("_" + ds) or ("_" + ds + "_") in subdir:
                dataset = ds
                break
        if dataset is None:
            print(f"  Skipping {subdir}: could not detect dataset")
            continue

        # SFT if dir name contains "SFT" or ends with _sft
        sft = "SFT" in subdir or subdir.endswith("_sft")

        experiments.append({
            "checkpoint": checkpoint_path,
            "base_model": base_model,
            "dataset": dataset,
            "sft": sft,
            "dir_name": subdir,
        })

    return experiments


def discover_sft_experiments():
    """Auto-discover SFT checkpoints that have adapter_config.json."""
    experiments = []
    if not os.path.isdir(SFT_BASE):
        return experiments

    for subdir in sorted(os.listdir(SFT_BASE)):
        full_path = os.path.join(SFT_BASE, subdir)
        adapter_config = os.path.join(full_path, "adapter_config.json")
        if not os.path.isfile(adapter_config):
            continue

        with open(adapter_config) as f:
            base_model = json.load(f)["base_model_name_or_path"]

        # Parse directory name: SFT_{org}_{model}_{dataset}
        name = subdir.replace("SFT_", "", 1)

        dataset = None
        for ds in DATASET_PROMPTS.keys():
            if name.endswith("_" + ds):
                dataset = ds
                break
        if dataset is None:
            print(f"  Skipping {subdir}: could not detect dataset")
            continue

        experiments.append({
            "checkpoint": full_path,
            "base_model": base_model,
            "dataset": dataset,
            "sft": True,
            "dir_name": subdir,
        })

    return experiments


def load_model(checkpoint_path):
    """Load a LoRA adapter merged into its base model."""
    peft_config = PeftConfig.from_pretrained(checkpoint_path)
    base_model_name = peft_config.base_model_name_or_path

    # Load tokenizer first to check if embeddings were resized during training
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        except (ValueError, ImportError):
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Resize embeddings to match adapter if tokenizer was extended during training
    model.resize_token_embeddings(len(tokenizer))

    # Also check adapter state dict for embedding size mismatch
    import safetensors.torch
    adapter_file = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        adapter_state = safetensors.torch.load_file(adapter_file)
        for key, tensor in adapter_state.items():
            if "embed_tokens" in key or "wte" in key:
                target_vocab_size = tensor.shape[0]
                if target_vocab_size != model.get_input_embeddings().weight.shape[0]:
                    model.resize_token_embeddings(target_vocab_size)
                break
            if "lm_head" in key:
                target_vocab_size = tensor.shape[0]
                if target_vocab_size != model.lm_head.weight.shape[0]:
                    model.resize_token_embeddings(target_vocab_size)
                break

    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()
    model = model.to("cuda")
    model.eval()

    return model, tokenizer, base_model_name


def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7,
             top_k=15, top_p=0.9):
    """Generate text from a prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def build_model_id(method, base_model, dataset, sft):
    """Build a consistent model_id string."""
    model_family, model_size = MODEL_SIZE_MAP.get(base_model, ("Unknown", "Unknown"))
    sft_tag = "sft" if sft else "nosft"
    return f"{method}_{model_family}-{model_size}_{dataset}_{sft_tag}"


def run_generation(method, checkpoint, dataset, sft, output_dir,
                    base_model_override=None, n_runs=N_RUNS):
    """Run generation for one model-dataset pair across n_runs seeds.

    Saves to output_dir/run1/, output_dir/run2/, etc.
    Returns list of saved file paths.
    """
    print(f"\n  Loading checkpoint: {checkpoint}")
    model, tokenizer, base_model = load_model(checkpoint)
    if base_model_override:
        base_model = base_model_override

    model_id = build_model_id(method, base_model, dataset, sft)

    print(f"  model_id: {model_id}")
    print(f"  base_model: {base_model} | dataset: {dataset} | sft: {sft}")

    prompts = DATASET_PROMPTS.get(dataset)
    if not prompts:
        print(f"  No prompts for dataset '{dataset}', skipping.")
        del model
        torch.cuda.empty_cache()
        return []

    saved = []
    for run_idx in range(1, n_runs + 1):
        seed = BASE_SEED + run_idx
        set_seed(seed)
        print(f"\n  ── run{run_idx} (seed={seed}) ──")

        results = []
        for i, prompt in enumerate(prompts):
            print(f"    Prompt {i+1}: {prompt[:60]}...")
            start = time.time()

            response = generate(
                model, tokenizer, prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
            )

            elapsed = time.time() - start
            print(f"    Generated in {elapsed:.1f}s")

            results.append({
                "question_id": i + 1,
                "model_id": model_id,
                "question": prompt,
                "answer": response,
                "metadata": {
                    "method": method,
                    "dataset": dataset,
                    "base_model": base_model,
                    "sft": sft,
                    "seed": seed,
                    "run": run_idx,
                    "temperature": TEMPERATURE,
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "generation_time_seconds": round(elapsed, 2),
                },
            })

        run_dir = os.path.join(output_dir, f"run{run_idx}")
        os.makedirs(run_dir, exist_ok=True)
        filename = f"{model_id}.json"
        filepath = os.path.join(run_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  Saved {len(results)} generations to {filepath}")
        saved.append(filepath)

    del model
    torch.cuda.empty_cache()

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dpo", "orpo", "sft", "all"], default="all")
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help=f"Number of runs per experiment (default {N_RUNS})")
    args = parser.parse_args()

    n_runs = args.runs
    saved_files = []

    # ── DPO generations ──────────────────────────────────────────────────
    if args.method in ("dpo", "all"):
        os.makedirs(DPO_OUTPUT_DIR, exist_ok=True)
        dpo_experiments = discover_dpo_experiments()
        print(f"\n{'='*60}")
        print(f"  DPO GENERATION ({len(dpo_experiments)} models x {n_runs} runs)")
        print(f"{'='*60}")

        for exp in dpo_experiments:
            paths = run_generation(
                method="DPO",
                checkpoint=exp["checkpoint"],
                dataset=exp["dataset"],
                sft=exp["sft"],
                output_dir=DPO_OUTPUT_DIR,
                base_model_override=exp.get("base_model"),
                n_runs=n_runs,
            )
            saved_files.extend(paths)

    # ── ORPO generations ─────────────────────────────────────────────────
    if args.method in ("orpo", "all"):
        os.makedirs(ORPO_OUTPUT_DIR, exist_ok=True)
        orpo_experiments = discover_orpo_experiments()
        print(f"\n{'='*60}")
        print(f"  ORPO GENERATION ({len(orpo_experiments)} models x {n_runs} runs)")
        print(f"{'='*60}")

        for exp in orpo_experiments:
            paths = run_generation(
                method="ORPO",
                checkpoint=exp["checkpoint"],
                dataset=exp["dataset"],
                sft=exp["sft"],
                output_dir=ORPO_OUTPUT_DIR,
                base_model_override=exp.get("base_model"),
                n_runs=n_runs,
            )
            saved_files.extend(paths)

    # ── SFT generations ──────────────────────────────────────────────────
    if args.method in ("sft", "all"):
        os.makedirs(SFT_OUTPUT_DIR, exist_ok=True)
        sft_experiments = discover_sft_experiments()
        print(f"\n{'='*60}")
        print(f"  SFT GENERATION ({len(sft_experiments)} models x {n_runs} runs)")
        print(f"{'='*60}")

        for exp in sft_experiments:
            paths = run_generation(
                method="SFT",
                checkpoint=exp["checkpoint"],
                dataset=exp["dataset"],
                sft=exp["sft"],
                output_dir=SFT_OUTPUT_DIR,
                base_model_override=exp.get("base_model"),
                n_runs=n_runs,
            )
            saved_files.extend(paths)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"All generations complete! {len(saved_files)} files saved across {n_runs} runs.")
    for f in saved_files:
        print(f"  {f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
