'''
Enhanced guided sampling inference for CVX-DPO models from cvx_rebuttal_models_5090.
Generates text guided by trained/finetuned cvxNN preference models.
Output in FastChat/MT-Bench compatible JSON format for LLM judge evaluation.

Each cvxNN model x prompt is run N_RUNS times (default 3) with different seeds
so that mean ± std and 95% CIs can be reported (Tables 1 & 7).
Results saved under run1/, run2/, run3/ sub-folders.

Usage:
    python guidance_sampling_pool2.py                # run all, 3 runs
    python guidance_sampling_pool2.py --runs 5       # 5 runs instead of 3
'''

import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Keep JAX off GPU — only needed for pickle loading

import torch
import numpy as np
import jax.numpy as jnp
from jax.nn import relu
import pickle
import json
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import glob


# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_output_rebuttal2_COALA")
CVX_BASE = os.path.join(BASE_DIR, "cvx_rebuttal_models_5090")
SFT_BASE = os.path.join(BASE_DIR, "SFT_models")

N_RUNS = 3
BASE_SEED = 42
GUIDANCE_SCALE = 2.0
TOP_K = 15
TOP_P = 0.9
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 256
GUIDE_EVERY_N = 5

# Map short model names (used in CVX dir names) to HuggingFace model IDs
MODEL_HF_MAP = {
    "distilgpt2": "distilbert/distilgpt2",
    "gpt2": "openai-community/gpt2",
    "dolphin7b": "cognitivecomputations/dolphin-2.1-mistral-7b",
    "dolphin": "cognitivecomputations/dolphin-2.1-mistral-7b",
    "llama8b": "meta-llama/Llama-3.1-8B",
    "llama": "meta-llama/Llama-3.1-8B",
    "mistral7b": "mistralai/Mistral-7B-v0.1",
    "mistral": "mistralai/Mistral-7B-v0.1",
}

# Map short model names to SFT checkpoint directory prefixes in SFT_BASE
MODEL_SFT_MAP = {
    "distilgpt2": "SFT_distilbert_distilgpt2",
    "gpt2": "SFT_openai-community_gpt2",
    "dolphin7b": "SFT_dolphin2.1-7B",
    "dolphin": "SFT_dolphin2.1-7B",
    "llama8b": "SFT_meta-llama_Llama-3.1-8B",
    "llama": "SFT_meta-llama_Llama-3.1-8B",
    "mistral7b": "SFT_mistralai_Mistral-7B-v0.1",
    "mistral": "SFT_mistralai_Mistral-7B-v0.1",
}

# Fallback SFT models ranked by size (largest first) for when no direct match
# Used when a model x dataset SFT checkpoint doesn't exist
SFT_FALLBACK_ORDER = [
    "SFT_meta-llama_Llama-3.1-8B",
    "SFT_mistralai_Mistral-7B-v0.1",
    "SFT_dolphin2.1-7B",
    "SFT_openai-community_gpt2",
    "SFT_distilbert_distilgpt2",
]

# Same prompts as generate_competitors.py for fair comparison
DATASET_PROMPTS = {
    "helpsteer": [
        "How do I write a professional email to request time off?",
        "What are some effective strategies for managing stress?",
        "Explain the pros and cons of remote work.",
        "How can I improve my public speaking skills?",
        "What steps should I take to start learning a new programming language?",
    ],
    "edu": [
        "Explain the concept of photosynthesis to a high school student.",
        "What is the difference between mitosis and meiosis?",
        "Describe how the water cycle works in simple terms.",
        "What are the main causes of climate change?",
        "Explain Newton's three laws of motion.",
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


def discover_experiments():
    """Auto-discover all cvxNN models in cvx_rebuttal_models_5090.

    Handles three directory patterns:
      - cvxNN_trained_{base|SFT}_{model}_{dataset}
      - finetuned_{base|SFT}_{model}_{dataset}         (lowercase)
      - Finetuned_cvx_{model}_{dataset}_inference_ready (uppercase)
    """
    experiments = []

    for subdir in sorted(os.listdir(CVX_BASE)):
        full_path = os.path.join(CVX_BASE, subdir)
        if not os.path.isdir(full_path):
            continue

        # Must have a pkl file
        pkl_files = glob.glob(os.path.join(full_path, "*_cvx_mlp.pkl"))
        if not pkl_files:
            continue
        pkl_path = pkl_files[0]

        name = subdir

        # ── Pattern 1: cvxNN_trained_{base|SFT}_{model}_{dataset} ────────
        if name.startswith("cvxNN_trained_"):
            stage = "cronos"
            rest = name[len("cvxNN_trained_"):]

            if rest.startswith("SFT_"):
                sft = True
                rest = rest[len("SFT_"):]
            elif rest.startswith("base_"):
                sft = False
                rest = rest[len("base_"):]
            else:
                print(f"  Skipping {subdir}: no SFT/base prefix")
                continue

            model_name, dataset = _parse_model_dataset(rest)

        # ── Pattern 2: finetuned_{base|SFT}_{model}_{dataset} (lowercase)
        elif name.startswith("finetuned_"):
            stage = "finetuned"
            rest = name[len("finetuned_"):]

            if rest.startswith("SFT_"):
                sft = True
                rest = rest[len("SFT_"):]
            elif rest.startswith("base_"):
                sft = False
                rest = rest[len("base_"):]
            else:
                print(f"  Skipping {subdir}: no SFT/base prefix")
                continue

            model_name, dataset = _parse_model_dataset(rest)

        # ── Pattern 3: Finetuned_cvx_{model}_{dataset}_inference_ready ───
        elif name.startswith("Finetuned_cvx_"):
            stage = "finetuned_cvx"
            rest = name[len("Finetuned_cvx_"):]
            # Strip _inference_ready suffix
            rest = rest.replace("_inference_ready", "")
            # rest is like "distilgpt2_edu", "dolphin_imdb", "llama_ultra"
            model_name, dataset = _parse_model_dataset(rest)
            # Finetuned_cvx models were finetuned from SFT checkpoints
            sft = True

        else:
            continue

        if model_name is None or dataset is None:
            print(f"  Skipping {subdir}: could not parse model/dataset")
            continue

        sft_tag = "sft" if sft else "nosft"
        model_id = f"COALA_{stage}_{model_name}_{dataset}_{sft_tag}"

        experiments.append({
            "model_id": model_id,
            "cvx_pkl": pkl_path,
            "model_name": model_name,
            "dataset": dataset,
            "sft": sft,
            "stage": stage,
            "dir_name": subdir,
        })

    return experiments


def _parse_model_dataset(rest):
    """Parse '{model}_{dataset}' from remainder string.

    Tries longest model name first to avoid partial matches
    (e.g. 'dolphin7b' before 'dolphin').
    """
    for known_model in sorted(MODEL_HF_MAP.keys(), key=len, reverse=True):
        if rest.startswith(known_model + "_"):
            dataset = rest[len(known_model) + 1:]
            return known_model, dataset
    return None, None


def _find_sft_dir_by_dataset(dataset):
    """Find all valid SFT dirs ending with _{dataset}, case-insensitive on dir name."""
    matches = []
    if not os.path.isdir(SFT_BASE):
        return matches
    for d in os.listdir(SFT_BASE):
        if d.lower().endswith("_" + dataset.lower()) and _is_valid_sft(os.path.join(SFT_BASE, d)):
            matches.append(d)
    return sorted(matches)


def _get_sft_base_model(sft_path):
    """Read the base_model_name_or_path from an SFT checkpoint's adapter_config."""
    with open(os.path.join(sft_path, "adapter_config.json")) as f:
        return json.load(f)["base_model_name_or_path"]


def find_sft_checkpoint(model_name, dataset):
    """Find matching SFT checkpoint for a model x dataset pair.

    Only returns checkpoints whose LoRA adapter is architecturally compatible
    with the generation model (same base_model_name_or_path in adapter_config).

    Strategy:
      1. Direct match: MODEL_SFT_MAP[model_name] + dataset (exact)
      2. Fuzzy scan: keyword match for inconsistent naming (e.g. dolphin-2.1-7b)
      3. Fallback: largest SFT model with correct dataset + same architecture
    All steps verify architecture compatibility before returning.
    """
    gen_hf_id = MODEL_HF_MAP.get(model_name, "")

    # 1. Direct match (exact)
    prefix = MODEL_SFT_MAP.get(model_name)
    if prefix:
        sft_path = os.path.join(SFT_BASE, f"{prefix}_{dataset}")
        if _is_valid_sft(sft_path):
            sft_base = _get_sft_base_model(sft_path)
            if sft_base == gen_hf_id:
                return sft_path
            else:
                print(f"    Direct SFT match {prefix}_{dataset} has incompatible "
                      f"base ({sft_base} != {gen_hf_id}), skipping")

    # 2. Fuzzy scan for dirs with same model keyword + dataset
    #    Handles inconsistent naming (e.g. SFT_dolphin-2.1-7b_imdb vs SFT_dolphin2.1-7B)
    if prefix:
        after_sft = prefix.split("SFT_", 1)[1] if "SFT_" in prefix else prefix
        keyword = after_sft.split("-")[0].split("_")[0].rstrip("0123456789.").lower()
        for d in sorted(os.listdir(SFT_BASE)):
            if (keyword in d.lower() and
                    d.lower().endswith("_" + dataset) and
                    _is_valid_sft(os.path.join(SFT_BASE, d))):
                sft_path = os.path.join(SFT_BASE, d)
                sft_base = _get_sft_base_model(sft_path)
                if sft_base == gen_hf_id:
                    print(f"    Fuzzy SFT match: {d}")
                    return sft_path

    # 3. Fallback: largest available SFT model with correct dataset + same architecture
    for fallback_prefix in SFT_FALLBACK_ORDER:
        sft_path = os.path.join(SFT_BASE, f"{fallback_prefix}_{dataset}")
        if not _is_valid_sft(sft_path):
            continue
        sft_base = _get_sft_base_model(sft_path)
        if sft_base == gen_hf_id:
            print(f"    No direct SFT for {model_name}/{dataset}, "
                  f"falling back to {fallback_prefix}_{dataset}")
            return sft_path

    print(f"    No compatible SFT for {model_name}/{dataset}, using base model")
    return None


def _is_valid_sft(path):
    """Check if an SFT checkpoint directory exists and has an adapter config."""
    return (os.path.isdir(path) and
            os.path.isfile(os.path.join(path, "adapter_config.json")))


class CVXGuidedSampler:
    def __init__(self, cvx_pkl_path, model_name, dataset, use_sft=False):
        # ── Load cvxNN ──────────────────────────────────────────────────
        print(f"  Loading cvxNN from {os.path.basename(cvx_pkl_path)}")
        with open(cvx_pkl_path, 'rb') as f:
            cvx_model = pickle.load(f)
        self.theta1 = np.array(cvx_model.theta1)
        self.theta2 = np.array(cvx_model.theta2)
        print(f"  theta1: {self.theta1.shape}, theta2: {self.theta2.shape}")

        # ── Load LLM ───────────────────────────────────────────────────
        hf_model_id = MODEL_HF_MAP[model_name]
        sft_checkpoint = find_sft_checkpoint(model_name, dataset) if use_sft else None

        if sft_checkpoint:
            print(f"  Loading SFT model from {sft_checkpoint}")
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            )
            # Resize embeddings if the SFT checkpoint extended the tokenizer
            try:
                sft_tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
                self.model.resize_token_embeddings(len(sft_tokenizer))
            except (ValueError, ImportError, OSError):
                # SFT checkpoint may not have a loadable tokenizer (e.g. different arch),
                # try the base model's tokenizer size — PeftModel will error if mismatch
                pass
            self.model = PeftModel.from_pretrained(self.model, sft_checkpoint)
            self.model = self.model.merge_and_unload()
        else:
            if use_sft:
                print(f"  WARNING: No SFT checkpoint found for {model_name}/{dataset}, using base model")
            print(f"  Loading base model: {hf_model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            )

        self.model = self.model.to("cuda")
        self.model.eval()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        except (ValueError, ImportError):
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.device = next(self.model.parameters()).device
        self.hidden_size = self.model.config.hidden_size
        print(f"  Model on {self.device}, hidden_size={self.hidden_size}")

    def _batch_score(self, generated, topk_indices):
        """Score all top-k candidates in a single batched forward pass."""
        k = topk_indices.shape[1]
        expanded = generated.expand(k, -1)
        candidates = torch.cat([expanded, topk_indices[0].unsqueeze(1)], dim=1)

        with torch.no_grad():
            outputs = self.model(candidates, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            features = last_hidden.mean(dim=1)
            features = features.cpu().float().numpy()

        activations = np.maximum(0, features @ self.theta1)
        scores = activations @ self.theta2
        return torch.tensor(scores, device=self.device)

    def guided_generate(self, prompt, max_new_tokens=256, top_k=15, top_p=0.9,
                        temperature=0.7, repetition_penalty=1.1, guide_every_n=5,
                        guidance_scale=2.0):
        """Generate text with batched cvxNN guidance applied every N tokens."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]
        tokens_since_guidance = 0

        for _ in tqdm(range(max_new_tokens), desc="Generating", leave=False):
            with torch.no_grad():
                outputs = self.model(generated)
                next_token_logits = outputs.logits[:, -1, :]

            if repetition_penalty > 1.0:
                for token_id in set(generated[0].tolist()):
                    next_token_logits[0, token_id] /= repetition_penalty

            if guidance_scale > 0 and tokens_since_guidance >= guide_every_n:
                topk_values, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
                guidance_scores = self._batch_score(generated, topk_indices)

                score_range = guidance_scores.max() - guidance_scores.min()
                if score_range > 1e-6:
                    guidance_scores = (guidance_scores - guidance_scores.min()) / score_range
                else:
                    guidance_scores = torch.zeros_like(guidance_scores)

                logits_adjustment = torch.zeros_like(next_token_logits)
                logits_adjustment[0, topk_indices[0]] = (guidance_scale * guidance_scores).half()
                next_token_logits = next_token_logits + logits_adjustment
                tokens_since_guidance = 0
            else:
                tokens_since_guidance += 1

            next_token_logits = next_token_logits / temperature

            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                probs = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs)

            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token[0, 0] == self.tokenizer.eos_token_id:
                break

        text = self.tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        return text


def run_experiment(exp, n_runs=N_RUNS):
    """Run guided generation for one cvxNN model across n_runs seeds.

    Saves to OUTPUT_DIR/run1/, OUTPUT_DIR/run2/, etc.
    Returns list of saved file paths.
    """
    dataset = exp["dataset"]
    prompts = DATASET_PROMPTS.get(dataset)
    if prompts is None:
        print(f"  No prompts defined for dataset '{dataset}', skipping.")
        return []

    print(f"\n{'='*60}")
    print(f"  {exp['model_id']}")
    print(f"  Stage: {exp['stage']} | Model: {exp['model_name']} | "
          f"Dataset: {dataset} | SFT: {exp['sft']}")
    print(f"{'='*60}")

    sampler = CVXGuidedSampler(
        cvx_pkl_path=exp["cvx_pkl"],
        model_name=exp["model_name"],
        dataset=dataset,
        use_sft=exp["sft"],
    )

    saved = []
    for run_idx in range(1, n_runs + 1):
        seed = BASE_SEED + run_idx
        set_seed(seed)
        print(f"\n  ── run{run_idx} (seed={seed}) ──")

        results = []
        for i, prompt in enumerate(prompts):
            print(f"    Prompt {i+1}: {prompt[:60]}...")
            start = time.time()

            response = sampler.guided_generate(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                top_k=TOP_K,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                guide_every_n=GUIDE_EVERY_N,
                guidance_scale=GUIDANCE_SCALE,
            )

            elapsed = time.time() - start
            print(f"    Generated in {elapsed:.1f}s")

            results.append({
                "question_id": i + 1,
                "model_id": exp["model_id"],
                "question": prompt,
                "answer": response,
                "metadata": {
                    "method": "COALA",
                    "stage": exp["stage"],
                    "base_model": MODEL_HF_MAP[exp["model_name"]],
                    "model_short": exp["model_name"],
                    "dataset": dataset,
                    "sft": exp["sft"],
                    "seed": seed,
                    "run": run_idx,
                    "guidance_scale": GUIDANCE_SCALE,
                    "temperature": TEMPERATURE,
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "guide_every_n": GUIDE_EVERY_N,
                    "generation_time_seconds": round(elapsed, 2),
                },
            })

        run_dir = os.path.join(OUTPUT_DIR, f"run{run_idx}")
        os.makedirs(run_dir, exist_ok=True)
        filename = f"{exp['model_id']}.json"
        filepath = os.path.join(run_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  Saved {len(results)} generations to {filepath}")
        saved.append(filepath)

    del sampler
    torch.cuda.empty_cache()

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help=f"Number of runs per experiment (default {N_RUNS})")
    args = parser.parse_args()

    n_runs = args.runs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    experiments = discover_experiments()
    print(f"Discovered {len(experiments)} cvxNN models:\n")
    for exp in experiments:
        print(f"  {exp['model_id']}")

    # Skip already-completed experiments (all N runs present)
    saved_files = []
    skipped = 0
    for exp in experiments:
        all_done = all(
            os.path.isfile(os.path.join(OUTPUT_DIR, f"run{r}", f"{exp['model_id']}.json"))
            for r in range(1, n_runs + 1)
        )
        if all_done:
            skipped += 1
            continue
        paths = run_experiment(exp, n_runs=n_runs)
        saved_files.extend(paths)

    if skipped:
        print(f"\n  Skipped {skipped} already-completed experiments.")

    print(f"\n{'='*60}")
    print(f"All generations complete! {len(saved_files)} files saved across {n_runs} runs.")
    print(f"Output directory: {OUTPUT_DIR}")
    for f in saved_files:
        print(f"  {f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
