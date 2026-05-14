"""
ORPO rebuttal experiments: Llama-3.2-3B SFT models on edu, imdb, ultra.
Runs ORPO for 1 epoch per dataset, logs reward margins to wandb + CSV.

Usage:
    python orpo_train_demo.py
    python orpo_train_demo.py --experiments edu imdb
"""

import os
import gc
import csv
import glob
import time
import argparse
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, PeftModel, PeftConfig, prepare_model_for_kbit_training
from trl import ORPOConfig, ORPOTrainer

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(BASE_DIR, "ORPO_rebuttal")
CSV_PATH    = os.path.join(BASE_DIR, "orpo_rebuttal_results.csv")
DATA_DIR    = os.path.join(BASE_DIR, "from_downloads")

BASE_MODEL_ID = "meta-llama/Llama-3.2-3B"

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "edu": {
        "sft_checkpoint": os.path.join(BASE_DIR, "from_downloads", "SFT_meta-llama_Llama-3.2-3B_edu"),
        "dataset_path":   os.path.join(DATA_DIR, "train_test_edu_dataset_full.json"),
        "sft_tag":        "sft",
        "run_name":       "ORPO_Llama-3.2-3B_edu_sft",
    },
    "imdb": {
        "sft_checkpoint": os.path.join(BASE_DIR, "from_downloads", "SFT_meta-llama_Llama-3.2-3B_imdb"),
        "dataset_path":   os.path.join(DATA_DIR, "train_test_imdb_dataset_full.json"),
        "sft_tag":        "sft",
        "run_name":       "ORPO_Llama-3.2-3B_imdb_sft",
    },
    "ultra": {
        "sft_checkpoint": os.path.join(BASE_DIR, "from_downloads", "SFT_meta-llama_Llama-3.2-3B_ultra"),
        "dataset_path":   os.path.join(DATA_DIR, "train_test_ultra_dataset_full.json"),
        "sft_tag":        "sft",
        "run_name":       "ORPO_Llama-3.2-3B_ultra_sft",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ORPO HYPERPARAMETERS — tuned for RTX 4090 24GB, Llama-3.2-3B in 4-bit
# ─────────────────────────────────────────────────────────────────────────────

ORPO_CFG = {
    "use_4bit":   True,
    "lora_r":     16,
    "lora_alpha": 32,
    "batch_size": 2,
    "grad_accum": 8,
    "grad_ckpt":  True,
    "lr":         8e-6,
    "max_len":    512,
    "target_modules": ["q_proj", "v_proj"],
}

MAX_TRAIN_SAMPLES = 7000

# TFLOPS estimation (same as cronos_trainer.py)
PEAK_BF16_TFLOPS = 330
EFFICIENCY = 0.70

def estimate_tflops(duration_seconds):
    gflops_per_sec = PEAK_BF16_TFLOPS * EFFICIENCY * 1000
    return (gflops_per_sec * duration_seconds) / 1000


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_preference_dataset(dataset_path, max_samples=MAX_TRAIN_SAMPLES):
    """Load JSONL preference dataset, cap train size, split 95/5."""
    ds = load_dataset("json", data_files=dataset_path, split="train")
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=1024).select(range(max_samples))
    split = ds.train_test_split(test_size=0.05, seed=1024)
    return split["train"], split["test"]


# ─────────────────────────────────────────────────────────────────────────────
# REWARD MARGIN CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class RewardMarginCallback(TrainerCallback):
    def __init__(self):
        self.rows = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step

        train_margin = logs.get("rewards/margins", logs.get("reward_margin", None))
        train_acc    = logs.get("rewards/accuracies", logs.get("reward_accuracy", None))
        train_loss   = logs.get("loss", None)
        if train_margin is not None:
            self.rows.append({"split": "train", "step": step,
                              "reward_margin": train_margin,
                              "reward_accuracy": train_acc, "loss": train_loss})

        eval_margin = logs.get("eval_rewards/margins", logs.get("eval_reward_margin", None))
        eval_acc    = logs.get("eval_rewards/accuracies", logs.get("eval_reward_accuracy", None))
        eval_loss   = logs.get("eval_loss", None)
        if eval_margin is not None:
            self.rows.append({"split": "eval", "step": step,
                              "reward_margin": eval_margin,
                              "reward_accuracy": eval_acc, "loss": eval_loss})


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CSV
# ─────────────────────────────────────────────────────────────────────────────

MASTER_HEADER = [
    "run_name", "model", "sft_tag", "dataset", "split", "step",
    "reward_margin", "reward_accuracy", "loss",
    "total_time_sec", "estimated_tflops", "peak_gpu_mb", "checkpoint_dir",
]

def init_master_csv():
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(MASTER_HEADER)

def flush_to_master_csv(run_name, sft_tag, dataset_key, callback,
                        total_time, tflops, peak_gpu_mb, checkpoint_dir):
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_HEADER)
        for row in callback.rows:
            w.writerow({
                "run_name":        run_name,
                "model":           "Llama-3.2-3B",
                "sft_tag":         sft_tag,
                "dataset":         dataset_key,
                "split":           row["split"],
                "step":            row["step"],
                "reward_margin":   row["reward_margin"],
                "reward_accuracy": row["reward_accuracy"],
                "loss":            row["loss"],
                "total_time_sec":  round(total_time, 1),
                "estimated_tflops": round(tflops, 1),
                "peak_gpu_mb":     round(peak_gpu_mb, 0),
                "checkpoint_dir":  checkpoint_dir,
            })
    print(f"  Flushed {len(callback.rows)} rows -> {CSV_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_orpo(experiment_key):
    exp      = EXPERIMENTS[experiment_key]
    run_name = exp["run_name"]
    sft_ckpt = exp["sft_checkpoint"]
    sft_tag  = exp["sft_tag"]
    out_dir  = os.path.join(OUTPUT_BASE, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ORPO : {run_name}")
    print(f"  Base : {BASE_MODEL_ID}")
    print(f"  SFT  : {sft_ckpt}")
    print(f"  Data : {exp['dataset_path']}")
    print(f"{'='*60}")

    # ── Load dataset ──────────────────────────────────────────────────────
    train_dataset, eval_dataset = load_preference_dataset(exp["dataset_path"])
    print(f"  Train: {len(train_dataset):,}  |  Eval: {len(eval_dataset):,}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # ── Load base model ───────────────────────────────────────────────────
    load_kwargs = {
        "device_map":  "auto",
        "use_cache":   False,
        "torch_dtype": torch.bfloat16,
    }
    if ORPO_CFG["use_4bit"]:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)

    # ── Merge SFT adapter ─────────────────────────────────────────────────
    print(f"  Merging SFT adapter: {sft_ckpt}")
    model = PeftModel.from_pretrained(model, sft_ckpt)
    model = model.merge_and_unload()

    if ORPO_CFG["use_4bit"]:
        model = prepare_model_for_kbit_training(model)

    # ── Fresh ORPO LoRA ───────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=ORPO_CFG["lora_r"],
        lora_alpha=ORPO_CFG["lora_alpha"],
        lora_dropout=0.05,
        bias="none",
        target_modules=ORPO_CFG["target_modules"],
        task_type="CAUSAL_LM",
    )

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb.init(
        project="rebuttal",
        name=run_name,
        config={
            "model":          "Llama-3.2-3B",
            "base_model":     BASE_MODEL_ID,
            "sft_checkpoint": sft_ckpt,
            "sft_tag":        sft_tag,
            "dataset":        experiment_key,
            "method":         "ORPO",
            "lora_r":         ORPO_CFG["lora_r"],
            "lr":             ORPO_CFG["lr"],
            "batch_size":     ORPO_CFG["batch_size"],
            "grad_accum":     ORPO_CFG["grad_accum"],
        },
    )

    # ── ORPO config ───────────────────────────────────────────────────────
    orpo_config = ORPOConfig(
        output_dir=out_dir,
        beta=0.1,
        max_length=ORPO_CFG["max_len"],
        num_train_epochs=1,
        per_device_train_batch_size=ORPO_CFG["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=ORPO_CFG["grad_accum"],
        gradient_checkpointing=ORPO_CFG["grad_ckpt"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=ORPO_CFG["lr"],
        lr_scheduler_type="linear",
        warmup_steps=10,
        max_grad_norm=0.3,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        report_to="wandb",
        push_to_hub=False,
    )

    callback = RewardMarginCallback()

    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[callback],
    )

    # ── Train ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    peak_mem = (torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available() else 0.0)
    tflops = estimate_tflops(total_time)

    trainer.save_model(out_dir)

    wandb.log({
        "total_run_time_seconds": total_time,
        "estimated_tflops": tflops,
        "peak_gpu_memory_mb": peak_mem,
    })

    flush_to_master_csv(run_name, sft_tag, experiment_key, callback,
                        total_time, tflops, peak_mem, os.path.abspath(out_dir))

    # Save per-run reward margin CSV
    margin_csv = os.path.join(out_dir, "reward_margins.csv")
    if callback.rows:
        with open(margin_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["split", "step", "reward_margin", "reward_accuracy", "loss"])
            w.writeheader()
            w.writerows(callback.rows)

    print(f"  Checkpoint -> {out_dir}")
    print(f"  Time: {total_time/60:.1f} min | TFLOPS: {tflops:.1f} | Peak VRAM: {peak_mem:.0f} MB")

    wandb.finish()
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+",
                   default=list(EXPERIMENTS.keys()),
                   choices=list(EXPERIMENTS.keys()),
                   help="Which experiments to run (default: all)")
    return p.parse_args()


if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    args = parse_args()
    init_master_csv()

    failed = []
    for exp_key in args.experiments:
        try:
            run_orpo(exp_key)
        except Exception as e:
            print(f"\n  FAILED: {exp_key}  --  {e}\n")
            failed.append(exp_key)
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  COMPLETE  {len(args.experiments) - len(failed)}/{len(args.experiments)} succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  CSV -> {CSV_PATH}")
    print(f"  Checkpoints -> {OUTPUT_BASE}/")
    print(f"{'='*60}")
