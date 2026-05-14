"""
DPO rebuttal experiments: Llama-3.2-3B SFT models on edu, helpsteer, imdb, ultra.
Runs DPO for 1 epoch per dataset, logs reward margins to wandb + CSV.

Usage:
    python dpo_train_demo.py
"""

import os
import time
import csv
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, PeftModel, PeftConfig
from trl import DPOTrainer, DPOConfig

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(BASE_DIR, "DPO_rebuttal")
DATA_DIR = os.path.join(BASE_DIR, "from_downloads")
CSV_PATH = os.path.join(BASE_DIR, "dpo_rebuttal_results.csv")

# Model-dataset pairs: (SFT checkpoint, dataset JSON, short name)
EXPERIMENTS = [
    (
        os.path.join(BASE_DIR, "from_downloads", "SFT_meta-llama_Llama-3.2-3B_edu"),
        os.path.join(DATA_DIR, "train_test_edu_dataset_full.json"),
        "edu",
    ),
    (
        os.path.join(BASE_DIR, "from_downloads", "SFT_meta-llama_Llama-3.2-3B_helpsteer"),
        os.path.join(DATA_DIR, "train_test_helpsteer_dataset_full.json"),
        "helpsteer",
    ),
    (
        os.path.join(BASE_DIR, "from_downloads", "SFT_meta-llama_Llama-3.2-3B_imdb"),
        os.path.join(DATA_DIR, "train_test_imdb_dataset_full.json"),
        "imdb",
    ),
    (
        os.path.join(BASE_DIR, "from_downloads", "SFT_meta-llama_Llama-3.2-3B_ultra"),
        os.path.join(DATA_DIR, "train_test_ultra_dataset_full.json"),
        "ultra",
    ),
]

NUM_EPOCHS = 1
MAX_SEQ_LENGTH = 512
MAX_PROMPT_LENGTH = 256
LOGGING_STEPS = 25

# RTX 4090 peak bf16 TFLOPS (same calculation used in cronos_trainer.py)
PEAK_BF16_TFLOPS = 330
EFFICIENCY = 0.70  # 70% efficiency assumption


def estimate_tflops(duration_seconds):
    """Estimate TFLOPs used, matching cronos_trainer.py calculation."""
    gflops_per_sec = PEAK_BF16_TFLOPS * EFFICIENCY * 1000  # 231000
    tflops_used = (gflops_per_sec * duration_seconds) / 1000
    return tflops_used


class RewardMarginCallback(TrainerCallback):
    """Captures reward margin metrics at each logging step for CSV export."""

    def __init__(self):
        self.margins = []  # list of (step, train_reward_margin, train_reward_accuracy)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        margin = logs.get("rewards/margins", logs.get("reward_margin", None))
        accuracy = logs.get("rewards/accuracies", logs.get("reward_accuracy", None))
        # Also check older TRL naming
        if margin is None:
            margin = logs.get("train_rewards/margins", None)
        if accuracy is None:
            accuracy = logs.get("train_rewards/accuracies", None)

        if margin is not None:
            self.margins.append({
                "step": step,
                "reward_margin": margin,
                "reward_accuracy": accuracy,
            })


MAX_TRAIN_SAMPLES = 7000

def load_dpo_dataset(json_path):
    """Load JSONL, split into train/eval, cap train at MAX_TRAIN_SAMPLES."""
    ds = load_dataset("json", data_files=json_path, split="train")
    split = ds.train_test_split(test_size=0.1, seed=1024)
    train = split["train"]
    if len(train) > MAX_TRAIN_SAMPLES:
        train = train.shuffle(seed=1024).select(range(MAX_TRAIN_SAMPLES))
    return train, split["test"]


def run_dpo(sft_checkpoint, dataset_path, dataset_name):
    """Run DPO finetuning for one model-dataset pair."""
    run_name = f"DPO_Llama-3.2-3B_{dataset_name}_sft"
    output_dir = os.path.join(OUTPUT_BASE, run_name)

    print(f"\n{'='*60}")
    print(f"  DPO: {run_name}")
    print(f"  SFT checkpoint: {sft_checkpoint}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # ── Load SFT adapter and merge into base model ──────────────────────
    peft_config_loaded = PeftConfig.from_pretrained(sft_checkpoint)
    base_model_name = peft_config_loaded.base_model_name_or_path

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
    )

    # Load and merge SFT adapter so DPO starts from SFT weights
    model = PeftModel.from_pretrained(model, sft_checkpoint)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # ── Dataset ─────────────────────────────────────────────────────────
    train_dataset, eval_dataset = load_dpo_dataset(dataset_path)
    print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # ── LoRA for DPO ────────────────────────────────────────────────────
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=32,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # ── DPO config ──────────────────────────────────────────────────────
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=5e-5,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=1000,
        bf16=True,
        tf32=True,
        push_to_hub=False,
        report_to="wandb",
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        beta=0.1,
        loss_type="sigmoid",
    )

    # ── Wandb init ──────────────────────────────────────────────────────
    wandb.init(project="rebuttal", name=run_name, config={
        "model": "meta-llama/Llama-3.2-3B",
        "sft_checkpoint": sft_checkpoint,
        "dataset": dataset_name,
        "num_epochs": NUM_EPOCHS,
        "beta": 0.1,
        "loss_type": "sigmoid",
        "lora_r": 256,
        "lora_alpha": 128,
    })

    # ── Reward margin callback ──────────────────────────────────────────
    margin_callback = RewardMarginCallback()

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model,
        ref_model=None,  # no ref model needed with PEFT
        peft_config=lora_config,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[margin_callback],
    )

    # ── Train ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    # ── Save ────────────────────────────────────────────────────────────
    trainer.save_model()

    # ── Metrics ─────────────────────────────────────────────────────────
    estimated_tflops = estimate_tflops(total_time)

    peak_memory = "N/A"
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    wandb.log({
        "total_run_time_seconds": total_time,
        "estimated_tflops": estimated_tflops,
        "peak_gpu_memory_mb": peak_memory,
    })

    # ── Save reward margins to per-run CSV ──────────────────────────────
    margin_csv = os.path.join(output_dir, "reward_margins.csv")
    if margin_callback.margins:
        with open(margin_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "reward_margin", "reward_accuracy"])
            writer.writeheader()
            writer.writerows(margin_callback.margins)
        print(f"  Saved {len(margin_callback.margins)} margin entries to {margin_csv}")

    print(f"  Completed {run_name} in {total_time:.1f}s | TFLOPS: {estimated_tflops:.1f}")

    # ── Cleanup ─────────────────────────────────────────────────────────
    del model, trainer
    torch.cuda.empty_cache()
    wandb.finish()

    return {
        "run_name": run_name,
        "dataset": dataset_name,
        "total_time_seconds": round(total_time, 2),
        "estimated_tflops": round(estimated_tflops, 2),
        "peak_gpu_memory_mb": round(peak_memory, 2) if isinstance(peak_memory, float) else peak_memory,
        "output_dir": output_dir,
        "num_margin_steps": len(margin_callback.margins),
    }


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    all_results = []

    for sft_checkpoint, dataset_path, dataset_name in EXPERIMENTS:
        result = run_dpo(sft_checkpoint, dataset_path, dataset_name)
        all_results.append(result)

    # ── Summary CSV ─────────────────────────────────────────────────────
    if all_results:
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSummary saved to {CSV_PATH}")

    print("\nAll DPO experiments complete!")
    print(f"Checkpoints saved at: {OUTPUT_BASE}/")
    for r in all_results:
        print(f"  {r['run_name']}: {r['total_time_seconds']}s, {r['estimated_tflops']} TFLOPS")


if __name__ == "__main__":
    main()
