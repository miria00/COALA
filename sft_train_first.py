"""
SFT training script for Llama-3.2-3B on edu, ultra, imdb, and helpsteer datasets.
Optimized for single RTX 4090 (24GB VRAM).

Usage:
    python sft_train_first.py

Trains 1 epoch on each dataset sequentially, saves LoRA adapters to from_downloads/.
"""

import os
import json
import glob
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

# ── Config ──────────────────────────────────────────────────────────────────
BASE_MODEL = "meta-llama/Llama-3.2-3B"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(BASE_DIR, "from_downloads")
DATA_DIR = os.path.join(BASE_DIR, "from_downloads")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

DATASETS = {
    "edu":       os.path.join(DATA_DIR, "train_test_edu_dataset_full.json"),
    "ultra":     os.path.join(DATA_DIR, "train_test_ultra_dataset_full.json"),
    "imdb":      os.path.join(DATA_DIR, "train_test_imdb_dataset_full.json"),
    "helpsteer": None,  # built from txt files
}

# LoRA config — matches existing adapters in from_downloads/
LORA_R = 32
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Training — tuned for RTX 4090 speed
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4  # effective batch = 32
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LOGGING_STEPS = 50
SAVE_STEPS = 500


def build_helpsteer_json():
    """Build a JSONL file from datasets/helpsteer/{pos,neg} txt files."""
    pos_dir = os.path.join(DATASETS_DIR, "helpsteer", "pos")
    neg_dir = os.path.join(DATASETS_DIR, "helpsteer", "neg")
    out_path = os.path.join(DATA_DIR, "train_test_helpsteer_dataset_full.json")

    if os.path.exists(out_path):
        print(f"  HelpSteer JSON already exists at {out_path}")
        return out_path

    pos_files = sorted(glob.glob(os.path.join(pos_dir, "*.txt")))
    neg_files = sorted(glob.glob(os.path.join(neg_dir, "*.txt")))

    print(f"  Building HelpSteer JSON from {len(pos_files)} pairs...")
    with open(out_path, "w", encoding="utf-8") as out:
        for pf, nf in zip(pos_files, neg_files):
            with open(pf, "r", encoding="utf-8") as f:
                pos_text = f.read()
            with open(nf, "r", encoding="utf-8") as f:
                neg_text = f.read()

            # Split on the assistant tag to separate prompt from response
            # Format: <|im_start|>system...user...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
            pos_split = pos_text.rsplit("<|im_start|>assistant\n", 1)
            neg_split = neg_text.rsplit("<|im_start|>assistant\n", 1)

            if len(pos_split) == 2 and len(neg_split) == 2:
                prompt = pos_split[0]
                chosen = "<|im_start|>assistant\n" + pos_split[1]
                rejected = "<|im_start|>assistant\n" + neg_split[1]
                row = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Saved {out_path}")
    return out_path


def load_sft_dataset(json_path):
    """Load JSONL and format for SFT (prompt + chosen as training text)."""
    ds = load_dataset("json", data_files=json_path, split="train")
    split = ds.train_test_split(test_size=0.1, seed=1024)

    def format_text(batch):
        texts = []
        for prompt, chosen in zip(batch["prompt"], batch["chosen"]):
            prompt = " ".join(prompt) if isinstance(prompt, list) else str(prompt)
            if isinstance(chosen, list):
                chosen = " ".join(
                    item["text"] if isinstance(item, dict) and "text" in item else str(item)
                    for item in chosen
                )
            else:
                chosen = str(chosen)
            texts.append(prompt + chosen)
        return {"text": texts}

    train_ds = split["train"].map(format_text, batched=True, num_proc=4, remove_columns=split["train"].column_names)
    eval_ds = split["test"].map(format_text, batched=True, num_proc=4, remove_columns=split["test"].column_names)
    return train_ds, eval_ds


def train_on_dataset(dataset_name, json_path, model, tokenizer):
    """Train LoRA adapter for one dataset, save checkpoint."""
    output_dir = os.path.join(OUTPUT_BASE, f"SFT_meta-llama_Llama-3.2-3B_{dataset_name}")

    if os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        print(f"  Checkpoint already exists at {output_dir}, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"  Training on: {dataset_name}")
    print(f"  Data: {json_path}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    train_ds, eval_ds = load_sft_dataset(json_path)
    print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # Fresh LoRA adapter for each dataset
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_SEQ_LENGTH,
        packing=True,
        dataset_text_field="text",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        report_to="wandb",
        run_name=f"SFT_Llama-3.2-3B_{dataset_name}",
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Free the adapter before next dataset
    del trainer, peft_model
    torch.cuda.empty_cache()

    print(f"  Saved adapter to {output_dir}")


def main():
    # ── Load base model once (shared across all datasets) ───────────────────
    print(f"Loading base model: {BASE_MODEL}")

    # 4-bit quantization to fit 3B model comfortably on 4090
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # ── Build helpsteer JSON if needed ──────────────────────────────────────
    helpsteer_path = build_helpsteer_json()
    DATASETS["helpsteer"] = helpsteer_path

    # ── Train on each dataset ───────────────────────────────────────────────
    for ds_name, json_path in DATASETS.items():
        train_on_dataset(ds_name, json_path, model, tokenizer)

    print("\n All training complete!")
    print(" Checkpoints saved to:")
    for ds_name in DATASETS:
        print(f"   {os.path.join(OUTPUT_BASE, f'SFT_meta-llama_Llama-3.2-3B_{ds_name}')}")


if __name__ == "__main__":
    main()
