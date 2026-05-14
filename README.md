<h1 align="center">COALA</h1>

<p align="center">
  <b>Convex Optimization for Alignment and Preference Learning</b><br>
  A lightweight, reference-free framework for preference fine-tuning of LLMs on a single GPU.
</p>

<p align="center">
  <a href="paper/Coala_preflearn_icml2026.pdf"><img alt="paper" src="https://img.shields.io/badge/paper-ICML%202026-blue.svg"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="jax" src="https://img.shields.io/badge/jax-0.4.33-orange.svg">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green.svg">
</p>

---

COALA (`Convex Optimization for Alignment and preference Learning Algorithm`) re-casts
preference fine-tuning of large language models as a **convex optimization problem**
over the last layer of a two-layer ReLU network (`cvxNN`) stacked on top of a
pre-trained backbone. By replacing the non-convex DPO objective with a convex
reformulation that admits an ADMM solver (`CRONOS`), COALA

  - eliminates the reference model required by DPO (cutting VRAM by `~2x`),
  - reaches stable, monotonically increasing reward margins in `~17.6%` of DPO's TFLOPS,
  - runs end-to-end on a single RTX 4090 (24 GB), and
  - inherits the convergence guarantees of convex programming — no `1e-9` learning
    rates, no grid-searched hyperparameters.

Across five backbones (`DistilGPT2`, `GPT-2`, `Mistral-7B`, `Dolphin-2.6-7B`,
`LLaMA-3.1-8B`) and three datasets (`EduFeedback`, `UltraFeedback`, `IMDb`),
COALA matches or beats DPO and ORPO on AlpacaEval2 length-controlled win rate,
and wins **39.1%** / **42.7%** of head-to-head matchups against the strongest
baseline in our 107-person human study (paper Table 2).

A summary of the method appears below; full derivations, proofs and experiments
are in [`paper/Coala_preflearn_icml2026.pdf`](paper/Coala_preflearn_icml2026.pdf).

## Method

For a pre-trained backbone `f_pre`, COALA stacks a convex two-layer ReLU network
`g_{Θ_1, θ_2}` on the frozen features and trains it in two phases:

  1. **Phase I — CRONOS.** Solve the convex reformulation of the two-layer ReLU
     network (Pilanci & Ergen 2020) by ADMM with preconditioned conjugate
     gradients. Produces `(Θ_1, θ_2)` with an `O(1/K)` convergence guarantee.
  2. **Phase II — Convex preference fine-tuning.** Freeze `Θ_1`. The reference-free
     COALA loss is

     ```
     min_{θ_2}  E [ log( 1 + exp( -β y_w θ_2^T (Θ_1 f_pre(x))_+ + γ ) ) ]
     ```

     which is convex in `θ_2` and solved by AdamW with an `O(1/k^2)`
     accelerated-gradient guarantee (Theorem 4.3).

The implementation is in JAX with `jit` compilation; the ADMM subproblems
reduce to a vector add and one matrix–vector product, which the JAX backend
parallelises efficiently on a single GPU.

## Repository layout

```
COALAgit/
├── paper/                         # ICML 2026 manuscript
├── solve/                         # CRONOS / cvxNN core
│   ├── models/                    #   cvx_relu_mlp, relu_mlp, two_layer_mlp, ...
│   ├── optimizers/                #   admm, pcg, adamW, varpro, dadapt_adamW
│   ├── preconditioner/            #   nystrom
│   ├── training/                  #   train, train_no_jit
│   ├── experiments/               #   lr_experiment
│   └── utils/                     #   gpt2_dataloader, load_data, model_utils, ...
├── utils/                         # data loading / tokenisation helpers
├── dataset_utils/                 # preference-data preprocessing
├── handy_functions/               # misc helpers
│
├── sft_train_first.py             # (optional) Stage 0 — SFT base
├── extract.py                     # Stage 1 — feature extraction (chosen / rejected)
├── extract_prep_prefdata.py       #         preference-data prep for extract.py
├── cronos_trainer.py              # Stage 2 — Phase I (CRONOS, Algorithm 2)
├── finetune_cvxdpo.py             # Stage 3 — Phase II (COALA loss, Algorithm 1)
├── defrun.py                      #         run wrapper for CRONOS
├── test_model_weights.py          #         sanity-check saved cvxNN
├── guided_inference.py            # Stage 4 — IMDb-style guided sampling
├── guidance_sampling_pool.py      #         general attention-pooled sampling
├── guidance_sampling_pool2.py     #         updated multi-prompt batched sampling
│
├── dpo_train_demo.py              # DPO baseline (TRL)
├── orpo_train_demo.py             # ORPO baseline (TRL)
│
├── download_datasets.py           # pull / format HelpSteer, UltraFeedback, IMDb, EduFeedback
├── generate_competitors.py        # generate text from DPO/ORPO/SFT for comparison
├── run_pairwise_judge.py          # GPT-4 pairwise judge (Table 2 / Table 7)
│
├── plot_*.py / scatter_tflops.py  # paper figures
│
└── *.sh                           # end-to-end pipeline drivers
```

## Installation

COALA depends on JAX (with GPU support), PyTorch, HuggingFace `transformers`,
`trl`, `peft`, and `datasets`. We recommend a fresh Python 3.10 environment:

```bash
git clone <your-fork-url> COALAgit
cd COALAgit

python -m venv .venv && source .venv/bin/activate

# JAX 0.4.33 with CUDA 12 — adjust for your CUDA version
pip install --upgrade "jax[cuda12]==0.4.33"

# Core dependencies
pip install torch transformers trl peft datasets accelerate bitsandbytes \
            numpy pandas scikit-learn wandb optax tqdm matplotlib seaborn

# Verify JAX sees your GPU
python jaxtest.py
```

## Quickstart

The four-stage pipeline reproduces COALA on a single dataset/backbone pair.

```bash
# (Optional) Stage 0 — SFT pretraining
python sft_train_first.py

# Stage 1 — extract frozen features (chosen / rejected pairs)
python extract.py \
    --model_path  <hf-model-or-sft-ckpt> \
    --data_path   datasets/edu/ \
    --pool        attn \
    --output_base extracted_features/

# Stage 2 — Phase I: train cvxNN via CRONOS (ADMM)
python cronos_trainer.py --model_name <model_name>

# Stage 3 — Phase II: convex preference fine-tune θ_2
python finetune_cvxdpo.py \
    --model_path  cvxNN_trained_<model_name>/ \
    --output_dir  Finetuned_cvxmlp_<model_name>/

# Stage 4 — guided generation
python guidance_sampling_pool2.py
```

Driver scripts batch the above across all five backbones × three datasets:

```bash
./run_coala_pipeline_final.sh        # main COALA pipeline (Stages 2–3)
./extract_features.sh                # batch feature extraction
```

The DPO and ORPO baselines used in the paper are reproduced via
`dpo_train_demo.py` and `orpo_train_demo.py`.

## Evaluation

Generate text from all four methods (COALA / DPO / ORPO / SFT) for head-to-head
comparison, then judge with GPT-4:

```bash
# Sample outputs from each method × backbone × dataset (3 runs each)
python generate_competitors.py

# GPT-4 pairwise judge — produces win-rate tables (paper Table 2 / Table 7)
python run_pairwise_judge.py
```

Set `OPENAI_API_KEY` in `.env` (loaded automatically) before running the
pairwise judge.

> **Note.** Several scripts contain absolute paths (e.g. `/home/miria/COALA/...`)
> from the original development environment. Adjust `BASE_DIR`, dataset, and
> checkpoint paths at the top of each script to match your layout before
> running.

## Datasets

  - **EduFeedback** — `26,621` GPT-4o-generated tutor conversations across 11
    fields of study (introduced in this paper, paper §5.1 and Appendix E.3).
    The *Alternating Population Strategy* expands these to `65,606` preference
    triplets without an external reward model.
  - **UltraFeedback** — `64k` samples from
    [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).
  - **IMDb** — `11k`-sample sentiment subset following Rafailov et al. (2024).

Run `python download_datasets.py` to fetch HelpSteer / UltraFeedback and
format them into `datasets/<name>/{pos,neg}/`. EduFeedback samples are
released alongside the paper.

## Citing

If you use COALA in your work, please cite

```bibtex
@inproceedings{feng2026coala,
  title  = {Convex Optimization for Alignment and Preference Learning on a Single GPU},
  author = {Feng, Miria and Pilanci, Mert},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year   = {2026},
  series = {PMLR 306},
}
```

The convex reformulation and CRONOS solver underpinning Phase I are due to:

  - Pilanci & Ergen, *Neural Networks are Convex Regularizers*, ICML 2020.
  - Feng, Frangella & Pilanci, *CRONOS: Enhancing Deep Learning with Scalable
    GPU-Accelerated Convex Neural Networks*, NeurIPS 2024.

## Authors

  - **Miria Feng** — `miria00 [at] stanford.edu`
  - **Mert Pilanci**

Stanford University, Department of Electrical Engineering.

## License

MIT. See [`LICENSE`](LICENSE).
