# Project Guide

This document walks through the entire project: what it does, how it's organized, and how to run every experiment from data preparation through final reporting.

## What This Project Does

We train antibody protein language models (PLMs) from scratch using the RoFormer architecture (same design as AntiBERTa2) with a per-amino-acid tokenizer. The core research question is whether **biologically-informed masking strategies** (e.g., upweighting CDR regions during masked language modeling) produce better representations than standard uniform random masking.

The model is trained via **masked language modeling (MLM)** on 500K antibody heavy-chain (VH) sequences from the Observed Antibody Space (OAS). We then evaluate the learned representations through zero-shot metrics, downstream supervised tasks, and distributional analyses.

## Project Structure

```
antibody-mlm/
├── configs/                    # All YAML configurations
│   ├── baseline.yaml           # Small model (validation only)
│   ├── medium.yaml             # Uniform masking, medium model
│   ├── cdr_medium.yaml         # CDR-weighted masking, medium model
│   ├── experiments.yaml        # Registry mapping experiments to configs/checkpoints
│   └── downstream/             # Downstream task configs
│       ├── paratope_probe.yaml
│       ├── binding_probe.yaml
│       └── developability_probe.yaml
│
├── data/                       # Data loading and preprocessing
│   ├── download.py             # OAS data downloader
│   ├── preprocessing.py        # Sequence filtering, JSONL generation
│   ├── dataset.py              # AntibodyDataset (PyTorch Dataset with CDR masks)
│   └── benchmarks/             # Downstream benchmark datasets
│       ├── paratope.py         # TDC SAbDab_Liberis (paratope prediction)
│       ├── binding.py          # CoV-AbDab (SARS-CoV-2 neutralization)
│       ├── developability.py   # TDC TAP (5 developability metrics)
│       └── ab_bind.py          # AB-Bind (mutation effect ddG)
│
├── masking/                    # Masking strategies (pluggable via registry)
│   ├── base.py                 # BaseMaskingStrategy ABC
│   ├── uniform.py              # Standard 15% uniform random masking
│   ├── cdr.py                  # CDR-weighted masking (higher prob on CDR regions)
│   └── collator.py             # MLMDataCollator (applies masking during batching)
│
├── models/
│   └── model.py                # Model factory (creates RoFormer from config)
│
├── training/
│   ├── config.py               # Typed dataclass configs loaded from YAML
│   └── trainer.py              # HuggingFace Trainer wrapper with early stopping
│
├── evaluation/                 # All evaluation infrastructure
│   ├── mlm_accuracy.py         # MLM accuracy + per-region perplexity
│   ├── pseudo_loglikelihood.py  # Pseudo-log-likelihood (PLL) scoring
│   ├── mutation_scoring.py     # Zero-shot mutation effect via delta-PLL
│   ├── infilling.py            # CDR infilling / restoration evaluation
│   ├── infilling_quality.py    # AA frequency analysis of infilled CDRs
│   ├── embeddings.py           # Embedding extraction (mean/CLS pooling)
│   ├── visualize.py            # UMAP and PCA plots
│   ├── compare.py              # Experiment discovery and comparison tables
│   ├── report.py               # LaTeX, Markdown, and plot generation
│   ├── significance.py         # Bootstrap CIs and paired tests
│   └── downstream/             # Supervised downstream evaluation
│       ├── base.py             # BaseDownstreamTask ABC (multi-seed runner)
│       ├── config.py           # DownstreamConfig dataclass
│       ├── encoder.py          # EncoderWrapper (frozen hidden state extraction)
│       ├── heads.py            # TokenClassification, SequenceClassification, Regression heads
│       ├── embedding_cache.py  # Cache embeddings to disk for probe training
│       ├── collator.py         # DownstreamCollator (pads without masking)
│       ├── trainer.py          # Probe and fine-tune training loops
│       ├── paratope.py         # Paratope prediction task
│       ├── binding.py          # Binding specificity task
│       └── developability.py   # Developability prediction task
│
├── scripts/                    # CLI entry points
│   ├── download_data.py        # Download and preprocess OAS sequences
│   ├── annotate_cdrs.py        # Add CDR annotations via ANARCI
│   ├── train.py                # Train a model from a config
│   ├── evaluate.py             # MLM accuracy + embedding extraction
│   ├── evaluate_zeroshot.py    # PLL, infilling, perplexity, infilling quality
│   ├── benchmark_mutations.py  # AB-Bind zero-shot mutation benchmark
│   ├── run_downstream.py       # Run a single downstream task
│   ├── compare.py              # Side-by-side experiment comparison
│   ├── run_all_evaluations.py  # Batch runner: all evals for all experiments
│   └── generate_report.py      # Generate LaTeX, Markdown, plots from results
│
└── utils/
    ├── tokenizer.py            # Tokenizer loading (AntiBERTa2)
    ├── seed.py                 # Reproducibility (set_seed)
    ├── annotate_cdrs.py        # ANARCI CDR annotation utilities
    └── io.py                   # File I/O helpers
```

## Setup

```bash
conda create -n abmlm python=3.10 -y
conda activate abmlm
pip install -r requirements.txt
pip install accelerate rjieba
```

`rjieba` is needed by the AntiBERTa2 tokenizer. `accelerate` is needed by the HuggingFace Trainer.

## Step-by-Step Workflow

### 1. Download and Prepare Data

Download 500K VH sequences from OAS and preprocess into JSONL:

```bash
python scripts/download_data.py --config configs/medium.yaml
```

This creates `data/processed/oas_vh_500k.jsonl`. Each line is a JSON object with a VH amino acid sequence.

Then annotate CDR regions (required for CDR masking and region-stratified evaluation):

```bash
python scripts/annotate_cdrs.py --input data/processed/oas_vh_500k.jsonl
```

This adds `cdr_annotations` to each record, marking which residues belong to CDR1/CDR2/CDR3.

### 2. Train Models

Each training config specifies the masking strategy, model size, data path, and all hyperparameters. Training uses the HuggingFace Trainer with early stopping.

**Uniform masking (baseline):**

```bash
python scripts/train.py --config configs/medium.yaml
```

**CDR-weighted masking:**

```bash
python scripts/train.py --config configs/cdr_medium.yaml
```

Both save checkpoints to `models/checkpoints/{experiment_name}/`. Training produces a `training_summary.json` with eval history, final metrics, and experiment metadata.

Key training parameters (medium models):
- 38M parameters (6 layers, 512 hidden, 8 heads)
- 500K sequences, 90/10 train/val split
- 20 epochs with early stopping (patience 5)
- Effective batch size 64 (32 x 2 gradient accumulation)
- FP16 mixed precision

### 3. Evaluate Models

There are multiple levels of evaluation, from cheap zero-shot metrics to supervised downstream tasks.

#### 3a. MLM Accuracy and Embeddings

Basic evaluation: region-stratified MLM accuracy, embedding extraction, UMAP/PCA plots.

```bash
python scripts/evaluate.py \
    --config configs/medium.yaml \
    --checkpoint models/checkpoints/uniform_medium \
    --output-dir evaluation_outputs/uniform_medium
```

Produces: `metrics.json`, `embeddings.npy`, `umap.png`, `pca.png`.

#### 3b. Zero-Shot Evaluation Suite

PLL scoring, CDR infilling, region-stratified perplexity, and (optionally) infilling quality analysis:

```bash
python scripts/evaluate_zeroshot.py \
    --config configs/medium.yaml \
    --checkpoint models/checkpoints/uniform_medium \
    --output-dir evaluation_outputs/uniform_medium \
    --device cuda \
    --infilling-quality
```

Flags to skip expensive components: `--skip-pll`, `--skip-infilling`, `--skip-perplexity`.

Produces: `metrics_zeroshot.json` with PLL, infilling accuracy per CDR, perplexity per region, and AA frequency distributions.

#### 3c. Mutation Effect Benchmark

Zero-shot mutation scoring using AB-Bind (correlates delta-PLL with experimental ddG):

```bash
python scripts/benchmark_mutations.py \
    --checkpoint models/checkpoints/uniform_medium \
    --output-dir evaluation_outputs/uniform_medium \
    --device cuda
```

Produces: `mutation_benchmark.json` with overall Spearman rho, Pearson r, binary AUROC, and per-complex correlations.

#### 3d. Downstream Tasks

Supervised tasks that measure representation quality. Each task trains a small head on frozen encoder embeddings (probe mode).

**Run a single task:**

```bash
python scripts/run_downstream.py --config configs/downstream/paratope_probe.yaml
```

Available tasks and what they measure:

| Task | Config | What it tests | Key metric |
|------|--------|---------------|------------|
| Paratope prediction | `paratope_probe.yaml` | Per-residue antigen contact prediction | AUROC, AUPRC |
| Binding specificity | `binding_probe.yaml` | SARS-CoV-2 neutralization classification | AUROC, F1 |
| Developability | `developability_probe.yaml` | 5 computed biophysical properties | Spearman rho |

Each task runs with multiple random seeds and reports mean +/- std. Results go to `downstream_outputs/{task}_{mode}/results.json`.

To evaluate a different checkpoint, edit the `checkpoint` field in the YAML config.

### 4. Run Everything at Once

The batch runner evaluates all experiments registered in `configs/experiments.yaml`:

```bash
python scripts/run_all_evaluations.py --device cuda
```

This runs MLM accuracy, zero-shot suite, mutation benchmark, and all downstream tasks for every experiment. Results are merged into `evaluation_outputs/{name}/all_metrics.json`.

Useful flags:
- `--experiments uniform_medium` -- run only specific experiments
- `--skip-pll` -- skip PLL (slowest component)
- `--skip-downstream` -- skip downstream tasks
- `--skip-mutations` -- skip mutation benchmark
- `--infilling-quality` -- include AA frequency analysis

### 5. Compare and Report

#### Quick comparison table

```bash
python scripts/compare.py
```

Prints a side-by-side table of all discovered experiments and saves training curve plots.

#### Full report generation

```bash
python scripts/generate_report.py --output-dir comparison_outputs
```

Produces:
- `comparison_table.json` / `.csv` -- full metrics
- `comparison_table.tex` -- LaTeX table (booktabs style, ready for paper)
- `summary.md` -- Markdown summary
- `training_curves.png` -- overlaid training loss/accuracy curves
- `metric_comparison.png` -- grouped bar charts

## How the Masking Strategies Work

All masking follows the 80/10/10 BERT convention (80% [MASK], 10% random, 10% keep) at 15% mask probability. The strategies differ in *which* positions get selected:

**Uniform:** every non-special token has equal probability of being masked. This is the standard baseline used by BERT, ESM, and AntiBERTa2.

**CDR-weighted:** positions in CDR regions are upweighted so the model sees more CDR masking during training. Weights are configurable per region:

```yaml
masking:
  strategy: "cdr"
  params:
    framework_weight: 1.0
    cdr1_weight: 3.0
    cdr2_weight: 3.0
    cdr3_weight: 6.0    # CDR3 is the most diverse, gets highest weight
```

The hypothesis is that forcing the model to predict CDR residues more often produces better CDR representations, which matter most for downstream antibody tasks.

## How the Evaluation Framework Works

Evaluation is organized into three tiers:

**Zero-shot (no training needed):** Measures the quality of the pretrained model directly.
- *PLL*: sum of log p(true token) across all positions. Higher = better sequence modeling.
- *CDR infilling*: mask an entire CDR, predict via argmax, measure accuracy and edit distance.
- *Perplexity*: exp(average NLL) stratified by CDR vs framework regions.
- *Mutation scoring*: delta-PLL correlates with experimental binding energy changes (ddG).
- *Infilling quality*: compare AA frequency distributions of generated vs natural CDRs (JSD).

**Linear probe (frozen encoder):** Train a small head on frozen encoder embeddings. Tests whether the representations contain task-relevant information.
- Embedding caching: encoder hidden states are extracted once and saved to disk, so probe training only trains the small head.
- Multiple seeds with mean/std reporting.

**Full fine-tune:** Train encoder + head end-to-end with separate learning rates. Tests whether the representations provide a good initialization.

## Adding a New Masking Strategy

1. Create `masking/your_strategy.py`, subclassing `BaseMaskingStrategy`
2. Implement `select_mask_positions(input_ids, special_tokens_mask, metadata=None)`
3. Register it in `masking/__init__.py`
4. Create a config YAML (copy `configs/medium.yaml`, change `masking.strategy` and `training.output_dir`)
5. Add the experiment to `configs/experiments.yaml`

## Adding a New Downstream Task

1. Create `data/benchmarks/your_task.py` with data loading and a PyTorch Dataset
2. Create `evaluation/downstream/your_task.py` subclassing `BaseDownstreamTask`
3. Use `@register_task("your_task")` decorator
4. Add `import evaluation.downstream.your_task` to `evaluation/downstream/__init__.py`
5. Create `configs/downstream/your_task_probe.yaml`
6. Add the config path to `configs/experiments.yaml` under `downstream_tasks`

## Output Directory Convention

```
models/checkpoints/{strategy}_{size}/     # Trained model checkpoints
    training_summary.json                  # Eval history, final metrics
evaluation_outputs/{strategy}_{size}/      # All evaluation results
    metrics.json                           # MLM accuracy
    metrics_zeroshot.json                  # PLL, infilling, perplexity
    mutation_benchmark.json                # AB-Bind correlations
    all_metrics.json                       # Merged (from batch runner)
    embeddings.npy                         # Extracted embeddings
    umap.png, pca.png                      # Visualizations
downstream_outputs/{task}_{mode}/          # Downstream task results
    results.json                           # Per-seed and aggregated metrics
    embedding_cache/                       # Cached encoder hidden states
comparison_outputs/                        # Cross-experiment reports
    comparison_table.{json,csv,tex}
    summary.md
    training_curves.png
    metric_comparison.png
```

## Config Reference

Training configs (`configs/*.yaml`) control:
- `data.processed_path` -- path to preprocessed JSONL
- `masking.strategy` -- which masking strategy to use (`"uniform"`, `"cdr"`)
- `masking.params` -- strategy-specific parameters (e.g., CDR weights)
- `model.model_size` -- `"small"` or `"medium"`
- `model.from_pretrained` -- always `false` (we train from scratch)
- `training.*` -- batch size, LR, epochs, early stopping, etc.

Downstream configs (`configs/downstream/*.yaml`) control:
- `task` -- registered task name
- `checkpoint` -- path to pretrained model
- `mode` -- `"probe"` (frozen encoder) or `"finetune"` (end-to-end)
- `num_seeds` -- how many random seeds to average over
- Training hyperparameters (LR, epochs, batch size, patience)
