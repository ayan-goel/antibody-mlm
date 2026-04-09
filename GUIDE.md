# Project Guide

This document walks through the entire project: what it does, how it's organized, and how to run every experiment from data preparation through final reporting.

## What This Project Does

We train antibody protein language models (PLMs) from scratch using the RoFormer architecture (same design as AntiBERTa2) with a per-amino-acid tokenizer. The core research question is whether **biologically-informed masking strategies** produce better representations than standard uniform random masking.

The strategies we compare span structural, functional, and evolutionary signal:

- **uniform** -- standard 15% random masking (baseline)
- **cdr** -- upweights complementarity-determining regions
- **span** -- SpanBERT-style contiguous span masking
- **structure** -- masks 3D-spatial neighborhoods via predicted contact-map kNN
- **interface** -- biases toward predicted paratope (antigen-contacting) residues
- **germline** -- biases toward somatic hypermutation sites (residues that differ from germline)
- **multispecific** -- multi-policy masking for paired VH+VL sequences (paratope, shared-light, VH-VL interface)
- **hybrid** -- stochastic mixture of sub-strategies with optional curriculum scheduling

Models are trained via **masked language modeling (MLM)** on antibody heavy-chain (VH) sequences (or paired VH+VL sequences) from the Observed Antibody Space (OAS). We then evaluate the learned representations through zero-shot metrics, downstream supervised tasks, and distributional analyses.

## Project Structure

```
antibody-mlm/
├── configs/                            # All YAML configurations
│   ├── medium.yaml                     # Uniform masking, medium model
│   ├── cdr_medium.yaml                 # CDR-weighted masking
│   ├── span_medium.yaml                # SpanBERT-style span masking
│   ├── structure_medium.yaml           # 3D-neighborhood masking
│   ├── interface_medium.yaml           # Paratope-biased masking
│   ├── germline_medium.yaml            # Germline / SHM-biased masking
│   ├── multispecific_medium.yaml       # Paired VH+VL multi-policy masking
│   ├── hybrid_curriculum_medium.yaml   # Curriculum mixture (single-chain)
│   ├── hybrid_paired_medium.yaml       # Curriculum mixture (paired)
│   ├── experiments.yaml                # Registry mapping experiments to configs/checkpoints
│   └── downstream/                     # Downstream task configs
│       ├── paratope_probe.yaml
│       ├── binding_probe.yaml
│       └── developability_probe.yaml
│
├── data/                               # Data loading and preprocessing
│   ├── download.py                     # OAS single-chain downloader
│   ├── download_paired.py              # OAS paired (VH+VL) downloader
│   ├── preprocessing.py                # Sequence filtering, JSONL generation
│   ├── preprocessing_paired.py         # Paired sequence filtering
│   ├── dataset.py                      # AntibodyDataset (loads metadata sidecars)
│   ├── dataset_paired.py               # PairedAntibodyDataset
│   └── benchmarks/                     # Downstream benchmark datasets
│       ├── paratope.py                 # TDC SAbDab_Liberis (paratope prediction)
│       ├── binding.py                  # CoV-AbDab (SARS-CoV-2 neutralization)
│       ├── developability.py           # TDC TAP (5 developability metrics)
│       └── ab_bind.py                  # AB-Bind (mutation effect ddG)
│
├── masking/                            # Masking strategies (pluggable via registry)
│   ├── base.py                         # BaseMaskingStrategy ABC + registry
│   ├── uniform.py                      # 15% uniform random masking
│   ├── cdr.py                          # CDR-weighted Bernoulli
│   ├── span.py                         # Geometric-length contiguous spans
│   ├── structure.py                    # 3D-neighborhood seed-and-grow
│   ├── interface.py                    # Paratope-weighted Bernoulli
│   ├── germline.py                     # Mutation-weighted Bernoulli
│   ├── multispecific.py                # Paired-chain multi-policy masking
│   ├── hybrid.py                       # Mixture meta-strategy with curriculum
│   └── collator.py                     # MLMDataCollator (applies masking during batching)
│
├── models/
│   ├── model.py                        # Model factory (single-chain + multispecific)
│   └── checkpoints/                    # Trained model checkpoints
│
├── training/
│   ├── config.py                       # Typed dataclass configs loaded from YAML
│   ├── trainer.py                      # HuggingFace Trainer wrapper
│   └── callbacks.py                    # HybridMaskingCallback (curriculum scheduling)
│
├── evaluation/                         # Evaluation infrastructure
│   ├── base.py                         # BaseEvaluator ABC
│   ├── mlm_accuracy.py                 # MLM accuracy + region-stratified perplexity
│   ├── pseudo_loglikelihood.py         # PLL scoring
│   ├── mutation_scoring.py             # Zero-shot mutation effect via delta-PLL
│   ├── infilling.py                    # CDR infilling / restoration
│   ├── infilling_quality.py            # AA frequency analysis of infilled CDRs
│   ├── embeddings.py                   # Embedding extraction (mean/CLS pooling)
│   ├── visualize.py                    # UMAP and PCA plots
│   ├── compare.py                      # Experiment discovery + comparison tables
│   ├── report.py                       # LaTeX, Markdown, plot generation
│   ├── significance.py                 # Bootstrap CIs and paired tests
│   └── downstream/                     # Supervised downstream evaluation
│       ├── base.py                     # BaseDownstreamTask ABC (multi-seed runner)
│       ├── config.py                   # DownstreamConfig dataclass
│       ├── encoder.py                  # EncoderWrapper (frozen hidden state extraction)
│       ├── heads.py                    # Token / sequence classification + regression heads
│       ├── embedding_cache.py          # Cache encoder embeddings to disk
│       ├── collator.py                 # DownstreamCollator (pads without masking)
│       ├── trainer.py                  # Probe and fine-tune training loops
│       ├── paratope.py                 # Paratope prediction task
│       ├── binding.py                  # Binding specificity task
│       └── developability.py           # Developability prediction task
│
├── scripts/                            # CLI entry points
│   ├── download_data.py                # Download single-chain OAS
│   ├── download_paired_data.py         # Download paired OAS
│   ├── annotate_cdrs.py                # Add CDR annotations
│   ├── compute_paratope_labels.py      # Soft paratope labels (interface strategy)
│   ├── compute_germline_labels.py      # Per-residue germline / SHM labels
│   ├── compute_multispecific_labels.py # Paratope + interface + germline for paired
│   ├── predict_structures.py           # ESM2 contact maps -> kNN sidecar
│   ├── convert_knn_int16.py            # kNN tensor format conversion
│   ├── train.py                        # Train one or more models
│   ├── benchmark_mutations.py          # AB-Bind zero-shot mutation benchmark
│   ├── run_downstream.py               # Run a single downstream task
│   ├── run_all_evaluations.py          # Batch runner: all evals for all experiments
│   ├── compare.py                      # Side-by-side experiment comparison
│   └── generate_report.py              # Generate LaTeX, Markdown, plots from results
│
└── utils/
    ├── tokenizer.py                    # Tokenizer loading (single + multispecific)
    ├── seed.py                         # Reproducibility (set_seed)
    ├── annotate_cdrs.py                # ANARCI CDR annotation utilities
    └── io.py                           # File I/O helpers
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

Download VH sequences from OAS and preprocess into JSONL:

```bash
python scripts/download_data.py --config configs/medium.yaml
```

This creates `data/processed/oas_vh_500k.jsonl`. Each line is a JSON object with a VH amino acid sequence.

For paired sequences (needed for the `multispecific` and `hybrid_paired` models):

```bash
python scripts/download_paired_data.py --config configs/multispecific_medium.yaml
```

This creates `data/processed/oas_paired_500k.jsonl` (VH+VL pairs).

#### Annotate CDRs

CDR annotations are required for CDR-weighted masking and region-stratified evaluation:

```bash
python scripts/annotate_cdrs.py --input data/processed/oas_vh_500k.jsonl
```

This adds `cdr1_aa`, `cdr2_aa`, `cdr3_aa` fields to each record. Pass `--anarci-fallback` to use ANARCI for sequences without OAS CDR fields.

#### Compute structural / functional metadata

The non-uniform strategies that need biological signal load it from precomputed sidecar `.pt` files aligned 1:1 with the JSONL. Run once before training:

```bash
# Soft paratope labels (used by interface, hybrid, multispecific)
python scripts/compute_paratope_labels.py \
    --input data/processed/oas_vh_500k.jsonl \
    --output data/structures/oas_vh_500k_paratope.pt

# Per-residue germline / SHM labels (used by germline, hybrid)
python scripts/compute_germline_labels.py \
    --input data/processed/oas_vh_500k.jsonl \
    --output data/structures/oas_vh_500k_germline.pt

# Predicted spatial neighborhoods via ESM2 contact maps (used by structure, hybrid)
python scripts/predict_structures.py \
    --input data/processed/oas_vh_500k.jsonl \
    --output data/structures/oas_vh_500k_coords.pt \
    --batch_size 64 --k_neighbors 32

# All-in-one for paired data: paratope + interface + germline labels per chain
python scripts/compute_multispecific_labels.py \
    --input data/processed/oas_paired_500k.jsonl \
    --output-dir data/structures \
    --prefix oas_paired_500k
```

Each config's `data.coords_path` / `paratope_path` / `germline_path` / `interface_path` fields point to the resulting sidecars.

### 2. Train Models

`scripts/train.py` supports two modes:

**Multi-model shorthand** (preferred for the full comparison run). Trains the listed models sequentially in command-line order, redirecting each model's stdout/stderr to `logs/<name>.log`. The terminal stays clean and shows only short status banners between models:

```bash
python scripts/train.py --uniform --cdr --span --structure --interface --germline
python scripts/train.py --multispecific --hybrid_curriculum --hybrid_paired
```

Available flags: `--uniform`, `--cdr`, `--span`, `--structure`, `--interface`, `--germline`, `--multispecific`, `--hybrid_curriculum`, `--hybrid_paired`.

**Legacy single-config mode** (no log redirection):

```bash
python scripts/train.py --config configs/medium.yaml
```

Both save checkpoints to `models/checkpoints/{experiment_name}/final/`. Training produces a `training_summary.json` with eval history, final metrics, and experiment metadata.

Key training parameters (medium models):
- ~50M parameters (12 layers, 512 hidden, 8 heads, 2048 intermediate)
- Equal-step comparison: every model trains for exactly `max_steps=125000` optimizer updates with `warmup_steps=6250` (5% of max_steps) and a cosine LR schedule. `num_epochs` is ignored.
- Effective batch size 64 (32 × 2 gradient accumulation, single-chain). Paired models use batch_size 16 × 4 grad accumulation = 64.
- FP16 mixed precision
- Early stopping is **disabled** (`patience=0`) so equal-step comparisons are fair
- The final checkpoint at exactly `max_steps` is reported (no `load_best_model_at_end`); selecting an intermediate checkpoint per model based on a noisy eval metric would defeat the equal-compute fairness goal
- Single-chain models use `data/processed/oas_vh_500k.jsonl` (`max_length=160`); paired models use `data/processed/oas_paired_500k.jsonl` (`max_length=384`)

### 3. Evaluate Models

Evaluation is done through the unified `scripts/run_all_evaluations.py` runner. It loads each checkpoint once and runs MLM accuracy, infilling, PLL, region-stratified perplexity, infilling quality, AB-Bind mutation benchmark, and downstream tasks sequentially. All metrics for an experiment are merged into one `all_metrics.json`.

```bash
# Run everything for every experiment in configs/experiments.yaml
python scripts/run_all_evaluations.py --device cuda

# A specific experiment only
python scripts/run_all_evaluations.py --experiments uniform_medium

# Skip expensive sections
python scripts/run_all_evaluations.py --skip-pll --skip-mutations --skip-downstream
```

Useful flags:
- `--experiments uniform_medium cdr_medium` -- run only specific experiments
- `--skip-mlm`, `--skip-infilling`, `--skip-pll`, `--skip-perplexity` -- skip individual zero-shot sections
- `--skip-mutations` -- skip AB-Bind mutation benchmark
- `--skip-downstream` -- skip all downstream tasks
- `--skip-infilling-quality` -- skip AA frequency analysis
- `--max-pll-sequences`, `--max-infilling-samples`, `--max-infilling-quality-samples` -- per-section sample caps
- `--ab-bind-dir data/ab_bind` -- AB-Bind data directory

The runner is **resumable**: existing sections in `all_metrics.json` are preserved across reruns, so a crashed run can be re-executed and only the missing or failed sections will be recomputed. Per-task downstream results are also saved incrementally.

For all experiments, MLM accuracy and region-stratified perplexity are computed under a **uniform reference masking** regardless of the model's training-time strategy -- comparing each model under its own training-time mask distribution is unfair, since span/cdr/germline/interface skew toward harder positions and would make those models look worse than uniform-trained models even when their representations are equally good.

The held-out eval split uses a fixed `EVAL_SPLIT_SEED=42` for the train/eval `random_split`, regardless of each model's training-time `config.seed`. This guarantees the eval set is identical across models so cross-model metric comparisons are well-defined. **Don't change this without re-running every experiment.**

#### Standalone mutation benchmark

You can also run AB-Bind on its own (rarely needed; `run_all_evaluations.py` already calls it):

```bash
python scripts/benchmark_mutations.py \
    --checkpoint models/checkpoints/uniform_medium/final \
    --output-dir evaluation_outputs/uniform_medium \
    --device cuda
```

The benchmark uses **wildtype-marginal scoring** (the ESM-style recipe from Meier et al., NeurIPS 2021): one masked forward pass per record, scored only at the mutation positions. Reports per-complex Spearman / AUROC as the primary metric and pooled Spearman / Pearson / AUROC as secondary diagnostics. Skipped automatically for paired models (AB-Bind sequences are single-chain only).

#### Standalone downstream task

To run a single downstream task on a single checkpoint outside the batch runner:

```bash
python scripts/run_downstream.py --config configs/downstream/paratope_probe.yaml
```

| Task | Config | What it tests | Key metric |
|------|--------|---------------|------------|
| Paratope prediction | `paratope_probe.yaml` | Per-residue antigen contact prediction | AUROC, AUPRC |
| Binding specificity | `binding_probe.yaml` | SARS-CoV-2 neutralization classification | AUROC, F1 |
| Developability | `developability_probe.yaml` | TDC TAP biophysical metrics | Spearman rho |

Each task runs with multiple random seeds (3 for paratope/binding, 5 for developability) and reports mean ± std. From the batch runner, results land in `downstream_outputs/{task}_{mode}/{experiment}/results.json`. To evaluate a different checkpoint with `run_downstream.py`, edit the `checkpoint` field in the YAML config.

### 4. Compare and Report

#### Quick comparison table

```bash
python scripts/compare.py
```

Auto-discovers all experiments from `models/checkpoints/` and `evaluation_outputs/`, prints a side-by-side table, and saves training-curve and embedding-comparison plots to `comparison_outputs/`.

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

All single-position strategies follow the BERT 80/10/10 convention (80% [MASK], 10% random, 10% keep) at 15% mask probability. The strategies differ in *which* positions get selected (and how, in the case of span / structure / hybrid):

- **uniform** -- every non-special token has equal probability of being masked. Standard baseline used by BERT, ESM, and AntiBERTa2.
- **cdr** -- CDR positions are upweighted via per-region weights. Defaults: framework 1.0, CDR1 / CDR2 = 3.0, CDR3 = 6.0 (CDR3 is the most diverse, gets the highest weight).
- **span** -- contiguous spans with geometric-length sampling (defaults `geometric_p=0.2`, `max_span_length=10`, mean span ~5 tokens). Spans never cross special tokens, preserving chain boundaries for paired models.
- **structure** -- per-residue kNN from ESM2 contact maps (default `k_neighbors=32`). Seed-and-grow algorithm: pick a random residue, mask it and its k nearest 3D neighbors, repeat until the budget is filled. Falls back to uniform when coordinates are missing.
- **interface** -- Bernoulli sampling weighted by per-residue paratope probabilities (defaults `paratope_weight=6.0`, `non_paratope_weight=1.0`). Concentrates ~50-60% of the mask budget on paratope residues despite paratopes being only ~15-25% of the sequence. Falls back to uniform when paratope labels are missing.
- **germline** -- Bernoulli sampling weighted by per-residue mutation labels (defaults `mutated_weight=6.0`, `germline_weight=1.0`). Addresses the AbLang-2 observation that ~85% of VH residues match germline, so uniform masking overwhelmingly trains on germline residues rather than functionally important SHM mutations.
- **multispecific** -- three policies sampled from a categorical distribution per step:
  - *Policy A:* module-isolated paratope masking (mask within one module's paratope)
  - *Policy B:* shared-light-chain masking (cross-module conditional infilling)
  - *Policy C:* VH-VL interface masking (bias toward cross-chain packing residues)

  Configurable via `params.policy_weights`. Requires paired data with `module_ids` and `chain_type_ids` metadata.
- **hybrid** -- mixture of sub-strategies. Each sample is generated by exactly one sub-strategy sampled from `policy_weights`. Sub-strategies whose required metadata is missing for a given sample are excluded from that sample's mixture and weights are re-normalized. Optional curriculum schedule with linearly interpolated breakpoints (see `configs/hybrid_curriculum_medium.yaml` for an example 4-stage schedule that warms up with general syntax, specializes the binding site, then introduces mutation realism).

Example CDR-weighted config block:

```yaml
masking:
  strategy: "cdr"
  params:
    framework_weight: 1.0
    cdr1_weight: 3.0
    cdr2_weight: 3.0
    cdr3_weight: 6.0
```

The hypothesis is that injecting biological structure into the masking distribution forces the model to predict residues that matter most for antibody function and produces representations that transfer better to downstream antibody tasks.

## How the Evaluation Framework Works

Evaluation is organized into three tiers, all driven by `run_all_evaluations.py`:

**Zero-shot (no training needed):** Measures the quality of the pretrained model directly.
- *MLM accuracy & region-stratified perplexity* -- under a uniform reference mask.
- *PLL* -- sum of log p(true token) across all positions; higher = better sequence modeling. Uses the dataset's pre-tokenized inputs so single-chain and paired models are scored on identical token streams (the older string-roundtrip path silently dropped non-standard amino acids and made the two incomparable).
- *CDR infilling* -- mask an entire CDR, predict via argmax, measure accuracy and edit distance per CDR.
- *Infilling quality* -- compare AA frequency distributions of generated vs natural CDRs (JSD).
- *Mutation scoring* -- AB-Bind wildtype-marginal scores correlate with experimental ddG; reports per-complex Spearman / AUROC as the primary metric and pooled Spearman / Pearson / AUROC as secondary diagnostics. Skipped for paired models.

**Linear probe (frozen encoder):** Train a small head on frozen encoder embeddings. Tests whether the representations contain task-relevant information.
- Embeddings are extracted once and cached to disk so probe training only updates the small head.
- Multiple seeds with mean ± std reporting.

**Full fine-tune:** Train encoder + head end-to-end with separate learning rates. Tests whether the representations provide a good initialization. Set `mode: finetune` in the downstream config.

## Adding a New Masking Strategy

1. Create `masking/your_strategy.py`, subclassing `BaseMaskingStrategy`
2. Implement `select_mask_positions(input_ids, special_tokens_mask, metadata=None)` and decorate the class with `@register_strategy("your_strategy")`
3. Add `import masking.your_strategy` to `masking/__init__.py` (the import triggers registration)
4. Create a config YAML (copy `configs/medium.yaml`, change `masking.strategy`, `masking.params`, and `training.output_dir`)
5. Add the experiment to `configs/experiments.yaml`
6. (Optional) Register a `--your_strategy` shorthand in the `MODEL_REGISTRY` dict in `scripts/train.py`
7. (If your strategy needs precomputed metadata) add a `compute_<thing>_labels.py` script under `scripts/` that produces an aligned `.pt` sidecar, then point to it from `data.<your>_path` in the config

## Adding a New Downstream Task

1. Create `data/benchmarks/your_task.py` with data loading and a PyTorch Dataset
2. Create `evaluation/downstream/your_task.py` subclassing `BaseDownstreamTask`
3. Use the `@register_task("your_task")` decorator
4. Add `import evaluation.downstream.your_task` to `evaluation/downstream/__init__.py`
5. Create `configs/downstream/your_task_probe.yaml`
6. Add the config path to `configs/experiments.yaml` under `downstream_tasks`

## Output Directory Convention

```
models/checkpoints/{experiment}/                # Trained checkpoints
    final/                                       # Final model + tokenizer
    training_summary.json                        # Eval history, final metrics
evaluation_outputs/{experiment}/                 # Per-experiment evaluation results
    all_metrics.json                             # Merged metrics from run_all_evaluations
downstream_outputs/{task}_{mode}/{experiment}/   # Downstream task results
    results.json                                 # Per-seed and aggregated metrics
    embedding_cache/                             # Cached encoder hidden states
logs/{experiment}.log                            # Per-model training logs (multi-model mode)
comparison_outputs/                              # Cross-experiment reports
    comparison_table.{json,csv,tex}
    summary.md
    training_curves.png
    metric_comparison.png
```

## Config Reference

Training configs (`configs/*.yaml`) control:
- `data.processed_path` -- path to preprocessed JSONL
- `data.max_length` / `data.min_length` -- sequence length filters (160 for VH, 384 for paired)
- `data.coords_path` / `paratope_path` / `germline_path` / `interface_path` -- optional metadata sidecars used by structure / interface / germline / multispecific / hybrid
- `data.paired` / `data.bispecific` -- toggle the paired pipeline
- `masking.strategy` -- one of `uniform`, `cdr`, `span`, `structure`, `interface`, `germline`, `multispecific`, `hybrid`
- `masking.params` -- strategy-specific parameters (see "How the Masking Strategies Work")
- `model.model_size` -- `"small"` (~5M params, 6L/256h, validation), `"medium"` (~50M, 12L/512h, main experiments), or `"full"` (~200M, 16L/1024h, matches AntiBERTa2)
- `model.from_pretrained` -- always `false` (we train from scratch)
- `training.max_steps` -- if > 0, training stops at exactly this many optimizer steps and `num_epochs` is ignored. Set to 125000 for the equal-step comparison.
- `training.warmup_steps`, `learning_rate`, `weight_decay`, `fp16`, `gradient_accumulation_steps`, `early_stopping_patience`

Downstream configs (`configs/downstream/*.yaml`) control:
- `task` -- registered task name
- `checkpoint` -- path to pretrained model (overridden per-experiment by the batch runner)
- `mode` -- `"probe"` (frozen encoder) or `"finetune"` (end-to-end)
- `num_seeds` -- how many random seeds to average over
- Training hyperparameters (LR, epochs, batch size, patience)
