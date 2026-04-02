# Evaluation Framework Implementation Plan

This document breaks the evaluation framework into concrete implementation phases.
Each phase builds on the previous one and can be validated independently before
moving to the next.

Current state: we have `BaseEvaluator` (ABC), `MLMAccuracyEvaluator` (top-1/top-5),
embedding extraction (mean/CLS pooling), UMAP/PCA visualization, and `metrics.json`
output. All downstream evaluation beyond MLM accuracy is unimplemented.

---

## Phase 1: Enhanced Pretraining Metrics and Comparison Infrastructure

**Goal**: Make it trivially easy to compare masking strategies on pretraining quality
before any downstream fine-tuning. This is the foundation everything else builds on.

### 1A. Per-region MLM accuracy breakdown

The current `MLMAccuracyEvaluator` reports a single top-1 and top-5 number across
all positions. We need per-region breakdowns to directly measure each strategy's
effect on CDR vs framework learning.

**What to build:**
- Extend `MLMAccuracyEvaluator` (or create `RegionMLMEvaluator`) that accepts
  `cdr_mask` metadata and reports:
  - `mlm_accuracy_cdr` (CDR1+CDR2+CDR3 combined)
  - `mlm_accuracy_cdr3` (CDR3 specifically, the most diverse region)
  - `mlm_accuracy_framework` (framework only)
  - `mlm_accuracy_overall` (same as current)
  - Same breakdown for top-5
- This requires the same metadata pipeline we built for CDR masking (dataset returns
  `cdr_mask`, collator threads it through). The evaluator needs access to the
  `cdr_mask` alongside predictions.

**Files:**
- `evaluation/mlm_accuracy.py` -- extend or create new evaluator
- `scripts/evaluate.py` -- wire up region-aware evaluation

### 1B. Model comparison utility

Currently you have to manually parse `training_summary.json` and `metrics.json`
files from different experiments. We need a single script that loads results from
multiple experiments and produces a comparison.

**What to build:**
- `scripts/compare.py` -- CLI that takes multiple experiment output dirs and generates:
  - Side-by-side training curves (eval_loss and mlm_accuracy vs step)
  - Final metrics comparison table (printed to terminal + saved as JSON/CSV)
  - Embedding visualization comparison (UMAP/PCA from multiple models on same data)
- The script should auto-discover experiments from a parent directory pattern
  (e.g., `models/checkpoints/*/training_summary.json`).

**Files:**
- `scripts/compare.py` -- comparison CLI
- `evaluation/compare.py` -- core comparison logic (load summaries, compute deltas,
  generate plots)

### 1C. Standardized experiment naming convention

Establish a mapping from config -> experiment name -> output paths so that results
are automatically organized:

```
models/checkpoints/{strategy}_{model_size}/     -- training outputs
evaluation_outputs/{strategy}_{model_size}/      -- eval outputs
```

This is already roughly in place (uniform_medium, cdr_medium) but should be enforced.

**Estimated effort:** 1-2 sessions.

---

## Phase 2: Zero-Shot Evaluation Suite

**Goal**: Evaluate masking strategies without any supervised fine-tuning. These are
the cheapest evaluations and can be run immediately after pretraining.

### 2A. Pseudo-log-likelihood (PLL) scoring

This is the standard zero-shot method for measuring how well a masked LM captures
sequence fitness/function. For each position, mask it, compute log p(true_token),
sum across positions.

**What to build:**
- `evaluation/pseudo_loglikelihood.py`:
  - `compute_pll(model, tokenizer, sequence, device) -> float`
  - `compute_pll_batch(model, tokenizer, sequences, device) -> list[float]`
  - Efficient implementation: one forward pass per position (or approximate with
    batched masking for speed, accepting slight approximation).
- `evaluation/mutation_scoring.py`:
  - `score_mutation(model, tokenizer, wildtype, mutant, device) -> float`
  - Returns `PLL(mutant) - PLL(wildtype)` (positive = mutant is more likely).
  - `score_mutations_batch(...)` for multiple mutants of the same wildtype.
- These are zero-shot -- no training needed. Just load checkpoint and score.

**Files:**
- `evaluation/pseudo_loglikelihood.py`
- `evaluation/mutation_scoring.py`

### 2B. CDR restoration / infilling evaluation

Directly measures how well the model can reconstruct CDR content when it's masked
out -- the most diagnostic zero-shot metric for CDR-focused masking.

**What to build:**
- `evaluation/infilling.py`:
  - Mask an entire CDR region (e.g., CDR-H3) and predict each position via argmax.
  - Metrics:
    - Per-residue accuracy within the masked CDR
    - Exact match rate (entire CDR predicted correctly)
    - Edit distance between predicted and true CDR
  - Stratify by CDR type (H1/H2/H3) and by CDR length (short/medium/long).
  - Also test N-terminus restoration (mask positions 1-30 in IMGT numbering).

**Files:**
- `evaluation/infilling.py`

### 2C. Region-stratified perplexity

Complement to 1A but using proper perplexity (exp of average NLL) rather than
accuracy. More sensitive to confidence calibration.

**What to build:**
- Add perplexity computation to the region evaluator from 1A.
- Report: `perplexity_cdr`, `perplexity_cdr3`, `perplexity_framework`, `perplexity_overall`.

**Estimated effort:** 1-2 sessions.

---

## Phase 3: Fine-Tuning Infrastructure

**Goal**: Build the shared machinery for all supervised downstream tasks. This is
built once and reused for every task in Phases 4-6.

### 3A. Downstream task base classes

**What to build:**
- `evaluation/downstream/base.py`:
  - `BaseDownstreamTask(ABC)`:
    - `load_data() -> (train_dataset, val_dataset, test_dataset)` -- handles
      downloading, preprocessing, and splitting.
    - `build_head(hidden_size) -> nn.Module` -- creates the task-specific head.
    - `train(encoder, head, train_data, val_data, config) -> dict` -- trains
      the head (with encoder frozen or unfrozen depending on mode).
    - `evaluate(encoder, head, test_data) -> dict[str, float]` -- returns metrics.
    - `run(encoder, mode="probe") -> dict` -- end-to-end convenience method.
  - `DownstreamConfig` dataclass:
    - `mode: str` ("zero_shot", "probe", "finetune")
    - `learning_rate`, `epochs`, `batch_size`
    - `num_seeds: int` (for uncertainty estimation)
  - Three evaluation tiers (matching the framework document):
    - **Zero-shot**: frozen encoder, no head, use PLL or embedding similarity
    - **Linear probe**: frozen encoder + linear head
    - **Full fine-tune**: unfreeze encoder + head

### 3B. Embedding extraction for downstream tasks

Extend the existing `extract_embeddings` to support:
- Per-token embeddings (not just pooled) for token classification tasks
- Region-specific pooling (CDR-only mean pool, framework-only, etc.)
- Caching: save extracted embeddings to disk so probe training doesn't rerun
  the encoder every epoch.

**Files:**
- `evaluation/downstream/__init__.py`
- `evaluation/downstream/base.py`
- `evaluation/downstream/config.py`
- `evaluation/downstream/heads.py` (token classification, sequence classification,
  regression heads)

### 3C. Downstream training harness

A lightweight training loop for fine-tuning/probing that:
- Handles frozen vs unfrozen encoder
- Supports multi-seed runs
- Logs metrics per epoch
- Saves best model by validation metric
- Returns a standardized metrics dict

This should NOT use HuggingFace Trainer (overkill for small heads on frozen
encoders). A simple PyTorch training loop with early stopping is cleaner.

**Files:**
- `evaluation/downstream/trainer.py`
- `scripts/run_downstream.py` -- CLI to run any downstream task for any checkpoint

**Estimated effort:** 2-3 sessions.

---

## Phase 4: Paratope Prediction

**Goal**: First downstream task. The most standard antibody PLM evaluation and
the most directly sensitive to CDR/interface masking.

### 4A. Dataset

**Source:** TDC SAbDab_Liberis paratope task (standardized, easy to download via
`pip install PyTDC` and `from tdc.single_pred import Paratope`).

**Preprocessing:**
- Each sample: antibody sequence + per-residue binary labels (1 = paratope contact
  within 4.5A of antigen).
- Tokenize with our tokenizer, align labels to tokens.
- Two split modes:
  - Random (TDC default) for comparability
  - Cluster-based (MMseqs2 on VH sequences) for OOD evaluation

### 4B. Task implementation

- Subclass `BaseDownstreamTask` as `ParatopePredictionTask`.
- Head: single linear layer per token -> sigmoid.
- Loss: weighted binary cross-entropy (positive class weight ~5-10x due to imbalance).

### 4C. Metrics

- Primary: AUPRC, AUROC
- Thresholded: F1, MCC (threshold chosen on validation via Youden's J)
- Per-loop: H1/H2/H3/L1/L2/L3 AUROC (requires IMGT region mapping)

**Files:**
- `evaluation/downstream/paratope.py`
- `data/benchmarks/paratope.py` (data downloading/preprocessing)

**Estimated effort:** 1-2 sessions.

---

## Phase 5: Binding Specificity and Mutation-Effect Prediction

### 5A. Binding specificity (sequence classification)

**Source options (pick one for initial implementation, can add more later):**
- CoV-AbDab: binary (SARS-CoV-2 binder vs non-binder). Simpler, well-defined.
- mBLM: multiclass (7 specificity categories). Richer but harder to obtain.

**Implementation:**
- Subclass `BaseDownstreamTask` as `BindingSpecificityTask`.
- Head: mean-pool encoder outputs -> linear -> softmax (or sigmoid for binary).
- Metrics: macro-F1, per-class AUROC (multiclass) or AUROC+AUPRC (binary).

**Files:**
- `evaluation/downstream/binding.py`
- `data/benchmarks/binding.py`

### 5B. Mutation-effect prediction

**Two modes:**

**Zero-shot (Phase 2 infrastructure):**
- Use PLL scoring from Phase 2A on AB-Bind or SKEMPI 2.0 mutants.
- Correlate PLL(mutant) - PLL(wildtype) with experimental ddG.
- No training needed.

**Supervised:**
- Subclass `BaseDownstreamTask` as `MutationEffectTask`.
- Head: mean-pool -> linear -> scalar output.
- Loss: MSE for regression.
- Also evaluate as binary classification (improved vs weakened binders).
- Split by complex (hold out entire parent antibodies).

**Metrics:**
- Spearman rho (primary), Pearson r
- Binary: AUROC, AUPRC

**Files:**
- `evaluation/downstream/mutation_effect.py`
- `data/benchmarks/mutation_effect.py` (AB-Bind downloader/preprocessor)

**Estimated effort:** 2-3 sessions.

---

## Phase 6: Developability and Infilling Quality

### 6A. Developability prediction

**Source:** TDC TAP dataset (242 antibodies, 5 computed developability metrics) --
easiest to obtain via PyTDC.

**Implementation:**
- Subclass `BaseDownstreamTask` as `DevelopabilityTask`.
- Head: mean-pool -> linear -> R^5 (multi-target regression).
- Loss: MSE, optionally per-target weighted.
- Metrics: Spearman rho per assay, macro-average Spearman.

**Files:**
- `evaluation/downstream/developability.py`
- `data/benchmarks/developability.py`

### 6B. Infilling quality evaluation (enhanced)

Extend Phase 2B with additional metrics:
- Amino acid frequency distribution of generated CDRs vs natural CDRs
- Developability proxy scores on infilled sequences (if Phase 6A is done)

**Estimated effort:** 1-2 sessions.

---

## Phase 7: Comparison Dashboard and Reporting

**Goal**: One command that runs all evaluations for all checkpoints and produces a
complete comparison.

### 7A. Batch evaluation runner

- `scripts/run_all_evaluations.py`:
  - Takes a list of checkpoint dirs (or auto-discovers from `models/checkpoints/`)
  - For each checkpoint, runs:
    - MLM accuracy (overall + per-region)
    - Zero-shot PLL on mutation benchmarks
    - CDR infilling
    - All downstream tasks
  - Saves all results to `evaluation_outputs/{experiment_name}/all_metrics.json`

### 7B. Comparison report generator

- `scripts/generate_report.py`:
  - Loads `all_metrics.json` from each experiment
  - Generates:
    - Summary table (strategies as rows, tasks as columns)
    - Per-task detailed comparison plots
    - LaTeX table for paper inclusion
    - Markdown summary for quick review

### 7C. Statistical significance

- Multiple seeds for supervised tasks (3-5 seeds minimum)
- Bootstrap confidence intervals for zero-shot metrics
- Paired significance tests (e.g., paired bootstrap) across strategies

**Estimated effort:** 1-2 sessions.

---

## Implementation Priority

The phases are ordered by dependency and impact. Recommended execution order:

| Priority | Phase | Why |
|----------|-------|-----|
| 1 | Phase 1 (comparison infra) | Needed now -- we have uniform + CDR to compare |
| 2 | Phase 2 (zero-shot) | Cheap, no fine-tuning, immediately useful |
| 3 | Phase 3 (fine-tune infra) | Prerequisite for all supervised tasks |
| 4 | Phase 4 (paratope) | Most directly relevant downstream task |
| 5 | Phase 5 (binding + mutation) | Core research evaluation |
| 6 | Phase 6 (developability) | Important but lower priority |
| 7 | Phase 7 (dashboard) | Polish, can be built incrementally |

Phases 1+2 together give us a meaningful comparison framework for the CDR vs uniform
experiment that's about to run. Phase 3+4 should follow immediately so we can
show downstream task improvement in the paper.

---

## File Structure (target)

```
evaluation/
  __init__.py
  base.py                          # BaseEvaluator (exists)
  mlm_accuracy.py                  # MLMAccuracyEvaluator (exists, extend for regions)
  embeddings.py                    # extract/save embeddings (exists)
  visualize.py                     # UMAP/PCA (exists)
  compare.py                       # Model comparison utilities (new)
  pseudo_loglikelihood.py           # PLL scoring (new)
  mutation_scoring.py               # Zero-shot mutation scoring (new)
  infilling.py                      # CDR restoration evaluation (new)
  downstream/
    __init__.py
    base.py                         # BaseDownstreamTask ABC (new)
    config.py                       # DownstreamConfig (new)
    heads.py                        # Reusable classification/regression heads (new)
    trainer.py                      # Lightweight fine-tune/probe loop (new)
    paratope.py                     # Paratope prediction task (new)
    binding.py                      # Binding specificity task (new)
    mutation_effect.py              # Mutation-effect prediction task (new)
    developability.py               # Developability prediction task (new)

data/
  benchmarks/
    __init__.py
    paratope.py                     # TDC paratope data download/prep (new)
    binding.py                      # CoV-AbDab/mBLM data download/prep (new)
    mutation_effect.py              # AB-Bind/SKEMPI data download/prep (new)
    developability.py               # TDC TAP data download/prep (new)

scripts/
  evaluate.py                       # Single model eval (exists, extend)
  compare.py                        # Multi-model comparison (new)
  run_downstream.py                 # Run downstream tasks for a checkpoint (new)
  run_all_evaluations.py            # Batch runner (new)
  generate_report.py                # Comparison report (new)
```
