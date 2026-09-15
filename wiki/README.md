# antibody-mlm wiki

Internal reference for the **antibody-mlm** project. The repo's research question:

> **Does *where* you mask residues during MLM pretraining of an antibody language model change what the model learns and how well it transfers to downstream antibody tasks?**

We train RoFormer (AntiBERTa2-architecture) models **from scratch** on ~500K OAS heavy-chain (VH) sequences, varying only the **masking strategy**, then compare the resulting checkpoints across a battery of zero-shot and supervised probes — all under identical compute and identical evaluation conditions.

This wiki is a navigation/context layer over the code. Pages cite real files; where the code and the prose docs (`README.md`, `GUIDE.md`) disagree, the wiki notes the discrepancy and trusts the code/configs.

## Pages

| # | Page | What's in it |
|---|------|--------------|
| 01 | [Overview & workflow](01-overview.md) | Goal, hypothesis, the data→train→eval→compare→figure pipeline, entry points |
| 02 | [Repository map](02-repository-map.md) | Directory-by-directory tour + file-format conventions (JSONL, `.pt`, YAML) |
| 03 | [Masking strategies](03-masking-strategies.md) | The core contribution: registry, base API, collator, all 9 strategies + rationale |
| 04 | [Hybrid & curriculum](04-hybrid-curriculum.md) | The hybrid mixture meta-strategy, curriculum scheduling, all 8 hybrid variants |
| 05 | [Data](05-data.md) | Pretraining corpus schema, metadata sidecars, benchmark datasets, provenance |
| 06 | [Models, training & configs](06-models-training-configs.md) | Architecture, MLM training loop, YAML config schema |
| 07 | [Evaluation & metrics](07-evaluation-and-metrics.md) | The two eval families + the full 73-column metric catalog (incl. the 6 "radar" metrics in depth) |
| 08 | [Experiments](08-experiments.md) | The 17-experiment matrix, naming convention, output artifacts, reproduction recipe |
| 09 | [Figures](09-figures.md) | Paper figures, their generator scripts, the unified color scheme, the render environment |
| 10 | [MLCB 2026 rebuttal notes](10-rebuttal-notes.md) | **Working doc.** Reviewer concerns, verified split facts w/ code refs, contamination axes, seed + binding experiment plans |
| – | [Results & % claims](results.md) | Per-task gains over uniform, the appendix defense table, and the abstract's "+6% median / 4.5× CDR3" claims |
| – | [Rebuttal response pack](rebuttal.md) | **Start here for the response.** Reviewer concerns clustered, the empirical result answering each, draft reply text, retraction list, per-reviewer assembly index |

## Key facts at a glance

- **Model:** RoFormer / AntiBERTa2 architecture, per-amino-acid tokenizer (`alchemab/antiberta2`, vocab 28). Main size **medium** = 12 layers, 512 hidden, 8 heads, ~50M params. Trained **from scratch** (`from_pretrained: false`).
- **Pretraining data:** `data/processed/oas_vh_500k.jsonl` — ~497K human VH sequences from the Observed Antibody Space (OAS).
- **Masking budget:** BERT-standard 15% with the 80/10/10 (mask/random/keep) split. Strategies change *which* positions are selected, not the budget.
- **Equal-compute comparison:** every model trains for exactly **125,000 steps** (warmup 6,250 = 5%, cosine LR, effective batch 64, FP16, early stopping disabled).
- **Fair evaluation:** zero-shot MLM/perplexity is always measured under a **uniform reference mask**; the held-out eval split is fixed (`EVAL_SPLIT_SEED=42`) across all models.
- **Comparison table:** `comparison_outputs/comparison_table.csv` — **17 experiments × 73 metric columns** (all single-chain).
- **Render env for `paper/figures`:** `/usr/scratch/thomasawalton/envs/protein_env/bin/python` (see [09-figures](09-figures.md)).

## Provenance of this wiki

Built from a full read of `masking/`, `training/`, `models/`, `evaluation/`, `data/`, `configs/`, `scripts/`, plus the repo's own `README.md` and `GUIDE.md`, and the authoritative `comparison_outputs/comparison_table.csv`. Cross-checked against the actual configs (not just prose). Last built 2026-06-15.
