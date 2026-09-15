# 08 · Experiments, artifacts & reproduction

## The experiment matrix (17 rows in `comparison_table.csv`)

All are **single-chain, medium (~50M), 125K steps**, differing only in masking. Grouped by role:

**Control (1)**
| Experiment | Config | Notes |
|---|---|---|
| `untrained_medium` | `untrained_medium.yaml` | random-init, **never trained** — the floor; isolates architecture from pretraining |

**Specialists (8)** — one masking prior each
| Experiment | Config | Strategy + key params |
|---|---|---|
| `uniform_medium` | `medium.yaml` | `uniform` (baseline) |
| `cdr_medium` | `cdr_medium.yaml` | `cdr` (cdr3_weight 6) |
| `span_medium` | `span_medium.yaml` | `span` (p 0.2, max 10) |
| `interface_medium` | `interface_medium.yaml` | `interface` (paratope_weight 6) |
| `germline_medium` | `germline_medium.yaml` | `germline` (mutated_weight 6) |
| `intersection_medium` | `intersection_medium.yaml` | `intersection` (paratope ∩ germline) |
| `structure_medium` | `structure_medium.yaml` | `structure` **k=5, sep=0** (IgFold) |
| `structure_longrange_medium` | `structure_longrange_medium.yaml` | `structure` **k=5, sep=4** (long-range only) — figure label "structure-LR" |

**Hybrids (8)** — mixtures/curricula ([04-hybrid-curriculum](04-hybrid-curriculum.md))
`hybrid_curriculum_medium`, `hybrid_stretched_medium`, `hybrid_reverse_medium`, `hybrid_warmstart_medium`, `hybrid_perbatch_medium`, `hybrid_weighted_medium`, `hybrid_adaptive_medium`, `hybrid_intersection_medium`.

> `strategy`/`model_size` are blank for `untrained_medium` in the CSV. The `structure` vs `structure-LR` pair is the **same strategy**, distinguished only by `min_seq_separation` (0 vs 4).

### Implemented but NOT in the comparison table
- `multispecific_medium`, `hybrid_paired_medium` — paired VH+VL runs; configs and code exist, but no rows in `comparison_table.csv` (the reported comparison is single-chain only).
- **binding** (CoV-AbDab) downstream task — loader exists (`data/benchmarks/binding.py`), but no `binding_probe.yaml` and no `ds_binding_*` columns.

### Figure subset note
`fig3_hybrids` ranks **14** of the 17: the 9 specialists+control (`untrained, uniform, cdr, span, interface, germline, intersection, structure, structure-LR`) plus **5** hybrids (`weighted, warmstart, stretched, reverse, perbatch`). `hybrid_curriculum`, `hybrid_adaptive`, `hybrid_intersection` are evaluated but omitted there. The radars (`fig_radar_*`) show the **8 specialists** (incl. structure-LR). See [09-figures](09-figures.md).

## Output artifacts

```
models/checkpoints/<exp>/
    final/                       # model + tokenizer (the reported 125K-step checkpoint)
    training_summary.json        # eval/train history, final metrics, metadata
evaluation_outputs/<exp>/
    all_metrics.json             # merged zero-shot + downstream metrics (resumable)
downstream_outputs/<task>_<mode>/<exp>/
    results.json                 # per-seed + aggregated probe metrics
    embedding_cache/             # cached frozen embeddings (train/val/test)
logs/<exp>.log                   # per-model training log (multi-model train.py path)
comparison_outputs/
    comparison_table.{csv,json,tex}   # 17×73 — csv is consumed by paper figures
    summary.md, *.png                 # human-readable summary + grouped bar charts
```

The batch runner is **resumable**: existing sections in `all_metrics.json` are preserved across reruns, so only missing/failed sections recompute.

## Scripts catalog (`scripts/`, 22 files)

| Stage | Scripts |
|-------|---------|
| Data build | `download_data.py`, `download_paired_data.py`, `annotate_cdrs.py`, `preprocessing*` (in `data/`) |
| Sidecars | `compute_paratope_labels.py`, `compute_germline_labels.py`, `predict_structures*.py`, `compute_multispecific_labels.py`, `build_sabdab_real_coords.py`, `convert_knn_int16.py` |
| Train | `train.py` (multi-model flags or `--config`) |
| Evaluate | `run_all_evaluations.py` (the orchestrator), `benchmark_mutations.py`, `run_downstream.py` |
| Aggregate / plot | `compare.py`, `generate_report.py`, `refresh_evaluations.py`, `generate_split_plots.py` |

**`run_all_evaluations.py`** loads each checkpoint once and runs MLM accuracy, perplexity, infilling (+quality), PLL, the AB-Bind mutation benchmark, and the downstream probes, merging into `all_metrics.json`. Useful flags: `--experiments <names…>`, `--device cuda`, `--skip-{mlm,infilling,pll,perplexity,mutations,downstream,infilling-quality}`, `--max-{pll-sequences,infilling-samples,infilling-quality-samples}`, `--ab-bind-dir`.

## Reproduction recipe

```bash
# 0. Setup
conda create -n abmlm python=3.10 -y && conda activate abmlm
pip install -r requirements.txt && pip install accelerate rjieba

# 1. Data + CDR annotation
python scripts/download_data.py --config configs/medium.yaml
python scripts/annotate_cdrs.py --input data/processed/oas_vh_500k.jsonl

# 2. Metadata sidecars (needed by structure/interface/germline/intersection/hybrid)
python scripts/compute_paratope_labels.py --input data/processed/oas_vh_500k.jsonl --output data/structures/oas_vh_500k_paratope.pt
python scripts/compute_germline_labels.py --input data/processed/oas_vh_500k.jsonl --output data/structures/oas_vh_500k_germline.pt
python scripts/predict_structures.py      --input data/processed/oas_vh_500k.jsonl --output data/structures/oas_vh_500k_igfold.pt --k_neighbors 32

# 3. Train (each → models/checkpoints/<exp>/final/, log → logs/<exp>.log)
python scripts/train.py --uniform --cdr --span --structure --interface --germline
#   plus the post-paper variants via their flags / configs (structure_longrange, intersection, hybrid_*)

# 4. Evaluate everything in configs/experiments.yaml
python scripts/run_all_evaluations.py --device cuda

# 5. Aggregate → comparison_outputs/comparison_table.{csv,json,tex}, summary.md, plots
python scripts/generate_report.py

# 6. (paper) regenerate figures from the CSV
#   see 09-figures.md (use the protein_env python)
```

To reproduce **one number**: find its column in [07-evaluation-and-metrics](07-evaluation-and-metrics.md), the producing module there, and the experiment row above; the value flows checkpoint → `all_metrics.json`/`results.json` → `comparison_table.csv`.
