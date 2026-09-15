# 01 · Overview & workflow

## The question

Antibody language models are usually pretrained with **uniform** masked-language-modeling (MLM): 15% of residues are masked at random. But antibodies are not uniform objects — function concentrates in the **CDR loops**, the **paratope** (antigen-contacting residues), and the **somatically hypermutated** positions that differ from germline. This project asks whether **biasing the MLM mask toward those functionally important positions** yields representations that transfer better to downstream antibody tasks.

From `README.md`:
> "Studying whether biologically-informed masking strategies improve antibody protein language models. We train RoFormer-based models from scratch on antibody sequences from OAS and compare a range of masking strategies that inject CDR, structural, paratope, germline, and paired-chain priors into the MLM objective."

The central design choice that makes the comparison clean: **everything is held fixed except the masking strategy** — same architecture, same data, same 125K-step compute budget, same fixed eval split, same uniform reference mask at eval time. Differences in downstream performance are therefore attributable to the masking prior.

## The strategies (the independent variable)

| Strategy | Biases the mask toward | Prior source |
|----------|------------------------|--------------|
| `uniform` | random positions (baseline) | — |
| `cdr` | CDR1/2/3 loops | OAS CDR annotation |
| `span` | contiguous SpanBERT-style spans | none (region-agnostic) |
| `interface` | paratope (antigen-contacting) residues | predicted paratope labels |
| `germline` | somatic-hypermutation sites | per-residue germline comparison |
| `intersection` | paratope **∩** germline (mutated binding residues) | product of two priors |
| `structure` | 3-D spatial neighborhoods (keeps neighbors visible) | IgFold Cα kNN |
| `hybrid` | a stochastic mixture of the above, optionally on a curriculum | all of the above |
| `multispecific` | paired VH+VL policies (paratope / shared-light / VH–VL interface) | paired sidecars |

`multispecific` (and the paired hybrid) are **implemented but not part of the final single-chain comparison table** — see [08-experiments](08-experiments.md). Details and exact parameters: [03-masking-strategies](03-masking-strategies.md) and [04-hybrid-curriculum](04-hybrid-curriculum.md).

## End-to-end pipeline

```
                 scripts/download_data.py
   OAS  ───────────────────────────────────────►  data/processed/oas_vh_500k.jsonl
                 scripts/annotate_cdrs.py            (sequence + cdr{1,2,3}_aa + v/j_call)
                                                          │
        scripts/compute_paratope_labels.py               │  (1:1 aligned .pt sidecars)
        scripts/compute_germline_labels.py    ───────────┤  oas_vh_500k_paratope.pt
        scripts/predict_structures*.py                   │  oas_vh_500k_germline.pt
                                                          │  oas_vh_500k_igfold.pt
                                                          ▼
                 scripts/train.py --<strategy>     ┌──────────────────────┐
   configs/<exp>.yaml  ─────────────────────────►  │  MLM pretraining     │
   (masking.strategy + params)                      │  RoFormer, 125K steps│
                                                     └──────────┬───────────┘
                                                                ▼
                                          models/checkpoints/<exp>/final/  (+ training_summary.json)
                                                                │
                 scripts/run_all_evaluations.py                 ▼
                 ┌───────────────────────────────────────────────────────────┐
                 │ zero-shot: MLM acc, perplexity, PLL, infilling, mutation,   │
                 │            attention   (uniform reference mask)             │
                 │ probes:    paratope, contact_map, structure_probe,          │
                 │            developability  (frozen encoder + trained head)  │
                 └───────────────────────────────┬───────────────────────────┘
                                                  ▼
              evaluation_outputs/<exp>/all_metrics.json
              downstream_outputs/<task>_<mode>/<exp>/results.json
                                                  │
                 scripts/generate_report.py       ▼
              comparison_outputs/comparison_table.{csv,json,tex}, summary.md, plots
                                                  │
                 paper/figures/make_fig*.py        ▼
              paper/figures/*.{pdf,png}   (the manuscript figures)
```

## Entry points (in pipeline order)

| Stage | Command | Output |
|-------|---------|--------|
| Download | `python scripts/download_data.py --config configs/medium.yaml` | `data/processed/oas_vh_500k.jsonl` |
| Annotate CDRs | `python scripts/annotate_cdrs.py --input data/processed/oas_vh_500k.jsonl` | adds `cdr{1,2,3}_aa` fields |
| Sidecars | `python scripts/compute_paratope_labels.py …` / `compute_germline_labels.py …` / `predict_structures*.py …` | `data/structures/*.pt` |
| Train | `python scripts/train.py --uniform --cdr --span --structure --interface --germline` | `models/checkpoints/<exp>/final/` |
| Evaluate | `python scripts/run_all_evaluations.py --device cuda` | `evaluation_outputs/<exp>/all_metrics.json` |
| Report | `python scripts/generate_report.py` | `comparison_outputs/comparison_table.*` |
| Figures | `python paper/figures/make_fig*.py` | `paper/figures/*.{pdf,png}` |

`scripts/train.py` accepts one flag per experiment (`--uniform`, `--cdr`, …); it trains the listed models sequentially and redirects each model's output to `logs/<name>.log`. See [08-experiments](08-experiments.md) for the full flag list and the reproduction recipe.

## Two fairness decisions worth remembering

1. **Equal compute.** Every model trains for exactly `max_steps=125000` with early stopping disabled (`patience=0`) and no `load_best_model_at_end`. Selecting a "best" intermediate checkpoint per model on a noisy eval metric would defeat the equal-compute comparison, so the final 125K-step checkpoint is always the one reported. (`GUIDE.md` §2)
2. **Uniform reference masking at eval.** MLM accuracy and region-stratified perplexity are computed under a uniform mask for *every* model, regardless of its training-time strategy. A cdr/germline/interface model evaluated under its own skewed mask would look artificially worse because it masks harder positions. (`GUIDE.md` §3) The eval split itself is pinned by `EVAL_SPLIT_SEED=42` so all models see the identical held-out set.

## Primary source docs

- `README.md` — one-screen summary + quick-start.
- `GUIDE.md` — the project's own full walkthrough (412 lines): structure, every command, how to add a strategy/task, how evaluation works. This wiki expands on it and reconciles it with the actual configs.
