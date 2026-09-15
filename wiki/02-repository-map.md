# 02 · Repository map

## Top-level directories

| Path | Role |
|------|------|
| `configs/` | YAML experiment configs (one per training run) + `experiments.yaml` registry + `downstream/` probe configs. **No `.py` here.** |
| `data/` | Data loading/preprocessing code (`dataset.py`, `download*.py`, `preprocessing*.py`), the `benchmarks/` loaders, and the `processed/` + `structures/` + `ab_bind/` data artifacts. See [05-data](05-data.md). |
| `masking/` | The pluggable masking-strategy package — the core contribution. See [03-masking-strategies](03-masking-strategies.md). |
| `models/` | `model.py` (RoFormer factory) + `checkpoints/<exp>/` trained weights. See [06](06-models-training-configs.md). |
| `training/` | HF-Trainer wrapper: `config.py` (dataclasses), `trainer.py` (loop), `callbacks.py` (curriculum). |
| `evaluation/` | Zero-shot evals + `downstream/` probe framework + `compare.py`/`report.py` aggregation. See [07](07-evaluation-and-metrics.md). |
| `scripts/` | CLI entry points for every pipeline stage (22 scripts). See [08-experiments](08-experiments.md). |
| `utils/` | `tokenizer.py`, `seed.py`, `annotate_cdrs.py` (ANARCI), `io.py`. |
| `tests/` | Unit/integration tests (10 files). |
| `comparison_outputs/` | Cross-experiment artifacts: `comparison_table.{csv,json,tex}`, `summary.md`, plots. The CSV is the canonical results table consumed by the paper figures. |
| `evaluation_outputs/` | Per-experiment `all_metrics.json` (merged zero-shot + downstream metrics). |
| `downstream_outputs/` | Per-task probe results: `<task>_<mode>/<exp>/results.json` + cached embeddings. |
| `logs/` | Per-model training logs (`<exp>.log`) from the multi-model `train.py` path. |
| `paper/` | The NeurIPS-2026 manuscript (`neurips_2026.tex`) and `figures/` (generators + outputs). See [09-figures](09-figures.md). |
| `markdown/` | An article (`presentation.tex`) + beamer slides (`slides.tex`) + `BUILD.md`; pulls figures from `comparison_outputs/`. |
| `README.md`, `GUIDE.md` | Repo's own prose docs. `claude.md` is Claude-Code instructions, not project docs. |

## Module file inventory

- `masking/` (12 py): `__init__.py`, `base.py`, `collator.py`, `uniform.py`, `cdr.py`, `span.py`, `interface.py`, `germline.py`, `intersection.py`, `structure.py`, `hybrid.py`, `multispecific.py`
- `training/` (4): `__init__.py`, `config.py`, `trainer.py`, `callbacks.py`
- `models/` (2): `__init__.py`, `model.py`
- `evaluation/` (13): `base.py`, `mlm_accuracy.py`, `pseudo_loglikelihood.py`, `mutation_scoring.py`, `infilling.py`, `infilling_quality.py`, `attention_analysis.py`, `embeddings.py`, `visualize.py`, `significance.py`, `compare.py`, `report.py`, `__init__.py`
- `evaluation/downstream/` (13): `base.py`, `config.py`, `encoder.py`, `heads.py`, `trainer.py`, `collator.py`, `embedding_cache.py`, `_metric_utils.py`, `paratope.py`, `contact_map.py`, `structure_probe.py`, `developability.py`, `__init__.py`
- `data/` (7): `dataset.py`, `dataset_paired.py`, `download.py`, `download_paired.py`, `preprocessing.py`, `preprocessing_paired.py`, `__init__.py`
- `data/benchmarks/` (7): `paratope.py`, `contact_map.py`, `structure_probe.py`, `developability.py`, `ab_bind.py`, `binding.py`, `__init__.py`
- `configs/` (24 yaml): 17 training configs + `experiments.yaml` + `medium.yaml` (=uniform) + `downstream/{paratope,contact_map,structure_probe,developability}_probe.yaml`

> Note: `GUIDE.md`'s tree lists a few names that differ from disk (e.g. `data/download.py` vs a `download_data.py` script, a `binding_probe.yaml` that is **not** present in `configs/downstream/`). Trust the on-disk listing above; the prose docs drifted slightly.

## File-format conventions

These three formats recur throughout the repo; knowing them unlocks most of the data flow.

### JSONL (pretraining corpus)
`data/processed/oas_vh_500k.jsonl` — one JSON object per line, no newlines inside a record. Core fields: `sequence` (VH amino acids, uppercase), `cdr1_aa`/`cdr2_aa`/`cdr3_aa` (substrings of `sequence`), `v_call`/`j_call` (IMGT allele strings, e.g. `IGHV1S45*01`). Loaded via `utils/io.load_jsonl` → `list[dict]`. Full schema + the paired variant: [05-data](05-data.md).

### `.pt` metadata sidecars (1:1 with the JSONL)
Files under `data/structures/` are `torch.save`'d **`list[dict | None]`** objects whose length equals the JSONL record count — entry `i` corresponds to record `i` (or `None` if generation failed/was filtered). Each dict holds one or more per-residue tensors keyed by name (`paratope_labels`, `germline_labels`, `knn_indices`). Loaded with `torch.load(path, weights_only=False)`. This is how the biological priors reach the masking strategies at training time. Schemas: [05-data](05-data.md).

### YAML configs
`configs/*.yaml` deserialize into typed dataclasses (`training/config.py`). A config has four sections — `data`, `masking`, `model`, `training` (+ a top-level `seed`). `masking.params` is a free-form dict forwarded as `**kwargs` to the strategy constructor. Full schema + every field: [06-models-training-configs](06-models-training-configs.md).

## How to navigate to answer a question

- *"What does strategy X actually do / its parameters?"* → [03](03-masking-strategies.md) (and [04](04-hybrid-curriculum.md) for hybrids).
- *"Where does the paratope/germline/structure signal come from?"* → [05-data](05-data.md) (sidecars + generation scripts).
- *"What is metric `ds_…`/`mut_…`/`infill_…`?"* → [07-evaluation-and-metrics](07-evaluation-and-metrics.md) (full 73-column catalog).
- *"Which experiments exist and how do I reproduce a number?"* → [08-experiments](08-experiments.md).
- *"How was figure N made / what colors?"* → [09-figures](09-figures.md).
