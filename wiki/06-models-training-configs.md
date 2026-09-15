# 06 · Models, training & configs

## Architecture (`models/model.py`)

RoFormer (rotary-position transformer — the **AntiBERTa2** architecture), via HuggingFace `RoFormerForMaskedLM`. Per-amino-acid tokenizer loaded from `alchemab/antiberta2` (`utils/tokenizer.py`). Models are built **from scratch** (`from_pretrained: false`) so the masking strategy is the only thing that shapes the learned representation.

**Shared config defaults:** `vocab_size=28`, `max_position_embeddings=256`, dropout 0.1, `type_vocab_size=1`, `pad_token_id=0`.

**Named sizes (`MODEL_SPECS`):**

| size | layers | hidden | heads | intermediate | ~params | use |
|------|--------|--------|-------|--------------|---------|-----|
| `small` | 6 | 256 | 8 | 1024 | ~4.8M | pipeline validation |
| **`medium`** | **12** | **512** | **8** | **2048** | **~50M** | **all reported experiments** |
| `full` | 16 | 1024 | 16 | 4096 | ~202M | matches AntiBERTa2 (unused in the comparison) |

`build_model(model_name, from_pretrained, model_size)` → `RoFormerForMaskedLM`. A multispecific variant (`build_multispecific_model`) extends the vocab to 32 (adds `[MOD1]`, `[MOD2]`, `[H]`, `[L]`) and `max_position_embeddings` to 512 for ~384-token paired sequences. Chain identity is carried by framing tokens, not `token_type_ids`.

## Training loop (`training/`)

Standard MLM via a thin HuggingFace `Trainer` wrapper (`training/trainer.py`):

1. Build dataset (`AntibodyDataset`/`PairedAntibodyDataset`) + seeded train/eval split.
2. Resolve the masking strategy with `get_strategy(masking.strategy, tokenizer, **masking.params)` and wrap it in `MLMDataCollator`.
3. Build the model from `model.*`.
4. Construct `TrainingArguments` from `training.*` and run `Trainer.train()`.
5. For `hybrid`, `callbacks.HybridMaskingCallback` advances the curriculum step each training step.
6. On finish, write `models/checkpoints/<exp>/final/` + `training_summary.json` (eval/train history, final metrics, metadata).

**Standard medium-model hyperparameters** (from `configs/medium.yaml`, identical across single-chain runs):

| Setting | Value |
|---|---|
| objective | MLM, BERT 80/10/10 at `mask_prob=0.15` |
| `max_steps` | **125000** (equal-compute; `num_epochs` ignored) |
| `warmup_steps` | 6250 (5%), cosine LR |
| `learning_rate` | 5e-5 (`hybrid_warmstart`: 2e-5) |
| effective batch | 64 = `batch_size 32` × `gradient_accumulation_steps 2` (paired: 16×4) |
| precision | FP16 |
| `weight_decay` | 0.01 |
| `save_steps` / `eval_steps` / `logging_steps` | 2000 / 5000 / 100 |
| `early_stopping_patience` | **0 (disabled)** — fair equal-step comparison; no `load_best_model_at_end` |
| `dataloader_num_workers` | 4 |
| `seed` | 42 |

The final 125K-step checkpoint is always the one reported (intentionally **not** the best-eval checkpoint) — see the fairness note in [01-overview](01-overview.md).

**Entry point.** `scripts/train.py` maps one CLI flag per experiment to a config + experiment name (`MODEL_REGISTRY`), trains the listed models sequentially, and redirects each to `logs/<name>.log`:

```bash
python scripts/train.py --uniform --cdr --span --structure --interface --germline
python scripts/train.py --config configs/medium.yaml     # legacy single-config mode
```

## Config schema (`configs/*.yaml` → `training/config.py`)

Four sections plus a top-level `seed`. `masking.params` is forwarded verbatim as `**kwargs` to the strategy constructor.

```yaml
seed: 42

data:
  processed_path: "data/processed/oas_vh_500k.jsonl"
  max_length: 160          # 384 for paired
  min_length: 80
  train_split: 0.9
  valid_amino_acids: "ACDEFGHIKLMNPQRSTVWY"
  coords_path:   ""        # → oas_vh_500k_igfold.pt   (structure / hybrid)
  paratope_path: ""        # → oas_vh_500k_paratope.pt (interface / intersection / hybrid)
  germline_path: ""        # → oas_vh_500k_germline.pt (germline / intersection / hybrid)
  # interface_path / paired / bispecific — paired pipeline only

masking:
  strategy: "uniform"      # any registered name (see 03)
  mask_prob: 0.15
  mask_token_ratio: 0.8
  random_token_ratio: 0.1
  params: {}               # strategy-specific kwargs (see 03 / 04)

model:
  model_name: "alchemab/antiberta2"
  from_pretrained: false
  model_size: "medium"

training:
  output_dir: "models/checkpoints/<exp>"
  batch_size: 32
  learning_rate: 5.0e-5
  num_epochs: 20           # ignored when max_steps > 0
  max_steps: 125000
  warmup_steps: 6250
  weight_decay: 0.01
  logging_steps: 100
  save_steps: 2000
  eval_steps: 5000
  fp16: true
  dataloader_num_workers: 4
  gradient_accumulation_steps: 2
  early_stopping_patience: 0
```

**Naming convention.** Config filename → experiment name. `medium.yaml` is the `uniform_medium` baseline; `<strategy>_medium.yaml` → `<strategy>_medium`. The `_medium` suffix encodes `model.model_size: "medium"`; it is not a separate field. `configs/experiments.yaml` is the registry that maps experiment names → config + checkpoint paths and lists the downstream tasks; it's consumed by `run_all_evaluations.py` and `generate_report.py`.

**Downstream configs** (`configs/downstream/*_probe.yaml`): `task`, `checkpoint` (overridden per experiment by the batch runner), `mode` (`probe` = frozen encoder / `finetune` = end-to-end), `num_seeds`, and probe-training hyperparameters. On-disk: `paratope_probe`, `contact_map_probe`, `structure_probe_probe`, `developability_probe`. (No `binding_probe.yaml` — the binding task isn't wired into the table.)

See [08-experiments](08-experiments.md) for the full config↔experiment table and [03](03-masking-strategies.md)/[04](04-hybrid-curriculum.md) for the `masking.params` of each strategy.
