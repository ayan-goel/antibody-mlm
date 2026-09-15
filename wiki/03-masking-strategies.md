# 03 · Masking strategies (`masking/`)

The core contribution. A pluggable registry of MLM masking strategies, all sharing one interface, selected by name from a config. Each strategy keeps the **same 15% budget and BERT 80/10/10 split** but changes *which* positions get masked.

> Hybrid + curriculum get their own page: [04-hybrid-curriculum](04-hybrid-curriculum.md).

## Registry & factory

- Strategies subclass `BaseMaskingStrategy` and self-register via the `@register_strategy("name")` decorator into a global registry (`masking/base.py`).
- `masking/__init__.py` imports each concrete module so the decorators fire at import time.
- `get_strategy(name, tokenizer, **kwargs)` instantiates by name; `**kwargs` come straight from a config's `masking.params`. Unknown names raise `KeyError` listing the registered set.

**Registered names (9):** `uniform`, `cdr`, `span`, `structure`, `interface`, `germline`, `intersection`, `multispecific`, `hybrid`.

There are **no separate "structure-LR" / "structure_raw" registry entries** — those are the same `structure` strategy with different `min_seq_separation` params. (The figure/label names `structure` vs `structure-LR` are presentation aliases for `structure_medium` vs `structure_longrange_medium`.)

## Base API (`masking/base.py`)

```python
class BaseMaskingStrategy(ABC):
    def __init__(self, tokenizer, mask_prob=0.15,
                 mask_token_ratio=0.8, random_token_ratio=0.1, ...): ...

    @abstractmethod
    def select_mask_positions(self, input_ids, special_tokens_mask,
                              metadata=None) -> torch.Tensor:  # bool (seq_len,)
        ...

    def apply(self, input_ids, special_tokens_mask, metadata=None
              ) -> tuple[masked_input_ids, labels]:
        ...
```

- `select_mask_positions` returns a **boolean mask** of shape `(seq_len,)`; `True` = this position will be masked. Special tokens (`special_tokens_mask == 1`, incl. padding) are never masked.
- `apply` calls `select_mask_positions`, then applies the BERT corruption: of the chosen positions, **80%** → `[MASK]`, **10%** → a random canonical amino acid, **10%** kept unchanged; and builds `labels` (true token id at masked positions, `-100` elsewhere). The random-AA pool is restricted to the 20 canonical residues (no X/B/Z).
- Strategies that need biological signal **fall back to uniform** when their metadata is absent (logged once + periodically), so a missing sidecar degrades gracefully rather than crashing.

### `metadata` dict keys (per-sample, optional)

| Key | Shape / dtype | Meaning | Consumed by |
|-----|---------------|---------|-------------|
| `cdr_mask` | `Long(L,)` ∈ {0,1,2,3} | framework / CDR1 / CDR2 / CDR3 | `cdr`, `intersection` |
| `paratope_labels` | `Float(L,)` ∈ [0,1] | per-residue paratope probability | `interface`, `intersection`, `multispecific`, `hybrid` |
| `germline_labels` | `Float(L,)` ∈ {0, 0.5, 1} | germline (0) / CDR3-junction (0.5) / SHM (1) | `germline`, `intersection`, `hybrid` |
| `knn_indices` | `Long/Int16(L, k)` | precomputed 3-D nearest neighbors | `structure` |
| `coords_ca` | `Float(L, 3)` | Cα coordinates (kNN computed on the fly) | `structure` (fallback path) |
| `module_ids`, `chain_type_ids` | `Long(L,)` | module {0,1,2} / chain {special,H,L} | `multispecific` |
| `interface_labels` | `Float(L,)` ∈ [0,1] | VH–VL interface contact probability | `multispecific` policy C |

## Collator (`masking/collator.py`)

`MLMDataCollator(tokenizer, strategy, pad_to_multiple_of=8, return_metadata=False)` turns tokenized examples into MLM batches:

1. Pad `input_ids`/`attention_mask`/`special_tokens_mask` to a common length (multiple of 8); padding counts as special.
2. **Per-sample** masking: for each example, build its metadata dict and call `strategy.apply(...)`. The 15% budget is applied per sample.
3. **Zero-fill** missing metadata keys (matching shape/dtype from a reference example) so the batch is stackable even when some examples lack a key.
4. If the strategy defines `begin_batch()`, the collator calls it once per batch — used by hybrid's `per_batch` sampling.

## Sampling mechanisms

Two distinct samplers sit under the strategies — don't conflate them.

**1. Weighted Bernoulli (per-position, independent).** Used by `uniform`, `cdr`, `interface`, `germline`, `intersection`. Each non-special position `i` gets a weight `wᵢ`, normalized to a per-position probability and sampled with an **independent** coin flip (e.g. `masking/cdr.py:75–82`):

```
pᵢ = min( mask_prob · wᵢ / mean(w over non-special positions), 1.0 )
mᵢ ~ Bernoulli(pᵢ)
```

What this means in practice:
- The mask count `|M|` is the **expected** budget `mask_prob · (#maskable)` (≈15%), but is **random per sequence** — this is *not* a weighted draw without replacement to an exact `⌊0.15·|S|⌋`.
- The `min(·, 1.0)` clamp caps saturated positions at `p = 1`, so when a few weights are very large (e.g. `intersection` gives paratope∩germline residues `wᵢ = 36`) the realized rate can dip slightly below 15%.
- `uniform` is the flat case (`wᵢ ≡ 1` ⇒ `pᵢ = mask_prob`).
- `interface`/`germline` interpolate by the soft label, `wᵢ = w_low + (w_high − w_low)·labelᵢ`; `intersection` takes the element-wise product of the per-prior weight vectors.

**2. Constructive, fixed-budget.** Used by `span` and `structure`. These fill a fixed budget `⌊mask_prob · (#maskable)⌋` *by construction*, not by independent coin flips: `span` lays down contiguous geometric-length spans until the budget is met; `structure` greedily masks dispersed, contact-preserving positions until the budget is met. So these do hit an (approximately) exact budget — but **not** via the proportional `pᵢ` rule above.

> The single `pᵢ ∝ wᵢ` formula therefore describes only the four weighted-Bernoulli strategies; `span` and `structure` need their own description. (This corrected an earlier draft that presented one weighted-without-replacement equation for all of them.)

## Strategy-by-strategy

### `uniform` — baseline
Independent Bernoulli at `mask_prob` over non-special positions. No metadata. The control every other strategy is measured against.

### `cdr` — CDR-weighted
Per-region weights mapped from `cdr_mask`. **Defaults:** `framework_weight=1.0`, `cdr1_weight=3.0`, `cdr2_weight=3.0`, `cdr3_weight=6.0`. CDR3 (most diverse) gets the heaviest weight. Falls back to uniform without `cdr_mask`. *Rationale:* CDRs are the antigen-binding loops and carry most of the functional diversity.

### `span` — SpanBERT-style
Masks contiguous spans. Span length ~ truncated Geometric. **Defaults:** `geometric_p=0.2` (mean ≈ 5 residues), `max_span_length=10`. Spans never cross special tokens (preserves chain boundaries). No metadata. *Rationale:* forces use of longer-range sequence context; spans approximate structural motifs.

### `structure` — 3-D contact-**preserving** (IgFold kNN)
**Defaults:** `k_neighbors=5`, `min_seq_separation=0`.
Algorithm (per the code): shuffle maskable positions; greedily mask a residue and mark its `k` nearest 3-D neighbors as **protected** (they will *not* be masked); continue until the budget is filled (relaxing protection if the budget can't otherwise be met). Uses `knn_indices` when present, else computes kNN from `coords_ca`. Falls back to uniform if neither exists.

> **Correction to `GUIDE.md`/`README.md` prose.** Those describe a "seed-and-grow … mask it *and* its k nearest neighbors" and an "ESM2 contact-map" prior. The shipped code does the **opposite** — it *keeps* each masked residue's neighbors visible (so the model must infer the residue *from* its spatial context) — and the configs point at an **IgFold** sidecar (`oas_vh_500k_igfold.pt`), not ESM2. Trust the code/configs.

`min_seq_separation` drops sequence-adjacent neighbors (i±1…i±sep) from the kNN list. Because the backbone is rigid, raw nearest-neighbors are dominated by ±1/±2 residues; filtering them isolates genuine long-range structural coupling. This is exactly the `structure` (sep=0) vs `structure-LR` (sep=4) distinction — both use `k_neighbors=5`. See [08-experiments](08-experiments.md).

### `interface` — paratope-weighted
Bernoulli weighted by `paratope_labels`. **Defaults:** `paratope_weight=6.0`, `non_paratope_weight=1.0` (linear interpolation by the soft label). Concentrates ~50–60% of the budget on paratope residues (which are only ~15–25% of the chain). Falls back to uniform without labels. *Rationale:* antigen-binding residues are the core functional determinants.

### `germline` — SHM-weighted
Bernoulli weighted by `germline_labels`. **Defaults:** `mutated_weight=6.0`, `germline_weight=1.0`. *Rationale:* ~85% of VH residues match germline, so uniform masking trains overwhelmingly on invariant positions; this redirects the budget onto the somatic-hypermutation sites that actually encode specificity. (CDR3-junction positions are labeled 0.5 — intermediate.)

### `intersection` — product of priors
Multiplies the per-residue weight vectors of two or more priors, concentrating the budget where **all** priors agree. **Default `priors=["paratope","germline"]`** → "mutated binding-site residues." Built-in `PRIOR_SPECS` (key, high, low): `paratope`→(`paratope_labels`,6,1), `germline`→(`germline_labels`,6,1), `cdr`→(`cdr_mask`,3,1). Falls back to uniform if any requested prior's metadata is missing.

Each prior contributes a per-residue weight by **smooth interpolation** between its low/high endpoints (`intersection.py:106–110`), and the final weight is the element-wise product across priors:

$$w_i \;=\; \prod_{k=1}^{K}\Bigl[\,\ell_{0,k} + (h_k-\ell_{0,k})\,s_k(i)\,\Bigr], \qquad s_k(i)\in[0,1],$$

where $s_k(i)$ is the (soft) value of label $k$ at residue $i$, and $(h_k,\ell_{0,k})$ are its high/low weights — the weights a residue approaches as $s_k(i)\to 1$ ("interesting") versus $\to 0$. The resulting $w_i$ then enters the weighted-Bernoulli sampler above.

> **Not a hard indicator.** This is interpolation, not $h_k\mathbf{1}[\ell_k(i)] + \ell_{0,k}\mathbf{1}[\lnot\ell_k(i)]$ — the two coincide only for binary labels. Here `paratope_labels` are soft teacher probabilities and `germline_labels` ∈ {0, 0.5, 1}, so weights are continuous. For the default paratope∩germline with $(h,\ell_0)=(6,1)$, $w_i=[1+5\,s_\text{para}(i)][1+5\,s_\text{germ}(i)]$ ranges over $[1,36]$; the value **36 is the ceiling** (both labels $=1$), not what every paratope∩germline residue receives.
>
> Caveat: the $s_k\in[0,1]$ assumption holds for `paratope`/`germline` but **not** for `cdr` as a prior (`cdr_mask ∈ {0,1,2,3}`), which would give CDR2/CDR3 factors of 5/7 rather than 3. The default `intersection` uses only paratope+germline, so this doesn't affect `intersection_medium`.

### `multispecific` — paired VH+VL (not in the main table)
Samples one of three policies per call (weights via `policy_weights`):
- **A — module-isolated paratope:** bias toward paratope within one module; a small leak elsewhere prevents collapse.
- **B — shared light chain:** heavily mask the light chain (`shared_chain_boost=3.0`) → conditional VL-given-VH infilling.
- **C — VH–VL interface:** weight by `interface_labels`.

Requires paired data with `module_ids` + `chain_type_ids` (falls back to uniform otherwise). Implemented and trainable, but `multispecific_medium`/`hybrid_paired_medium` are **absent from `comparison_table.csv`** — see [08-experiments](08-experiments.md).

### `hybrid` — mixture / curriculum
A meta-strategy that samples one sub-strategy per sample (or per batch) from a weighted mixture, optionally on a step-based curriculum. Full treatment: [04-hybrid-curriculum](04-hybrid-curriculum.md).

## Quick reference

| Strategy | Key params (defaults) | Needs metadata | Fallback |
|----------|----------------------|----------------|----------|
| `uniform` | — | — | — |
| `cdr` | fw 1, cdr1 3, cdr2 3, cdr3 6 | `cdr_mask` | uniform |
| `span` | `geometric_p` 0.2, `max_span_length` 10 | — | — |
| `structure` | `k_neighbors` 5, `min_seq_separation` 0 | `knn_indices`/`coords_ca` | uniform |
| `interface` | `paratope_weight` 6, `non_paratope_weight` 1 | `paratope_labels` | uniform |
| `germline` | `mutated_weight` 6, `germline_weight` 1 | `germline_labels` | uniform |
| `intersection` | `priors=[paratope,germline]` | the priors' keys | uniform |
| `multispecific` | `policy_weights`, `shared_chain_boost` 3 | `module_ids`,`chain_type_ids` | uniform |
| `hybrid` | `sub_strategies`, `policy_weights`, `curriculum`, `sampling_mode`, `adaptive` | per sub-strategy | uniform |
