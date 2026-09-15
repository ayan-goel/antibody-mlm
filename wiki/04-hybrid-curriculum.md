# 04 · Hybrid masking & curriculum (`masking/hybrid.py`)

`hybrid` is a meta-strategy: each training **sequence** is masked by **exactly one** sub-strategy, drawn from a mixture distribution over the specialists. The mixture can be **static**, follow a **curriculum** that reweights the specialists over training, or **adapt** to training loss. Eight hybrid variants are registered as experiments; they share the architecture and compute of the other runs and differ principally along a single axis each — the curriculum schedule, the pool of priors, the sampling granularity, the weighting rule, or the initialization.

## Formulation

At step `t` the sampler holds a mixing distribution `π(t) = (p₁(t), …, p_K(t))` over the `K` specialists (`Σ_s p_s(t) = 1`). For each sequence it draws one specialist `k ∼ π(t)` and masks that sequence with the chosen specialist's own sampler ([03](03-masking-strategies.md)). Key points, worth getting right because the notation invites misreadings:

- `π(t)` is a **full distribution over all specialists at every step**. The curriculum *reweights how often each specialist is sampled*; it does **not** anneal from one pure strategy to the next. Equivalently, each `p_s(t)` is its own piecewise-linear curve over training (e.g. uniform/span start high and decay; germline grows from a small share).
- `t` is the **optimizer step**, not the epoch. Breakpoints are step counts against `max_steps = 125000`; the schedule holds the last breakpoint's weights thereafter.
- The draw is **per sequence** — so a single batch is a *heterogeneous* mix of specialists in proportions ≈ `π(t)` (the mixture `π(t)` is shared across the batch, but each sequence rolls its own `k`). The exception is `hybrid_perbatch`, which draws once per minibatch.
- Every specialist masks the same 15% budget, so the *amount* of masking is invariant to the draw — only its *placement* changes.

## Mechanics

- **`sub_strategies`** — list of registry names to mix, e.g. `["uniform","cdr","span","structure","interface","germline"]`. Each is instantiated once (with its own `sub_strategy_params`).
- **`policy_weights`** — base mixture probabilities over the sub-strategies (same length/order).
- **Per-sample availability filtering** — if a sampled sub-strategy's required metadata is missing for that sample, it is dropped from that sample's mixture and the weights renormalize. So a sample lacking paratope labels can still be masked by `uniform`/`span`/`cdr` rather than being lost.
- **`sampling_mode`** — `"per_sample"` (default; each example in a batch can use a different sub-strategy) or `"per_batch"` (one sub-strategy for the whole minibatch, via the collator's `begin_batch()` hook — a "pure specialist" gradient per step).
- **`curriculum`** — optional list of `{step, weights}` breakpoints; at step *t* the weights are **linearly interpolated** between surrounding breakpoints (clamped to the first/last). The trainer pushes the current step via `callbacks.HybridMaskingCallback`, and the weights live in a **shared-memory tensor** so forked DataLoader workers see the updates.
- **`adaptive`** — when `True`, tracks an EMA of per-sub-strategy loss and blends a softmax over it into the weights (`adaptive_decay=0.99`, `adaptive_temperature` configurable), steering budget toward sub-strategies the model is doing worst on.

### Curriculum YAML syntax (from `configs/hybrid_curriculum_medium.yaml`)

```yaml
masking:
  strategy: "hybrid"
  params:
    sub_strategies: ["uniform", "cdr", "span", "structure", "interface", "germline"]
    policy_weights:  [0.30, 0.15, 0.30, 0.10, 0.10, 0.05]
    sub_strategy_params:
      cdr:       {cdr3_weight: 6.0}
      span:      {geometric_p: 0.2, max_span_length: 10}
      structure: {k_neighbors: 32}        # NB: 32 here, vs 5 in standalone structure_medium
      interface: {paratope_weight: 6.0}
      germline:  {mutated_weight: 6.0}
    curriculum:
      - {step: 0,     weights: [0.30, 0.15, 0.30, 0.10, 0.10, 0.05]}
      - {step: 6250,  weights: [0.10, 0.30, 0.10, 0.15, 0.25, 0.10]}
      - {step: 18750, weights: [0.10, 0.20, 0.10, 0.15, 0.20, 0.25]}
      - {step: 40000, weights: [0.10, 0.20, 0.10, 0.15, 0.20, 0.25]}
```

> The hybrid's `structure` sub-strategy uses `k_neighbors: 32`, whereas the standalone `structure_medium` experiment uses `k_neighbors: 5`. The IgFold sidecar stores 32 neighbors per residue; both draw from it.

## The eight hybrid variants

Shared unless noted: pool `[uniform, cdr, span, structure, interface, germline]` (weight vectors are in that order), from-scratch RoFormer-medium, `max_steps=125000`, `lr=5e-5`, per-sequence draw; curricula hold the final breakpoint's weights for the rest of training. All schedules below are verbatim from the configs.

### A. Curriculum-shape variants (vary the trajectory `π(t)` only)

**`hybrid_curriculum`** — canonical "general → specialized → balanced":

> Schedules below are given as booktabs `tabular`s (need `\usepackage{booktabs}`); weights are in column order and sum to 1 per row. Wrap in a `table` float with a caption for the paper.

```latex
\begin{tabular}{rcccccc}
\toprule
step & uniform & cdr & span & structure & interface & germline \\
\midrule
0     & 0.30 & 0.15 & 0.30 & 0.10 & 0.10 & 0.05 \\
6250  & 0.10 & 0.30 & 0.10 & 0.15 & 0.25 & 0.10 \\
18750 & 0.10 & 0.20 & 0.10 & 0.15 & 0.20 & 0.25 \\
40000 & 0.10 & 0.20 & 0.10 & 0.15 & 0.20 & 0.25 \\
\bottomrule
\end{tabular}
```

Stage 1 region-agnostic (uniform+span = 60%); stage 2 raises the binding-site priors (cdr+interface); stage 3 raises germline; stage 4 (held to 125K) is balanced.

**`hybrid_stretched`** — same arc, longer stages (finishes at 90K), sharper start:

```latex
\begin{tabular}{rcccccc}
\toprule
step & uniform & cdr & span & structure & interface & germline \\
\midrule
0     & 0.50 & 0.05 & 0.40 & 0.00 & 0.05 & 0.00 \\
20000 & 0.10 & 0.30 & 0.10 & 0.10 & 0.30 & 0.10 \\
50000 & 0.05 & 0.15 & 0.05 & 0.15 & 0.30 & 0.30 \\
90000 & 0.10 & 0.20 & 0.10 & 0.15 & 0.20 & 0.25 \\
\bottomrule
\end{tabular}
```

**`hybrid_reverse`** — specialized-first (≈ `stretched` traversed backwards), a control for whether curriculum *direction* matters:

```latex
\begin{tabular}{rcccccc}
\toprule
step & uniform & cdr & span & structure & interface & germline \\
\midrule
0      & 0.05 & 0.10 & 0.05 & 0.15 & 0.30 & 0.35 \\
35000  & 0.05 & 0.15 & 0.05 & 0.15 & 0.30 & 0.30 \\
75000  & 0.10 & 0.30 & 0.10 & 0.10 & 0.30 & 0.10 \\
105000 & 0.50 & 0.05 & 0.40 & 0.00 & 0.05 & 0.00 \\
\bottomrule
\end{tabular}
```

**`hybrid_weighted`** — schedule biased toward the strongest specialist: a region-agnostic warm-up, then an interface-dominant phase, then interface+germline, then a consolidated mix.

```latex
\begin{tabular}{rcccccc}
\toprule
step & uniform & cdr & span & structure & interface & germline \\
\midrule
0     & 0.40 & 0.05 & 0.45 & 0.00 & 0.10 & 0.00 \\
6250  & 0.10 & 0.10 & 0.10 & 0.05 & 0.50 & 0.15 \\
18750 & 0.05 & 0.10 & 0.05 & 0.10 & 0.35 & 0.35 \\
40000 & 0.10 & 0.10 & 0.20 & 0.10 & 0.30 & 0.20 \\
\bottomrule
\end{tabular}
```

### B. Composition variant (vary the pool)

**`hybrid_intersection`** — adds `intersection` (paratope∩germline, with its own `(h,ℓ₀)=(6,1)`) as a 7th prior, introduced only after the component priors are learned. Order `[…, germline, intersection]`:

```latex
\begin{tabular}{rccccccc}
\toprule
step & uniform & cdr & span & structure & interface & germline & intersection \\
\midrule
0     & 0.40 & 0.10 & 0.40 & 0.00 & 0.05 & 0.05 & 0.00 \\
18750 & 0.10 & 0.25 & 0.10 & 0.10 & 0.25 & 0.20 & 0.00 \\
50000 & 0.05 & 0.10 & 0.05 & 0.10 & 0.20 & 0.20 & 0.30 \\
90000 & 0.10 & 0.10 & 0.10 & 0.15 & 0.15 & 0.15 & 0.25 \\
\bottomrule
\end{tabular}
```

(Changes both the pool and the schedule, so not a strict one-knob ablation.)

### C. Sampling-granularity variant

**`hybrid_perbatch`** — the cleanest one-knob ablation: identical schedule to `hybrid_curriculum`, but `sampling_mode: per_batch` draws a single specialist per minibatch (every sequence in the batch shares it). Tests per-sequence mixing vs a per-batch "pure" objective.

### D. Weighting-rule variant (no fixed schedule)

**`hybrid_adaptive`** — self-paced. Base weights `[.20,.15,.20,.10,.20,.15]`; tracks an EMA of each specialist's MLM loss (`adaptive_decay=0.99`) and reweights toward the worst-reconstructed specialist via a softmax over the loss EMA (`adaptive_temperature=0.5`). The schedule emerges from training dynamics rather than being prescribed.

### E. Initialization variant (no curriculum)

**`hybrid_warmstart`** — continued pretraining, not a curriculum. The mixture is **static** at the balanced end-state `[.10,.20,.10,.15,.20,.25]`, but the model is **initialized from the trained `interface_medium` checkpoint** (`from_pretrained: true`) and trained for `max_steps=50000` at `lr=2e-5` (warmup 2500). Tests layering the remaining priors on top of the strongest specialist. It is the only hybrid **not** trained from scratch on the 125K-step budget — flag it as continued-pretraining when comparing.

## Reporting

All eight hybrids above are reported. Structural notes: `curriculum`/`stretched`/`reverse` form a controlled triple (same six priors; differ only in schedule timing and direction), and `perbatch` is the cleanest single-knob ablation. The current `fig3_hybrids` rank chart plots five of the eight (`weighted`, `warmstart`, `stretched`, `reverse`, `perbatch`); regenerate it ([09-figures](09-figures.md)) if all eight should appear. A paired-data hybrid (`hybrid_paired_medium`) exists as a config but is **not** in the comparison table ([08-experiments](08-experiments.md)).
