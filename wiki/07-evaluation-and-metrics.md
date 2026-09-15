# 07 · Evaluation & metrics

Two families of evaluation, both orchestrated by `scripts/run_all_evaluations.py`, merged per experiment into `evaluation_outputs/<exp>/all_metrics.json`, then aggregated by `evaluation/compare.py` → `comparison_outputs/comparison_table.csv` (**17 experiments × 73 columns**).

## The two families

**A. Zero-shot** (`evaluation/*.py`) — measure the pretrained model directly, no task training.
**B. Frozen-embedding probes** (`evaluation/downstream/`) — train a small head on **frozen** last-hidden-layer embeddings; tests what the representation encodes.

### Shared probe protocol (family B)
- Encoder is **frozen** (`mode="probe"`); last hidden states are extracted **once** and cached to disk (`embedding_cache.py`), so only the head trains. (`mode="finetune"` tunes encoder+head end-to-end — supported, not used for the main table.)
- Split: **contact_map** and **structure_probe** use 60/20/20 **by PDB** (seed 42, no complex leakage); **paratope** (SAbDab_Liberis) and **developability** (TAP) use TDC's provided splits. Full procedure + hyperparameters below.
- **Multi-seed:** N independent heads (3 for most tasks; 5 for developability), reported as `<metric>_mean` (± std). Decision thresholds (e.g. Youden's J) are fit on validation, never test.
- Heads (`heads.py`): **TokenClassification** (paratope), **bilinear ContactMap**, **Hewitt–Manning StructureProbe** (predicts squared Cα distances), **mean-pooled Regression** (developability). Special/framing tokens are excluded from pooling for fair paired-vs-single comparison.

### Zero-shot evaluations (family A)
| Module | Produces | Notes |
|--------|----------|-------|
| `mlm_accuracy.py` | MLM accuracy/top-5 + region-stratified perplexity | always under a **uniform reference mask** |
| `pseudo_loglikelihood.py` | `pll`, `pll_normalized` | Σ log p(true token), each position masked in turn |
| `infilling.py` | per-CDR accuracy / exact-match / edit-distance, N-terminus, scattered-k | mask a whole region, argmax-decode |
| `infilling_quality.py` | per-CDR `*_jsd` (+ AA-freq, CDR3-length stats) | distributional naturalness, not exact recovery |
| `mutation_scoring.py` (+ `scripts/benchmark_mutations.py`) | per-complex Spearman/AUROC vs ΔΔG | **wildtype-marginal** PLL scoring on AB-Bind |
| `attention_analysis.py` | attention entropy, head-ablation importance, attention↔contact correlation | model-internals diagnostics |

Two fairness rules (see [01](01-overview.md)): uniform reference mask at eval; fixed `EVAL_SPLIT_SEED=42`. Mutation metrics are aggregated **per complex** (then averaged) to avoid Simpson's paradox; paired models skip the (single-chain) AB-Bind benchmark.

## Downstream probing procedure

All four probes freeze the encoder and train only a light head on cached **last-hidden-layer** embeddings (`mode: probe`; a `finetune` mode exists but is unused for the table). Optimizer **AdamW** (weight decay 0.01), head **LR 1e-3**, **linear-warmup→cosine** schedule (warmup fraction 0.1), gradient-norm clip 1.0, dropout 0.1, **full precision** (no AMP); `base_seed=42`, `seed = 42+i`. Per seed: re-initialize the head → train to early stopping → restore the best-validation checkpoint → (paratope only) fit a Youden's-J threshold on validation → score the test set. Results are mean ± sample std (ddof = 1) over seeds. Early stopping monitors each task's headline validation metric; developability trains a fixed budget (patience 0).

| task | head | LR | epochs | batch | patience | seeds | early-stop metric |
|---|---|---|---|---|---|---|---|
| paratope | per-token linear (H→1) | 1e-3 | 50 | 32 | 10 | 3 | AUPRC |
| contact_map | bilinear over residue pairs | 1e-3 | 50 | 16 | 10 | 3 | precision@L |
| structure_probe | Hewitt–Manning linear (rank 128) | 1e-3 | 100 | 4 | 15 | 3 | Spearman ρ |
| developability | mean-pool + linear (H→5) | 1e-3 | 100 | 16 | 0 (none) | 5 | spearman-macro |

Defaults (`config.py`): `weight_decay 0.01`, `max_grad_norm 1.0`, `warmup_fraction 0.1`, `encoder_learning_rate 1e-5` (finetune-only, **unused**), `base_seed 42`. H = encoder hidden size = 512 (medium).

**Heads** (`heads.py`): paratope = dropout(0.1)→linear(H→1), masked BCE with a training-set positive-class weight; contact_map = dropout(0.1)→**bilinear** over upper-triangle residue pairs (AA positions only), masked BCE; structure_probe = bias-free linear `B` (rank 128) predicting `‖B hᵢ − B hⱼ‖²`, masked MSE on true squared Cα distances; developability = **mean-pool over non-special tokens** → dropout(0.1)→linear(H→5), MSE on z-scored targets (metrics on original scale).

**Data sources / splits:** paratope = TDC SAbDab_Liberis (antigen contact ≤ 4.5 Å), TDC split; contact_map = SAbDab real X-ray Cα (8 Å Cα–Cα), 60/20/20 **by PDB** (seed 42); structure_probe = **AB-Bind + SAbDab** structures (`max_length 256`, others 160), 60/20/20 **by PDB**; developability = TDC TAP (5 properties, z-scored), TDC split.

**Caveats.** (1) The downstream YAMLs hard-code `checkpoint: models/checkpoints/uniform_medium/final` as a placeholder — `run_all_evaluations.py` **overrides it and probes every experiment's own checkpoint**, so per-strategy results reflect each strategy's model. (2) "Split by complex" holds only for contact_map/structure_probe; paratope/developability inherit TDC splits. (3) Zero-shot evals (MLM, PLL, infilling, mutation) train nothing.

## Aggregation & column naming (`compare.py` / `report.py`)

`build_comparison_table()` discovers each experiment's `all_metrics.json` (+ `downstream_outputs/<task>_<mode>/<exp>/results.json`) and flattens it, applying prefixes:

- MLM + zero-shot keys → **no prefix** (`mlm_accuracy`, `pll_mean`, `infill_cdr3_exact_match`, …)
- mutation keys → **`mut_`**
- attention keys → **`attn_`** (note: the keys already start with `attn_entropy_…`, so columns read **`attn_attn_entropy_…`** — a real double-prefix quirk)
- downstream keys → **`ds_<task>_`** (e.g. `ds_paratope_auprc_mean`)

`report.py` additionally emits `comparison_table.tex` (booktabs, best-value bolded), `summary.md`, and grouped bar charts. `significance.py` provides bootstrap CIs and a paired bootstrap test for comparing two strategies on the same items.

## Full metric catalog (the 73 columns)

**Identity / training (9):** `experiment`, `strategy`, `model_size`, `total_params`, `dataset`, `train_eval_loss`, `train_mlm_accuracy`, `total_train_steps`, `best_step`, `best_train_mlm_accuracy`. *(These describe the run; `total_train_steps`=125000 for trained models.)*

**MLM accuracy — zero-shot, uniform ref mask (8, ↑):** `mlm_accuracy`, `mlm_top5_accuracy`, and `_cdr` / `_cdr3` / `_framework` region variants of each.

**Perplexity (4, ↓):** `perplexity_overall`, `perplexity_cdr`, `perplexity_cdr3`, `perplexity_framework`.

**PLL (2, ↑):** `pll_mean`, `pll_normalized_mean`.

**Infilling — exact recovery (15):** per CDR (`infill_cdr{1,2,3}_accuracy`, `_exact_match` ↑), CDR3 length strata (`infill_cdr3_{short,medium,long}_accuracy` ↑), N-terminus (`nterm_accuracy`, `nterm_exact_match` ↑), scattered (`scattered_accuracy_k{1,5,10}` ↑). *(edit-distance variants exist in the JSON; not all surfaced as columns.)*

**Infilling quality — distributional (3, ↓):** `cdr1_jsd`, `cdr2_jsd`, `cdr3_jsd` (Jensen–Shannon divergence of AA frequencies; lower = more natural).

**Mutation — AB-Bind, `mut_` (6):** `mut_mean_per_complex_spearman_rho` (↑, **headline**), `mut_median_per_complex_spearman_rho` (↑), `mut_mean_per_complex_auroc` (↑), `mut_median_per_complex_auroc` (↑), `mut_n_complexes`, `mut_n_mutants_total` (counts).

**Attention — `attn_attn_` (4):** `attn_attn_entropy_mean`, `attn_attn_entropy_layer{0,5,11}` (diagnostic; no single direction).

**Downstream `ds_` (22):**
- `ds_paratope_` (4, ↑): `auroc_mean`, `auprc_mean`, `f1_mean`, `mcc_mean`
- `ds_contact_map_` (8, ↑): `auroc_mean`, `precision_at_L_mean`, `precision_at_L2_mean`, `precision_at_L5_mean`, `long_range_auroc_mean`, `long_range_precision_at_L_mean`, `long_range_precision_at_L5_mean`, `medium_long_auroc_mean`
- `ds_developability_` (7): `spearman_macro_mean` (↑), `spearman_{CDR_Length,PSH,PPC,PNC,SFvCSP}_mean` (↑), `mse_original_scale_mean` (↓)
- `ds_structure_probe_` (3): `spearman_distance_mean` (↑), `contact_precision_at_L_mean` (↑), `rmse_distance_angstrom_mean` (↓)

> There are **no `ds_binding_*` columns** — the CoV-AbDab binding probe is implemented in code but not part of this table.

## The six "radar" metrics, in depth

These are the axes of `fig_radar_*` ([09-figures](09-figures.md)). All oriented higher = better. The four `ds_*` ones are **frozen-embedding probes** (they measure what the pretrained representation encodes, not a fine-tuned model); the two zero-shot ones use the pretrained MLM directly.

| Radar label | Column | Family |
|---|---|---|
| **Para** | `ds_paratope_auprc_mean` | probe |
| **Cont** | `ds_contact_map_long_range_precision_at_L_mean` | probe |
| **Struc** | `ds_structure_probe_spearman_distance_mean` | probe |
| **Dev** | `ds_developability_spearman_macro_mean` | probe |
| **Mut** | `mut_mean_per_complex_spearman_rho` | zero-shot |
| **CDR3** | `infill_cdr3_exact_match` | zero-shot |

**Para — paratope prediction (AUPRC).** Per-residue binary classification of antigen-contacting residues (TDC **SAbDab_Liberis**, contact ≤ 4.5 Å). Frozen embeddings → linear token head, class-weighted BCE, threshold fit on val; area under the precision–recall curve, mean over 3 seeds. *Covers:* the antigen-binding contact surface.

**Cont — contact map (long-range P@L).** Precision of the top-**L** predicted contacts (L = chain length) among **long-range** pairs (|i−j| ≥ 24); contacts at 8 Å Cα–Cα from real **SAbDab** X-ray coords. Frozen embeddings → bilinear pairwise head; averaged over test complexes and seeds. *Covers:* global tertiary fold (short-range, trivial contacts excluded).

**Struc — structure probe (Spearman, distance).** Hewitt–Manning probe: a learned linear map predicts squared Cα distances; score is the Spearman correlation between predicted and true pairwise distances (**AB-Bind + SAbDab** structures), mean over 3 seeds. *Covers:* how linearly the embedding encodes inter-residue geometry.

**Dev — developability (macro Spearman).** TDC **TAP**, 5 properties (`CDR_Length`, `PSH`, `PPC`, `PNC`, `SFvCSP`). Frozen embeddings → mean-pooled regression head; per-property Spearman, macro-averaged, mean over (5) seeds. *Covers:* manufacturability / biophysical liabilities.

**Mut — mutation effect (per-complex Spearman).** Zero-shot **wildtype-marginal** PLL: mask the mutated position(s) in the wild-type sequence and score the wild-type-vs-mutant log-prob difference; correlate with experimental ΔΔG (**AB-Bind**) within each complex (≥3 mutants), then average across complexes. *Covers:* affinity-maturation / variant-effect prediction. **In practice this is near zero for all trained models (several negative), and the untrained control ranks highest — treat it as a negative result, not a comparison axis (values below).**

**CDR3 — CDR3 infilling (exact match).** Zero-shot: mask the entire CDR3 in held-out **OAS** sequences and argmax-decode in one pass; fraction of sequences reconstructed **exactly** (all positions correct). Heavy-chain only. *Covers:* generative design of the principal binding loop.

## Specialist results table — column mapping

The paper's specialist and hybrid results tables (`tab:results_specialist`, `tab:results_hybrid`) report **seven** metrics — the radar's Para/Cont/Struc/Dev/CDR3 plus paratope MCC and contact AUROC, with the dropped Mut excluded. Header → exact CSV column (all ↑):

| Table header | CSV column | metric |
|---|---|---|
| CDR3 | `infill_cdr3_exact_match` | CDR3 exact-match infilling (held-out OAS) |
| P.~AUPRC | `ds_paratope_auprc_mean` | paratope AUPRC (SAbDab_Liberis, 4.5 Å) |
| P.~MCC | `ds_paratope_mcc_mean` | paratope MCC |
| C.~AUROC | `ds_contact_map_auroc_mean` | contact-map AUROC (SAbDab, 8 Å Cα) |
| C.~P@L | `ds_contact_map_long_range_precision_at_L_mean` | contact-map long-range precision@L |
| Str.~ρ | `ds_structure_probe_spearman_distance_mean` | structure-probe Spearman ρ (AB-Bind + SAbDab) |
| Dev.~ρ | `ds_developability_spearman_macro_mean` | developability macro-Spearman ρ (TDC TAP) |

Gotchas caught when verifying these tables: the structure column is **Spearman ρ**, not precision@L; **SKEMPI is not used anywhere in the repo** (only AB-Bind, `sirin2016abbind`); and the `hybrid_warmstart` row had two transcribed values corrected against the CSV (C.~AUROC 0.978→0.974, C.~P@L 0.673→0.638).

## Results tables (LaTeX)

Both tables report the same seven metrics (Mut dropped; all higher-is-better), best-per-column bolded, values verbatim from `comparison_table.csv`. The specialist table includes the `untrained` control; the hybrid table covers the seven reported hybrids (`hybrid_adaptive` omitted).

**Specialists** — `tab:results_specialist`:
```latex
\begin{table}[h]
  \caption{Specialist strategy results. Metrics are defined in Section~\ref{sec:experiments} and strategies are defined in Section~\ref{sec:specialist}. \stratuntrained{} is a randomly initialized control (no pretraining). Higher is better on all reported metrics.}
  \label{tab:results_specialist}
  \centering
  \footnotesize
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{lccccccc}
    \toprule
    Strategy & CDR3 & P.~AUPRC & P.~MCC & C.~AUROC & C.~P@L & Str.~$\rho$ & Dev.~$\rho$ \\
    \midrule
    \stratuntrained{}  & 0.000 & 0.130 & 0.164 & 0.589 & 0.071 & 0.089 & 0.100 \\
    \midrule
    \stratuniform{}    & 0.052 & 0.832 & 0.628 & 0.970 & 0.606 & 0.590 & 0.336 \\
    \stratcdr{}        & \textbf{0.236} & 0.792 & 0.622 & 0.964 & 0.549 & 0.597 & 0.324 \\
    \stratspan{}       & 0.205 & 0.849 & 0.622 & 0.968 & 0.601 & 0.609 & \textbf{0.385} \\
    \stratstruct{}     & 0.000 & 0.802 & 0.617 & 0.974 & 0.645 & 0.594 & 0.321 \\
    \stratstructLR{}   & 0.019 & 0.813 & 0.614 & 0.980 & 0.693 & 0.619 & 0.268 \\
    \stratint{}        & 0.100 & \textbf{0.872} & 0.664 & \textbf{0.982} & \textbf{0.720} & \textbf{0.627} & 0.340 \\
    \stratgerm{}       & 0.146 & 0.846 & 0.610 & 0.975 & 0.648 & 0.618 & 0.347 \\
    \stratinter{}      & 0.153 & \textbf{0.872} & \textbf{0.684} & 0.974 & 0.636 & 0.616 & 0.305 \\
    \bottomrule
  \end{tabular}}
\end{table}
```

**Hybrids** — `tab:results_hybrid` (values verified; `warmstart` C.~AUROC/C.~P@L corrected, two rounding nits fixed, Dev.~$\rho$ added):
```latex
\begin{table}[h]
  \caption{Hybrid strategy results. Metrics are defined in Section~\ref{sec:experiments} and strategies are defined in Section~\ref{sec:hybrid}. Higher is better on all reported metrics. Each strategy underwent the same training setup with the exception of \strathybws{} which was trained for an additional 50,000 steps.}
  \label{tab:results_hybrid}
  \centering
  \footnotesize
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{lccccccc}
    \toprule
    Strategy & CDR3 & P.~AUPRC & P.~MCC & C.~AUROC & C.~P@L & Str.~$\rho$ & Dev.~$\rho$ \\
    \midrule
    \strathyb{}        & 0.196 & 0.831 & 0.599 & 0.969 & 0.587 & 0.596 & 0.294 \\
    \strathybst{}      & \textbf{0.205} & 0.833 & 0.648 & \textbf{0.980} & \textbf{0.690} & 0.619 & 0.300 \\
    \strathybws{}      & 0.172 & \textbf{0.877} & 0.679 & 0.974 & 0.638 & \textbf{0.621} & \textbf{0.358} \\
    \strathybrv{}      & 0.191 & 0.859 & 0.659 & 0.974 & 0.641 & 0.605 & 0.296 \\
    \strathybinter{}   & 0.195 & 0.845 & 0.642 & 0.966 & 0.572 & 0.590 & 0.331 \\
    \strathybpb{}      & 0.201 & 0.864 & 0.661 & 0.973 & 0.632 & 0.607 & 0.327 \\
    \strathybw{}       & 0.200 & 0.861 & \textbf{0.683} & 0.976 & 0.648 & 0.586 & 0.263 \\
    \bottomrule
  \end{tabular}}
\end{table}
```

## Radar Mut/Dev values (not in the specialist table)

`Mut` and `Dev` exist for every experiment but were excluded from the specialist table. Values (both ↑), from `comparison_table.csv`:

| experiment | Mut (`mut_mean_per_complex_spearman_rho`) | Dev (`ds_developability_spearman_macro_mean`) |
|---|---|---|
| untrained | 0.107 | 0.100 |
| uniform | 0.013 | 0.336 |
| cdr | 0.005 | 0.324 |
| span | 0.020 | 0.385 |
| structure | −0.034 | 0.321 |
| structure-LR | −0.016 | 0.268 |
| interface | 0.010 | 0.340 |
| germline | −0.035 | 0.347 |
| intersection | −0.045 | 0.305 |
| hybrid-curriculum | 0.037 | 0.294 |
| hybrid-stretched | −0.032 | 0.300 |
| hybrid-reverse | −0.008 | 0.296 |
| hybrid-warmstart | 0.020 | 0.358 |
| hybrid-perbatch | −0.030 | 0.327 |
| hybrid-weighted | 0.001 | 0.263 |
| hybrid-adaptive | −0.009 | 0.232 |
| hybrid-intersection | −0.062 | 0.331 |

> **Mut is effectively uninformative at this scale**: every trained model is near zero (several negative) and the untrained control (0.107) outranks all of them, so the zero-shot AB-Bind ΔΔG correlation carries no usable signal here (small medium models; few mutants per complex). This is almost certainly why Mut was left out of the specialist table — report it as a limitation, not a comparison axis. **Dev does carry signal** (trained 0.26–0.39 vs untrained 0.10; `span` 0.385 and `hybrid-warmstart` 0.358 highest).

## Paper section — Experiments

Merged paper Experiments section (setup / Data / Metrics); full linear-probe details in the appendix that follows. The two `\section{}` blocks drop into the manuscript body and appendix respectively.

```latex
\section{Experiments}
\label{sec:experiments}
We study function-aware masking in the pretraining setting, training models parameterized by the RoFormer architecture~\cite{su2024roformer}. Specifically, we instantiate a medium-sized RoFormer model utilizing the AntiBERTa2 tokenizer~\citep{barton2024antiberta2}, AdamW optimizer (peak LR $5\!\times\!10^{-5}$, 5\,\% warmup, cosine decay), batch size 64, fp16, 125{,}000 training steps, and the same fixed seed for initialization. All masking strategies are evaluated using this setup with the exception of \strathybws{}, which trains for an additional 50{,}000 steps at a reduced learning rate ($2\!\times\!10^{-5}$).

\textbf{Data.} We sample 500{,}000 heavy-chain variable-domain (VH) sequences from the Observed Antibody Space~\cite{kovaltsuk2018oas} and filter to the 20 canonical amino acids and lengths in $[80,160]$, leaving 497{,}309 training sequences. CDR1/2/3 boundaries are taken from the OAS IMGT annotations, with ANARCI~\cite{dunbar2014anarci} as a fallback for unannotated sequences. Per-residue paratope probabilities are predicted by a teacher model: AntiBERTa2 fine-tuned with a per-token classification head on the TDC SAbDab\_Liberis paratope set~\cite{liberis2018parapred,dunbar2014sabdab}, producing per-residue labels in $[0,1]$. Per-residue germline-mutated labels are derived by comparing each residue to a per-gene consensus computed from the corpus itself, grouped by the OAS V/J-gene calls. Structures are predicted with IgFold~\cite{ruffolo2023igfold} (experimental X-ray structures exist for $<1\,\%$ of OAS); the predicted $C_\alpha$ geometry is reduced to a per-residue nearest-neighbor graph for the structure-aware strategies. Figure~\ref{fig:2}b details the most commonly sampled mask locations under each specialist strategy.

\textbf{Evaluation.} We assess seven metrics designed to evaluate the diverse biological functions of antibodies, with each metric isolating a distinct functional property captured by learned representations. To evaluate generative recovery of the antigen-binding loop, we measure exact-match CDR3 infilling (CDR3) zero-shot on held-out OAS heavy chains. All remaining metrics are evaluated either with a linear or bilinear probe, where the pretrained model is frozen and a lightweight head is trained on the resulting last-layer embeddings. For functional binding, paratope AUPRC (Para) and MCC evaluate the identification of antigen-contacting residues on TDC SAbDab\_Liberis (4.5\,\AA{} contact threshold). To assess structural functions, contact map AUROC and long-range precision-at-$L$ (Cont; sequence separation $\geq$ 24) measure tertiary fold recovery on SAbDab crystal structures (8\,\AA{} $C_\alpha$) utilizing a bilinear head. Furthermore, the structure probe Spearman $\rho$ (Struct) assesses how linearly the embedding space encodes three-dimensional geometry via the rank correlation of $C_\alpha$ distances on AB-Bind and SAbDab structures~\cite{sirin2016abbind}. Finally, developability macro-Spearman $\rho$ (Dev) captures the recovery of critical biophysical properties for therapeutic applications across the five TDC TAP metrics. Full details regarding probe architectures, optimization, and data splits are provided in Appendix~\ref{ap:experiments}.

```

## Appendix — Linear probing details

```latex
\section{Linear probing details}
\label{ap:experiments}
Six of the seven evaluation metrics in Section~\ref{sec:experiments}---all
except zero-shot CDR3 infilling---are obtained by linear probing on frozen
representations across four probing tasks: paratope, contact map, structure, and
developability. For each task the pretrained encoder is held fixed, its
last-layer token embeddings are extracted once and cached, and only a
task-specific head is trained (the encoder is never fine-tuned). Each model is
probed with its own pretrained checkpoint. Heads are optimized with AdamW (weight
decay $0.01$) at a head learning rate of $10^{-3}$ under a linear-warmup--cosine
schedule (warmup fraction $0.1$), with gradient-norm clipping at $1.0$, in full
precision. For each task we train multiple independent probes (seeds $42, 43,
\dots$; counts in Table~\ref{tab:probe_hparams}). At every epoch the head is
scored on the held-out validation split using that task's early-stopping metric
(Table~\ref{tab:probe_hparams}); the best-validation checkpoint is retained and
restored for the single test-set evaluation, and training halts once that metric
has not improved for the task's patience budget (developability, patience $0$,
runs the full epoch budget). For the binary paratope task a decision threshold is
additionally fit on validation by maximizing Youden's J for the
threshold-dependent metric (MCC); AUPRC and AUROC are threshold-free. The seeds
re-randomize head initialization, dropout, and minibatch order only---the data
split and cached embeddings are identical across seeds---so the reported
across-seed mean $\pm$ sample standard deviation (ddof $=1$) measures
probe-optimization variance on a fixed split, not data-resampling variance.

\paragraph{Heads.} Paratope (Para): a per-token linear classifier ($H\!\to\!1$)
trained with class-weighted binary cross-entropy, scored by AUPRC and (at the
fitted threshold) MCC. Contact map (Cont): a bilinear form over residue-pair
embeddings (upper triangle, amino-acid positions only) trained with masked binary
cross-entropy, scored by AUROC and long-range precision-at-$L$. Structure probe
(Struct): a Hewitt--Manning linear structural probe~\citep{hewitt2019structural}---a
bias-free linear map of rank $128$---that predicts squared $C_\alpha$ distances,
trained with masked MSE and scored by the Spearman $\rho$ between predicted and
true distances. Developability (Dev): mean-pooling over residue tokens followed by
a linear regressor ($H\!\to\!5$) trained with MSE on $z$-scored targets, scored by
the macro-averaged Spearman $\rho$ over the five TAP properties. Here $H = 512$ is
the encoder hidden size.

\paragraph{Splits.} Contact-map (SAbDab) and structure-probe (AB-Bind + SAbDab)
data are split $60/20/20$ by PDB (seed $42$) so that no complex appears in more
than one split; paratope (SAbDab\_Liberis) and developability (TAP) use the
train/validation/test splits provided by TDC.

\begin{table}[h]
  \caption{Linear-probe configurations. Patience $0$ disables early stopping
  (developability trains its full budget).}
  \label{tab:probe_hparams}
  \centering
  \footnotesize
  \begin{tabular}{lccccl}
    \toprule
    Task & Epochs & Batch & Patience & Seeds & Early-stop metric \\
    \midrule
    Paratope        & 50  & 32 & 10 & 3 & AUPRC \\
    Contact map     & 50  & 16 & 10 & 3 & precision@$L$ \\
    Structure probe & 100 & 4  & 15 & 3 & Spearman $\rho$ \\
    Developability  & 100 & 16 & 0  & 5 & macro Spearman $\rho$ \\
    \bottomrule
  \end{tabular}
\end{table}
```

Corrections vs. the original draft (verified against code): `warmstart` = `interface`-init + 50k @ lr 2e-5; paratope labels from AntiBERTa2 fine-tuned on SAbDab_Liberis (soft labels, not Parapred / 0.5-thresholded); germline from a per-gene corpus consensus (not IgBLAST; `ye2013igblast` removed); CDR from OAS IMGT (ANARCI fallback); structure probe on AB-Bind + SAbDab; metrics now seven (developability added, Mut dropped, structure metric = Spearman rho).

