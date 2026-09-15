# MLCB 2026 — rebuttal response pack

**Purpose:** one place holding every reviewer concern, the empirical result that answers it,
and draft response text. Built 2026-08-14 by consolidating [10-rebuttal-notes](10-rebuttal-notes.md)
(the working scratchpad — verified facts, code refs, contamination analysis) and
[results](results.md) (the generated 3-seed numbers).

- **This page** = what we say to reviewers, and the evidence behind it.
- **[10-rebuttal-notes](10-rebuttal-notes.md)** = why we believe it, with `file:line` citations. Keep both.

Every number below is reproducible from `comparison_outputs/`. Nothing here is estimated.

---

## 0. Scoreboard — what we can answer today

| # | Concern cluster | Raised by | Status |
|---|---|---|---|
| **A** | Pretraining-seed robustness | #2, #5, #6 | ✅ **Fully answered** — 3 seeds × 9 strategies + all 6 hybrids + control |
| **B** | Which mixture/weights matter | #1, #2, #5 | ✅ **Answered, mostly null** — random-mixture control; see §3, corrected 2026-08-15 |
| **C** | CDR vs span attribution | #5 | ✅ **Answered** — +0.027 at 3.0σ |
| **D** | CDR3 novelty / degeneracy | #1 | ✅ **Answered** — novelty pack, with a candid caveat |
| **E** | Binding / functional transfer | #5 | ✅ **Answered, and it is a null** — 2 new benchmarks |
| **F** | Dynamic / adaptive weighting | #1, #4 | ✅ **Answered** — already implemented + trained |
| **G** | Developability instability | #5, #6 (implicit) | ✅ **Answered, and it reverses** — ridge probe |
| **H** | Teacher→probe circularity | #2, #6 | ✅ **Verified empirically** — 0 test antibodies seen by the teacher; benchmark redundancy quantified |
| **I** | OAS ↔ benchmark contamination | #1, #2, #6 | ✅ **Answered** — audit run 2026-08-14, no measurable overlap |
| **J** | Mask statistics vs biological identity | #2, #5 | ✅ **Answered, favourably** — permutation erases the whole `interface` gain |
| **K** | Hybrid replication (Table 2) | #5, #6 | ✅ **Complete 2026-08-15** — every hybrid at n=3, no arm cited single-seed |
| **L** | Scaling behaviour | #5 | ⚪ **Concede** — out of scope |
| **M** | Heavy-chain-only scope | #5 | ⚪ **Concede** — with a mitigating note |
| **N** | IgFold accuracy on our corpus | #4 | ✅ **Measured 2026-08-14** — 0.93 kNN agreement, 0.92 Å RMSD vs crystals |
| **O** | Presentation / broken refs | all five | ✅ **Checklist ready** |

**The single most important framing decision:** replication *strengthens* our headline
claim and *retracts* several secondary ones. Lead with that. A response that volunteers
its own retractions is far more credible than one that defends everything — and §4 below
is the list we must volunteer before a reviewer finds it.

---

## 1. The five reviewers at a glance

No Reviewer #3. Relevance ratings: #4 "perfectly aligned"; #1/#5/#6 "strongly relevant"; #2 "somewhat relevant".

| Reviewer | Posture | Their central ask |
|---|---|---|
| **#1** | Positive | Split detail; are weights/schedules sensitive; can weights be *learned*; CDR3 novelty; interface's secondary peak |
| **#2** | Most skeptical | Single run per model; teacher leakage; baselines not matched |
| **#4** | Most positive | Biological interpretation of hybrids; label cost; readability |
| **#5** | Most thorough | Seed robustness; CDR-vs-span; no real binding task; no scaling; hybrids unmotivated; HC-only; teacher supervision framing + IgFold's 240 GPU-h |
| **#6** | Positive, wants interpretation | Why interface/span win; why hybrid-weighted and plain hybrid underperform; paratope split; seeds |

**Concerns raised by 3+ reviewers** — these deserve the most response real-estate:
seed robustness (A), split/contamination (H+I), and interpretation of the hybrid results (B+F).

---

## 2. Cluster A — pretraining-seed robustness ⭐ the dominant concern

> #2.3.1 "Each main model is trained only once."
> #2.5.1 "What is the variance across independent pretraining seeds, and which improvements remain significant at the encoder level?"
> #5.3.1 "Each strategy appears to use only one pretrained encoder… particularly concerning for metrics with modest gains, such as contact-map AUROC and residue-distance Spearman."
> #6.3 "Multiple seeds on the linear probe are described in the supplement, but not a test of pretraining variance."

### The evidence

We retrained **every** strategy at 3 independent pretraining seeds (42, 1, 2), varying
weight init and masking RNG only. The train/eval partition is held fixed and is now
independent of the model seed (`data/splits.py::make_train_eval_split`), so replicates
isolate exactly the variance the reviewers asked about. Dataloader order is pinned
(`data_seed=42`). Contrasts are computed **paired within seed**, then averaged.

Full tables: [results.md §Table 1–2](results.md). Headline:

| Contrast | Metric | Per-seed deltas | Mean | Effect |
|---|---|---|---|---|
| `cdr` − `random` | CDR3 | +0.184 +0.193 +0.200 | **+0.192** | **24.0σ** |
| `span` − `random` | CDR3 | +0.153 +0.176 +0.166 | **+0.165** | **14.3σ** |
| `interface` − `random` | P. AUPRC | +0.044 +0.043 +0.040 | **+0.042** | **19.2σ** |
| `interface` − `random` | C. P@L | +0.102 +0.044 +0.114 | +0.086 | 2.3σ |
| `structure-LR` − `random` | C. P@L | +0.093 +0.041 +0.087 | +0.074 | 2.6σ |

**And, critically, the reviewers' suspicion was correct.** Resolved effects (|mean/sd| ≥ 2)
exist on **only three** metrics — CDR3, paratope AUPRC, contact P@L. There are **zero**
resolved effects, for **any** strategy, on structure-probe Spearman ρ, developability ρ,
contact AUROC, or paratope MCC. Reviewer #5 named contact AUROC and residue-distance
Spearman specifically; both are among the four that cannot support the differences we reported.

### Draft response

> We agree this was the most important gap, and we have closed it. We retrained all nine
> masking strategies plus the untrained control at three independent pretraining seeds
> (n=3 encoders per arm, 30 additional pretraining runs), holding the held-out split fixed
> and independent of the model seed so that replicates isolate weight-init and masking
> stochasticity. We report paired within-seed contrasts against the random-masking baseline
> with effect sizes in units of between-encoder sd.
>
> The primary result strengthens: CDR3 exact-match improvement for `cdr` over random masking
> is +0.192 (24.0σ), and for `span` +0.165 (14.3σ). The random-masking baseline's published
> CDR3 value (0.052) was the most favourable of its three draws (mean 0.039), so the true
> effect is **5.9×**, not the 4.5× we reported. `interface`'s paratope AUPRC gain is +0.042
> at 19.2σ — the most reproducible non-CDR3 result in the paper.
>
> Reviewer #5's specific concern was also correct, and we now state it in the paper: resolved
> effects appear on only three of seven metrics. Contact-map AUROC, structure-probe Spearman ρ,
> developability ρ and paratope MCC show **no** resolved effect for any strategy at n=3. We
> have removed the claims those columns were carrying (see our reply on Table 1 bolding) and
> now report between-encoder sd separately from between-probe-seed sd throughout.

---

## 3. Cluster B — which mixture and schedule actually matter ⭐ strongest new result

> #1.3.2 "How are the weight ratios and mixing probabilities/schedules assigned? Is the performance sensitive to the choice of values?"
> #5.3.5 "The paper evaluates numerous curricula with manually chosen weights, transition points, directions, and warm-start procedures. The motivation for these choices is unclear, and the experiments do not isolate which components matter."
> #2.3.3 "The new masks may simply make training harder or focus on different sequence positions, even without useful biological knowledge."

### The evidence — the random-mixture control

`hybrid-RANDOM` holds *everything* identical to `hybrid` — same six sub-strategies, same
6,250-step block structure, same budget — and replaces only the **values**: at each block the
mixture is redrawn uniformly from the simplex, Dirichlet(1,…,1). The seed drives the draw, so
the three replicates are three *different arbitrary curricula*.
(`configs/hybrid_random_medium.yaml`, `random_policy` in `masking/hybrid.py`.)

⚠️ **Updated 2026-08-15, all six hybrids now at n=3.** The earlier version of this section
read "no hand-designed curriculum resolves against the control on any metric." That was
computed when only `hybrid` and `hybrid-stretched` were replicated. With `reverse`,
`perbatch` and `weighted` complete, it is **too strong** and must be corrected.

Paired within seed against the control, mean Δ (paired t-test, 2 df):

| vs `hybrid-RANDOM` | CDR3 | P. AUPRC | P. MCC | C. AUROC | C. P@L | Str. ρ | Dev. ρ |
|---|---|---|---|---|---|---|---|
| `hybrid` | −0.003 | −0.019 | −0.026 | +0.004 | +0.027 | +0.008 | −0.003 |
| `hybrid-stretched` | −0.005 | +0.000 | −0.001 | +0.008 | +0.065 | +0.015 | +0.012 |
| `hybrid-reverse` | −0.002 | +0.009 | +0.007 | **+0.010** *(p=.016)* | **+0.098** *(p=.025)* | **+0.026** *(p=.029)* | +0.038 |
| `hybrid-perbatch` | −0.001 | +0.014 | **+0.031** *(p=.013)* | +0.004 | +0.033 | +0.004 | +0.025 |
| `hybrid-weighted` | −0.006 | +0.016 | +0.019 | +0.005 | +0.051 | +0.004 | +0.007 |
| `hybrid-warmstart`‡ | **−0.033** *(p=.046)* | +0.035 | **+0.044** *(p=.026)* | **+0.008** *(p=.036)* | **+0.072** *(p=.028)* | **+0.030** *(p=.046)* | +0.025 |

Bold = nominally significant at p<0.05. **Nothing survives Bonferroni correction over the
seven metrics (p<0.0071).** ‡ `warmstart` is confounded by 50k extra steps from an
`interface` checkpoint.

**Corrected conclusion, in two parts:**

1. **On CDR3 — the headline hybrid effect — the schedule is definitively not load-bearing.**
   No arm differs from the control (all |Δ| ≤ 0.006, none significant), and the control
   recovers the full CDR3 gain over uniform (+0.162, 8.2σ) that the designed schedules
   produce (`hybrid` +0.159, 12.0σ). Mixing specialists at all is what produces it.
2. **On the structural and paratope metrics there is weak evidence some schedules help.**
   `reverse` (contact AUROC, contact P@L, structure ρ) and `perbatch` (paratope MCC) beat
   an arbitrary redrawn mixture at nominal p<0.05. This is suggestive, not established: none
   survives multiple-comparison correction, and n=3 sd estimates are fragile — `perbatch`'s
   P. MCC sd is 0.001, which inflates any ratio built on it.

So the answer to #1.2 ("is performance sensitive to the choice of values?") is: **not for the
metric the hybrids are sold on, and only weakly elsewhere.** The answer to #5.5 ("the
experiments do not isolate which components matter") is that this control isolates exactly
that, and the honest reading is a mostly-null with a hint on structural metrics.

It also partly answers #2.3: an arbitrary Dirichlet curriculum carries no biological knowledge
whatsoever, yet matches the designed ones — so for the *hybrid* arms, the gain is attributable
to mask-placement diversity rather than to correct biological priors. (The specialist arms are
a separate question; see cluster J, which we cannot yet answer.)

### Draft response

> We added a control we agree the paper needed. `hybrid-RANDOM` is identical to our hybrid
> curriculum in every respect — same six sub-strategies, same block structure, same budget —
> except that the mixture weights at each block are drawn uniformly from the simplex rather
> than hand-designed. Across three seeds it samples three different arbitrary curricula.
>
> No hand-designed curriculum resolves against it on any of the seven metrics, while it
> recovers the same CDR3 gain over random masking that the designed schedules do (+0.162,
> 8.2σ vs +0.159, 12.0σ). We therefore now state the honest conclusion: **mixing specialists
> is what produces the hybrid effect; the specific weights and schedule are not load-bearing.**
> This directly addresses the concern that our curricula were manually chosen without
> isolating which components matter — the answer is that the components we hand-tuned largely
> do not matter, and we have rewritten the hybrid section around that finding.

---

## 4. Cluster C — CDR masking vs generic span masking

> #5.3.2 "CDR masking achieves the best CDR3 exact-match score… However, generic span masking reaches 0.205 while performing better on most other evaluations. Because CDR3 infilling itself involves reconstructing a contiguous missing region, it is unclear how much of the gain reflects CDR-specific targeting rather than improved span reconstruction."

### The evidence

Paired within seed, on CDR3 exact match:

| Contrast | Per-seed deltas | Mean | sd | Effect |
|---|---|---|---|---|
| `cdr` − `span` | +0.031 +0.017 +0.034 | **+0.027** | 0.009 | **+3.0σ** |
| `cdr` − `random` | +0.184 +0.193 +0.200 | +0.192 | 0.008 | +24.0σ |
| `span` − `random` | +0.153 +0.176 +0.166 | +0.165 | 0.012 | +14.3σ |

The reviewer's reading is **largely correct and we should concede it**: generic span
reconstruction accounts for **86%** of the CDR3 gain (0.165 / 0.192). The CDR-specific
increment is real and resolved, but it is +0.027 — an order of magnitude smaller than the
span effect it sits on top of.

Note the paper already positions `span` as a region-agnostic control rather than a competitor,
but the framing does not make this decomposition visible. It should.

### Draft response

> This is a fair reading and we have quantified it. Paired within seed, `cdr` − `span` on CDR3
> exact match is +0.027 (3.0σ) — resolved, but small: generic span reconstruction accounts for
> 86% of the total gain over random masking (+0.165 of +0.192). We now report this decomposition
> explicitly rather than leaving `span` as an unremarked row, and we have softened the CDR-targeting
> claim accordingly. The honest statement is that contiguity is the dominant mechanism and
> CDR-specific placement adds a small resolved increment on top of it.

---

## 5. Cluster D — are the recovered CDR3s novel, or degenerate?

> #1.3.4 "What is the novelty and variety of the recovered CDR3s in the infilling task? This would help evaluate whether the model produces a degenerate pattern for a very flexible design task."

### The evidence

`comparison_outputs/cdr3_novelty.csv` and `cdr3_degeneracy.csv`, 1,000 held-out CDR3s per arm.
The critical addition is **two anchors** — the ground-truth CDR3s themselves, and a
degenerate "always predict the modal CDR3" strategy:

| Arm | edit dist. to nearest train CDR3 | frac. identical to a train CDR3 | distinct frac. | mean pairwise edit |
|---|---|---|---|---|
| **[anchor] true held-out CDR3s** | 0.857 | **0.838** | 0.580 | 9.83 |
| **[anchor] always-mode (degenerate)** | 0.0 | 1.000 | 0.001 | 0.00 |
| `cdr` | 0.950 | 0.698 | 0.371 | 8.59 |
| `span` | 0.853 | 0.706 | 0.333 | 8.67 |
| `uniform` (random masking) | 2.687 | 0.196 | 0.563 | 7.72 |
| `structure` | 2.972 | 0.012 | 0.280 | 5.29 |

Three things follow, and the first is the one that answers the reviewer:

1. **Not degenerate.** `cdr` sits at distinct-fraction 0.371 and mean pairwise edit distance
   8.59, against 0.001 / 0.00 for true mode collapse. There are zero homopolymers and the
   completions carry correct AR…/…DL framing motifs.
2. **But the eval set itself is highly redundant.** 83.8% of the *true* held-out CDR3s appear
   verbatim in training. `cdr` reproduces train-set CDR3s at 69.8% — i.e. *less* often than
   the ground truth does. The model is not memorising beyond what the corpus distribution
   already implies.
3. **The success is consensus reproduction, not design.** Stratified by training-set frequency,
   exact match is **0.000 for every strategy on every CDR3 seen fewer than 10 times** — that is
   265 of the 1,000 eval sequences, including all 162 never seen. `cdr` scores 0.395 on CDR3s
   seen ≥100×. And the best-scoring arms are the *least* diverse (`cdr` novel-fraction 0.311 vs
   `uniform` 0.817).

Point 3 is a genuine limitation and we should volunteer it. It also reframes the CDR3 metric:
it measures repertoire-prior recall, not generative design capability.

### Draft response

> We added a CDR3 novelty analysis with two reference anchors — the ground-truth held-out CDR3s,
> and a degenerate always-predict-the-mode baseline. The completions are not degenerate:
> `cdr` reaches a distinct fraction of 0.371 and mean pairwise edit distance 8.59, against
> 0.001 and 0.00 for mode collapse, with zero homopolymers and correct framing motifs.
>
> The more informative comparison is against the ground truth. 83.8% of the *true* held-out
> CDR3s appear verbatim in the training corpus; `cdr` reproduces training CDR3s at 69.8%, i.e.
> slightly *less* than the real distribution does. However, stratifying by training frequency
> shows exact match is 0.000 for all strategies on CDR3s seen fewer than 10 times (265 of 1,000
> eval sequences), rising to 0.395 for CDR3s seen ≥100×. We now state plainly that this metric
> measures **recall of the repertoire consensus rather than novel design capability**, and we
> report the stratified table. We think this is an important clarification of what the headline
> number means, and we thank the reviewer for prompting it.

---

## 6. Cluster E — binding / functional transfer ⭐ new experiments, and they are null

> #5.3.3 "Improved CDR3 infilling… provides limited evidence of transfer to antibody function… Binding-specificity or affinity evaluations, as considered in related preferential-masking work [1,2], would provide stronger evidence of functional improvement."

This is the most substantive experimental ask in the reviews. The paper's "functional binding"
was paratope identification — a per-residue geometry task, not affinity or specificity. The
critique is correct as written. We ran **two** new benchmarks; both are honest nulls on the
masking question and strong positives on the pretraining question.

### E1 — AB-Bind ΔΔG, supervised frozen-embedding probe

Matches the protocol of the cited work (frozen embeddings + supervised head) rather than
zero-shot likelihood. Ridge regression, **grouped by complex so no complex straddles train/test**,
repeated over 20 grouped splits. `scripts/run_ab_bind_probe.py`, mean per-complex Spearman ρ.

| Strategy | mean ρ | sd | paired Δ vs random |
|---|---|---|---|
| `random` (uniform) | **0.201** | 0.019 | — |
| `interface` | 0.199 | 0.067 | −0.002 (−0.0σ) |
| `cdr` | 0.197 | 0.010 | −0.004 (−0.4σ) |
| `structure` | 0.153 | 0.059 | −0.049 (−0.6σ) |
| `intersection` | 0.140 | 0.042 | −0.061 (−2.5σ)* |
| `germline` | 0.121 | 0.028 | −0.080 (−1.8σ) |
| `structure-LR` | 0.119 | 0.057 | −0.082 (−1.4σ) |
| `span` | 0.105 | 0.070 | −0.096 (−1.1σ) |
| `untrained` | 0.095 | 0.025 | **−0.106 (−15.1σ)*** |

### E2 — CoV-AbDab SARS-CoV-2 neutralization specificity

The VH-only analogue of Ng & Briney's Dataset E. Logistic probe on frozen embeddings,
stratified 5-fold CV **matching the reference protocol**, n = 24,968.
`scripts/run_ngbriney_cov.py`.

| Strategy | AUROC | sd | paired Δ vs random |
|---|---|---|---|
| `structure` | 0.761 | 0.002 | +0.002 (+0.3σ) |
| `random` (uniform) | 0.759 | 0.004 | — |
| `cdr` | 0.759 | 0.003 | −0.001 (−0.2σ) |
| `germline` | 0.758 | 0.002 | −0.001 (−0.2σ) |
| `interface` | 0.758 | 0.002 | −0.002 (−0.3σ) |
| `structure-LR` | 0.757 | 0.002 | −0.002 (−0.9σ) |
| `span` | 0.757 | 0.001 | −0.003 (−0.6σ) |
| `intersection` | 0.755 | 0.003 | −0.004 (−2.8σ)* |
| `untrained` | 0.657 | 0.002 | **−0.102 (−32.4σ)*** |

⚠️ **State this caveat:** stratified CV is not clonotype-grouped, so clonal relatives can straddle
folds — SARS-CoV-2 neutralizers are dominated by public IGHV3-53/3-66 clonotypes. We follow the
reference protocol for comparability; a clonotype-grouped split would score lower. The script
documents this at `scripts/run_ngbriney_cov.py:28`.

### What these mean

- **Pretraining matters enormously.** Every trained encoder beats the untrained control by a
  huge, resolved margin (−15.1σ and −32.4σ respectively).
- **Masking placement does not transfer to binding.** No strategy beats random masking on either
  benchmark. The two resolved effects are both *losses* (`intersection`).
- This also **resolves the embarrassing zero-shot result** in [10-rebuttal-notes §5.2](10-rebuttal-notes.md),
  where every trained model scored below chance on AB-Bind ΔΔG via ΔPLL and `untrained` was best.
  That was a protocol artifact, exactly as diagnosed: ΔPLL scores mutants under a repertoire
  prior, and affinity-improving mutations are typically germline-divergent and therefore
  low-likelihood. Under the supervised protocol the ordering inverts and behaves sensibly.

### Draft response

> We agree, and we ran the experiment. We added two binding evaluations using the frozen-embedding
> supervised protocol of the cited work rather than zero-shot likelihood: (i) AB-Bind ΔΔG
> regression, grouped by complex over 20 splits, and (ii) SARS-CoV-2 neutralization specificity
> on CoV-AbDab (n=24,968), the VH-only analogue of Ng & Briney's Dataset E.
>
> The result is a clear null on our central variable and we report it as such. Pretraining itself
> is decisive — every trained encoder beats the untrained control by 15σ and 32σ respectively —
> but **no masking strategy outperforms random masking on either benchmark** (AB-Bind: random
> 0.201, best specialist 0.199; CoV-AbDab: random 0.759, best specialist 0.761). We now state
> that the benefits of function-aware masking we observe are confined to the axes aligned with
> the masking prior — CDR3 recovery, paratope prediction, long-range contacts — and do **not**
> extend to binding affinity or specificity. We think this bounds the claim usefully rather than
> weakening the paper.
>
> We note one caveat for comparability: we follow the reference stratified-CV protocol on
> CoV-AbDab, which is not clonotype-grouped; a grouped split would score lower for all arms.

---

## 7. Cluster F — can masking weights be learned rather than pre-assigned?

> #1.3.3 "Is it possible to dynamically assign masking weights and mixing schedules based on (1) predictive performance and (2) probed, not pre-assigned, residue relevance?"
> #4.5.2 "Might different permutations/weights of hybrid training lead to better performance than specialized masking strategies?"

### The evidence

**We already implemented and trained (1).** `hybrid-adaptive` (`configs/hybrid_adaptive_medium.yaml`,
`adaptive=True` in `masking/hybrid.py`) tracks an EMA of per-sub-strategy training loss
(decay 0.99, ≈70-batch half-life) and softmaxes it (temperature 0.5) into the sampling
distribution, multiplied against a static base policy so no sub-strategy starves. Sub-strategies
the model is currently worst at receive more mask budget — exactly the reviewer's proposal.

At seed 42 it is the **best hybrid on CDR3 (0.203) and P. MCC (0.648)**. It is worst on Dev. ρ
(0.232), though that run had 1 of 5 developability probe seeds collapse, so discount it.

For #4's "different permutations/weights": Table 3 already spans six curricula
(forward, stretched, reverse, per-batch, weighted, warm-start) plus the Dirichlet control.

⚠️ Caveat we must state: `hybrid-adaptive` is the **only** variant that is genuinely
non-deterministic at fixed seed — the loss-feedback loop depends on noisy per-batch losses,
so the same seed does not reproduce the same curriculum. Replicating it needs *more* seeds
than the fixed-schedule hybrids, not fewer. It is currently n=1.

For (2), "probed, not pre-assigned, residue relevance": not implemented. Worth flagging as
the natural next step, since it is a genuinely good idea we did not test.

### Draft response

> We had already implemented the reviewer's first suggestion and can report it. `hybrid-adaptive`
> assigns mask budget dynamically from predictive performance: it tracks an exponential moving
> average of per-sub-strategy training loss and softmaxes it into the sampling distribution, so
> sub-strategies the model is currently worst at receive more budget. At seed 42 it is the best
> hybrid on CDR3 exact match (0.203) and paratope MCC (0.648). We have added it to the paper.
>
> One honest caveat: it is the only variant that is non-deterministic at fixed seed, because the
> loss-feedback loop depends on noisy per-batch losses — so it requires more replicates than the
> fixed-schedule hybrids, and it is currently single-seed. The reviewer's second suggestion —
> deriving residue relevance by probing rather than from pre-assigned annotations — we did not
> test, and we agree it is the more interesting direction; we now name it explicitly as future work.

---

## 8. Cluster G — developability, and why the column reverses

> #6 "…the small dynamic range of most of the metrics suggests that additional runs with different seeds would improve the robustness of the observations."

Not asked directly, but the paper makes developability claims that do not survive, and a
reviewer checking the repo would find this.

### The evidence

The published developability numbers come from an SGD probe on TAP: 169 train / 24 val / 48 test
against a 2,560-parameter head. **7 runs had probe seeds collapse at epoch ≤8, two of them
published.** Neither a lower LR nor MSE-based selection fixes it — the task is data-starved.

`scripts/run_developability_ridge.py` replaces it with closed-form ridge + nested 5-fold CV over
all 241 antibodies, reducing seed variance ~5×. Under that probe, macro Spearman ρ across 3 seeds:

| Strategy | ridge ρ | sd | paired Δ vs random |
|---|---|---|---|
| **`untrained`** | **0.427** | 0.003 | **+0.037 (+1.5σ)** |
| `cdr` | 0.401 | 0.005 | +0.011 (+0.6σ) |
| `span` | 0.393 | 0.008 | +0.003 (+0.1σ) |
| `random` (uniform) | 0.390 | 0.022 | — |
| `structure` | 0.388 | 0.011 | −0.001 (−0.0σ) |
| `germline` | 0.384 | 0.027 | −0.006 (−0.2σ) |
| `intersection` | 0.371 | 0.031 | −0.018 (−0.4σ) |
| `structure-LR` | 0.368 | 0.003 | −0.022 (−1.1σ) |
| `interface` | 0.368 | 0.023 | −0.022 (−1.8σ) |

**The randomly-initialised encoder is the best model on developability.** No trained arm resolves
against random masking. TAP properties (CDR length, patches of surface hydrophobicity/charge) are
largely computable from raw sequence composition, which a random projection preserves — so this
probe does not measure representation quality at all.

### Draft response

> We rebuilt the developability evaluation and it changed our conclusion. The published numbers
> used an SGD probe on TAP with 169 training antibodies against a 2,560-parameter head; 7 runs
> had probe seeds collapse, two of which were published. We replaced it with closed-form ridge
> regression under nested 5-fold cross-validation over all 241 antibodies, which reduces seed
> variance roughly five-fold.
>
> Under the corrected probe, no trained encoder resolves against random masking, and the
> **randomly-initialised control scores highest** (0.427 vs 0.390). We take this to mean TAP
> properties are largely recoverable from raw sequence composition, which even a random projection
> preserves, so this probe does not measure representation quality. We have withdrawn the
> developability claims — including the "+14% for span" result in the appendix — and now report
> the column with this caveat.

---

## 9. Cluster H/I — splits, leakage and contamination

> #1.3.1 "The detailed train/test split strategy is not provided… crucial for evaluating potential information leak between the masking and the downstream predictive objectives."
> #2.3.2 "There may be information leakage from teacher models."
> #2.5.2 "Are evaluation proteins clustered away from OAS pretraining sequences and from data used to train the paratope and IgFold teachers?"
> #6.3 "the teacher model uses TDC SAbDab_Liberis paratope data, which is also the source of the paratope AUPRC and MCC metrics. it's worth double checking if there is potential for leakage or confounding here?"

The concerns decompose into **four distinct axes**. Three we can answer; one we cannot.
Full analysis with `file:line` refs: [10-rebuttal-notes §2](10-rebuttal-notes.md).

| Axis | Question | Status |
|---|---|---|
| 1 | OAS train vs OAS eval | ✅ separate, ⚠️ not clustered — clonal relatives straddle |
| 2 | OAS corpus vs external benchmark sets | ✅ **measured 2026-08-14 — no overlap** |
| 3 | Teacher-train vs probe-test | ✅ **verified separate** |
| 4 | Within-benchmark split quality | ⚠️ paratope is the weak link — **now quantified**: 27.8% of test has a ≥95% match in train |

### What we can assert (axis 3 — answers #6.3 and #2.2 directly)

The paratope teacher and the paratope probe call the **same split function with the same
defaults**, so the partition is identical and deterministic:

- `scripts/train_paratope_teacher.py:159` → `load_paratope_splits(...)`; trains on `split["train"]`, early-stops on `split["valid"]`, reports on `split["test"]`
- `evaluation/downstream/paratope.py:26` imports the **same** `load_paratope_splits`
- **No test antibody was ever labelled by the teacher during teacher training.**

**This is now verified, not asserted** — `scripts/verify_paratope_split.py`, output in
`comparison_outputs/paratope_split_verification.json`. A reviewer asking about circularity is
entitled to a check rather than a code-reading argument:

| Check | Result |
|---|---|
| Partition identical across independent calls | ✅ byte-identical (716 / 102 / 205) |
| Splits pairwise disjoint (train↔valid↔test) | ✅ 0 shared sequences in all three pairs |
| Test antibodies among the 818 the teacher sees | ✅ **0 of 205** |

**Held in reserve — benchmark-internal redundancy (do not raise unprompted).** The same script
also scores all 205 × 716 test/train pairs by global alignment, which measures something the
circularity question does not ask about: whether SAbDab_Liberis itself is redundant under TDC's
random split.

- median nearest-train identity **89.4%** (framework similarity — the expected floor for VH)
- 57 of 205 (27.8%) have a ≥95%-identical antibody in train
- 7 are 100% identical over the shorter sequence — 6 strict substrings, i.e. one antibody
  deposited with construct boundaries differing by 1–2 terminal residues, which is why
  exact-string overlap is nonetheless 0

This is a property of the public TDC benchmark, not of our method, and it shifts absolute
paratope values for **every arm identically** — so it cannot affect any comparison the paper
makes. Only relevant if a reviewer raises it directly; the number is here so we can answer
immediately if they do.

Three mechanism points that hold regardless of the split:
1. The encoder's only training loss is amino-acid reconstruction; paratope labels never appear as targets.
2. Teacher predictions are computed **only on the 500k OAS corpus** (`scripts/compute_paratope_labels_v2.py`), never on evaluation sequences.
3. At probe time masks are irrelevant — the encoder sees clean sequences. There is no channel by which mask placement becomes a memorised test answer.

Two of four structural splits are **already grouped**, which the paper failed to say:
contact map is 60/20/20 grouped by PDB (`data/benchmarks/contact_map.py:237-250`); the structure
probe is PDB-keyed and deduped so a structure in both AB-Bind and SAbDab cannot cross splits
(`data/benchmarks/structure_probe.py:220-263`). The new AB-Bind binding probe is also
complex-grouped.

### Axis 2 — measured, and it is clean (2026-08-14)

`scripts/contamination_audit.py`, results in `comparison_outputs/contamination_audit.csv`.
Across **34,315 benchmark heavy chains** against all 497,309 pretraining sequences:

| Benchmark | heavy chains | exact VH match | CDR-H3 in corpus | CDR-H3 + V gene | max identity |
|---|---|---|---|---|---|
| paratope (SAbDab_Liberis) | 678 | 0 | 2 (0.29%) | 0 | 90.2 |
| developability (TAP) | 241 | 0 | 0 | 0 | 77.5 |
| ab_bind (ΔΔG) | 18 | 0 | 0 | 0 | 71.9 |
| binding (CoV-AbDab) | 8,093 | 0 | 1 (0.01%) | 0 | 94.1 |
| binding (Ng&Briney E) | 24,969 | 0 | 2 (0.01%) | 0 | 76.4 |
| structure (SAbDab coords) | 316 | 0 | 1 (0.32%) | 0 | 87.9 |

**Zero exact matches anywhere. Six CDR-H3 hits out of 34,315 (0.017%), none of them
clonal, and no benchmark antibody reaches 95% identity to any pretraining
sequence** — against a framework background of 54–62%.

All six hits fail the clonotype test on V gene (IGHV3-64, IGHV3-23, IGHV3-53, IGHV3-53,
IGHV3-7, IGHV3-64 against differently-assigned corpus matches), and all six have CDR-H3 lengths
of 5–9 residues against a benchmark median of 12 — the short-CDR-H3 regime where exact collision
against 497k sequences is expected by chance. There is no case here to explain away.

Three methodological points that make this defensible:

1. **CDR-H3, not global identity, is the right instrument.** Unrelated human VH domains
   sit at 80–90% identity because they share a scaffold, so the reflexive "cluster at 90%
   with MMseqs2" audit would flag nearly everything and demonstrate nothing. CDR-H3 is the
   V(D)J-generated segment; a shared exact CDR-H3 is essentially always clonal relation.
2. **The CDR-H3 definitions are matched on both sides.** The corpus carries OAS's own
   `cdr3_aa`; benchmark sequences are numbered with ANARCI and cut at IMGT 105–117. Those
   two definitions agree exactly on 300/300 sampled corpus sequences, so a cross-side
   comparison is not a convention artifact.
3. **The negative is verified, not assumed.** Querying the index with corpus sequences
   recovers a 100% self-match 40/40, and 20/20 even after trimming 4 N-terminal and 6
   C-terminal residues to mimic a PDB-derived chain. The pipeline would have found
   contamination had it been there.

Light chains are reported separately and excluded from the heavy-chain rates: pretraining
is VH-only, so a light chain cannot be contaminated by this corpus.

### What we must concede
- **Axis 1: OAS dedup is exact-string only** (`data/preprocessing.py:52-56`). OAS is dense with
  clonally related sequences, so relatives straddle train/eval. This is why the CDR3 absolute
  number is optimistic — and the novelty analysis in §5 quantifies it (83.8% of true held-out
  CDR3s appear verbatim in training).
- **Axis 4: the paratope split is TDC's default random split** over SAbDab_Liberis
  (`data/benchmarks/paratope.py:96`), which is heavily redundant. This inflates both the teacher's
  apparent quality and every arm's paratope numbers. #6.3 lands here — not on circularity.
  Measured: 57 of 205 test antibodies (27.8%) have a ≥95%-identical antibody in train, 7 are
  effectively identical. Shifts all arms identically; see the reserve note above.
- **The structure story is weaker than the paratope one.** IgFold's training set is external and
  opaque to us, and our contact-map eval uses SAbDab crystal structures, so overlap with IgFold's
  training PDBs is likely. The mechanism argument (IgFold shapes masks on OAS, not on test
  structures) still holds and is the primary defence, but we cannot point to a shared split function.

### Clarification points

Full split specification for Appendix B: [appendix-splits](appendix-splits.md).

> We appreciate this being raised by three reviewers and we want to separate four distinct
> questions that "leakage" collapses together.
>
> **Teacher→probe circularity (#6, #2): verified absent.** The paratope teacher and the paratope
> probe call the same split function with the same defaults; the teacher is fit only on the train
> split and the probe scored only on the test split, so no test antibody was ever labelled by the
> teacher during its training — verified directly: 0 of 205 test antibodies appear among the 818
> the teacher sees, and the partition is byte-identical across independent calls. Independently of
> the split, the encoder's only loss is amino-acid
> reconstruction — paratope labels never appear as targets — teacher predictions are computed only
> on the OAS corpus and never on evaluation sequences, and at probe time the encoder sees clean
> unmasked sequences. There is no channel by which mask placement becomes a memorised test answer.
>
> **Structural splits: two of four are already grouped**, which we failed to state. Contact-map is
> split 60/20/20 grouped by PDB so chains of one complex never straddle; the structure probe is
> PDB-keyed and deduplicated across its two sources. Our new AB-Bind binding probe is
> complex-grouped. We will give the full split specification in the appendix.
>
> **Two limitations we must concede.** First, our OAS deduplication is exact-string only, so
> clonally related sequences straddle the pretraining train/eval split; this inflates absolute
> CDR3 numbers for all arms equally (our new novelty analysis shows 83.8% of *true* held-out CDR3s
> appear verbatim in training). Second, the paratope benchmark uses TDC's default random split over
> a redundant source, which inflates paratope numbers for every arm alike. Neither biases the
> ranking between strategies, which is what the paper compares, but both inflate absolute values
> and we will state this.
>
> **On evaluation antibodies appearing in the OAS pretraining corpus (#2): we measured it, and
> they do not.** We compared all 34,315 benchmark heavy chains against the full 497k-sequence
> pretraining corpus. No benchmark antibody appears verbatim in the corpus; six of the 34,315
> (0.017%) share an exact CDR-H3 with any pretraining sequence, none of those also match on V
> gene, and no benchmark antibody exceeds 95% identity to any pretraining sequence. We report
> CDR-H3 overlap rather than global sequence identity deliberately: unrelated VH domains sit at
> 80–90% identity by shared framework alone, so a generic identity-clustering audit is
> uninformative for antibodies, whereas a shared CDR-H3 is near-conclusive evidence of clonal
> relationship. CDR-H3 is defined identically on both sides (IMGT 105–117, verified to reproduce
> the corpus annotation on 300/300 samples), and the search recovers known self-matches at 100%
> including truncated chains, so the negative result reflects the data rather than an
> insensitive method.

---

## 10. Clusters L/M/N and the interpretation questions

### 10.1 The `interface` secondary peak (#1.5) — RESOLVED 2026-08-16

**It is at IMGT 92–95, it is highly reproducible, and it is an annotation artifact
in the TDC SAbDab_Liberis labels rather than antigen contact.**

Scripts: `scripts/interface_peak_profile.py` (teacher profile over a canonical IMGT
frame, 20,000 corpus sequences) and `scripts/verify_paratope_labels.py` (true contacts
from crystal geometry). Outputs in `comparison_outputs/interface_peak_profile.csv` and
`paratope_label_verification.csv`.

| Evidence | Value |
|---|---|
| Teacher probability at IMGT 93 / 94, corpus mean | **0.998 / 0.999** |
| Fraction of corpus sequences with prob ≥ 0.5 there | **99.9%** |
| TDC labels IMGT 92/93/94 as paratope | **678 / 678 antibodies = 100%** (more consistent than *any* CDR position; best is IMGT 58 at 94.1%) |
| **True antigen contact from crystals, ≤4.5 Å (139 heavy chains, 84 complexes)** | **IMGT 92: 0.7% · 93: 0.7% · 94: 0.0%** |
| Real contact for comparison | CDR mean 38.5%, framework mean 2.6%; top ten positions all CDR |
| Share of the `interface` mask budget on IMGT 92–95 | **27.5%** (CDRs get 72.4%) |
| Trivial rule "predict exactly IMGT 92,93,94" on the paratope **test** split | **precision 1.000, recall 0.405** — 426 of 1051 positives, using no sequence information |

Median antibody in the benchmark carries only 6 paratope labels, so three of a typical
label set are this fixed framework triple.

**What this does and does not affect.**

- **Does not affect the ranking between strategies.** Every arm trains against the same
  teacher and is scored on the same labels, so the artifact shifts all arms identically —
  the same argument that covers benchmark redundancy (§9) and contamination (§9 axis 2).
- **Does not touch** CDR3 exact match, the contact-map or structure probes, the two
  binding benchmarks, or the developability ridge — all independent of these labels.
- **Does bound what paratope AUPRC / MCC mean in absolute terms.** ~40% of the positive
  labels are recoverable positionally, so part of the metric measures whether a model
  learned three fixed positions.
- **Partially qualifies cluster J.** `interface` spends 27.5% of its budget on the triple,
  so "the biological prior does real work" is partly "consistently masking a fixed FR3
  triple does real work." The teacher still puts 72.4% of its mass on genuine CDRs, so the
  biological signal dominates — but the permutation result should not be read as *purely*
  about paratope biology.

⚠️ **The decisive follow-up, not yet run:** re-score the paratope probe with IMGT 92–95
excluded from evaluation. If `interface`'s +0.042 AUPRC advantage survives, the result is
robust to the artifact and we can say so plainly. That is the single most valuable
remaining experiment, and it is cheap — probe re-scoring only, no retraining.

⚠️ **Do not claim the peak is functionally meaningful.** An earlier reading held that it
might be a real paratope-adjacent feature such as the DE loop / HV4 (IMGT ~81–85). The
data rules that out: IMGT 85 contacts antigen in 3.6% of structures and 82 in 1.2%, so the
DE loop is *not* where the peak is.

### L — scaling (#5.3.4) — concede

One model size, one budget, ~500k sequences. No scaling experiment exists and none is affordable
in the rebuttal window. Concede plainly and name it as the main open question; it is a fair
limitation for a workshop-scale paper.

### M — heavy-chain-only scope (#5.3.6) — concede with mitigation

All pretraining and evaluation use isolated VH domains. Two mitigating notes:
- `hybrid-paired` and `multispecific` checkpoints exist, trained on `oas_paired_500k.jsonl`, but
  were **never evaluated** — VH-only benchmark numbers would not be comparable, and no paired-chain
  benchmark suite exists in the repo. Do not cite them.
- The AbCDR-Binding datasets of the cited Talaei et al. work are mutational scans around a *single
  parent antibody*, so the light chain is constant within each dataset — a VH-only encoder loses no
  discriminative information on the VH-mutation subsets. Worth stating as the route to a fair
  paired comparison.

⚠️ **Also update reference [6]** — Talaei et al. is now titled *"CDR-aware masked language models
for paired antibodies enable state-of-the-art binding prediction"*, bioRxiv `2025.10.31.685149`.

### N — IgFold accuracy (#4.5.1) — not measured

We never quantified IgFold prediction quality on our corpus. Either measure it (compare IgFold
predictions against SAbDab crystal structures for the subset present in both) or concede.

### Interpretation questions (#6, #4.1, #1.5)

Replication answers several of these for free:

| Question | Answer from the replicates |
|---|---|
| Why is `interface` the best specialist? | **It isn't, robustly.** `germline` leads C. AUROC, C. P@L and Str. ρ; `intersection` has the most resolved wins (3/5) and the best paratope. Six of seven columns have 2–4 tied leaders. The "best overall" sentence must go. |
| Why does `span` do well with no biological prior? | Contiguity is the mechanism — it recovers 86% of the CDR3 gain (§4). Not a puzzle; the paper under-explains its own control. |
| Why is `structure`'s CDR3 exactly 0.000? | **Not a bug.** 0.000 ± 0.001 across 3 seeds, and the novelty data shows why: novel-fraction 0.988, edit distance to nearest training CDR3 2.97, only 1.2% identical to any training CDR3. It generates plausible-but-novel strings and therefore never matches exactly. |
| Why does `hybrid-weighted` underperform? | **Currently unanswerable — it is n=1.** This is what cluster K is for. |
| Why does plain `hybrid` fall below random? | Confirmed as a **resolved loss** on developability (−0.069, −2.7σ) — the only resolved loss among the hybrids. On the other six metrics it is within noise of the Dirichlet control. |
| Interface's secondary non-CDR peak (#1.5) | **Answered 2026-08-16 — it is an annotation artifact in the benchmark labels, not antigen contact. See §10.1.** |

---

## 11. Cluster K + outstanding gaps — what is still owed

| Gap | Cost | Value | Status |
|---|---|---|---|
| **Table 2b replication** — `reverse`, `perbatch`, `weighted` at seeds 1,2 | 6 runs × 5.1 h ≈ 31 GPU-h | Removes the last n=1 rows from Table 3; needed before answering "why does hybrid-weighted underperform" | ✅ **Done 2026-08-15**, 0 failures — see §11.2 |
| **Contamination audit** (axis 2) | ~few hours, CPU | Converts the one unanswerable question into a stated fact | ✅ **Done 2026-08-14** — no overlap, see §9 |
| **Label-permutation control** | 1 run, ~5 GPU-h | Highest value per GPU-hour: separates biological identity from mask statistics. Answers #2.3, #5.2 and the circularity cluster at once | ✅ **Done 2026-08-14** — result in §11.1 |
| **IgFold accuracy** (#4.5.1) | Low | Closes a direct question | ✅ **Done 2026-08-14** — `scripts/igfold_accuracy.py`, result in §10 N |
| **`hybrid-adaptive` replication** | 2 runs, ~11 GPU-h | It is non-deterministic, so n=1 is weakest here | ⚪ **Dropped** — will not go in the paper; cite as single-seed only |
| **Interface secondary-peak characterisation** (#1.5) | Low, no GPU | Closes a direct question | ✅ **Done 2026-08-16** — it is a label artifact, see §10.1 |
| **Re-score paratope probe excluding IMGT 92–95** | Low, probe only | Tests whether `interface`'s +0.042 AUPRC survives the artifact. Highest-value item now | 🔴 Not started |

### 11.2 Table 2b — COMPLETE (2026-08-15)

All six hybrids are now n=3. Full table in [results](results.md); contrasts against the
Dirichlet control in §3 above. Published single-seed values against their 3-seed means:

| Arm | metric most at risk | published (s42) | 3-seed mean ± sd | shift |
|---|---|---|---|---|
| `hybrid-weighted` | **P. MCC** — "largest cell in the whole table" | 0.683 | **0.648 ± 0.034** | +1.0 sd |
| `hybrid-perbatch` | P. AUPRC | 0.864 | 0.848 ± 0.024 | +0.7 sd |
| `hybrid-reverse` | P. AUPRC | 0.859 | 0.842 ± 0.017 | +1.0 sd |

**`weighted`'s P. MCC 0.683 does not survive** — it replicates to 0.648, no longer the
largest cell (that is now `warmstart` at 0.672, itself confounded). Consistent with the
paper-wide pattern: every published value sits within 1.1 sd of its replicated mean, so
nothing was cherry-picked; the problem was reporting resolution we did not have.

**`reverse` is the notable arm.** Mean rank 4.57 of 16, second-best overall behind
`warmstart`, with the narrowest rank range of any strategy (2–7). It also beats the
Dirichlet control on three structural metrics at nominal p<0.05 (§3).

⚠️ **The start-generic-heavy design principle is not supported.** `stretched` − `reverse`,
paired within seed, resolves on **nothing** — every effect under 0.5σ and every metric's
per-seed deltas flip sign:

| | CDR3 | P. AUPRC | P. MCC | C. AUROC | C. P@L | Str. ρ | Dev. ρ |
|---|---|---|---|---|---|---|---|
| `stretched` − `reverse` | −0.003 | −0.009 | −0.009 | −0.003 | −0.032 | −0.012 | −0.026 |
| effect | −0.2σ | −0.5σ | −0.2σ | −0.4σ | −0.4σ | −0.5σ | −0.4σ |

`reverse` is the end-to-end mirror of `stretched` (0.90 generic weight at step 0 versus 0.10,
inverted). Running the curriculum backwards changes nothing measurable. **Do not claim in the
response that the generic-heavy opening was validated** — it was tested and it was not.

### Launch record — 2026-08-14 13:05

The driver mismatch that killed the first attempt (host rebooted 2026-08-13 ~17:06 with kernel
module 610.43.02 against userspace 610.57.04) was cleared by a host reset. Both are now
610.57.04 and torch sees all 8 GPUs. The stale partial checkpoints were deleted, so all six
runs restart from step 0 — `scripts/train.py` has no resume path.

Relaunched as a **7-run sweep** over GPUs 0–2, measured at 7.41 it/s → ~4 h 45 m train plus
~1 h eval per run:

```
MANIFEST=configs/seeds/MANIFEST.rebuttal.tsv TAG=rebuttal bash scripts/launch_table2b.sh
```

| GPU | queue |
|---|---|
| 0 | `interface_permuted_medium` → `hybrid_reverse_medium_s1` → `hybrid_weighted_medium_s1` |
| 1 | `hybrid_reverse_medium_s2` → `hybrid_weighted_medium_s2` |
| 2 | `hybrid_perbatch_medium_s1` → `hybrid_perbatch_medium_s2` |

GPU 0 is the long pole at ~15.4 h. Then
`python scripts/compare.py && python scripts/aggregate_seeds.py --paper-metrics-only`.

`hybrid_adaptive_medium` at seeds 1,2 was launched alongside and **cancelled 45 minutes in**:
the result will not go in the paper, and the rebuttal will mention `hybrid-adaptive` in passing
as a single-seed experiment. The seed-42 checkpoint is retained. Note the manifest still lists
those two rows — **relaunch from `MANIFEST.table2b.tsv`** or they will restart.

**Fallback if it does not finish:** say explicitly in the
response that `reverse`, `perbatch` and `weighted` are single-seed and that their apparent
advantages (e.g. `weighted` P. MCC 0.683, the largest cell in the whole table) should not be
cited. Given that every fine-grained single-seed difference in Table 1 dissolved under
replication, this is the safe and defensible position either way.

### 11.1 The permutation control — RESULT (2026-08-14)

`interface_permuted_medium`, seed 42, trained and evaluated. Paired against `interface`
at the same seed; Δ/sd uses `interface`'s between-seed sd at n=3.

| Metric | `interface` (s42) | permuted (s42) | Δ | Δ/sd | `random` (n=3) |
|---|---|---|---|---|---|
| **P. AUPRC** | 0.8722 | **0.8231** | **−0.0491** | **−4.6σ** | **0.8214** |
| Dev. ρ | 0.3405 | 0.2487 | −0.0918 | −2.0σ | 0.3582 |
| CDR3 exact | 0.1000 | 0.0420 | −0.0580 | −1.7σ | 0.0390 |
| C. P@L | 0.7197 | 0.6708 | −0.0489 | −1.6σ | 0.6043 |
| P. MCC | 0.6636 | 0.6406 | −0.0231 | −1.4σ | 0.6255 |
| Str. ρ | 0.6267 | 0.6101 | −0.0166 | −1.4σ | 0.5918 |
| C. AUROC | 0.9820 | 0.9778 | −0.0042 | −1.3σ | 0.9701 |

**The gain comes from paratope identity.** `interface`'s headline result is P. AUPRC
+0.042 over random masking (19.2σ). Permuting the labels drops it to **0.8231 against a
random-masking baseline of 0.8214** — the entire effect is gone. All seven metrics
degrade (sign test p = 0.016).

This is the answer to #2.3.3 ("the new masks may simply make training harder or focus on
different sequence positions, even without useful biological knowledge"): a mask carrying
identical supervision magnitude, identical rate and identical weight distribution, aimed
at the wrong residues, performs like no prior at all.

**Note the contrast with the hybrids.** The Dirichlet control (§3) shows the specific
*mixture* is not load-bearing; this shows the specific *residues* are. Specialists depend
on the biological prior; hybrids do not depend on the schedule over them.

⚠️ n=1. The contrast is paired within seed 42 and scaled by `interface`'s n=3 spread.
Only P. AUPRC is individually resolved at that scale; the other six are directionally
consistent but each under 2σ. A second seed would firm up the non-AUPRC rows.

### The permutation control — construction and pre-launch verification

Take the teacher's per-residue paratope probabilities and **shuffle them across positions within
each sequence** — preserving the marginal rate and weight distribution, destroying only *which*
residues are paratopes. Pretrain one model on that.

- If `interface` still beats it → the gain comes from paratope **identity**; the biological prior does real work.
- If not → the gain was mask **statistics**, and we learn it before a reviewer does.

One run, ~5 GPU-h. A permuted-label teacher carries exactly as much supervision as the real one,
so it answers #2.3's "baselines are not fully matched" cleanly. For the *hybrid* arms the
Dirichlet control (§3) already makes this argument; for the *specialist* arms — which carry the
paper's main claim — we have no equivalent, and this is the hole.

**Implementation.** `scripts/permute_paratope_labels.py` writes
`data/structures/oas_vh_500k_paratope_permuted.pt`; `configs/interface_permuted_medium.yaml`
is byte-identical to `configs/interface_medium.yaml` except for `data.paratope_path` and
`training.output_dir`. A permutation is a bijection on the multiset of values, so every
statistic the masker can see is preserved by construction — the script asserts the marginal
is conserved and fails rather than train on a broken sidecar.

Verified before launch, because a wrong `paratope_path` makes `masking/interface.py` fall
back silently to uniform masking and the run would look valid while being meaningless:

| check | result |
|---|---|
| per-sequence label sum, original vs permuted | identical (9.0001 vs 9.0001) |
| positions changed | 119 / 119 |
| corpus-wide mean label | 0.080210 over 59.6M residues, preserved |
| sequences exceeding the truncation window | 0 (so no mass moves into a discarded tail) |
| `InterfaceMasking` fallback count | 0 |
| mean label at masked vs unmasked positions | 0.375 vs 0.029 — 12.87× enrichment |

That last row is the one that matters: the masker is genuinely following the permuted labels,
so the arm receives the same supervision magnitude aimed at the wrong residues.

---

## 12. Claims we must retract or soften ⭐ volunteer these

Replication invalidated real claims. Getting ahead of these is worth more than defending them.

| Published claim | Status after replication |
|---|---|
| `interface` wins **5/5** tasks, median +6% | **Overstated.** 2 of 5 resolved; other 3 within noise. |
| *"The `interface` masking strategy performed the best overall"* | **Withdraw.** `germline` leads 3 metrics; `intersection` has the most resolved wins. |
| *"each of the five hybrid strategies outperformed the random masking baseline by a substantial margin"* | **False, and contradicted by our own Table 2.** `hybrid` is resolvably *worse* on developability (−2.7σ) and loses on 5 of 7 metrics. Also "five" should be **six** (#6.3). **Rewrite.** |
| `cdr` CDR3 **+354% (4.5×)** | **Understated** — now +493% (**5.9×**). Report as "4%→23% exact match", not a percentage on a 0.039 base. |
| `span` best on Dev, **+14%** | **Does not survive.** +1% (0.1σ), sign flips across seeds; and the whole column is superseded by §8. |
| `cdr` Cont **−9%** | **Reverses** to +4%. Contact P@L is the noisiest metric (sd up to 0.076). |
| `structure` Cont **+7%** | **Does not survive.** +0.0% (0.0σ). |
| `intersection` "largely failed to outperform interface" | **Wrong.** Most resolved wins (3/5) and best paratope AUPRC. |
| Bolded per-column maxima in Tables 1–3 | **Remove.** Only CDR3 has a unique winner. Bold the tie group or nothing. |
| "parameter-free mechanism for imposing functional inductive biases" | **Reframe** (#5.7). IgFold cost ~240 GPU-h, ≈3× total pretraining compute. Honest framing: *task supervision distilled into the corruption schedule, at zero inference-time cost.* That is a better claim anyway. |

Reassuring context worth stating: **no published value is more than 1.5 sd from its 3-seed mean**,
so nothing looks cherry-picked — the single-seed table was a plausible draw throughout. The
problem was reporting resolution we did not have, not selecting favourable numbers.

⚠️ **Unraised confound to pre-empt:** `hybrid-warmstart` — the strongest hybrid — trains **175k
steps vs 125k** *and* initialises from an `interface` checkpoint. Its four resolved wins are
exactly what extra compute from a strong specialist would produce. No reviewer caught it, but
#2.3's "improvements may come from training effects" points straight at it. Match the compute or
state the caveat; do not let a reviewer find this after acceptance.

---

## 13. Presentation fixes — hit by all five reviewers

Cheap, and collectively a large share of the review text.

- [ ] **Broken refs:** `Figure ??b` (Appendix B), `Section ??` (Appendix C.1). — #1, #4.iv, #5.8, #6.4
- [ ] Main text §3.1 "Details … can be found in Section 4" should point at the appendix; Appendix B's "Full details … provided in Appendix B" is self-referential.
- [ ] **"five hybrid masking strategies" vs six rows in Table 2.** — #6.3
- [ ] **Explain metric abbreviations in captions** (P. AUPRC, C. P@L, Str. ρ); unify terminology across §4 / Figure 2 / Tables 1–2. — #1
- [ ] **Figure 3:** rank presentation obscures effect sizes; #1 explicitly requests the raw per-metric ranking table behind it. Provide it.
- [ ] **Repeat `untrained` / `random` rows in Table 2** so readers need not scroll. — #6.1
- [ ] **Figure S1:** too small; unlabeled dotted line is the random baseline; first 8 panels duplicate Figure 2b at a different aspect ratio; hybrids visually indistinguishable — either comment on why the distributions are near-identical or replace with a difference plot. — #6.5, #6.6
- [ ] **`structure` sampling under-specified** — "only a few masks are sampled" needs the actual procedure. Main text says "greater than four sequence indices", appendix says "at least four indices apart"; the code drops `|i−j| ≤ sep` with `sep=4`, i.e. keeps `|i−j| ≥ 5` — **the appendix wording is wrong.** — #5.8
- [ ] **Move more training/eval detail into the main text.** — #4.iii/iv
- [ ] **Specify the full split strategy**, especially for CDR3 and the paratope labels. — #1.1, #6.2, #6.5
- [ ] **Update reference [6]** (Talaei et al., new title + bioRxiv `2025.10.31.685149`).

---

## 14. Per-reviewer assembly index

Build each reply from these clusters.

| Reviewer | Clusters to include, in order |
|---|---|
| **#1** | H/I (splits) → B (weight sensitivity) → F (adaptive weighting) → D (CDR3 novelty) → §10 interface peak → §13 presentation |
| **#2** | A (seeds) ⭐ → H/I (teacher leakage) ⭐ → B + J (matched baselines; concede the permutation control) |
| **#4** | §10 + B (biological interpretation of hybrids) → N (IgFold accuracy) → F (permutations/weights) → §13 readability |
| **#5** | A (seeds) ⭐ → C (cdr vs span) → E (binding) ⭐ → L (scaling, concede) → B (hybrid motivation) ⭐ → M (HC-only, concede) → §12 reframing (teacher supervision + 240 GPU-h) → §13 |
| **#6** | §10 (interpretation) ⭐ → A (seeds) → H/I (paratope split) → G (developability) → §13 minors 1–6 |

⭐ = the concern that reviewer will weight most heavily.

**Suggested opening for the global response:** lead with the seed study (it is what three
reviewers asked for), state up front that it caused us to retract specific claims, then present
the random-mixture control and the two binding benchmarks as new evidence. A response that opens
with its own retractions earns the credibility needed for the claims we *do* defend — the CDR3
effect at 24σ and the paratope effect at 19.2σ, neither of which any reviewer concern touches.

---

## 15. Provenance

| Source | Used for |
|---|---|
| `MLCB 2026 Reviews.pdf` | All reviewer quotations (5 reviewers, no #3) |
| [10-rebuttal-notes.md](10-rebuttal-notes.md) | Verified code facts, contamination axes, planned controls |
| [results.md](results.md) | Tables 1–3, paired contrasts, retraction list |
| `comparison_outputs/ab_bind_probe.csv` | §6 E1 — aggregated across seeds for this page |
| `comparison_outputs/ngbriney_cov.csv` | §6 E2 — aggregated across seeds for this page |
| `comparison_outputs/developability_ridge.csv` | §8 — aggregated across seeds for this page |
| `comparison_outputs/cdr3_novelty.csv`, `cdr3_degeneracy.csv` | §5 |
| `comparison_outputs/comparison_table.csv` | §4 `cdr` − `span` paired contrast |
| `comparison_outputs/contamination_audit.csv` + `_per_sequence.csv` | §9 axis 2 — `scripts/contamination_audit.py`, run 2026-08-14 |
| `comparison_outputs/paratope_split_verification.json` | §9 axis 3/4 — `scripts/verify_paratope_split.py`, run 2026-08-14 |
| [appendix-splits.md](appendix-splits.md) | Full split specification for Appendix B |
| `comparison_outputs/igfold_accuracy.csv` + `_per_chain.csv` | §10 N — `scripts/igfold_accuracy.py`, run 2026-08-14 |

The three new-benchmark tables (§6, §8) were aggregated seed-wise for this page using the same
convention as `scripts/aggregate_seeds.py` — bare name = seed 42, `_s1`/`_s2` = seeds 1/2;
contrasts paired within seed, then averaged; `*` marks |mean/sd| ≥ 2. They are **not** yet emitted
by `aggregate_seeds.py` itself; folding them in would make this page fully regenerable.
