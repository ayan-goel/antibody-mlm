# 10 · MLCB 2026 rebuttal notes

**Status:** working document. Started 2026-08-10 after MLCB 2026 reviews returned.

Source documents in repo root: `MLCB 2026 Reviews.pdf`, `Antibody_MLCB_2026.pdf`, `Antibody_Appendix_MLCB.pdf`.
Five reviewers (#1, #2, #4, #5, #6 — no #3). Relevance: #4 "perfectly aligned"; #1/#5/#6 "strongly relevant"; #2 "somewhat relevant".

Purpose: collect (a) facts we can assert in a response, each backed by a code reference, and (b) the experiments we still owe. Everything in "Verified facts" has been checked against the code — cite the file:line, not the prose docs.

---

## 1. Verified facts we can assert

### 1.1 The paratope teacher never sees the probe's test set

Both the teacher and the downstream probe call the **same** split function, so the split is identical and deterministic:

- `scripts/train_paratope_teacher.py:159` → `load_paratope_splits(...)`; trains on `split["train"]`, early-stops on `split["valid"]`, reports on `split["test"]`
- `evaluation/downstream/paratope.py:26` imports the **same** `load_paratope_splits` with the same defaults
- `data/benchmarks/paratope.py:87-106` → `Paratope(name="SAbDab_Liberis").get_split()`

The teacher is fit only on the train split; the probe is scored only on the test split. **No test antibody was ever labeled by the teacher during teacher training.**

> ⚠️ **Confirm before asserting:** TDC's `get_split()` default is documented as `method='random', seed=42, frac=[0.7,0.1,0.2]`, but `tdc` is not installed in any env on this machine and this was not executed. Run it and paste the actual split sizes into the response.

Additional mechanism points (these are about *what the encoder can possibly learn*, and hold regardless of the split):

- The encoder's only training loss is amino-acid reconstruction. Paratope labels never appear as targets.
- Teacher predictions are computed **only on the 500k OAS corpus** (`scripts/compute_paratope_labels_v2.py`), never on evaluation sequences.
- At probe time masks are irrelevant — the encoder sees clean sequences. There is no channel by which mask placement becomes a memorized test answer.

### 1.2 Structural evaluation splits are by complex, not by chain

Worth stating explicitly — reviewers assumed all our splits were naive, and two of the four are not.

| Probe | Split | Code | Verdict |
|---|---|---|---|
| Contact map | 60/20/20 **grouped by PDB**, chains of one complex never straddle | `data/benchmarks/contact_map.py:237-250` | ✅ defensible |
| Structure probe | by PDB; AB-Bind + SAbDab merged into a unified PDB-keyed pool, deduped by PDB id so a structure present in both sources cannot cross splits | `data/benchmarks/structure_probe.py:220-263` | ✅ defensible |
| Paratope | TDC default **random** split over SAbDab_Liberis | `data/benchmarks/paratope.py:96` | ❌ weakest link (§2.4) |
| Developability | TDC default random split over TAP (small n) | `data/benchmarks/developability.py:100` | ⚠️ small-n |

### 1.3 The OAS pretraining eval split is genuinely held out

- `training/trainer.py:110-116` — `random_split(full_dataset, [train, eval], generator=manual_seed(config.seed))`, `train_split: 0.9`
- `scripts/run_all_evaluations.py:52,582` — `EVAL_SPLIT_SEED = 42`
- Every config in `configs/*.yaml` sets `seed: 42`

Both seeds are currently 42, so the evaluation split **is** the same held-out 10% excluded from training. Verified consistent across all 17 experiments. (But see §4.1 — this coupling breaks the moment we vary the seed.)

---

## 2. Contamination: the four axes

The reviewers said "leakage"; the concerns actually decompose into four separate things. Only one is genuinely unexamined.

### 2.1 Axis 1 — OAS train vs OAS eval ✅ separate, ⚠️ not clustered

Used for: CDR3 infilling, MLM accuracy, perplexity, PLL, JSD.

Separate (§1.3). **But it is a random split by index.** `data/preprocessing.py:52-56` deduplicates by **exact string match only**. OAS is dense with clonally related sequences — same donor, same V/J call, near-identical CDR3 — so clonal relatives straddle train/eval.

Impact: inflates absolute CDR3 infilling numbers (the headline 0.236 exact match) for **all arms equally**. Ranking is unbiased; the absolute number is optimistic. This is the axis where "we should have clustered" is correct.

Fix: cluster on CDR3 identity (MMseqs2, or Levenshtein ≥90% on CDR3) and split by cluster.

### 2.2 Axis 2 — OAS pretraining corpus vs external benchmark sets ❌ NEVER CHECKED

**This is the real gap, and it is what Reviewer #2 actually asked about.**

The probes evaluate on antibodies from *different sources* than the pretraining corpus: SAbDab_Liberis (paratope), SAbDab crystal structures (contact map), AB-Bind + SAbDab (structure probe), TDC TAP (developability). **Nothing anywhere in the repo checks whether any of those antibodies also appear in the 497k OAS pretraining corpus.** Grepping for dedup/overlap logic returns only within-corpus exact-string dedup.

This is a distinct question from §2.1. The OAS internal split being clean says nothing about whether a TAP or SAbDab antibody sits inside the OAS training 90%. Many SAbDab/TAP entries are well-characterized therapeutic or anti-viral mAbs, and OAS contains repertoires from immunized and infected donors, so exact matches are plausible and near-relatives are likely.

Impact: affects every arm equally → **does not bias the ranking**, but inflates absolute numbers and we currently cannot answer the question at all. Cheap to close (§4.2).

### 2.3 Axis 3 — teacher-train vs probe-test ✅ separate

Verified, §1.1. Assert this directly.

**Asymmetry to be aware of:** the paratope rebuttal is clean because teacher and probe share a split function. The **structure** story is weaker — IgFold's training set is external and opaque to our pipeline, and IgFold was trained on PDB antibody structures while our contact-map eval is SAbDab crystal structures, so overlap between IgFold's training data and our test PDBs is likely. The mechanism argument (IgFold shapes masks on OAS, not on test structures) still holds and is the primary defense, but we cannot point to a shared split function. Expect pushback here; answer with mechanism + the permutation control (§4.4).

### 2.4 Axis 4 — within-benchmark split quality ⚠️ paratope is the weak link

`load_paratope_splits` takes TDC's **default random split**. SAbDab is heavily redundant — the same antibody recurs across many PDB entries, plus large families of near-identical antibodies. A random split puts near-duplicates on both sides.

This inflates two separate things: the teacher's reported quality (consistent with its implausible AUPRC) **and** the paratope probe numbers in Table 1, for every strategy alike. It is independent of the teacher question entirely — Reviewer #6's "worth double checking the split for the paratope labels" lands here, not on circularity.

Fix: re-split SAbDab_Liberis by sequence-identity cluster and re-run the paratope probe. Note the teacher would also need retraining on the new train split for consistency.

---

## 3. Where the reviewers' framing is wrong, and where it survives

**Wrong:** "leakage" in the ordinary sense — test answers reaching the model — is not what happens. Push back on this, using §1.1.

**Survives (#5.7's framing, which is the accurate one):** this is an **attribution** issue, not a leakage issue. The `interface` arm had access to supervised paratope annotations, laundered through a teacher; the `random` arm had none. "Interface beats random on paratope prediction" compares a weakly-supervised pipeline against an unsupervised one. That does not invalidate the result — it changes what it means.

The abstract currently sells this as "a parameter-free mechanism for imposing functional inductive biases," while the appendix notes IgFold cost 240 GPU-hours, ~3× total pretraining compute. A reader who assembles those two facts feels misled. **Reframe:** task supervision distilled into the corruption schedule, at zero inference-time cost. That is a fine claim — arguably a more interesting one — and it costs a paragraph.

---

## 4. Open action items

### 4.1 ✅ FIXED (2026-08-10) — data-split seed decoupled from the model seed

**Was:** `training/trainer.py` used `config.seed` for the train/eval `random_split`, while `scripts/run_all_evaluations.py` hardcoded `EVAL_SPLIT_SEED = 42`. Both were 42, so past results are unaffected — but a run with `seed: 1` would have drawn a *different* 90/10 split while evaluation still sliced the seed-42 split, putting ~90% of the "held-out" set inside that run's training data. Silently inflated, not obviously broken.

**Now:** the partition is defined once in `data/splits.py::make_train_eval_split`, keyed on a new `DataConfig.data_split_seed` (default 42) that is independent of `config.seed`. `training/trainer.py`, `scripts/run_all_evaluations.py`, and `scripts/refresh_evaluations.py` all call it; both `EVAL_SPLIT_SEED` constants and the "keep this in sync" comment are gone.

What `seed` still varies for replicates: **weight init and masking RNG** (via `set_seed`). Dataloader order stays fixed — `trainer.py` already pinned `data_seed=42` in `TrainingArguments`. So replicates isolate init + mask stochasticity, which is what #5.1 asks for.

**Side effect, benign but worth knowing:** the two paths previously computed split sizes with different arithmetic — trainer `int(n·0.9)`, eval scripts `n − int(n·0.1)` — leaving them off by one. On n=497,309 the eval scripts scored **49,730** sequences while the true held-out set is **49,731**; the old set was a strict *subset*, so no training data ever leaked and published numbers stand. Post-fix both use 49,731.

The size delta is one sequence in ~49.7k, but the *offset* shifts by one, which matters more than it sounds for the sampled metrics: `evaluation/infilling.py:231` scores a deterministic **prefix** (`indices = list(range(n))`), not a random sample, so the CDR3 window gains `perm[447578]` and drops `perm[448578]`. That is 1 sequence of the 1,000 evaluated → CDR3 exact match can move by at most ±0.001. Negligible against the 0.031 `cdr`−`span` gap, but not literally zero. Don't mix pre- and post-fix numbers in one table without a footnote.

Tests: `tests/test_data_split.py` (10 cases) pins the guarantee — split invariant to `config.seed`, still responsive to `data_split_seed`, disjoint + exhaustive, sizes follow the trainer convention, and legacy configs without the field default to 42. Full suite: 152 passed.

### 4.2 Contamination audit (answers #2, #1.1, #6 minor 2)

MMseqs2 the 497k OAS corpus against each benchmark set (SAbDab_Liberis, SAbDab contact PDB chains, AB-Bind, TAP). Report the max-identity distribution and counts above 90 / 95 / 100%. One table, a few hours, converts an unanswerable question into a stated fact. Do this **before** deciding whether any re-splitting is actually necessary — if overlap is negligible, we say so and move on.

### 4.3 Seed replicates — plan

**Scope (decided 2026-08-10):** all 8 non-hybrid strategies, 2 additional seeds each. Hybrids deferred.

| Strategy | `train.py` flag | Base config | Seed-42 experiment name |
|---|---|---|---|
| `uniform` (paper's `random`) | `--uniform` | `configs/medium.yaml` | `uniform_medium` |
| `cdr` | `--cdr` | `configs/cdr_medium.yaml` | `cdr_medium` |
| `span` | `--span` | `configs/span_medium.yaml` | `span_medium` |
| `structure` | `--structure` | `configs/structure_medium.yaml` | `structure_medium` |
| `structure-lr` | `--structure_longrange` | `configs/structure_longrange_medium.yaml` | `structure_longrange_medium` |
| `interface` | `--interface` | `configs/interface_medium.yaml` | `interface_medium` |
| `germline` | `--germline` | `configs/germline_medium.yaml` | `germline_medium` |
| `intersection` | `--intersection` | `configs/intersection_medium.yaml` | `intersection_medium` |

**Also replicate `untrained`** (`configs/untrained_medium.yaml`). `scripts/create_untrained_baseline.py:51` seeds init from `config.seed`, so it has genuine seed variance, it costs minutes not hours, it is the floor for every metric in Table 1, and it is currently the *best* model on the AB-Bind ΔΔG columns (§5.2). Cheap and load-bearing.

**Cost.** Measured from `logs/cdr_medium.log`: `train_runtime = 18,988 s` = **5.27 h/run** (matches the appendix's ≈5 h). 16 runs ≈ **84 GPU-h** — about the same as the original 15-run budget. On the 8 GPUs used for IgFold: 2 waves, ~11 h wall clock. Evaluation is on top and is currently unmeasured; measure it on the first replicate before committing to a schedule.

**Mechanics.**

1. `scripts/train.py` has no `--seed` override — `MODEL_REGISTRY` (`train.py:42-64`) maps each flag to a fixed `(config, name)` pair. Generate 16 per-seed YAMLs instead and use the legacy `--config` path. Each is the base config plus two overrides:
   ```yaml
   seed: 1                                                    # was 42
   training:
     output_dir: "models/checkpoints/interface_medium_s1"     # was interface_medium
   ```
   Leave `data.data_split_seed` unset so it defaults to 42 (§4.1) — that is the whole point.
2. **Names must be distinct.** `evaluation/compare.py:60-94` discovers experiments by globbing `models/checkpoints/*/training_summary.json`, `evaluation_outputs/*/all_metrics.json`, and `downstream_outputs/*/`. Same-named runs would overwrite each other. Convention: keep seed 42 at its existing bare name, add `_s1` / `_s2`.
3. `scripts/generate_report.py` has a hardcoded experiment allowlist but accepts `--experiments`; pass the full list explicitly.
4. **Aggregation** is handled by `scripts/aggregate_seeds.py` (below) — `compare.py` itself still emits one row per experiment.

**Tooling (built 2026-08-10).**

| File | Purpose |
|---|---|
| `scripts/make_seed_configs.py` | Emits per-seed configs + `configs/seeds/MANIFEST.tsv`. Asserts each variant differs from its base in *only* `seed` and `training.output_dir` — any drift in `data`/`masking`/`model`/rest-of-`training` is a hard failure. |
| `configs/seeds/*.yaml` | 18 generated configs (9 strategies × seeds 1, 2). |
| `scripts/run_seed_replicates.sh` | Shards the manifest across GPUs round-robin. Idempotent — skips any run whose `final/` already exists, so it is safe to re-launch after an interruption. Dispatches `untrained` to `create_untrained_baseline.py`. |
| `scripts/aggregate_seeds.py` | Groups replicates by strategy → `seed_aggregate.csv` (mean/sd/n), `seed_contrasts.csv` (paired per-seed deltas vs `uniform_medium`, with effect size in sd units), `seed_summary.md` (paper-shaped table). |
| `tests/test_data_split.py`, `tests/test_aggregate_seeds.py` | 25 cases pinning split determinism and the contrast arithmetic. |

```bash
python scripts/make_seed_configs.py                      # already run; --seeds 1 2 3 to extend
bash scripts/run_seed_replicates.sh 0 2                  # shard 0 of 2 (one shell per GPU)
bash scripts/run_seed_replicates.sh 1 2                  # shard 1 of 2
# ...then evaluate each new checkpoint, regenerate comparison_table.csv, and:
python scripts/aggregate_seeds.py --paper-metrics-only
```

**Wall clock:** 2 GPUs → 8 training runs each → **~42 h**. 4 GPUs → ~21 h. 8 GPUs → ~11 h. Evaluation is additional and still unmeasured — time it on the first finished replicate.

**Evaluation protocol: match the paper exactly. Do not tune it here.**

The replicates exist to put error bars on the *published* numbers, so they must use the identical evaluation procedure. Confirmed from `evaluation_outputs/uniform_medium/all_metrics.json`: `infill_cdr3_exact_match_count = 1000`, `pll_num_sequences = 500` — i.e. the bare `run_all_evaluations.py` defaults, matching the documented invocation in `GUIDE.md:230`. `scripts/run_seed_replicates.sh` therefore passes **no** sample-count overrides; `INFILL_SAMPLES` / `PLL_SEQUENCES` exist only for a separate, separately-reported sweep and must be left unset.

Raising the infilling sample count was considered and **rejected** for these runs: it would confound pretraining-seed variance with an evaluation change, force re-evaluation of all nine seed-42 models, and create two incompatible versions of every published number. If higher-precision CDR3 measurement is wanted later, run it as its own uniform pass over *all* checkpoints and report it separately.

Note the motivating noise estimate was overstated. `evaluation/infilling.py:231` scores a deterministic *prefix*, so all models see the identical 1,000 sequences. The ≈1.3-point binomial se bounds uncertainty on the *absolute* 0.236; for model-vs-model comparisons the sample is shared and largely cancels, so the paired difference is considerably tighter. n=1,000 is adequate for the comparisons the paper makes.

**One unavoidable deviation.** The corrected split (§4.1) evaluates on 49,731 sequences where the published runs used 49,730, and because infilling takes a prefix the window shifts by one — up to ±0.001 on CDR3 exact match. This is forced: decoupling the split seed is what makes the seed study valid at all. Footnote it; do not try to reproduce the old off-by-one.

**Reporting.**

- Report **between-encoder sd** separately from **between-probe-seed sd**. This decomposition is the direct answer to #5.1 and #6 — the current 3–5 probe seeds randomize only head init, dropout, and batch order (`Appendix C.1`), which is exactly why reviewers discounted them.
- With n=3 per arm, avoid parametric testing. Compute each contrast (e.g. `interface − uniform`) **within seed** and then average; the paired difference is far tighter than comparing marginals, since all three seeds share the identical held-out set and evaluation sample.
- State each headline gap in units of observed pretraining sd.
- Prioritize the metrics #5.1 named: contact-map AUROC and structure-probe Spearman ρ.

**Early read (2026-08-11, 3 of 9 strategies complete at all three seeds).** Paired per-seed contrasts vs `uniform`, n=3, effect = mean/sd:

| Contrast | Metric | Per-seed deltas | Mean | sd | Effect |
|---|---|---|---|---|---|
| `cdr` − `uniform` | CDR3 | +0.184 +0.193 +0.200 | **+0.192** | 0.008 | **24.0** |
| `span` − `uniform` | CDR3 | +0.153 +0.176 +0.166 | **+0.165** | 0.012 | **14.3** |
| `span` − `uniform` | P. AUPRC | +0.016 +0.030 +0.045 | +0.030 | 0.015 | 2.1 |
| `cdr` − `uniform` | Dev. ρ | −0.012 −0.053 −0.062 | −0.043 | 0.027 | −1.6 |
| `cdr` − `uniform` | C. P@L | −0.057 +0.033 +0.093 | +0.023 | 0.076 | 0.3 |
| `span` − `uniform` | Dev. ρ | +0.048 −0.035 −0.001 | +0.004 | 0.042 | 0.1 |
| `span` − `uniform` | Str. ρ | +0.019 +0.033 −0.043 | +0.003 | 0.040 | 0.1 |

Two things follow, both worth knowing before the response is written:

1. **The CDR3 claims are extremely robust** — 24 σ and 14 σ. No reviewer concern touches them. And they get *stronger*: `uniform` CDR3 across seeds is 0.052 / 0.033 / 0.032 (mean 0.039), so the published 0.052 was the most favorable draw. On means the `cdr` gain is 5.9×, not 4.5×.

2. **Two published secondary claims do not survive.** `span`'s "+14% developability" (appendix Table 1) is +0.004 on paired means with the sign flipping across seeds — it was a seed artifact of the 0.336 baseline draw. `cdr`'s "−9% on Cont" likewise reverses: the paired mean is **+0.023**, and per-seed C. P@L for `cdr` spans 0.549 → 0.707. Contact-map P@L is the noisiest metric by a wide margin (sd up to 0.076), exactly as #5.1 predicted.

Caveats: n=3, only 2 of 8 strategies contrasted so far, and `interface` — which carries the headline "best specialist" claim — has not finished. Do not rewrite anything until all 8 land. But plan for the appendix relative-improvement table to need real revision, not just error bars.

**Deferred:** hybrids stay single-seed for now, so Table 2's claims remain single-seed — say so explicitly in the paper. Note that `hybrid-warmstart`, the strongest hybrid, additionally carries the +50k-step / warm-start confound (§6). Defending Table 2 properly will eventually need at least one replicated hybrid.

### 4.4 Label-permutation control (highest value per GPU-hour)

Take the teacher's per-residue paratope probabilities and **shuffle them across positions within each sequence** — preserving the marginal rate and weight distribution, destroying only *which* residues are paratopes. Pretrain one model on that.

- If `interface` still beats it → the gain comes from paratope **identity**; the biological prior is doing real work.
- If not → the gain was mask **statistics**, and we learn that before a reviewer does.

One run, ~5 GPU-h. Answers #2's "baselines are not fully matched," #5.2's "cdr vs span may just be span reconstruction," and the entire circularity cluster at once — a permuted-label teacher carries exactly as much supervision as the real one.

### 4.5 Binding task — see §5

---

## 5. Binding: what we have, what is broken, what to run

### 5.1 There is no binding task in the paper

Section 4's "functional binding" is **paratope identification** — a per-residue interface-geometry task. Not affinity, not specificity. #5.3's critique is correct as written.

`data/benchmarks/binding.py` (CoV-AbDab, binary SARS-CoV-2 neutralization from VH, stratified 70/15/15 at `binding.py:111-137`) exists but is **dead code**:
- nothing imports it (grep for `benchmarks.binding|load_binding_splits|BindingDataset` hits only the file itself); `data/benchmarks/__init__.py` is a bare docstring
- never registered — `@register_task` appears only in `contact_map.py:28`, `developability.py:29`, `paratope.py:58`, `structure_probe.py:31`
- no sequence-level classification head exists in `evaluation/downstream/heads.py`
- no binding columns in `comparison_outputs/comparison_table.csv`
- one loose thread: `evaluation/report.py:300` maps `"binding": "Binding (CoV-AbDab)"` — a label for a task that never emits anything

### 5.2 🔴 AB-Bind ΔΔG is already computed — and every trained model is below chance

`evaluation/mutation_scoring.py` + `scripts/benchmark_mutations.py` already score AB-Bind ΔΔG zero-shot via ΔPLL, for all 17 experiments, 29 complexes / 1341 mutants. Columns `mut_*` in the comparison table:

| Experiment | Spearman ρ | AUROC |
|---|---|---|
| `untrained_medium` | **+0.107** | **0.602** |
| `span_medium` | +0.020 | 0.481 |
| `uniform_medium` | +0.013 | 0.472 |
| `interface_medium` | +0.010 | 0.445 |
| `cdr_medium` | +0.005 | 0.412 |
| `germline_medium` | −0.035 | 0.404 |
| `hybrid_intersection_medium` | −0.062 | 0.421 |

Every trained model is at ρ≈0 and **below-chance AUROC**; the randomly initialized control is the best by a wide margin.

**Do not report these numbers as-is.** The most likely reading is that the *protocol* is wrong, not the method: ΔPLL scores a mutant by its likelihood under an OAS repertoire prior, and affinity-improving mutations are typically germline-*divergent* and therefore low-likelihood. A well-trained abLM that has learned the repertoire prior should anti-correlate with ΔΔG on this protocol — which is exactly the pattern above, including why `untrained` (no prior) escapes it. Compounding this, the model never sees the antigen, so there is a hard ceiling on achievable correlation.

Both referenced papers avoid this by using **supervised probes on frozen embeddings**, not zero-shot likelihood. We should match their protocol. Worth a sentence in the paper either way — an inverted zero-shot result that we can explain is more interesting than a silent omission, and reviewers may find it in the repo.

### 5.3 The referenced papers' actual protocols

**Ng & Briney** (Patterns 2025) — frozen embeddings + single feedforward head, 5-fold CV:
1. native vs. shuffled VH/VL pairing (Dataset C ~65k, Dataset D ~140k)
2. **binding specificity**: SARS-CoV-2-specific vs. not, Dataset E ~25k — **paired chains**

**Talaei et al.** — note the citation is stale: now titled *"CDR-aware masked language models for paired antibodies enable state-of-the-art binding prediction"*, bioRxiv `2025.10.31.685149`. **Update reference [6].** Released dataset **AbCDR-Binding**, Zenodo record `18762978`, CC-BY 4.0, `binding_affinity_curated.zip` (104.9 MB), six affinity datasets:

| Dataset | N | Content |
|---|---|---|
| anti-HR2 SARS-CoV-2 | 71,830 | combinatorial mutations, spike |
| anti-Fluorescein | 11,052 | single + combinatorial |
| G6 (HyHEL-10) | 4,275 | single-point, anti-lysozyme |
| D44 (D44.1) | 2,048 | single-point, anti-lysozyme |
| anti-H1 hemagglutinin | 1,038 | single + combinatorial |
| Trastuzumab | 422 | single-point, anti-HER2 |

**Key enabler for us:** each dataset is a mutational scan around a **single parent antibody**, so the light chain is *constant within a dataset*. A VH-only encoder therefore loses no discriminative information on the VH-mutation subsets — this sidesteps the heavy-chain-only objection (#5.6) for this benchmark specifically. Need to download and confirm which mutations fall in VH vs VL.

### 5.4 Recommended order

1. **AbCDR-Binding VH-mutation subsets** — exactly the benchmark #5 asked for, directly comparable to the cited work, public and licensed. Largest payoff.
2. **AB-Bind ΔΔG as a supervised frozen-embedding regression probe** — data already in `data/ab_bind/`, reuses existing probe infrastructure, cheapest add. Note AB-Bind PDBs are already used by the structure probe; flag the reuse.
3. **CoV-AbDab specificity** — the VH-only analogue of Ng & Briney's Dataset E. Needs a sequence-classification head + `@register_task` registration (~half a day) to finish the existing loader.

**Split warning for all three:** mutational scans need mutation-position-disjoint or held-out-combination splits, not random, or the probe trivially interpolates. CoV-AbDab needs clonotype-level splits — neutralizers are dominated by IGHV3-53/3-66 public clonotypes, and a random split will badly overstate performance.

---

## 6. Presentation fixes (cheap, hit by all five reviewers)

- Unresolved refs: `Figure ??b` (Appendix B), `Section ??` (Appendix C.1). Also main text §3.1 "Details ... can be found in Section 4" should point at the appendix; Appendix B's "Full details ... provided in Appendix B" is self-referential.
- "five hybrid masking strategies" (§5) vs **six** rows in Table 2. — #6.3
- **Unsupported claim:** "each of the five hybrid strategies outperformed the random masking baseline by a substantial margin" is false against our own Table 2. `hybrid` loses on 5 of 7 metrics (P.AUPRC, P.MCC, C.AUROC, C.P@L, Dev). Developability is a systematic hybrid weakness: 5 of 6 hybrids fall below random; only `hybrid-warmstart` beats random on all seven. The "substantial margin" is the CDR3 term dominating an average. **Rewrite this sentence.**
- **Unraised confound to pre-empt:** `hybrid-warmstart` is the strongest hybrid but trained 175k steps vs 125k *and* initialized from the `interface` checkpoint. No reviewer caught it; #2's "improvements may come from training effects" points straight at it. Either match compute or state the caveat.
- Terminology drift across §4 / Figure 2 / Tables 1–2; metric abbreviations (P. AUPRC, C. P@L, Str. ρ) unexplained in captions. — #1
- Figure 3 rank presentation obscures effect sizes; #1 explicitly requests the raw per-metric ranking table behind it.
- Repeat `untrained`/`random` rows in Table 2. — #6.1
- Figure S1: too small, unlabeled dotted line (random baseline), first 8 panels duplicate Figure 2b at a different aspect ratio, hybrids visually indistinguishable. — #6.5, #6.6
- `structure` sampling under-specified — "only a few masks are sampled" needs the actual procedure. Also main text says "greater than four sequence indices" while the appendix says "at least four indices apart"; per `masking/` the filter drops `|i−j| ≤ sep` with sep=4, i.e. keeps `|i−j| ≥ 5` — **the appendix wording is wrong**. — #5.8
- Move more training/eval detail into the main text. — #4.iii/iv

---

## 7. Interpretation questions still owed (#6, #4.1, #1.5)

No experiments needed, but the paper is weaker without answers:

- Why is `interface` the best specialist?
- Why does `span` do well with no biological prior?
- Why is `structure`'s CDR3 exact match **exactly 0.000**, identical to the untrained control? This reads like a bug — **check it before responding.**
- Why does `hybrid-weighted` underperform despite prioritizing the best specialist?
- Why does plain `hybrid` fall below random?
- What is the secondary non-CDR peak in the `interface` mask distribution, and is it functionally meaningful? (#1.5)
- Are recovered CDR3s diverse, or degenerate? (#1.4)
- How accurate are the IgFold predictions on our corpus? (#4.1)
