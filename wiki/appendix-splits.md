# Appendix draft — data splits

Drafted 2026-08-14 to answer #1.3.1 ("The detailed train/test split strategy is not provided
as mentioned in Appendix B"), and the related #6.2 and #2.5.2. Every number below is read
directly from the code path named beside it, not from memory.

⚠️ **One correction to carry into the paper:** pretraining is a **90/10 two-way** split, not
90/5/5. `data/splits.py::make_train_eval_split` returns exactly `(train, eval)`. There is no
third partition anywhere in the pretraining path.

---

## B.x Data splits

### B.x.1 Pretraining corpus

Pretraining uses 497,309 unique heavy-chain variable-domain (VH) sequences drawn from the
Observed Antibody Space. Sequences are uppercased, restricted to the 20 standard amino acids,
length-filtered to 80–200 residues, and deduplicated by exact string match
(`data/preprocessing.py::preprocess_sequences`).

The corpus is partitioned **90/10 into train and held-out sets — 447,578 and 49,731
sequences** — by `torch.utils.data.random_split` under a generator seeded with
`data_split_seed = 42` (`data/splits.py::make_train_eval_split`).

Three properties of this partition matter for interpreting the results:

1. **It is identical across every experiment.** Both the training loop
   (`training/trainer.py`) and the evaluation scripts call the same function, so all models
   are scored on exactly the same held-out sequences and cross-model comparisons are
   well-defined.
2. **It is independent of the experiment seed.** `data_split_seed` is a separate field from
   the `seed` that controls weight initialisation and masking RNG. Our seed replicates
   therefore vary only initialisation and masking stochasticity; had the split moved with the
   seed, a replicate would be evaluated on sequences it had trained on and its zero-shot
   metrics would be silently inflated.
3. **No model selection is performed on the held-out set.** Early stopping is disabled
   (`early_stopping_patience: 0`) and every strategy trains for a fixed 125,000 optimizer
   steps, so the equal-step comparison is fair and the held-out split functions as a clean
   test set despite being consumed as the `eval_dataset` during training. Dataloader order is
   pinned at `data_seed = 42`.

All held-out pretraining metrics — masked-token accuracy, pseudo-log-likelihood, and CDR-H3
infilling exact match — are computed on this 49,731-sequence partition.

### B.x.2 Downstream probes

Each downstream benchmark carries its own split, listed in full below. All are deterministic
at seed 42. "Grouped" indicates whether related sequences are constrained to a single split.

| Benchmark | Source | Split procedure | Grouped by | Train / val / test |
|---|---|---|---|---|
| Paratope prediction | TDC `SAbDab_Liberis` | TDC default `get_split()`: random, 70/10/20, seed 42 | — none | 716 / 102 / 205 (1,023) |
| Developability, published probe | TDC `TAP` | TDC default random 70/10/20, seed 42, applied per label then merged on antibody ID | — none | 169 / 24 / 48 (241) |
| Developability, corrected probe | TDC `TAP` | Nested 5-fold CV over all 241 antibodies, 5 repeats; ridge α selected on inner folds only | — none | all 241, out-of-fold |
| Contact map | SAbDab crystal structures | 60/20/20 by PDB entry, seed 42 | ✅ PDB | 367 / 121 / 130 (618) |
| Residue-distance (structure probe) | SAbDab + AB-Bind PDBs, deduplicated by PDB ID | 60/20/20 by PDB entry, seed 42 | ✅ PDB | 407 / 134 / 139 (680) |
| AB-Bind ΔΔG | AB-Bind | 60/20/20 bin-packed by complex, repeated over 20 independent split seeds (0–19); mean ± sd reported | ✅ complex | 849 / 284 / 278 (1,411) |
| Binding specificity, dataset loader | CoV-AbDab | 70/15/15 bin-packed by approximate clonotype, seed 42 | ✅ clonotype | 5,667 / 1,214 / 1,214 (8,095) |
| Binding specificity, reported result | Ng & Briney Dataset E (Zenodo 14019655) | Stratified 5-fold cross-validation, shuffled, seed 42 | — none, by design | 24,969, 5-fold |

Two of these deserve a sentence of explanation.

**Bin-packed grouped splits (AB-Bind, CoV-AbDab).** AB-Bind is severely skewed — a single
complex holds roughly 35% of all mutants — so splitting by complex *count* yields wildly
unbalanced *record* counts. Groups are instead shuffled and assigned largest-first to whichever
split is furthest below its target share. No complex, and no clonotype, ever straddles a split,
so a model cannot memorise a per-group offset and reuse it at test time. Clonotypes are the
standard approximation: same V gene, same CDR-H3 length, same CDR-H3 four-residue prefix.

**Why the reported CoV-AbDab result uses stratified rather than grouped CV.** We report the
stratified 5-fold protocol of the reference work so our numbers are directly comparable to
theirs. Stratified CV is not clonotype-grouped, and SARS-CoV-2 neutralisers are dominated by
public IGHV3-53/3-66 clonotypes, so clonal relatives can straddle folds and all arms — ours and
the reference's alike — score higher than a grouped split would give. A clonotype-grouped
loader is implemented (`data/benchmarks/binding.py::load_binding_splits`) and is the honest
protocol; we use it for internal comparison and the stratified one for external comparability.

### B.x.3 Teacher models

The paratope teacher (`scripts/train_paratope_teacher.py`) calls the **same**
`load_paratope_splits` function with the **same** defaults as the paratope probe
(`evaluation/downstream/paratope.py`), so the two partitions are byte-identical. The teacher is
fit only on the train split, early-stopped on validation, and reported on test. **No test
antibody was ever labelled by the teacher during teacher training.**

Independently of the split, three properties rule out label leakage into the encoder: the
encoder's only training objective is amino-acid reconstruction, so paratope labels never appear
as prediction targets; teacher predictions are computed only over the OAS pretraining corpus
(`scripts/compute_paratope_labels_v2.py`) and never over evaluation sequences; and at probe time
the encoder receives clean, unmasked sequences, so mask placement cannot become a memorised test
answer.

IgFold is used as an external pretrained predictor with no fitting on our part. Its training
set is not disclosed to us, so we cannot assert a shared partition with our structural
benchmarks; the mechanism argument above (IgFold shapes masks over OAS, never over evaluation
structures) is the operative defence.

### B.x.4 Pretraining corpus versus benchmark sets

We audited whether evaluation antibodies also appear in the pretraining corpus
(`scripts/contamination_audit.py`). Comparing all 34,315 benchmark heavy chains against all
497,309 pretraining sequences: **no benchmark antibody appears verbatim in the corpus**; six
(0.017%) share an exact CDR-H3 with some pretraining sequence, and **none of those six also
matches on V gene**, so none is clonally related. All six have unusually short CDR-H3s (5–9
residues, against a benchmark median of 12), the regime in which exact collision is expected by
chance. No benchmark antibody exceeds 95% identity to any pretraining sequence.

We report CDR-H3 overlap rather than global sequence identity deliberately: unrelated VH domains
share a common scaffold and routinely sit at 80–90% identity, so generic identity clustering is
uninformative for antibodies, whereas CDR-H3 is the V(D)J-generated segment that individuates a
lineage. CDR-H3 is defined identically on both sides (IMGT positions 105–117 via ANARCI, verified
to reproduce the corpus annotation on 300 of 300 sampled sequences), and the search recovers known
self-matches at 100% identity — including chains truncated at both termini to mimic PDB-derived
sequences — so the negative result reflects the data rather than an insensitive method.

### B.x.5 Scope of the comparison

All arms share every partition described above, so the comparisons the paper makes are between
models evaluated on identical data. Two properties of the underlying corpora bear on how the
**absolute** values should be read, and neither affects the ranking:

1. **OAS deduplication is exact-string.** The corpus retains clonally related sequences differing
   by one or two residues, so relatives appear on both sides of the pretraining split. Our CDR-H3
   novelty analysis quantifies the consequence directly: 83.8% of true held-out CDR-H3s appear
   verbatim in the training partition, and we interpret the CDR-H3 metric as recall of the
   repertoire consensus accordingly.
2. **The paratope and TAP benchmarks use TDC's default random split**, which we adopt unchanged
   so our numbers are directly comparable to other work on these benchmarks.

> **Not for the paper — internal note.** Point 2's redundancy is measured
> (`scripts/verify_paratope_split.py`): 27.8% of paratope test antibodies have a ≥95%-identical
> counterpart in train. This is a property of the public benchmark, shifts all arms identically,
> and was not raised by any reviewer. Keep the number available; do not volunteer it.

---

## Source map

| Claim | Code |
|---|---|
| 90/10 pretraining split, seed 42, seed-independent | `data/splits.py:28-51`, `training/config.py:19,24` |
| Fixed 125k steps, early stopping disabled | `configs/*.yaml` `max_steps`, `early_stopping_patience` |
| Exact-string dedup | `data/preprocessing.py::preprocess_sequences` |
| Paratope split = TDC default 70/10/20 seed 42 | `data/benchmarks/paratope.py:87-106` |
| TAP split = TDC default | `data/benchmarks/developability.py:82-140` |
| Contact map 60/20/20 by PDB | `data/benchmarks/contact_map.py:208-260` |
| Structure probe 60/20/20 by PDB, deduped | `data/benchmarks/structure_probe.py:197-263` |
| AB-Bind bin-packed by complex, 20 splits | `data/benchmarks/ab_bind.py:298-380`, `scripts/run_ab_bind_probe.py:146-156,196` |
| CoV-AbDab clonotype-grouped 70/15/15 | `data/benchmarks/binding.py:123-170` |
| Ng&Briney stratified 5-fold seed 42 | `scripts/run_ngbriney_cov.py:109-111,152-154` |
| Developability ridge nested 5-fold × 5 repeats | `scripts/run_developability_ridge.py:91-133` |
| Teacher and probe share a split function | `scripts/train_paratope_teacher.py:159`, `evaluation/downstream/paratope.py:26` |
| Contamination audit | `scripts/contamination_audit.py`, `comparison_outputs/contamination_audit.csv` |
