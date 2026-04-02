# Standardized Evaluation Framework for Antibody Masked-LM Masking Variants

A masking strategy only matters insofar as it changes *what the model learns* and *what it can transfer to*. In protein language modeling broadly, a core motivation for standardized benchmarks is that otherwise results become fragmented across datasets, splits, and metrics, making it hard to attribute gains to a modeling choice rather than evaluation variance [1]. 

Antibodies add an extra complication: many general protein benchmarks historically excluded antibody data, which makes it difficult to evaluate antibody-specific representation learning with comparable rigor [2]. The good news is that antibody resources and benchmarks have matured: curated structural antibody databases enable interface labeling [3], antibody-specific language models and paratope evaluations exist [4,5], and newer antibody-centric benchmark suites target binding/fitness and developability at scale [2]. This report defines a standardized downstream evaluation suite (tasks + datasets + splits + metrics + implementation) intended to compare **CDR-focused**, **span-masking**, **structure-aware**, **paratope/interface-aware**, **germline/SHM-hotspot-aware**, **multispecific-aware**, and **hybrid** masking strategies under *identical* model architecture, data, and compute budgets.

Key standardization principles used throughout (and reused per task):

- **Canonical region annotation**: delineate FR/CDR boundaries via **IMGT** definitions [30] and/or number sequences with **ANARCI** (which supports IMGT and other numbering schemes) [28].  
- **Leakage control as a first-class axis**: provide **two split regimes** where possible: (i) **random** (in-distribution) and (ii) **cluster/identity-based** (OOD) to reduce overly optimistic generalization (a central lesson emphasized in benchmark design work) [1,17]. 
- **Three evaluation tiers per task**: **zero-shot** (no supervised training), **linear probe** (frozen encoder + simple head), and **full fine-tune**, mirroring best practice in protein LM evaluation [1,16]. 
- **Report uncertainty**: at minimum, multiple random seeds for supervised heads and confidence intervals/bootstraps for test metrics; this is especially important for small structural datasets (e.g., interface labels) [4,12]. 

## Paratope prediction

**What it tests.** Paratope prediction tests whether the model localizes residues that physically contact antigen (i.e., binding site residues), a direct proxy for “does the representation encode interface geometry/chemistry?” [4,5]. This task is strongly motivated by the fact that while CDRs are the main binding loops, only a subset of residues participate in binding and some binding residues can fall outside traditional CDR definitions; Parapred explicitly motivates paratope prediction because fewer than ~20 residues typically participate in binding within ~40–50 hypervariable residues and because mapping paratopes enables rational engineering without disrupting binding [4]. 

{"layout":"carousel","aspect_ratio":"16:9","query":["antibody antigen complex structure paratope epitope","IgG antibody CDR loops diagram","antibody paratope epitope interface close-up"],"num_per_query":1}

**Dataset sources and recommended standard datasets.**

- **TDC “SAbDab_Liberis” paratope task** (token-level classification): curated from Parapred and structural antibody data, distributed as a standardized ML task with a default random split [9].  
- **Parapred’s original construction** (recommended “gold-label” protocol when you control preprocessing): Parapred defines a binding residue as having **≥1 atom within 4.5 Å** of any antigen atom [4]. It reports a dataset of **1,662 CDR sequences** derived from **277 Ab–Ag complexes × 6 CDRs** [4]. 
- **Structural source of truth**: SAbDab is an open-access curated repository of antibody structures, designed to facilitate standardized structural datasets and updated regularly [3].

**Label definition (standardized).** For an antibody chain with residues i=1,\ldots,L, define:

y_i=
\begin{cases}
1,& \min\limits_{a\in \text{atoms}(i),\ b\in \text{atoms(Ag)}} r_a-r_b\le 4.5\ \text{Å}
0,& \text{otherwise}
\end{cases}

This is exactly the convention used by Parapred [4]. 

**Implementation blueprint (token classification head).**  
Let the pretrained encoder output contextual embeddings h_i\in\mathbb{R}^d. A minimal head is:  
  
\hat p_i=\sigma(w^\top h_i + b)  
  
Train with weighted binary cross-entropy to handle extreme class imbalance:  
  
\mathcal{L}=-\sum_{i=1}^L\left[\alphay_i\log \hat p_i + (1-y_i)\log(1-\hat p_i)\right]  
  
Parapred itself evaluates residue-level classification and reports metrics including F-score and MCC after choosing a threshold (see below) [4].

**Standard splits and leakage control.**

- **Baseline**: use TDC’s random split for comparability and speed [9].
- **OOD split (recommended)**: cluster antibodies by sequence identity (e.g., MMseqs2 clusters) and ensure no cluster crosses train/test, mirroring core benchmark logic for biological generalization [1].
- **Region-aware stratification**: ensure heavy/light representation balance because light-chain paratope positions are fewer and skew can bias recall, consistent with observations in antibody paratope modeling discussions [5]. 

**Metrics (standardized).**

- **Primary**: AUPRC (preferred under heavy imbalance) + AUROC (secondary) [4,5].   
- **Thresholded**: F1 and MCC at a threshold chosen on validation; Parapred uses a threshold obtained by maximizing Youden’s index for certain analyses and reports F-score and MCC based on that thresholding approach [4]. 
- **Granular reporting**: per-loop (H1/H2/H3/L1/L2/L3) and per-chain, because Parapred reports loop-specific ROC AUC and antibody LMs often show loop-dependent behavior [4]. 

**Masking strategies this task is most sensitive to (expected).** Interface/paratope masking and structure-aware masking should most directly help, because they explicitly bias learning toward interface residues or local 3D neighborhoods; CDR-focused and span masking likely help secondarily because paratopes are enriched in CDRs [4,5]. 

## Binding specificity

**What it tests.** Binding specificity evaluation asks whether the model can distinguish *what* an antibody binds, i.e., whether sequence representations carry information that correlates with antigen class or epitope region [5].  This is a core unsolved problem in repertoire analysis: predicting binding specificity from sequence alone is challenging, and many analyses historically over-focus on CDRH3 despite binding depending on broader context [5]. 

**Dataset sources and recommended standard datasets.**

- **mBLM antibody specificity benchmark**: “An explainable language model for antibody specificity” describes a curated dataset of **5,561 human antibodies**, labeled into **seven specificity categories** spanning influenza HA head/stem and several viral targets, and reports classification performance (best F1 reported in the paper summary) [7]. 
- **CoV-AbDab**: a curated coronavirus antibody database with sequences and metadata (cross-neutralization evidence, origin, germline assignments, epitope region, and links to structures/homology models). As of Aug 5, 2020 it included **1,402** anti-coronavirus antibodies/nanobodies, of which **1,131** bind SARS‑CoV‑2 [6]. 
- **Optional paired Ab–Ag affinity proxy** (when you want an antibody+antigen sequence task): TDC’s **AntibodyAff (Protein_SAbDab)** defines a regression task over **493 Ab–Ag pairs**, recommends log-transforming affinity, and uses a random split [10].
- **Complex-level benchmark context**: AbBiBench argues that evaluating antibodies in isolation can be misleading and proposes evaluating models via Ab–Ag complex likelihood vs experimental affinity; it curates **>184,500 experimental measurements** across **14 antibodies** and **9 antigens**, and explicitly treats the Ab–Ag complex as the evaluation unit [31]. 

**Task formulations (standardized).**

**Single-antibody multiclass specificity (recommended primary).**  
Given antibody sequence(s) x (heavy only, light only, or paired VH/VL), predict antigen class y \in 1,\dots,K (e.g., influenza head vs stem vs SARS‑CoV‑2 epitope regions), matching mBLM-style setups [7].   
Model head:

\hat p(y=k\mid x)=\text{softmax}*k\left(Wg(h*{1:L})+b\right)

where g(\cdot) is mean pooling over regions (e.g., CDR-only pooling using IMGT boundaries) [30] or a learned [CLS]-style pooling [5]. 

**Antibody–antigen paired scoring (optional extension).**  
Using TDC AntibodyAff, encode antibody x^{Ab} and antigen x^{Ag}, then predict \log K_D or related affinity [10]. A standardized two-tower baseline is:

s(x^{Ab},x^{Ag})=\langle f_\theta(x^{Ab}), g_\phi(x^{Ag})\rangle

and regress \hat y = as+b to \log affinity, as TDC recommends log-transforming [10]. 

**Splits and leakage control.**

- **mBLM protocol replication**: replicate its identity-controlled split (reported as controlling maximal train–test similarity) to avoid trivial memorization [7].  
- **CoV-AbDab OOD split**: hold out by (i) **epitope region** (e.g., RBD vs NTD) or (ii) **V-gene family** or (iii) **time-split by publication/patent year** to test robustness to novelty; CoV-AbDab provides epitope-region and germline metadata necessary to implement these splits [6]. 

**Metrics (standardized).**

- **Multiclass**: macro-F1, per-class AUROC (one-vs-rest), and expected calibration error (ECE) if using predictions operationally; mBLM reports F1-style endpoints as a primary summary [7]. citeturn0search2  
- **Binary specificity** (e.g., SARS‑CoV‑2 binder vs non-binder): AUROC + AUPRC [6]. citeturn39view0  
- **Paired affinity regression**: Spearman’s \rho and Pearson’s r between predicted and measured \log K_D, consistent with common practice in fitness/affinity benchmarking [16,31]. citeturn14search15turn28view0

**Masking strategies this task is most sensitive to (expected).** CDR-focused and interface-aware masking should help most, because determinants of specificity concentrate in binding loops and interface residues; multispecific-aware masking should help when labels depend on joint VH/VL context rather than heavy chain alone [5,6]. citeturn30view0turn39view0  

## Mutation-effect prediction

**What it tests.** Mutation-effect prediction evaluates whether the learned distribution assigns higher probability (or better supervised predictions) to variants with improved binding/fitness and lower to deleterious variants—i.e., whether the model captures parts of the antibody fitness landscape [12,14,16]. citeturn13view0turn3view4turn14search15

**Dataset sources and recommended standard datasets.**

- **AB-Bind**: a curated dataset of **1,101 mutants** with experimentally measured binding free energy changes \Delta\Delta G across **32 complexes**; includes multiple assay types and both single and multiple mutations [12]. citeturn13view0  
- **SKEMPI 2.0**: **7,085** curated mutations for structurally resolved protein–protein interactions; provides cleaned PDBs and annotations including mutation location classes and homologous clusters (including antibody/antigen interactions) [13]. citeturn26view0  
- **Tite-Seq**: an experimental approach that measures binding titration curves and corresponding affinities for **thousands of antibody variants in parallel**, demonstrated on CDR1H and CDR3H of a scFv; importantly, titration curves help decouple affinity from expression/stability confounds common in DMS [14]. citeturn3view4  
- **AbBiBench** (strongly recommended if you want a modern consolidated benchmark): curates **>184,500** experimental affinity-related measurements across **9 antigens** and **14 antibodies**, and evaluates models via correlation between model likelihood and experimental affinity; it also emphasizes leakage control at the mutant-complex level [31]. citeturn28view0

**Two standardized evaluation modes.**

**Zero-shot mutation scoring via pseudo-log-likelihood (PLL).**  
Protein LMs can predict mutational effects in a zero-shot manner by using likelihood-based scoring; Meier et al. show that language models can capture mutational effects without supervision [15]. citeturn14search0turn14search4  
For an MLM, define PLL for a sequence x of length L:

\text{PLL}(x)=\sum_{i=1}^{L}\log p_\theta(x_i\mid x_{\setminus i})

where x_{\setminus i} indicates the sequence with position i masked (or marginalized), following standard masked-scoring practice in this line of work [15]. citeturn14search0  
For a mutant x^{(mut)} and wild-type x^{(wt)}:

s_{\text{mut}}=\text{PLL}(x^{(mut)})-\text{PLL}(x^{(wt)})

Then compare s_{\text{mut}} to experimental \Delta\Delta G or \log K_D using rank correlation, consistent with likelihood-fitness benchmarking frameworks [15,16]. citeturn14search0turn14search15  

**Supervised regression/classification.**  
A simple supervised head predicts a continuous fitness label (e.g., \Delta\Delta G, \log K_D):

\hat y = w^\top g(h_{1:L}) + b, \quad \mathcal{L}=y-\hat y_2^2

AB-Bind itself emphasizes both regression correlation limitations and binary classification utility (improved vs weakened binders) and reports ROC-AUC-style performance in that framing [12]. citeturn13view0  

**Splits and generalization controls.**

- **Per-complex split (key for AB-Bind)**: hold out entire parent complexes so the model must generalize across interfaces rather than interpolate within one antibody system [12]. citeturn13view0  
- **Homology-aware split (key for SKEMPI 2.0)**: use SKEMPI’s homologous-interaction annotations/clusters to avoid training and testing on homologous interfaces, which SKEMPI explicitly provides to help avoid overfitting and overly optimistic generalization [13]. citeturn26view0  
- **Landscape split (key for Tite-Seq / DMS)**: follow fitness benchmark best practice (e.g., neighborhood vs distant variants) to evaluate extrapolation beyond single-mutation neighborhoods, aligning conceptually with FLIP-style splitting logic [17]. citeturn14search3  
- **Leakage-aware benchmark (AbBiBench)**: AbBiBench explicitly aims to avoid leakage by ensuring the evaluated mutant Ab–Ag complexes are not present in training corpora even if wild-type components exist in public resources [31]. citeturn28view0

**Metrics (standardized).**

- **Regression**: Spearman \rho (primary) + Pearson r (secondary), and MAE/RMSE (diagnostic) [16,31]. citeturn14search15turn28view0  
- **Binary classification**: AUROC and AUPRC; AB-Bind reports ROC AUC for improved vs weakened binders and notes best AUC values for strong-effect subsets [12]. citeturn13view0  
- **DMS-style evaluation**: for enrichment-based or proxy affinity labels, report Spearman \rho and stratify by mutation count (1-mut vs multi-mut), consistent with benchmark practice for fitness landscapes [17]. citeturn14search3

**Masking strategies this task is most sensitive to (expected).** Germline/SHM-hotspot-aware and CDR-focused masking should have direct impact because mutation effects are concentrated in affinity maturation regions; span masking can help because variants often correspond to localized loop edits; structure-aware and interface-aware masking should help in datasets where changes reflect interface packing/geometry (AB-Bind, SKEMPI antibody/antigen subsets) [12,13,14]. citeturn13view0turn26view0turn3view4  

## Developability

**What it tests.** Developability tasks test whether representations encode signals related to manufacturability and clinical viability—properties like hydrophobic patches, charge balance, self-association, viscosity risk, aggregation propensity, thermostability, polyreactivity, and expression titer [18,2,19]. citeturn6view1turn6view2turn21view0

**Dataset sources and recommended standard datasets.**

- **TAP guidelines as computed labels**: “Five computational developability guidelines for therapeutic antibody profiling” derives guideline cutoffs for **five computed metrics**: total CDR length; extent/magnitude of surface hydrophobicity; CDR positive charge; CDR negative charge; and heavy–light charge asymmetry. It also provides the TAP tool to compute these measures from sequences via modeled structures [18]. citeturn6view1  
- **TDC Developability task**: standardizes developability prediction into datasets including **TAP (242 antibodies; regression on the five TAP metrics)** and “SAbDab_Chen” (a binary developability label derived from BIOVIA pipelines, with dataset selection criteria and statistics described) [11]. citeturn27view0  
- **FLAb2**: compiles developability/fitness labels at scale, stating it integrates datasets to yield a database of **~4M antibodies across 32 studies** and frames antibody fitness landscapes across multiple developability categories (e.g., thermostability, expression, aggregation, polyreactivity, immunogenicity, etc.) [2]. citeturn6view2  
- **GDPa1 dataset (Ginkgo Datapoints)**: provides assay data and metadata columns including cluster-based CV folds; its dataset card describes assays such as SEC aggregation, nanoDSF/DSF thermostability, HIC hydrophobicity, AC-SINS self-association, and polyreactivity measures, and provides recommended cluster/isotype-stratified folds for reporting [20]. citeturn36view1  
- **Blinded external test benchmark (AbDev competition)**: the 2025 AbDev competition reports a public training set of **246 antibodies** and a blinded held-out test set of **80 antibodies**, evaluated across five properties using Spearman correlation; it explicitly highlights overfitting and limited OOD generalization as a central finding [19]. citeturn21view0

**Standard task formulations.**

**Multi-target regression (recommended primary).**  
Given paired VH/VL sequences, predict a vector of developability assay outputs:

\hat{\mathbf{y}}=Wg(h^{VH},h^{VL})+b,\quad \mathbf{y}\in\mathbb{R}^m

where g concatenates pooled heavy/light embeddings (e.g., mean pooling over IMGT-defined regions) [30] or uses a learned pooling [18]. citeturn33search3turn6view1  
Loss:

\mathcal{L}=\sum_{j=1}^m \lambda_j(y_j-\hat y_j)^2

Choose m based on dataset: m=5 for TAP metrics (TDC TAP) [11], and m\ge 5 for GDPa1 subsets (as defined by the dataset card and competition) [19,20]. citeturn27view0turn21view0turn36view1  

**Binary developability classification (optional).**  
Use TDC “SAbDab_Chen” (binary label derived externally) when you need a classification endpoint [11]. citeturn27view0  

**Splits and reporting (standardized).**

- **Random split baseline**: use TDC’s provided random splits for TAP and SAbDab_Chen to enable community comparability [11]. citeturn27view0  
- **Cluster-based OOD split (recommended)**: GDPa1 explicitly provides hierarchical clustering folds (including isotype-stratified clustering) and recommends reporting on the isotype-stratified cluster fold for generalization [20]. citeturn36view1  
- **True held-out test**: adopt the AbDev competition pattern—train/validate on public set, evaluate on a held-out OOD test set—to detect overfitting (the competition reports systematically higher CV than held-out performance) [19]. citeturn21view0

**Metrics (standardized).**

- **Primary**: Spearman’s \rho per assay/property, because developability measurements are often non-normally distributed and Spearman is used in the AbDev benchmark reporting [19] and described as appropriate when distributions are non-normal in related analyses [18]. citeturn21view0turn6view1  
- **Secondary**: MAE/RMSE per assay; for binary tasks, AUROC/AUPRC [11]. citeturn27view0  
- **Aggregate**: macro-average Spearman across assays, but always report per-assay values because the AbDev results vary strongly by assay [19]. citeturn21view0

**Masking strategies this task is most sensitive to (expected).** Hybrid strategies and those that emphasize regions implicated in developability (often CDR physicochemical properties and surface patches) should help most; TAP explicitly motivates developability metrics tied to CDR hydrophobicity/charge and CDR length [18], while IgLM explicitly highlights improved in silico developability profiles for infilled CDR-H3 libraries [26]. citeturn6view1turn40view0  

## Multispecific evaluation

**What it tests.** “Multispecific-aware” evaluation asks whether the model can represent *multi-chain constraints* and *multi-binding designs*—including cases where antibodies must satisfy more than one binding interaction or format constraint. Multispecific modalities like T-cell engagers operate by bridging immune and target cells and forming immunological synapses, which motivates evaluating models on multi-binding and format-aware settings rather than single-chain abstractions [27]. citeturn22search1turn22search2

Given limited standardized public “multispecific function” datasets, this section focuses on two concrete, currently implementable benchmarks that stress multi-chain reasoning:

**Benchmark A: cognate heavy–light pairing and compatibility.**  
Compatibility of heavy (H) and light (L) chains is biologically and therapeutically crucial; ImmunoMatch frames this explicitly as distinguishing cognate from random H–L pairs and notes sensitivity to sequence differences at the H–L interface [23]. citeturn38view3  

- **Positive data**: PairedAbNGS introduces a paired H/L database with **58 bioprojects and >14M assembled productive sequences**, designed for large-scale analysis of native pairings [22]. citeturn38view2  
- **Negative data**: ImmunoMatch constructs pseudo-negatives by swapping light chains across pairs (with constraints such as matching CDRL3 length) to simulate non-cognate pairings under realistic length distributions [23]. citeturn38view3

**Benchmark B: common-light-chain feasibility for bispecific formats.**  
A practical engineering problem for bispecific antibodies is designing a single light chain compatible with two heavy chains. The LICHEN repository explicitly provides a function that takes **two heavy sequences** and “generate[s] a common light sequence for a bispecific antibody,” and also exposes a heavy–light log-likelihood scoring interface [24]. citeturn38view1

**Standard task formulations.**

**A. Pair classification and retrieval.**  
Given (H,L), predict y\in0,1 (cognate vs non-cognate):

\hat p=\sigma\left(w^\top g(h^{H},h^{L})+b\right)

where g concatenates pooled heavy/light embeddings and optional interface-region pooling (e.g., IMGT FR/CDR segmentation) [30]. citeturn33search3  
For retrieval: given H, rank a candidate set L_k by score and measure top-K accuracy / MRR (standard for pairing tasks, consistent with the goal of reconstructing pairs described for ImmunoMatch) [23]. citeturn38view3  

**B. Likelihood-based common-light scoring (bispecific proxy).**  
Concatenate sequences with separators:

x=[\text{H}_1\ \text{SEP}\ \text{H}*2\ \text{SEP}\ \text{L}]

Define a conditional PLL-style score over the light chain positions:

s(\text{H}1,\text{H}2,\text{L})=\sum{i\in \text{positions(L)}}\log p\theta(x_i\mid x*{\setminus i})

This aligns with the idea of extracting log-likelihood scores for heavy–light pairing described in LICHEN tooling [24]. citeturn38view1  

**Splits and reporting (standardized).**

- **Cluster-split** by heavy-chain CDRH3 identity (or overall VH identity) so that test heavy chains are not near-duplicates of train, protecting against memorization [1]. citeturn31view0  
- **Cross-source split** (PairedAbNGS bioproject holdout): hold out entire projects to test robustness across sequencing protocols and cohorts, consistent with the fact PairedAbNGS aggregates many bioprojects [22]. citeturn38view2

**Metrics (standardized).**

- **Binary classification**: AUROC + AUPRC (primary), calibration (secondary) [23]. citeturn38view3  
- **Retrieval**: MRR and top-K accuracy.  
- **Bispecific common-light proxy**: report average of s(\text{H}_1,\text{H}_2,\text{L}) across held-out heavy pairs and compare against baselines (e.g., random light, donor light). LICHEN’s interface explicitly supports this style of likelihood scoring [24]. citeturn38view1

**Masking strategies this task is most sensitive to (expected).** Multispecific-aware masking should help most because it forces joint modeling across chains; structure-aware masking may help because H–L compatibility is mediated by interface residues and geometry, consistent with ImmunoMatch’s sensitivity to H–L interface differences [23]. citeturn38view3  

## Antibody sequence infilling and restoration

**What it tests.** Infilling/restoration tasks probe whether the LM has learned robust conditional distributions over antibody sequence spans—especially CDR loops—and whether it can reconstruct plausible antibody-like content when segments are missing. This is tightly aligned with span masking objectives and highly diagnostic for CDR-focused masking variants [25,26]. citeturn23view1turn40view0

**Dataset sources and recommended standard datasets.**

- **AbLang restoration benchmark**: AbLang is positioned explicitly around completing missing antibody residues and reports that a large fraction of repertoire sequences can be missing N-terminus residues; it describes masking positions (e.g., 1–30) after IMGT numbering via ANARCI and evaluating restoration accuracy [25]. citeturn23view1  
- **IgLM infilling framing**: IgLM is described as a text-infilling generative LM trained on **558M natural antibody sequences** and highlights that its infilling formulation enables redesign of variable-length spans and can generate infilled CDR-H3 libraries with improved developability profiles [26]. citeturn40view0  
- **Region boundary definitions**: use IMGT FR/CDR definitions to standardize what counts as “CDR-H3 span” etc [30], and ANARCI to apply numbering at scale [28]. citeturn33search3turn33search1

**Standard infilling tasks (recommended minimal suite).**

**Task A: N-terminus restoration (AbLang-style).**  
Number sequences (IMGT scheme), mask a prefix region (e.g., positions 1–30), then predict masked residues and compute restoration accuracy, matching AbLang’s described evaluation procedure [25]. citeturn23view1  

**Task B: random scattered residue restoration.**  
Mask k\in1,5,10 random positions and compute percent correct recovery, also mirroring AbLang’s described evaluation setup [25]. citeturn23view1  

**Task C: CDR span infilling (span-masking diagnostic).**  
Using ANARCI+IMGT boundaries, mask a full CDR (e.g., CDRH3) and evaluate conditional generation quality; this directly reflects IgLM’s emphasis on infilling CDR loop libraries [26]. citeturn33search1turn33search3turn40view0  

**Scoring and metrics (standardized).**

- **Token accuracy**:

\text{Acc}=\frac{1}{|M|}\sum_{i\in M}\mathbf{1}[\hat x_i = x_i]

consistent with restoration framing used by AbLang (percent correctly predicted amino acids) [25]. citeturn23view1  
- **Span-level exact match**:

\text{EM}=\mathbf{1}[\hat x_{M}=x_{M}]

reported per-span (especially meaningful for CDRH3) and stratified by span length (CDR lengths vary) [30]. citeturn33search3  
- **Conditional negative log-likelihood (NLL)** over masked positions (primary zero-shot metric for an MLM), and **edit distance** (diagnostic) for spans, especially if using sampling/decoding.  
- **Downstream plausibility checks**: evaluate generated spans with developability predictors/metrics (e.g., TAP metrics or GDPa1-derived regressors) since IgLM explicitly emphasizes developability improvements for infilled libraries [18,26]. citeturn6view1turn40view0

**Masking strategies this task is most sensitive to (expected).** Span masking and CDR-focused masking should show the largest gains on CDR infilling and exact-match metrics, because these objectives directly train “replace spans conditioned on context,” paralleling IgLM’s infilling motivation and AbLang’s completion framing [25,26]. citeturn23view1turn40view0  

## Structure and contact awareness

**What it tests.** Structure/contact awareness tasks test whether learned representations encode information predictive of 3D proximity—an essential capability if the masking strategy explicitly injects structural neighborhoods or adjacency. This is a direct analog of classic protein LM structure-probing tasks (e.g., contact prediction in TAPE) [1]. citeturn31view0turn32view1

**Dataset sources and recommended standard datasets.**

- **Structural ground truth**: SAbDab provides curated antibody structures from the Protein Data Bank and is explicitly designed to support standardized structural dataset creation [3]. citeturn23view3  
- **Contact prediction evaluation protocol**: TAPE defines contact labels as residue pairs within **8 Å** and reports **precision@L/5** for medium/long-range contacts as a standardized metric (aligned with CASP reporting) [1]. citeturn32view0turn32view1

**Standard antibody-structure probing tasks.**

**Task A: intra-chain contact prediction (VH and VL separately).**  
Given a variable domain sequence x of length L, define contacts:

y_{ij}=\mathbf{1}[d(i,j)<8\ \text{Å}],\quad i<j

mirroring TAPE’s definition [1]. citeturn32view0

**Task B: inter-chain VH–VL contact prediction (paired setting).**  
For paired sequences, predict which VH residues contact which VL residues in the assembled Fv; this directly probes whether multi-chain representations encode interface packing, consistent with the importance of VH–VL orientation and contacts in antibody structure resources [3]. citeturn23view3

**Implementation blueprint (pairwise head).**
Given residue embeddings h_i,h_j, build a pair representation:

r_{ij}=[h_i;\ h_j;\ |h_i-h_j|;\ h_i\odot h_j]

and predict:

\hat p_{ij}=\sigma(w^\top r_{ij}+b)

Train with BCE over sampled pairs to control O(L^2) cost. This mirrors standard contact-prediction framing in benchmarks like TAPE (pairwise classification) [1]. citeturn32view0  

**Splits and leakage control.**

- **Identity filtering / cluster split**: TAPE emphasizes preventing information leakage across evolutionary relationships and describes identity-filtered splits for broad generalization [1]. Apply the same principle to antibody variable domains by clustering sequences and holding out clusters. citeturn31view0  
- **Structure-quality filtering**: if deriving contacts from experimental structures, apply resolution/method filters consistent with structural dataset practices; TDC’s developability dataset describes using crystal structures with resolution <3 Å when curating structure-quality antibody subsets, illustrating how resolution filters get used in antibody ML datasets [11]. citeturn27view0

**Metrics (standardized).**

- **Primary**: precision@L/5 on medium- and long-range contacts, exactly as specified in TAPE [1]. citeturn32view0turn32view1  
- **Secondary**: AUPRC over all contact pairs and precision@L and precision@L/2, mirroring TAPE’s expanded reporting [1]. citeturn32view0  
- **Inter-chain specificity**: report VH–VL contact precision separately from intra-chain, because multispecific-aware masking may primarily affect inter-chain modeling [22,23]. citeturn38view2turn38view3

**Masking strategies this task is most sensitive to (expected).** Structure-aware masking should show the clearest gains here because it explicitly conditions learning on structural neighborhoods; multispecific-aware masking may improve VH–VL contact prediction by strengthening inter-chain context modeling, consistent with pairing/interface sensitivity described in ImmunoMatch [23]. citeturn38view3  

## References

```text
[1] Rao R, Bhattacharya N, Thomas N, et al. “Evaluating Protein Transfer Learning with TAPE.” NeurIPS (2019).
    https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf

[2] Chungyoun M, et al. “Fitness Landscape for Antibodies 2: Benchmarking Reveals That Protein AI Models Cannot Yet Consistently Predict Developability Properties.” (2025).
    https://pmc.ncbi.nlm.nih.gov/articles/PMC12767642/

[3] Dunbar J, Krawczyk K, Leem J, et al. “SAbDab: the structural antibody database.” Nucleic Acids Research (2014).
    https://pmc.ncbi.nlm.nih.gov/articles/PMC3965125/

[4] Liberis E, Veličković P, Sormanni P, Vendruscolo M, Lio’ P. “Parapred: antibody paratope prediction using convolutional and recurrent neural networks.” Bioinformatics (2018).
    https://academic.oup.com/bioinformatics/article-pdf/34/17/2944/50581998/bioinformatics_34_17_2944.pdf

[5] Leem J, Mitchell LS, Farmery JHR, Barton J, Galson JD. “Deciphering the language of antibodies using self-supervised learning.” Patterns (2022).
    https://pmc.ncbi.nlm.nih.gov/articles/PMC9278498/

[6] Raybould MIJ, Kovaltsuk A, Marks C, Deane CM. “CoV-AbDab: the Coronavirus Antibody Database.” Bioinformatics (2020).
    https://pmc.ncbi.nlm.nih.gov/articles/PMC7558925/

[7] Wang Y, et al. “An explainable language model for antibody specificity.” (mBLM specificity dataset; paper summarizes 5,561 antibodies and 7 specificity categories).
    https://research.google/pubs/an-explainable-language-model-for-antibody-specificity     

[8] Huang K, et al. “Artificial intelligence foundation for therapeutic science.” (Therapeutics Data Commons overview) (2022).
    https://pmc.ncbi.nlm.nih.gov/articles/PMC9529840/

[9] Therapeutics Data Commons. “Paratope Prediction Task Overview (SAbDab_Liberis).”
    https://tdcommons.ai/single_pred_tasks/paratope/

[10] Therapeutics Data Commons. “Antibody-antigen Affinity Prediction Task Overview (Protein_SAbDab).”
     https://tdcommons.ai/multi_pred_tasks/antibodyaff/

[11] Therapeutics Data Commons. “Antibody Developability Prediction Task Overview (TAP; SAbDab_Chen).”
     https://tdcommons.ai/single_pred_tasks/develop/

[12] Sirin S, Apgar JR, Bennett EM, Keating AE. “AB-Bind: Antibody binding mutational database for computational affinity predictions.” Protein Science (2016).
     https://pmc.ncbi.nlm.nih.gov/articles/PMC4815335/

[13] Jankauskaitė J, Jiménez-García B, Dapkūnas J, Fernández-Recio J, Moal IH. “SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy, kinetics and thermodynamics upon mutation.” Bioinformatics (2019).
     https://pmc.ncbi.nlm.nih.gov/articles/PMC6361233/

[14] Adams RM, Mora T, Walczak AM, Kinney JB. “Measuring the sequence-affinity landscape of antibodies with massively parallel titration curves (Tite-Seq).” (2016/2017).
     https://pmc.ncbi.nlm.nih.gov/articles/PMC5268739/

[15] Meier J, Rao R, Verkuil R, Liu J, Sercu T, Rives A. “Language models enable zero-shot prediction of the effects of mutations on protein function.” NeurIPS (2021).
     https://proceedings.neurips.cc/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html

[16] Notin P, et al. “ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction.” (2023).
     https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/

[17] Dallago C, et al. “Benchmark tasks in fitness landscape inference for proteins (FLIP).” (2021).
     https://www.biorxiv.org/content/10.1101/2021.11.09.467890v2.full.pdf

[18] Raybould MIJ, Marks C, Krawczyk K, et al. “Five computational developability guidelines for therapeutic antibody profiling.” PNAS (2019).
     https://pmc.ncbi.nlm.nih.gov/articles/PMC6410772/

[19] van Niekerk L, et al. “2025 Ginkgo Datapoints Antibody Developability Competition outcomes: limited model performance and a call for data standardization.” mAbs (2026).
     https://www.researchgate.net/publication/401061076_2025_Ginkgo_Datapoints_Antibody_Developability_Competition_outcomes_limited_model_performance_and_a_call_for_data_standardization

[20] Ginkgo Datapoints / Hugging Face dataset card. “GDPa1: Antibody developability dataset.” (assays + suggested cluster/isotype folds described on the card)
     https://huggingface.co/datasets/ginkgo-datapoints/GDPa1

[21] Raybould MIJ, Marks C, Krawczyk K, et al. “Thera-SAbDab: the Therapeutic Structural Antibody Database.” (2020).
     https://pmc.ncbi.nlm.nih.gov/articles/PMC6943036/

[22] Dudzic P, et al. “Conserved heavy/light contacts and germline preferences revealed by a large-scale analysis of natively paired human antibody sequences and structural data.” (PairedAbNGS) (2025).
     https://www.nature.com/articles/s42003-025-08388-y.pdf

[23] Guo D, Dunn-Walters DK, Fraternali F, Ng JCF, et al. “ImmunoMatch learns and predicts cognate pairing of heavy and light immunoglobulin chains.” Nature Methods (2026).
     https://www.nature.com/articles/s41592-025-02913-x

[24] oxpig (repo). “LICHEN: Light-chain Immunoglobulin sequence generation Conditioned on the Heavy chain and Experimental Needs.” (includes bispecific common light-chain generation interface)
     https://github.com/oxpig/LICHEN

[25] Olsen TH, Moal I, Deane CM. “AbLang: an antibody language model for completing antibody sequences.” Bioinformatics Advances (2022).
     https://academic.oup.com/bioinformaticsadvances/article-pdf/2/1/vbac046/47086264/vbac046.pdf

[26] Shuai RW, Ruffolo JA, Gray JJ. “IgLM: Infilling language modeling for antibody sequence design.” Cell Systems (2023).
     https://www.sciencedirect.com/science/article/pii/S2405471223002715

[27] Kang X, et al. “Bispecific and multispecific T-cell engagers.” (review; emphasizes immunological synapse formation) (2025).
     https://pmc.ncbi.nlm.nih.gov/articles/PMC12813292/

[28] Dunbar J, Deane CM. “ANARCI: antigen receptor numbering and receptor classification.” Bioinformatics (2016).
     https://pmc.ncbi.nlm.nih.gov/articles/PMC4708101/

[29] Ye J, Ma N, Madden TL, Ostell JM. “IgBLAST: an immunoglobulin variable domain sequence analysis tool.” Nucleic Acids Research (2013).
     https://academic.oup.com/nar/article/41/W1/W34/1097536

[30] IMGT. “Definition of the FR-IMGT and CDR-IMGT regions.” (web reference for standardized FR/CDR boundaries)
     https://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html
```

