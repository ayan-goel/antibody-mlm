# Antibody MLM Masking Strategies and Implementation Methodology

## Shared setup for all strategies

Antibodies (immunoglobulins) are multi-chain proteins whose antigen specificity is primarily determined by the variable domains of a heavy chain (VH) and a light chain (VL). The antigen-binding site (paratope) is shaped largely by six hypervariable loops called complementarity-determining regions (CDRs): three in VH and three in VL. In structural antibody resources, five of the six CDR loops often fall into “canonical” conformational families, while CDR H3 is notably more variable and harder to model. These facts matter because any masking scheme that concentrates learning on CDR loops (especially CDR H3) explicitly targets the parts of the sequence space where antibodies express the most diversity and binding-relevant variation. [8] citeturn16view0

Training antibody-family language models at scale is feasible because large, curated repertoire sequence databases exist, foremost the Observed Antibody Space (OAS), which provides both unpaired and paired antibody sequence data through a web interface and specialized downloads. OAS also has an associated peer-reviewed description noting that the database was created to provide cleaned/annotated repertoire data and later updated to accommodate growing volumes and the appearance of paired VH/VL sequencing data. [9–10] citeturn6search2turn21view0 Experimental structures and antibody–antigen complexes are available through structure-focused databases such as SAbDab (structures, metadata, heavy/light pairing, antigen details, and in some cases affinity), as well as newer curated complex datasets that include explicit interface annotations (e.g., AACDB, which provides interaction residues via ΔSASA and atom-distance definitions). [8,16] citeturn16view0turn8view0

For all masking strategies below, assume a backbone masked language modeling (MLM) objective in the BERT family:

- Let an input token sequence be \(x = (x_1,\dots,x_L)\), where tokens are amino acids plus special tokens (e.g., chain separators).
- A masking strategy samples a set of target indices \(M \subset \{1,\dots,L\}\) (or spans / structured groups that imply such a set).
- A corruption function produces \(\tilde{x}\) by replacing each \(x_i, i\in M\), with:
  - a dedicated mask token 80% of the time,
  - a random token 10% of the time,
  - the original token 10% of the time,
  while predicting the original \(x_i\) at all \(i\in M\).
- The (unweighted) MLM loss is:
\[
\mathcal{L}_{\text{MLM}}(\theta; x) \;=\; - \sum_{i\in M} \log p_\theta(x_i \mid \tilde{x}).
\]
This 15%-of-positions (“mask budget”) and 80/10/10 corruption scheme is described in the original BERT paper and is a clean default for proteins as well. [1] citeturn7view0

Because antibodies are not “generic proteins,” multiple groups have published antibody-specific LMs trained on OAS-scale data and evaluated on antibody tasks (e.g., AntiBERTa for paratope prediction and structural contact-related signals in attention, AbLang for restoring missing residues, and AbLang-2 for explicitly addressing germline bias). [3–4,17] citeturn12view0turn13view0turn22view0 These provide practical evidence that (a) antibody-family modeling benefits from domain-specific inductive biases and (b) masking choices can be tuned to antibody biology rather than copied verbatim from generic protein MLMs.

Practical standardization is essential before masking:

- **Numbering / region annotation:** Use an antibody numbering system (IMGT, Kabat, Chothia, etc.) and a tool that can apply it at scale. ANARCI is an open tool that classifies and numbers antibody variable domain sequences and supports multiple popular schemes (including IMGT). [11] citeturn29view0
- **IMGT FR/CDR boundaries:** IMGT provides explicit FR and CDR region definitions and fixed position ranges (e.g., FR1 positions 1–26; CDR1 positions 27–38; CDR2 positions 56–65; CDR3 positions 105–117 in the IMGT unique numbering for V-domains). [12] citeturn3search7
- **Paired-chain representation:** If using VH+VL, concatenate with separators and include chain-type embeddings (segment embeddings) so the model can distinguish chains, just as BERT distinguishes segments. Multi-chain modeling is increasingly central for realistic antibody design and pairing tasks, as reflected by paired-sequence resources and paired-sequence ML papers. [10,21–23] citeturn21view0turn10view0turn24view0turn25view0

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["antibody structure diagram Fab Fc Fv CDR loops","antibody variable domain CDR loops ribbon structure","VH VL antibody binding site schematic CDR"] ,"num_per_query":1}

## CDR-focused masking

**What it is.** Allocate the MLM mask budget disproportionately (or exclusively) to the six CDR loops, optionally with extra emphasis on CDR H3, rather than sampling mask positions uniformly over the whole variable domain.

**Why it is biologically motivated.** Structural antibody resources describe the variable domains as containing three CDRs each, and note the special difficulty/variability of CDR H3. SAbDab explicitly frames antigen recognition as involving these CDR loops, emphasizes canonical clusters for most CDRs, and highlights the unusual variability of H3. [8] citeturn16view0 IMGT provides standardized FR/CDR delineations that can be used reproducibly at scale. [12] citeturn3search7

**Required inputs and preprocessing.**
1. Input sequences should be variable-domain sequences (VH, VL) or full chains with variable-domain extracted.
2. Number each domain in a consistent scheme (recommended IMGT for unambiguous boundaries across lengths), using ANARCI. [11–12] citeturn29view0turn3search7
3. Define a region label function \(r(i)\in\{\text{FR1},\text{CDR1},\text{FR2},\text{CDR2},\text{FR3},\text{CDR3},\text{FR4},\text{special}\}\) for each residue position.

**Mask sampler formulation.** Let the desired global mask fraction be \(m\) (e.g., \(m=0.15\) following BERT). Define per-region weights:
- \(w_{\text{CDR}} > w_{\text{FR}}\), optionally split into loop-specific weights \(w_{\text{H3}}\) etc.

A simple fixed-budget sampler that guarantees exactly \(\lfloor mL \rfloor\) masked residues:
1. Compute unnormalized scores \(s_i = w_{r(i)}\).
2. Sample without replacement \(K=\lfloor mL \rfloor\) indices \(M\) from \(\{1,\dots,L\}\) with probability proportional to \(s_i\).

Equivalent “Bernoulli” form (often easier for distributed training) with an expected mask budget:
\[
p_i \;=\; \min\!\left(1,\; m \cdot \frac{w_{r(i)}}{\frac{1}{L}\sum_{j=1}^L w_{r(j)}} \right),
\qquad
M \sim \{ i \; | \; \text{Bernoulli}(p_i)=1 \}.
\]

**Choice of weights (practical defaults).** A common starting point is:
- \(w_{\text{FR}}=1\),
- \(w_{\text{CDR1}}=w_{\text{CDR2}}=3\),
- \(w_{\text{CDR3}}=w_{\text{H3}}=6\),
then tune based on validation objectives (e.g., paratope prediction accuracy or CDR reconstruction error). The rationale is anchored in structural antibody literature emphasizing CDR loops as binding determinants and CDR H3 as especially diverse. [8] citeturn16view0

**Loss and evaluation emphasis.** The base loss remains \(\mathcal{L}_{\text{MLM}}\). However, because CDR-focused masking intentionally changes which residues are predicted, you should track:
- CDR-only reconstruction perplexity and accuracy (especially CDR3),
- whole-sequence perplexity (to ensure the model does not collapse on FRs),
- downstream paratope labeling performance (AntiBERTa explicitly shows antibody LM fine-tuning to paratope prediction and reports binding-site related signals in attention). [3] citeturn12view0

**Edge cases and guardrails.**
- Very long CDR3 loops: IMGT inserts can occur; rely on numbering output rather than raw indices. IMGT explicitly uses lengths and gaps in its scheme to preserve equivalence. [12] citeturn3search7
- Framework contributions to binding: paratopes are often concentrated in CDRs, but not strictly limited to them. To avoid teaching the model that FRs are “irrelevant,” keep a nonzero FR masking probability (e.g., 20–40% of \(M\) from FRs), especially for developability tasks and VH–VL interface modeling. This point aligns with paired-chain interface studies showing meaningful framework contributions to interfaces. [22–23] citeturn24view0turn10view0

## Span masking

**What it is.** Mask *contiguous spans* of residues—rather than independent positions—following the SpanBERT idea of span masking and, optionally, a span-boundary objective (SBO). [2] citeturn7view1

**Why it fits antibodies.**
- Antibody design and maturation often involve coordinated changes in contiguous regions (CDR loops are contiguous segments in sequence space even though they act in 3D).
- Antibody generation papers explicitly adopt “infilling” formulations that redesign variable-length spans using bidirectional context; IgLM frames antibody design as text-infilling and demonstrates infilling of CDR loops (including CDR H3 libraries). [7] citeturn14view0

**Required inputs and preprocessing.**
- No special labels are required beyond the sequence.
- Optional: region annotations enable a *CDR-aligned span sampler* (mask whole subsegments inside CDRs more often than in FRs).

**SpanBERT-style span selection.** SpanBERT describes an iterative procedure: sample spans until a masking budget (e.g., 15%) is used. [2] citeturn7view1

A protein-adapted span sampler:
1. Set total budget \(B=\lfloor mL \rfloor\).
2. While masked_count < \(B\):
   - Sample a span length \(\ell\) from a truncated geometric distribution:
     \[
     \ell \sim \text{Geom}(p)\ \text{truncated to } [\ell_{\min},\ell_{\max}]
     \]
     (SpanBERT uses span sampling with a budget-driven process; the exact distribution is a design choice guided by its framing.) [2] citeturn7view1
   - Sample a start index \(s\) (respecting chain boundaries; do not allow spans to cross VH/VL separators unless you intentionally want cross-chain spans).
   - Mask indices \(s, s+1, \dots, s+\ell-1\) that are not already masked, clipping to the chain end.

**Span-boundary objective (SBO) adaptation to amino acids.** SpanBERT introduces an auxiliary loss encouraging boundary representations to predict the masked span content without using internal token representations. [2] citeturn7view1

For a masked span \([s,e]\), define boundary token indices \(b_\text{L}=s-1\) and \(b_\text{R}=e+1\) (or boundary special tokens if the span touches an edge). Let \(h_i\) be the transformer hidden state for token \(i\). For each position \(k \in [s,e]\), define an offset embedding \(\pi_{k-s+1}\). A minimal SBO head:
\[
u_k = \text{MLP}\big([h_{b_\text{L}}, h_{b_\text{R}}, \pi_{k-s+1}]\big),
\qquad
\mathcal{L}_{\text{SBO}} = -\sum_{k=s}^{e}\log p_\theta(x_k \mid u_k).
\]
Total loss:
\[
\mathcal{L} = \mathcal{L}_{\text{MLM}} + \lambda_{\text{SBO}}\mathcal{L}_{\text{SBO}}.
\]
This mirrors SpanBERT’s stated mechanism (span masking + span boundary prediction). [2] citeturn7view1

**Antibody-specific span variants (recommended).**
- **CDR-span masking:** force a fraction of spans to start within CDR1/2/3 and remain inside that region (using IMGT boundaries). [12] citeturn3search7
- **CDR H3-focused infilling:** dedicate some training batches to masking 30–100% of CDR H3 (single-span or multi-span) to match infilling use cases highlighted by IgLM. [7] citeturn14view0
- **Junction-aware spans:** if V(D)J junction indices are available (from IgBLAST; see the SHM section), over-sample spans that include the CDR3 junction region because it is generated by V(D)J joining and is the most diverse. [14] citeturn18view0

## Structure-aware masking using 3D neighborhoods

**What it is.** Use predicted (or experimental) antibody structure to define *spatial neighborhoods* of residues, and mask residues in these 3D neighborhoods together—rather than by 1D adjacency in sequence.

**Why it is justified now.**
- IgFold provides fast antibody structure prediction from sequence, using embeddings from an antibody LM pre-trained on hundreds of millions of antibody sequences and graph networks to predict backbone coordinates, reporting runtimes under ~25 seconds and providing per-residue quality estimates. [5] citeturn17view0
- The idea that local geometric context strongly determines amino-acid identity underpins structure-based modeling; for example, ProteinMPNN evaluates neighborhood sizes defined by nearest Cα neighbors and reports saturation of performance around 32–48 neighbors, supporting the use of \(k\)-NN neighborhoods as a meaningful structural context scale. [24] citeturn28view0

**Required inputs and preprocessing.**
1. Antibody sequence(s), ideally paired VH/VL for conventional antibodies.
2. Predicted structures from IgFold:
   - backbone coordinates (at least Cα; IgFold predicts backbone atoms and provides per-residue error estimates), which you store as \(R = \{ \mathbf{r}_i \in \mathbb{R}^3\}_{i=1}^L\). [5] citeturn17view0
3. Optional: confidence gating using IgFold’s per-residue error prediction (down-weight unreliable regions or add noise).

**Neighborhood definitions.** Let \(d(i,j) = \|\mathbf{r}_i - \mathbf{r}_j\|_2\), typically using Cα coordinates. Define one of:

- **k-nearest neighbors (kNN):**
  \[
  \mathcal{N}_k(i) = \text{arg sort}_{j\neq i}\; d(i,j)\ \text{take top } k.
  \]
  Choose \(k\in[24,48]\) as a starting range, supported by ProteinMPNN’s neighbor saturation finding. [24] citeturn28view0

- **Radius neighbors:**
  \[
  \mathcal{N}_\delta(i) = \{j \neq i: d(i,j) \le \delta\},
  \]
  with \(\delta\) typically 8–12 Å as a design choice (tune with validation).

**Mask sampler: “seed-and-grow” 3D masking.**
1. Sample \(S\) seed residues \(\{i_1,\dots,i_S\}\) from a seed distribution (uniform, or CDR-weighted if you want structure-aware + CDR-aware hybridization).
2. Set:
   \[
   M = \bigcup_{s=1}^S \left( \{i_s\} \cup \mathcal{N}_k(i_s) \right),
   \]
   and if \(|M| > B=\lfloor mL\rfloor\), subsample \(M\) down to \(B\) indices.
3. Apply BERT-style corruption and standard MLM loss. [1] citeturn7view0

**Confidence-aware masking (recommended with predicted structures).**
IgFold provides a per-residue accuracy estimate. [5] citeturn17view0 Let \(q_i\) be a confidence score (e.g., inverse predicted error). Use it to avoid overfitting to incorrect neighborhoods:
- Seed sampling: \(\Pr(i \text{ seed}) \propto q_i^\alpha\) (choose \(\alpha\le 1\) so low-confidence regions aren’t ignored entirely).
- Loss weighting:
  \[
  \mathcal{L} = -\sum_{i\in M} w_i \log p_\theta(x_i\mid \tilde{x}),
  \quad w_i = \text{clip}(q_i, w_{\min}, w_{\max}).
  \]

**What this trains the model to learn.**
AntiBERTa reports that attention maps correspond to structural contacts and binding sites, suggesting that even sequence-only self-supervision can internalize structural signals; structure-aware masking pushes this further by explicitly tying prediction targets to 3D context. [3] citeturn12view0

**Implementation notes.**
- Compute IgFold structures offline for large sequence corpora (store only coordinates + confidence) to avoid repeatedly running inference during training. IgFold makes large-scale structure prediction feasible because of its reported speed. [5] citeturn17view0
- Ensure neighborhoods do not cross chain boundaries *unless desired*. For VH+VL, you often want neighborhoods that include cross-chain contacts because VH–VL interactions shape the paratope and stability, and paired-chain studies highlight meaningful inter-chain contacts involving CDR3 and framework residues. [21–23] citeturn10view0turn24view0turn25view0

## Interface and paratope masking using antibody–antigen complexes

**What it is.** Use antibody–antigen complex structures (or paratope predictors) to identify the antibody’s binding interface residues (paratope) and bias masking toward those residues, effectively making the MLM focus on binding determinants rather than generic sequence recovery.

**Why this is strongly evidence-backed.**
- AntiBERTa explicitly states it can be fine-tuned for paratope prediction and highlights that attention maps correspond to binding sites. [3] citeturn12view0
- SAbDab provides curated antibody structural data including antigen details and, when available, binding affinity, enabling antibody–antigen structural supervision. [8] citeturn16view0
- AACDB was introduced specifically to provide curated antigen–antibody complexes and, critically, it provides detailed interacting residue information via both ΔSASA and atom-distance methods (and notes commonly used distance thresholds, with an example interface plot cutoff <6 Å and support for filtering at alternative thresholds like 5 Å). [16] citeturn8view0
- When antigen is not available, sequence-based paratope predictors exist; Parapred is a sequence-based method for paratope prediction that operates on hypervariable region sequence input without antigen information. [15] citeturn2search0

**Two label generation modes.**

**Mode A: Structure-derived paratope labels (preferred).**
Given a complex structure with antibody residues \(i\in \mathcal{A}\) and antigen residues \(j\in \mathcal{G}\), define an “interface contact” distance between residues by minimum heavy-atom distance:
\[
d_{\min}(i,j) = \min_{a\in \text{atoms}(i), b\in \text{atoms}(j)} \|\mathbf{r}_a-\mathbf{r}_b\|_2.
\]
Define the paratope set:
\[
P(\delta) = \left\{ i\in \mathcal{A} : \min_{j\in \mathcal{G}} d_{\min}(i,j) \le \delta \right\}.
\]
AACDB explicitly supports distance-based interacting residue definitions and references a commonly used <6 Å interaction plot cutoff while allowing users to filter at 5 Å if preferred. [16] citeturn8view0

Optionally also define epitope \(E(\delta)\) analogously for antigen residues. If your MLM includes antigen sequences as additional segments, epitope masking becomes possible as well (see below).

**Mode B: Predicted paratope labels (fallback).**
Use a paratope predictor to approximate \(P\). Parapred is directly motivated as a sequence-based probabilistic algorithm for paratope prediction and is commonly used as an antigen-agnostic approach. [15] citeturn2search0 AntiBERTa itself is also positioned as fine-tunable for paratope prediction. [3] citeturn12view0

**Mask sampler formulation.** Let \(P\subset\{1,\dots,L\}\) be the (true or predicted) paratope index set.

A minimal paratope-biased sampler:
\[
s_i =
\begin{cases}
w_P & i\in P,\\
w_{\neg P} & i\notin P,
\end{cases}
\quad w_P \gg w_{\neg P},
\]
then sample \(M\) via weighted sampling exactly as in CDR-focused masking.

A “paratope-only epochs” variant (often useful for deliberate specialization):
- With probability \(\rho\) per batch, restrict masking to \(P\) only:
  \[
  M \subseteq P,\quad |M|=\min(\lfloor mL\rfloor, |P|),
  \]
- Else do a mixed mask as above to maintain general language modeling competence.

**Multi-segment interface MLM (optional but powerful).**
If you jointly model antibody + antigen sequences, define input:
\[
x = [\text{CLS}, \text{Ab tokens}, \text{SEP}, \text{Ag tokens}, \text{SEP}],
\]
and define mask sets \(M_{\text{Ab}}\) and \(M_{\text{Ag}}\). You can:
- mask only paratope residues on antibody and predict them from antigen context,
- or mask epitope residues and predict from antibody context,
- or do both.
AACDB’s explicit residue-level interface annotations make supervised construction of \(P\) and \(E\) feasible at scale. [16] citeturn8view01

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["antibody antigen complex structure paratope epitope","antigen antibody interface residues visualization","Fab antigen complex ribbon structure"],"num_per_query":1}

**Guardrails against label noise.**
- If using predicted paratopes (Mode B), treat \(P\) as a soft set with confidence \(c_i\in[0,1]\) and incorporate that into weights:
  \[
  s_i = w_{\neg P} + (w_P-w_{\neg P})c_i.
  \]
- If using structural contacts, consider dual definitions:
  - ΔSASA-based interface residues, and
  - distance-based interface residues,
  since AACDB highlights both and the absence of a universal interface definition across studies. [16] citeturn8view0

## Germline and SHM hotspot masking using germline alignment

**What it is.** Use germline alignment to identify (i) positions that mutated away from germline (nongermline residues in the mature antibody) and (ii) somatic hypermutation (SHM) hotspot contexts, then bias masking toward those biologically “mutation-active” positions.

**Why it is necessary for modern antibody LMs.**
- IgBLAST is explicitly designed for immunoglobulin sequence analysis, including matching to germline V/D/J genes, delineating FR and CDR regions, and supporting both nucleotide and protein query sequences in batch. [14] citeturn18view0
- Germline bias is a recognized challenge for antibody LMs: AbLang-2 frames antibody diversity as arising from V(D)J recombination and mutations (including outside CDRs), notes that a significant fraction of variable-domain residues remain germline, and argues that this biases pre-training toward germline residues even though nongermline mutations are often critical for specificity and potency. [17] citeturn22view0
- SHM targeting is known to be context-dependent, with classic hotspot motifs. Peer-reviewed analyses and reviews describe AID-related hotspots (WRCY / RGYW) and polymerase-related hotspots such as WA/TW in SHM targeting models. [18–19] citeturn26view0turn27search18 IMGT’s educational material also describes polymerase η hotspots including WA and TW. [20] citeturn27search4

**Required inputs and preprocessing.**
1. Prefer nucleotide sequences when available (because hotspot motifs are defined in DNA space). IgBLAST supports nucleotide and protein sequences, and explicitly provides germline gene matches and region delineations. [14] citeturn18view0
2. Run IgBLAST and output:
   - best germline V (and D/J if heavy chain) assignment,
   - an alignment of query to germline,
   - FR/CDR boundary mapping. [14] citeturn18view0
3. Construct:
   - an aligned germline sequence \(g\) (in nucleotide or amino acids, depending on pipeline),
   - a mutation indicator \(u_i\) for aligned positions (1 if mutated away from germline, else 0).

**Mutation set definition.**
For amino-acid indexed positions \(i\) in an alignment:
\[
S = \{ i : x_i \neq g_i \ \wedge\ \text{not-gap}(i)\}.
\]
Cohen et al. describe computing mutations by comparing rearranged sequences to germline via alignment, and discuss hotspot motifs being overrepresented in CDRs. [18] citeturn26view0

**Hotspot labeling (nucleotide-first recommended).**
Let the nucleotide sequence be \(X_{\text{nt}}\). Define a hotspot indicator for nucleotide position \(t\) if the local 4-mer context matches a known motif set:
- AID-type: WRCY (or reverse complement RGYW) as described in SHM hotspot analyses. [18–19] citeturn26view0turn27search18
- Polymerase η-type: WA (and TW) hotspots described in open literature and IMGT educational material. [20–21] citeturn27search4turn27search0

Map nucleotide hotspots to amino acids by marking residue \(i\) as hotspot if any nucleotide in its codon (or within a fixed window around it) is hotspot-labeled.

**Mask sampler.** Define a per-position weight combining:
- mutated-away-from-germline positions,
- hotspot contexts,
- baseline regional weights (CDR vs FR),
- and optionally “distance to junction” weights.

Example:
\[
s_i = w_0
\;+\;
w_S \cdot \mathbf{1}[i\in S]
\;+\;
w_H \cdot \mathbf{1}[i\in H]
\;+\;
w_{\text{CDR}}\cdot \mathbf{1}[r(i)\in\{\text{CDR1,CDR2,CDR3}\}],
\]
then sample \(M\) proportionally to \(s_i\) with budget \(B=\lfloor mL\rfloor\).

**Loss variants aligned to “predict nongermline well.”**
Because AbLang-2 frames germline bias as an imbalance where randomly masking residues rarely selects nongermline residues, a direct remedy is *mask reweighting* and/or *loss reweighting* toward nongermline predictions. [17] citeturn22view0 Concretely:
\[
\mathcal{L} = -\sum_{i\in M} \alpha_i \log p_\theta(x_i \mid \tilde{x}),
\quad
\alpha_i =
\begin{cases}
\alpha_{\text{NGL}} & i\in S,\\
1 & \text{otherwise.}
\end{cases}
\]
This is conceptually consistent with AbLang-2’s motivation to improve non-germline residue prediction by countering germline-dominated training signals. [17] citeturn22view0

**Operational notes.**
- Even if your LM consumes amino-acid sequences, store nucleotide-derived hotspot annotations whenever possible; SHM motifs are fundamentally DNA-context motifs. [18–21] citeturn26view0turn27search18turn27search0turn27search4
- If you only have amino acids and cannot recover nucleotides, treat hotspot masking as “mutation-set masking” (mask \(S\)) plus CDR weighting, and explicitly document that true motif-level hotspot annotation was unavailable.

## Multispecific-aware masking using multi-chain data and paratope labels

**What it is.** Masking policies that explicitly account for antibodies that have *multiple binding specificities* (most commonly bispecifics), or antibodies described as interacting with multiple targets/epitopes, by representing and masking *multiple chains / multiple Fv modules* together while keeping specificity structure explicit.

**Why it matters and what “multispecific” concretely implies.**
A bispecific antibody is described in recent reviews as a synthetic antibody with two targeted binding units that can simultaneously bind either two different antigens or two epitopes on the same antigen. [22] citeturn23view0 This implies at least two “binding submodules” whose paratopes may be partly independent and sometimes structurally coupled via shared scaffolds or engineered formats.

**Data sources enabling multispecific-aware training.**
- OAS explicitly supports paired sequences and has been updated to accommodate paired VH/VL sequencing data. [10] citeturn21view0
- PairedAbNGS introduces a very large paired heavy/light sequence database (over 14 million assembled productive sequences, 58 bioprojects) and links pairing preferences to structural contacts. [21] citeturn10view0
- ImmunoMatch demonstrates ML trained on paired H–L sequences to infer chain compatibility, and explicitly notes sensitivity to differences at the H–L interface. [23] citeturn24view0
- LICHEN provides a heavy-chain-conditioned light sequence generator and explicitly supports conditioning on germline and CDRs; notably, it states it can take two heavy sequences as input to find a common light sequence for a bispecific antibody—an explicit example of model design motivated by multispecific use cases. [23] citeturn25view0

**Representation: multi-module tokenization.**
Let there be \(K\) binding modules (for bispecific, \(K=2\)). Represent the input as concatenated segments with explicit module and chain tags:
\[
x = [\text{CLS},
\underbrace{\text{MOD}_1,\text{H},\text{VH}_1,\text{SEP},\text{L},\text{VL}_1}_{\text{module 1}},
\text{SEP},
\underbrace{\text{MOD}_2,\text{H},\text{VH}_2,\text{SEP},\text{L},\text{VL}_2}_{\text{module 2}},
\text{SEP}],
\]
and provide the transformer with:
- position embeddings,
- chain-type embeddings,
- module-ID embeddings.

This is aligned with how BERT uses segment embeddings [1] citeturn7view0 and how modern antibody pairing tasks treat H and L as distinct but interacting sequences. [21–23] citeturn10view0turn24view0turn25view0

**Multispecific-aware masking policies.** The defining feature is that masking is conditioned on module identity and (when available) paratope identity per module.

### Policy A: Module-isolated paratope masking
If you have paratope labels \(P_k\) for each module (from complex structures or paratope predictors), do:
1. Sample a module \(k \sim \text{Cat}(\pi)\) (often uniform).
2. Mask primarily in \(P_k\) while leaving other modules intact, forcing the model to learn module-specific binding content without conflating modules:
   \[
   s_i =
   \begin{cases}
   w_P & i\in P_k,\\
   w_{\neg P} & \text{otherwise},
   \end{cases}
   \quad w_P \gg w_{\neg P}.
   \]

This mirrors the interface/paratope masking idea but adds a “which specificity” conditioning layer. AntiBERTa’s paratope-focused fine-tuning and AACDB’s explicit interface labels provide the practical ingredients for \(P_k\) when structure is available. [3,16] citeturn12view0turn8view0

### Policy B: Cross-module consistency masking (shared light or engineered scaffolds)
Some multispecific designs share components (e.g., a common light chain). LICHEN explicitly frames generating a common light chain given two heavy sequences as a bispecific-relevant use case. [23] citeturn25view0 A masking strategy for shared components:
- Mask the shared chain more aggressively while conditioning on both heavy chains:
  \[
  M \subseteq \text{(shared chain indices)},\quad |M|=\lfloor mL_{\text{shared}}\rfloor.
  \]
This turns the MLM into a “conditional infilling” objective for shared-chain design.

### Policy C: VH–VL interface masking (pairing-aware)
PairedAbNGS and ImmunoMatch emphasize that VH–VL interface residues carry pairing information and that models can be sensitive to interface differences. [21–23] citeturn10view0turn24view0turn25view0 If you have IgFold structures (or known structures), define an interface neighborhood across VH and VL (3D neighbors across chains) and mask those residues using the structure-aware neighborhood mechanism (kNN across both chains). IgFold makes predicted 3D coordinates available quickly from sequence, enabling this at scale. [5] citeturn17view0

**Loss.** Use the standard MLM loss, but (optionally) stratify reporting by:
- per-module loss,
- paratope vs non-paratope loss,
- interface vs non-interface loss,
to ensure that “multispecific-aware” masking actually improves the intended subproblems (binding-module separation and/or shared-chain conditioning).

## Hybrid masking combining multiple strategies

**What it is.** A designed mixture of masking policies that, across training, exposes the model to multiple views of “what matters”: CDR-centric variability, span-level infilling, spatial context, binding interface residues, and germline/SHM mutation logic—without overfitting to any single proxy label source.

**Why a hybrid is usually the best default.**
- Antibody LMs are used for heterogeneous downstream tasks: missing-residue restoration (AbLang), paratope prediction (AntiBERTa), structure prediction features (AntiBERTy embeddings used in IgFold), infilling design (IgLM), and nongermline mutation suggestion (AbLang-2). These tasks cover different parts of antibody biology and are not simultaneously optimized by a single naive uniform-masking scheme. [3–7,17] citeturn12view0turn13view0turn14view0turn17view0turn15view0turn22view0

**Unified formulation.**
Let \(s\in\mathcal{S}\) be a discrete masking strategy choice among:
\[
\mathcal{S}=\{\text{CDR},\ \text{Span},\ \text{3D},\ \text{Paratope},\ \text{Germline/SHM},\ \text{Multispecific}\}.
\]
Define a mixture distribution \(q(s)\) (hyperparameter). For each sample \(x\sim D\):
1. Sample \(s\sim q\),
2. Sample a mask set \(M\sim f_s(\cdot\mid x)\),
3. Apply corruption and compute MLM loss.

Overall objective:
\[
\mathcal{L}(\theta) \;=\; \mathbb{E}_{x\sim D}\,\mathbb{E}_{s\sim q}\,\mathbb{E}_{M\sim f_s}\left[-\sum_{i\in M}\log p_\theta(x_i\mid \tilde{x})\right].
\]

**Recommended hybrid schedules (from antibody constraints, not arbitrary heuristics).**
- **Stage 1 (general antibody syntax):** Mostly uniform + span masking to learn broad antibody sequence patterns and infilling competence, consistent with BERT-style MLM foundations and span-masking motivations. [1–2] citeturn7view0turn7view1
- **Stage 2 (binding-site specialization):** Increase probability mass on CDR-focused and paratope/interface masking, grounded in literature that binding sites are in CDR loops and that LMs can learn binding-site-related signals and be fine-tuned for paratopes. [3,8,16] citeturn12view0turn16view0turn8view0
- **Stage 3 (mutation realism):** Increase weight on germline/SHM masking to directly address germline bias and improve nongermline predictions, aligned with AbLang-2’s framing. [17] citeturn22view0
- **Stage 4 (paired / multispecific):** Introduce multi-chain and multispecific-aware masking using paired datasets and explicit pairing objectives, consistent with paired-sequence resources and heavy–light pairing ML work. [10,21–23] citeturn21view0turn10view0turn24view0turn25view0

**How to implement hybrid masking robustly in practice.**
- Compute expensive annotations offline:
  - IMGT numbering outputs (ANARCI),
  - germline alignment outputs (IgBLAST),
  - predicted structures (IgFold),
  - complex-derived paratope labels (AACDB/SAbDab).
  This reduces training-time I/O and avoids nondeterminism across distributed workers. [5,8,14,16] citeturn17view0turn16view0turn18view0turn8view0
- Keep certain strategies “label-conditional”:
  - If a sample lacks structure, do not attempt 3D neighborhood masking; fall back to span or CDR.
  - If a sample lacks germline alignment, do not attempt hotspot-based masking; fall back to CDR+span.
- Track stratified validation:
  - CDR perplexity,
  - nongermline (mutation-set) perplexity (explicitly motivated by AbLang-2’s critique of standard perplexity being dominated by germline residues),
  - paratope reconstruction accuracy (where labels exist). [17,3] citeturn22view0turn12view0

---

## Reference list with numbering and verified sources

[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. citeturn7view0  
[2] SpanBERT: Improving Pre-training by Representing and Predicting Spans. citeturn7view1  
[3] AntiBERTa: Deciphering the language of antibodies using self-supervised learning. citeturn12view0  
[4] AbLang: an antibody language model for completing antibody sequences. citeturn13view0  
[5] IgFold: Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies. citeturn17view0  
[6] AntiBERTy: Deciphering antibody affinity maturation with language models and weakly supervised learning. citeturn15view0  
[7] IgLM: Infilling language modeling for antibody sequence design. citeturn14view0  
[8] SAbDab: the structural antibody database. citeturn16view0  
[9] Observed Antibody Space (OAS) database web resource. citeturn6search2  
[10] Observed Antibody Space: a diverse database of cleaned, annotated, and translated unpaired and paired antibody sequences. citeturn21view0  
[11] ANARCI: antigen receptor numbering and receptor classification. citeturn29view0  
[12] IMGT unique numbering: standardized FR/CDR delimitation for V-domains. citeturn3search7  
[13] Fifty years of antibody numbering schemes (review, useful for scheme differences and tool landscape). citeturn3search19  
[14] IgBLAST: an immunoglobulin variable domain sequence analysis tool. citeturn18view0  
[15] Parapred: antibody paratope prediction using convolutional and recurrent neural networks. citeturn2search0  
[16] AACDB: Antigen–Antibody Complex Database with interface residue annotations (ΔSASA and distance). citeturn8view0  
[17] Addressing the antibody germline bias and its effect on language models for improved antibody design (AbLang-2). citeturn22view0  
[18] Somatic hypermutation targeting and AID hotspot motifs (WRCY/RGYW) in Ig V regions. citeturn26view0  
[19] SHM targeting model motifs including WRCY/RGYW and WA/TW (review). citeturn27search18  
[20] IMGT education: polymerase η hotspots including WA and TW in SHM context. citeturn27search4  
[21] PairedAbNGS: large-scale paired heavy/light sequences and structural contacts. citeturn10view0  
[22] Bispecific antibodies review defining dual-binding-unit antibodies. citeturn23view0  
[23] ImmunoMatch (paired H–L compatibility) and LICHEN (heavy-conditioned light generation; bispecific use case). citeturn24view0turn25view0  
[24] ProteinMPNN (kNN structural neighborhood sizing evidence: saturation at 32–48 nearest Cα neighbors). citeturn28view0  

```text
[1]  https://aclanthology.org/N19-1423.pdf
[2]  https://www.cs.princeton.edu/~danqic/papers/tacl2020.pdf
[3]  https://www.sciencedirect.com/science/article/pii/S2666389922001052
[4]  https://academic.oup.com/bioinformaticsadvances/article/2/1/vbac046/6609807
[5]  https://pmc.ncbi.nlm.nih.gov/articles/PMC10129313/
[6]  https://arxiv.org/abs/2112.07782
[7]  https://www.sciencedirect.com/science/article/pii/S2405471223002715
[8]  https://pmc.ncbi.nlm.nih.gov/articles/PMC3965125/
[9]  https://opig.stats.ox.ac.uk/webapps/oas/
[10] https://pubmed.ncbi.nlm.nih.gov/34655133/
[11] https://academic.oup.com/bioinformatics/article/32/2/298/1743894
[12] https://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
[13] https://www.mdpi.com/2073-4468/13/4/99
[14] https://pmc.ncbi.nlm.nih.gov/articles/PMC3692102/
[15] https://academic.oup.com/bioinformatics/article/34/17/2944/4972995
[16] https://elifesciences.org/articles/104934
[17] https://academic.oup.com/bioinformatics/article/40/11/btae618/7845256
[18] https://pmc.ncbi.nlm.nih.gov/articles/PMC3109224/
[19] https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2013.00358/full
[20] https://www.imgt.org/IMGTeducation/Tutorials/IGandBcells/_UK/SomaticHypermutations/
[21] https://www.nature.com/articles/s42003-025-08388-y
[22] https://pmc.ncbi.nlm.nih.gov/articles/PMC12320366/
[23] https://www.nature.com/articles/s41592-025-02913-x
     https://www.nature.com/articles/s42003-026-09727-3
[24] https://www.bakerlab.org/wp-content/uploads/2022/09/Dauparas_etal_Science2022_Sequence_design_via_ProteinMPNN.pdf
```