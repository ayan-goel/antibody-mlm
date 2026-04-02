# Deep Research Map for Biology-Informed Masking in Antibody Protein Language Models

## What your annotated bibliography already gives us

Your annotated bibliography establishes a strong, coherent foundation around a single central mismatch: **standard masked language modeling (MLM) assumes “all residues are equally informative,” but antibody function and variation are concentrated in specific regions**, especially CDRs and binding-relevant structural neighborhoods. fileciteturn0file0 fileciteturn0file1

It already covers (at a high level) the *full end-to-end experimental story* you’re trying to tell:

- **Why antibodies (and especially multispecifics) break sequence-only assumptions**: *Synapse* shows that multispecific efficacy is an emergent property of topology/format and domain arrangement, motivating biologically grounded inductive biases rather than uniform sequence treatment. citeturn0search1turn0search5turn0search13  
- **The closest direct precedent for “masking distribution as inductive bias”**: *Ng & Briney* demonstrate that preferentially masking the non-templated CDR3 improves training efficiency and downstream performance, making them the most “adjacent” paper to your core idea. citeturn0search4turn24search7turn24search5  
- **A biophysical reason span/cluster masking might matter**: antibody binding landscapes exhibit substantial **epistasis**, and many non-additive effects concentrate in CDRs, arguing that independent token masking can be misaligned with functional coupling. citeturn5search2turn5search10turn5search11  
- **Why this matters in real design pipelines**: therapeutic design and developability workflows already leverage protein/antibody PLMs, so improved representation of CDR/interface residues has practical consequences. citeturn7search4turn9search3turn25search7  
- **A concrete downstream “binding specificity” task with interpretability**: mBLM is a strong example of a curated specificity dataset (influenza HA antibodies) + saliency-based analysis that tends to highlight CDR/interface residues—useful as both a benchmark and an interpretability template. citeturn7search6turn7search10turn23search3  
- **Core infrastructure sources**: the bibliography already points to a large-scale antibody sequence corpus (OAS) and foundational protein/task literature (ProteinBERT, DMS fitness prediction, SpanBERT). citeturn1search0turn10search3turn9search2  

What your bibliography *does not* fully supply (and what the rest of this report focuses on) is a **complete implementation-oriented map** of:  
(1) additional antibody PLMs and multimodal approaches you should compare to, (2) the tooling needed (numbering, germline assignment, structure prediction, paratope labeling), and (3) downstream datasets/benchmarks that are realistically runnable and defensible.

## Masking and self-supervised objectives we can directly borrow or adapt

Your work treats “masking” as a design choice in the **corruption process** of self-supervised learning. That’s aligned with the broader NLP pretraining view: BERT’s MLM framing, RoBERTa’s emphasis on training recipe and dynamic masking, and span-corruption/infilling approaches like SpanBERT, BART, and T5 provide a rich set of *well-studied noising operators* you can translate into antibody-aware variants. citeturn14search1turn14search0turn9search2turn14search3turn14search2  

A useful way to operationalize your masking policies, consistent with this literature, is: define a family of **mask distributions** over positions/spans/structural neighborhoods, while holding total corruption constant (e.g., ~15%), so improvements can be attributed to *where* learning pressure is applied rather than to “more masking.” This is exactly the kind of controlled comparison RoBERTa argues is necessary when evaluating pretraining changes. citeturn14search0  

Your bibliography already includes two “masking-adjacent” ideas that are particularly transferable:

- **Preferential / region-aware masking (antibody-specific)**: shifting masking probability toward non-templated regions (CDR3) improves AbLM learning, strongly motivating your “generalize from CDR3 to richer biology.” citeturn0search4turn24search7  
- **Curriculum/difficulty-aware masking (sequence-domain)**: CM-GEMS shows that progressively shifting masking toward harder spans can preserve performance while cutting training steps dramatically; while it’s on gene sequences, the “mask what matters / what’s hard” logic is directly reusable for antibody region/interface curricula. citeturn9search1turn9search9  

There is also antibody-specific evidence that “training schedule” matters:

- **Curriculum learning for paired vs unpaired antibody data**: Burbach & Briney propose a curriculum strategy for integrating unpaired and paired antibody sequences; this is relevant if you decide to incorporate paired VH–VL in training and want to keep the rest of the experimental design controlled. citeturn19search2turn19search10turn23search5  

Finally, span-based antibody “infill” objectives exist in-domain:

- **IgLM frames antibody generation as infilling in the style of NLP infilling/span corruption**, making it a direct conceptual cousin to your proposed CDR span masking—useful both for methodological precedent and for downstream generative evaluation ideas. citeturn20search1turn23search6turn20search4  

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["antibody structure diagram CDR regions labeled","antibody variable region framework CDR loops schematic","antibody paratope epitope interface diagram"],"num_per_query":1}

## Biological priors and infrastructure needed to implement your masking strategies

This section is the “toolchain backbone” for building region-, structure-, interface-, and evolution-aware masking in a way reviewers will consider reproducible.

A key theme across modern antibody modeling papers is that you get leverage by combining (a) a massive repertoire corpus, with (b) standardized annotation (CDR boundaries, germline calls), plus (c) structural or interface signals where possible.

The best-supported, widely used resources for those steps are:

- **Sequence corpora and CDR boundaries**  
  The entity["organization","Observed Antibody Space","antibody sequence database"] (OAS) provides cleaned and annotated antibody sequences, and it is widely used as a training source for antibody-specific LMs (including AbLang, AntiBERTa, and later paired models). citeturn1search0turn1search4turn23search0turn16view1  

- **Structural antibody complexes and affinity metadata**  
  entity["organization","SAbDab","structural antibody database"] is a central resource for antibody structures (including antibody–antigen complexes) annotated consistently, with curated experimental metadata and—in some cases—affinity data. This underpins paratope labeling from structures and any structure-aware benchmark creation. citeturn1search1turn1search9turn22search0turn22search8  

- **Therapeutic sequences and developability context**  
  entity["organization","Thera-SAbDab","therapeutic antibody database"] tracks WHO-recognized antibody therapeutics and links to structural representatives where available, enabling therapeutic-context sampling and evaluation sets. citeturn1search6turn22search2turn25search21  

- **Specialized binding-focused antibody databases (useful for downstream tasks)**  
  entity["organization","CoV-AbDab","coronavirus antibody database"] consolidates coronavirus-binding antibodies (sequences, and when available structures), and is widely used in binding-related ML tasks because it targets a well-defined antigen family. citeturn1search7turn1search3turn1search11  

To *compute* the priors your masking policy needs, you also need standardized annotation tools:

- **Numbering / defining CDRs and frameworks**  
  ANARCI is a widely used tool for assigning antibody numbering schemes (including IMGT) and classifying variable domains; it’s explicitly designed for antibody/TCR variable sequences. citeturn2search0turn2search8  

- **Germline assignment and V(D)J context (for hotspot masking)**  
  IgBLAST is a standard tool for germline V(D)J assignment and delineating framework/CDR regions during IG sequence analysis, enabling “mutation-from-germline” features used in hotspot masking. citeturn2search1turn2search5turn2search9  

- **Somatic hypermutation targeting priors**  
  SHM targeting is known to be biased by motif/context (e.g., canonical hotspot motifs such as RGYW/WRCY and WA/TW families), and there is substantial immunology literature characterizing these biases and their relationship to affinity maturation. These sources give you defensible grounding for “hotspot-aware masking.” citeturn13search4turn13search17turn13search15turn13search16  

For structure-aware masking, you need predicted (or experimental) structures at scale:

- **Antibody structure prediction tools**  
  IgFold provides a fast approach for antibody structure prediction from sequence, and can be used to precompute residue neighborhoods for 3D masking. citeturn3search0turn3search8turn3search19  
  DeepAb is another major antibody structure predictor, with published benchmarking. citeturn3search1turn3search9  
  ABlooper focuses on CDR loop structures and is useful if you need CDR-loop-focused uncertainty or structure priors. citeturn3search3turn3search14turn3search7  
  ABodyBuilder-family tools are widely used in antibody modeling pipelines and are integrated into broader tool suites like SAbPred. citeturn22search3turn22search15turn22search7  

For interface-aware masking, you need paratope prediction and/or labeled paratopes from complexes:

- **Paratope prediction methods and resources**  
  AntiBERTa itself is a strong paratope prediction reference point and includes public training/fine-tuning materials, which is valuable for reproducibility. citeturn16view1turn18view0turn20search17  
  Parapred is a well-cited deep-learning paratope predictor. citeturn2search2turn2search6  
  proABC-2 predicts antibody contact residues and interaction types using a CNN and provides code. citeturn2search7turn2search3  
  Newer PLM-embedding-based paratope predictors continue to appear (e.g., ParaAntiProt and Paraplume), and these are relevant because they show how PLM features translate into interface-label prediction pipelines. citeturn5search0turn5search4turn5search15  

## Downstream tasks and datasets for evaluating your masking policies

A defensible experimental section needs downstream tasks that are: (a) antibody-relevant, (b) runnable with public datasets, and (c) capable of showing that your masking actually improved *functional* learning rather than only MLM loss.

Below is a practical menu of downstream tasks aligned to your proposal’s aims, with sources you can use to implement each.

**Paratope prediction (sequence → per-residue interface labels)**  
This is one of the most standard antibody PLM evaluations because it directly tests whether the model’s internal representation captures binding-site information.

- AntiBERTa demonstrates fine-tuning for paratope prediction and points to public training/fine-tuning assets. citeturn16view1turn18view0  
- SAbDab provides antibody–antigen structures that can be used to generate paratope labels (e.g., contact-based definitions), and is the typical upstream resource for such labeling. citeturn1search5turn22search0turn6search2  

**Binding specificity / epitope classification (sequence → antigen region class)**  
This is where you can directly test whether “more CDR/interface learning pressure” improves antigen-specific prediction.

- mBLM provides a curated influenza hemagglutinin antibody dataset (mined from publications/patents) and trains an explainable model for epitope-region specificity. citeturn7search2turn7search6turn7search10  
- There are also public binder/non-binder classification datasets for therapeutically important targets (e.g., CTLA-4, PD-1) used explicitly to evaluate whether sequence models can predict binding. citeturn5search21  

**Mutation-effect prediction on antibody binding (sequence variant → affinity/fitness proxy)**  
This is one of the best places to show your claim about epistasis and “learning coupled residues,” especially if you evaluate performance specifically on CDR variants.

- Tite-Seq is a key experimental method for mapping antibody sequence–affinity landscapes at scale, producing mutation/affinity datasets that can be used for mutation-effect benchmarks. citeturn5search3turn5search7  
- Adams et al. analyze epistasis in an antibody-antigen binding landscape derived from such data, giving strong grounding for evaluating beyond additive mutation models. citeturn5search2turn5search10turn5search11  
- AB-Bind is a curated database of antibody binding ΔΔG values across mutants, useful for mutation-effect evaluation and classification of “improved vs weakened” binders. citeturn6search0turn6search16  
- SKEMPI 2.0 is broader (protein–protein interfaces), but contains structurally resolved binding ΔΔG data and is frequently used for interface mutation effect modeling; it can be filtered to antibody–antigen complexes when needed. citeturn6search1turn6search5  

**Antibody–antigen affinity regression (sequence pair → affinity)**  
If you want a supervised task closer to therapeutic ranking, there are multi-source merged resources emerging:

- The Therapeutics Data Commons includes an antibody–antigen affinity task derived from SAbDab pairs. citeturn6search22turn22search12  
- A new large unified dataset, ANDD, consolidates antibody/nanobody sequences, structures, antigens, and affinity values across many sources, aimed explicitly at design benchmarking. citeturn6search14turn22search16  

**Developability prediction (sequence → manufacturability/biophysical risk)**  
This is a high-value axis because reviewers understand why models that “only optimize binding” fail in practice.

- The “five computational developability guidelines” paper provides the canonical framing that developability issues cluster in antibody variable surface features (often involving CDRs), and introduces TAP as an assessment tool. citeturn9search3turn25search7turn25search2  
- TAP (Therapeutic Antibody Profiler) is available as an online tool and has follow-on work on computational developability assessment. citeturn25search0turn25search1turn25search15  
- TherAbDesign is a modern ML-guided framework targeting therapeutic-like properties (e.g., viscosity-related liabilities) and is a good downstream evaluation anchor if you want a “design-improves-developability” story. citeturn7search3turn7search11turn7search7  
- There are experimental datasets and ML studies on antibody viscosity and aggregation prediction that can serve as additional supervised proxies, though these datasets are often smaller. citeturn5search1turn5search16turn5search9  

**Multispecific-focused evaluation**  
If you want your proposal and experiments to clearly “own” the multispecific angle, you should include at least one explicit multispecific benchmark/task—not just motivation.

- Synapse (multispecific synthetic landscapes + graph models) provides a framework and code for benchmarking multispecific format/topology effects. citeturn0search1turn0search13turn0search5  
- EVA provides a closed-loop, format/topology/spacing-aware multispecific design case study (HER2×CD3), useful for framing what real multispecific optimization requires. citeturn7search1turn7search5turn7search9  
- AI-guided design of common light chains tackles manufacturability constraints in bispecifics, reinforcing that multispecific performance is constrained by VH–VL pairing/interface details (a good use case for interface- and hotspot-aware masking). citeturn8search0turn8search3  
- Clinical trispecific review sources give your paper’s motivation clinical realism (what’s being developed, why it’s hard). citeturn9search0turn9search4  

Across all downstream experiments, it helps to use **benchmark and split methodology** that is already considered credible in protein ML:

- TAPE and FLIP emphasize biologically meaningful generalization and careful splitting (e.g., testing on divergent sequences / out-of-distribution regimes), which you can adapt to antibody family-based or clonotype-based splits. citeturn11search1turn11search2turn11search5  
- ProteinGym provides a large-scale benchmark suite for mutation-effect prediction and evaluation regimes, useful as a methodological template even when your primary benchmarks are antibody-specific. citeturn11search0turn11search4  

## Source catalog with links organized by what you need to build and run the project

This is the “grab bag” you can use to implement masking algorithms, build the dataset pipeline, and choose downstream tasks. Each item is linked via citation.

### Core papers already in your annotated bibliography
Your bibliography includes (at minimum) these key sources, which map directly to your methods + evaluation plan. fileciteturn0file0

- Preferential masking in antibody LMs (CDR3 / non-templated regions). citeturn0search4turn24search7turn24search5  
- Multispecific function depends on topology/arrangement; Synapse benchmark + code. citeturn0search1turn0search13turn0search5  
- Epistasis/fitness landscapes in antibody binding; Tite-Seq foundations. citeturn5search2turn5search3turn5search7  
- Antibody specificity prediction with curated influenza HA antibodies (mBLM). citeturn7search6turn7search10turn23search3  
- Developability guidelines and TAP lineage. citeturn9search3turn25search0turn25search7  
- OAS antibody sequence corpus. citeturn1search0turn1search4  
- Span masking foundations (SpanBERT). citeturn9search2turn9search6  

### Antibody-specific language models and multimodal antibody representation learning
These are the main antibody PLM baselines/adjacent work you should know about and (selectively) compare against:

- AntiBERTa + public training/fine-tuning assets. citeturn20search17turn16view1turn18view0  
- AbLang (heavy/light chain models; sequence completion and embeddings). citeturn23search0turn21search6turn23search16  
- AbLang-2 (explicitly addresses germline bias and non-germline residues). citeturn13search2turn22search17turn22search1  
- AntiBERTy (affinity maturation trajectories; weak supervision). citeturn20search0turn20search3turn20search13  
- BALM (antibody LM; paired/unpaired comparisons show the value of native pairing). citeturn4search9turn4search8turn4search17  
- IgBert and IgT5 (large-scale paired antibody LMs trained on billions of OAS sequences). citeturn20search2turn20search11turn20search5  
- Contrastive sequence–structure pretraining (CSSP) / AntiBERTa2-CSSP as a key multimodal adjacent approach. citeturn12search3turn19search5turn19search3  
- S2ALM (sequence+structure antibody PLM built on ESM-2; modern multimodal direction). citeturn19search11turn12search23  
- AbMAP (hypervariable-region adaptation framework; strong for mutation-effect and paratope tasks). citeturn12search1turn12search5  

### Masking and corruption operators you can cite when defining your algorithms
These sources are useful to justify *why* your mask policy choices are principled and how to formalize them.

- BERT (MLM objective origins). citeturn14search1turn14search5  
- RoBERTa (dynamic masking; careful control of training recipe). citeturn14search0turn14search4  
- SpanBERT (span masking + span-boundary objective; strong precedent for span-style CDR masking). citeturn9search2turn9search10  
- BART and T5 (span infilling / span corruption as denoising pretraining). citeturn14search3turn14search2turn14search6  
- CM-GEMS (curriculum/difficulty-masked training; shows efficiency gains from smarter masking). citeturn9search1turn9search17  
- Curriculum learning for AbLMs (paired/unpaired schedule as a learning curriculum). citeturn19search2turn19search14  
- IgLM (in-domain antibody infilling objective; relevant for “mask spans in CDR loops”). citeturn20search1turn23search6turn20search4  

### Structure prediction and 3D resources for structure-aware masking
If you implement “mask 3D neighborhoods,” these are the main practical references:

- IgFold (fast antibody structure prediction; widely adopted). citeturn3search0turn3search19turn24search10  
- DeepAb (interpretable antibody structure prediction). citeturn3search1turn3search10  
- ABlooper (CDR loop prediction + confidence; useful for loop-focused masking). citeturn3search3turn3search14  
- ABodyBuilder2/3 + SAbPred tool suite (common infrastructure in antibody modeling pipelines). citeturn22search7turn22search3turn3search13  
- SAbDab for experimental structures and antibody–antigen complexes. citeturn1search5turn22search0turn6search2  
- Protein structure resources via the entity["organization","Protein Data Bank","protein structure database"] (PDB) underpin SAbDab and downstream contact labeling. citeturn1search1turn1search9  

### Paratope/interface labeling and prediction
These sources help you implement “interface-aware masking” and evaluate paratope prediction:

- AntiBERTa’s paratope fine-tuning materials (practical baseline). citeturn16view1turn18view0  
- Parapred (deep paratope prediction; classic baseline). citeturn2search2turn2search6  
- proABC-2 (contact prediction + code). citeturn2search7turn2search3  
- ParaAntiProt (PLM embeddings for paratope prediction). citeturn5search0turn2search19  
- Paraplume (recent sequence-based paratope prediction using PLM embeddings). citeturn5search4turn5search15  
- Structure-free paratope similarity/prediction methods (useful for scalable interface priors). citeturn5search12  

### Deep mutational scanning, binding affinity mutation sets, and evaluation benchmarks
These are the most directly useful sources for “mutation-effect prediction” and binding ΔΔG benchmarks:

- Tite-Seq method (assay foundation). citeturn5search3turn5search7  
- Antibody epistasis/freely energy landscape analysis (Adams et al.). citeturn5search2turn5search11  
- AB-Bind (antibody binding ΔΔG mutant database). citeturn6search0turn6search16  
- SKEMPI 2.0 (protein–protein ΔΔG; filter antibody–antigen if needed). citeturn6search1turn6search5  
- ProteinGym (mutation-effect benchmark methodology; large-scale evaluation patterns). citeturn11search0turn11search4  
- TAPE and FLIP (split/evaluation regimes emphasizing real generalization). citeturn11search1turn11search2turn11search5  
- Antibody-specific docking/complex benchmarks (ABAG-docking; PierceLab benchmark repo). citeturn6search3turn6search11turn6search15  

### Multispecific and therapeutic-context sources
These sources provide multispecific motivation and, importantly, candidates for multispecific evaluation tasks:

- Synapse multispecific benchmark + code. citeturn0search1turn0search13turn0search5  
- EVA closed-loop multispecific design platform. citeturn7search1turn7search5turn7search9  
- Common light chain design for bispecific manufacturability. citeturn8search0turn8search1  
- Mechanistic modeling of mono- vs bi-specific binding tradeoffs (IL-6R/IL-8R case). citeturn8search3turn8search15  
- Clinical trispecific review (immune-oncology). citeturn9search0turn9search4  

### Practical open implementations you can reuse directly
If your goal is to “run these experiments” efficiently, these are high-value implementation anchors (and you can cite them when describing reproducibility).

- AntiBERTa notebooks and assets via entity["company","GitHub","code hosting company"]. citeturn18view0  
- AntiBERTy repository. citeturn20search13  
- IgFold repository. citeturn3search19turn24search10  
- AbLang / AbLang-2 repositories. citeturn23search16turn22search17  
- AntiBERTa2-CSSP model card on entity["company","Hugging Face","ml model hub company"] (and the associated CSSP paper). citeturn19search3turn19search5turn19search0  

## How to turn these sources into an “everything we need” experimental blueprint

To convert this literature map into an implementation plan for your paper, the cleanest structure (and the one most aligned with how reviewers read) is:

**Define masking policies as distributions over positions/spans/structural neighborhoods**, with an explicit constraint that total masking rate is fixed, and run a controlled study where you vary only the policy. This is directly supported by the antibody-specific precedent (preferential masking) and by the broader lesson from pretraining methodology papers: you need controlled comparisons to claim causal improvement from pretraining changes. citeturn0search4turn14search0turn19search2  

Then, implement masking policies using the following “prior channels,” each backed by concrete tooling and datasets:

- **Region priors (CDR vs framework)** using OAS annotations or ANARCI. citeturn1search0turn2search0  
- **Structure priors (3D neighborhoods)** using predicted structures (IgFold/ABodyBuilder2) and validated structural sets (SAbDab). citeturn3search0turn22search3turn1search5  
- **Interface priors (paratope)** using paratope predictors (AntiBERTa, Parapred, proABC-2) or contact labels from antibody–antigen complexes in SAbDab. citeturn18view0turn2search2turn2search7turn6search2  
- **Evolution priors (SHM hotspots / non-germline residues)** using IgBLAST-based germline calls + motif-informed SHM bias grounding, and optionally AbLang-2/germline-bias literature as methodological framing. citeturn2search1turn13search4turn13search2turn22search9  

Finally, evaluate using a small set of **high-signal downstream tasks**:
- Paratope prediction (robust, standard, interpretable). citeturn16view1turn2search2  
- Binding specificity classification (mBLM influenza) + at least one binder/non-binder dataset. citeturn7search6turn5search21  
- Mutation-effect prediction (Tite-Seq / AB-Bind; optionally SKEMPI-filtered). citeturn5search3turn6search0turn6search1  
- Developability proxy prediction or developability-guided optimization references (TAP/TherAbDesign). citeturn25search0turn7search11  
- At least one explicit multispecific benchmark angle (Synapse and/or the bispecific common-light-chain setting). citeturn0search1turn8search0