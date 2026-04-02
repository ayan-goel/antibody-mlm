# Function- and Structure-Aware Masking for Antibody Protein Language Models  
## (Focused on Multispecific Antibodies)

---

## 1. Problem

Protein language models (PLMs) trained with masked language modeling (MLM) assume that all residues are equally informative. This assumption is fundamentally misaligned with antibody biology.

In antibodies:
- Functional signal is concentrated in **CDRs (complementarity-determining regions)**
- Binding depends on **3D structural neighborhoods**
- Residues interact through **epistasis and spatial coupling**
- Multispecific antibodies exhibit **emergent behavior across multiple binding domains**

Despite this, standard MLM applies **uniform random masking**, which distributes learning signal inefficiently.

---

## 2. Key Insight

**Masking is not just corruption — it is an inductive bias.**

By changing *where* the model is forced to predict missing information, we can control:
- what representations the model prioritizes
- which biological signals it learns

We propose:
> Designing biologically-informed masking distributions to better align PLM training with antibody function.

---

## 3. Research Goal

Develop and evaluate **function- and structure-aware masking strategies** for antibody-specific PLMs, with a focus on **multispecific antibody behavior**.

We aim to answer:

1. Can biologically-informed masking improve representation quality?
2. Which types of biological priors matter most?
3. Do these improvements transfer to real downstream tasks (binding, mutation effects)?
4. Can masking capture **multi-binding interactions** in multispecific antibodies?

---

## 4. Approach Overview

We keep **everything fixed** (model, data, training steps) and vary **only the masking distribution**.

### Base Model
- AntiBERTa or similar antibody-specific transformer
- MLM objective (~15% masking)

---

## 5. Masking Strategies

### 5.1 Baselines

- **Uniform MLM (BERT)**  
  Random masking across all residues  

- **CDR-focused masking (Ng & Briney-style)**  
  Higher masking probability in CDR regions  

- **Span masking (SpanBERT-style)**  
  Mask contiguous regions instead of independent tokens  

---

### 5.2 Proposed Methods

#### 1. Structure-Aware Masking
Mask residues based on **3D contact neighborhoods**

- Select a residue
- Mask all residues within radius r in 3D structure

**Hypothesis:**  
Improves learning of structural coupling and folding constraints

---

#### 2. Interface / Paratope Masking
Mask residues involved in **binding interfaces**

- Derived from:
  - SAbDab structures
  - predicted paratopes

**Hypothesis:**  
Improves binding site representation and specificity modeling

---

#### 3. Germline / Mutation-Hotspot Masking
Mask residues based on **somatic hypermutation likelihood**

- Use:
  - germline alignment
  - SHM hotspot motifs

**Hypothesis:**  
Aligns model learning with evolutionary pressure

---

#### 4. Multispecific-Aware Masking (NEW)
Mask **multiple functional regions jointly**

- Mask across:
  - multiple paratopes
  - different chains/domains
- Optionally:
  - co-mask interacting binding sites

**Hypothesis:**  
Captures cross-binding dependencies and multi-target behavior

---

#### 5. Hybrid Masking
Combine strategies with fixed masking budget

Example:
- 40% CDR
- 30% structure
- 30% uniform

---

## 6. Experimental Design

### Controlled Setup

- Same:
  - architecture
  - dataset
  - training steps
  - masking ratio (~15%)

- Only variable:
  - masking distribution

---

### Ablation Plan

We explicitly test:

- Each masking strategy independently
- Pairwise combinations
- Full hybrid model

Also vary:
- masking proportions
- span sizes
- structure radius

---

## 7. Datasets

### Pretraining
- Observed Antibody Space (OAS)

### Structural Data
- SAbDab (structures + binding sites)

### Downstream Evaluation

#### 1. Mutation Effect Prediction
- Deep mutational scanning (DMS) datasets

#### 2. Binding Prediction
- CoV-AbDab or similar binding datasets

#### 3. Paratope Prediction
- Structure-derived labels

#### 4. Multispecific Evaluation (if available)
- bispecific / trispecific datasets
- or synthetic multi-binding tasks

---

## 8. Evaluation Metrics

### Regression Tasks
- Spearman correlation
- Pearson correlation

### Classification Tasks
- ROC-AUC
- Accuracy
- F1 score

### Representation Analysis
- Embedding clustering (UMAP / PCA)
- Functional separation
- Attention map interpretability

---

## 9. Expected Contributions

1. **Reframing MLM masking as inductive bias design**
2. Demonstrating that **where you mask matters more than how much**
3. Showing that **biological priors improve representation learning**
4. Introducing **multispecific-aware masking**
5. Providing a **systematic evaluation framework** for masking strategies

---

## 10. Potential Pitfalls

- Noisy structure predictions → weak structure masking signal  
- Overfitting to CDR regions → loss of generalization  
- Limited multispecific datasets  
- Paratope labels may be imperfect  

---

## 11. Future Work

### 11.1 Learned Masking (Extension)
Instead of heuristics:
- learn masking distribution via model
- attention-based masking
- reinforcement learning masking policy

---

### 11.2 Generative Evaluation
- sequence design tasks
- binding optimization
- antibody engineering pipelines

---

## 12. Summary

We propose a new perspective:

> Masking is a controllable inductive bias that determines what protein language models learn.

By aligning masking with biological structure and function, especially in **multispecific antibodies**, we aim to build representations that are more useful, interpretable, and biologically grounded.

---