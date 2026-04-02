Perfect — this is exactly what you should be doing next. Here’s a **Cursor-ready `.md` file** that’s focused purely on **getting started (Week 1–2 execution plan)**.

This is not theory — it’s **what to actually build first**.

---

```md id="init-antibody-mlm"
# Project Setup: Function-Aware Masking for Antibody PLMs
## Initial Implementation Plan (Week 1–2)

---

## 🎯 Goal of This Phase

Establish a **clean, reproducible baseline pipeline** for antibody MLM training so that all future experiments are controlled.

By the end of this phase, you should have:
- A working MLM training pipeline
- Uniform masking baseline trained
- Evaluation pipeline for embeddings + downstream tasks
- Clean dataset + preprocessing

---

## 🧱 Step 1: Environment Setup

### Create project structure

```

antibody-mlm/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
├── training/
├── masking/
├── evaluation/
├── utils/
│
├── configs/
├── scripts/
│
└── README.md

````

---

### Install dependencies

```bash
pip install torch transformers datasets biopython pandas numpy scikit-learn matplotlib seaborn
````

Optional (recommended):

```bash
pip install umap-learn
```

---

## 📦 Step 2: Dataset (OAS)

### Goal

Get a **clean antibody sequence dataset**.

### Tasks

* Download subset of **Observed Antibody Space (OAS)**
* Extract:

  * heavy chain sequences (VH) first (simpler)
* Filter:

  * remove sequences with invalid tokens
  * length constraints (e.g., 80–200 AA)

---

### Output format

Save as:

```json
{
  "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTF..."
}
```

---

### Save to:

```
data/processed/oas_sequences.json
```

---

## 🔢 Step 3: Tokenization

Use amino acid vocabulary:

```python
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
```

Use HuggingFace tokenizer or custom:

* each AA = token
* add special tokens:

  * [MASK]
  * [CLS]
  * [PAD]

---

## 🧠 Step 4: Baseline Model (IMPORTANT)

### DO NOT overcomplicate this

Start with:

* small transformer (6 layers)
* or use pretrained AntiBERTa if available

---

### Minimal config

```python
hidden_size = 256
num_layers = 6
num_heads = 8
```

---

## 🎭 Step 5: Uniform Masking (Baseline)

This is your **most important control experiment**

### Implement:

* 15% of tokens selected
* of those:

  * 80% → [MASK]
  * 10% → random token
  * 10% → unchanged

---

### File:

```
masking/uniform_mask.py
```

---

## 🏋️ Step 6: Training Loop

### Objective

Standard MLM loss (cross-entropy)

---

### Key details

* batch size: 32–128
* learning rate: 1e-4
* epochs: start with 1–3

---

### Output

Save:

```
models/uniform_baseline.pt
```

---

## 📊 Step 7: Basic Evaluation (MANDATORY)

Before doing anything fancy, confirm:

### 1. Loss curve

* does training converge?

### 2. Token accuracy

* basic MLM accuracy

---

## 🧪 Step 8: Embedding Extraction

Create script:

```
evaluation/extract_embeddings.py
```

For each sequence:

* take CLS token OR mean pooling

Save embeddings:

```
data/processed/embeddings.npy
```

---

## 📉 Step 9: Embedding Visualization

Use UMAP:

```python
import umap
```

Check:

* do sequences cluster?
* any structure at all?

---

## 🧬 Step 10: Add CDR Annotation (VERY IMPORTANT NEXT STEP)

### Tool:

* ANARCI

---

### Goal:

Annotate each sequence with:

* CDR1, CDR2, CDR3
* framework regions

---

### Output format

```json
{
  "sequence": "...",
  "cdr_mask": [0,0,1,1,1,0,...]
}
```

---

## 🧩 Step 11: Prepare for Next Phase

Once baseline is working, next steps will be:

* CDR masking
* span masking
* structure-aware masking

---

## ✅ Milestone Checklist

Before moving on, confirm:

* [ ] Dataset loaded and cleaned
* [ ] Tokenizer working
* [ ] Uniform masking implemented
* [ ] Model trains without crashing
* [ ] Loss decreases
* [ ] Embeddings extracted
* [ ] UMAP visualization works
* [ ] CDR annotation pipeline ready

---

## 🚨 Common Mistakes (Avoid These)

* ❌ Changing model + masking at same time
* ❌ Using too large dataset too early
* ❌ No evaluation before new methods
* ❌ Not fixing random seeds
* ❌ Debugging on full dataset (use small subset first)

---

## 🔥 Key Principle

> You are not building the best model yet.
> You are building a **controlled experimental system**.

---

## ⏭️ What Comes Next (Preview)

Next phase:

* Implement:

  * CDR-focused masking
  * Span masking
* Compare against uniform baseline
* Measure performance differences
