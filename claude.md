# Cursor / Agent Guidelines: Antibody MLM Project

## 🎯 Purpose

This file defines:
- coding conventions
- architectural rules
- experimentation discipline
- how Cursor (or any AI agent) should assist

The goal is to ensure:
> reproducible experiments, clean abstractions, and fast iteration

---

## 🧠 Core Philosophy

1. **Controlled experiments > clever code**
2. **Masking is the ONLY variable**
3. **Everything must be reproducible**
4. **Simple > complex (until needed)**

---

## 📁 Project Structure Rules

Never violate this structure:

```

data/           → raw + processed datasets
masking/        → ALL masking strategies live here
models/         → model definitions only
training/       → training loops only
evaluation/     → metrics, visualization
configs/        → experiment configs
scripts/        → runnable entrypoints
utils/          → generic helpers ONLY

````

---

## ⚙️ Coding Conventions

### General

- Use **Python 3.10+**
- Follow **PEP8**, but prioritize readability
- Use **type hints everywhere**

Example:
```python
def mask_sequence(sequence: str) -> tuple[list[int], list[int]]:
````

---

### Naming

* snake_case → variables, functions
* PascalCase → classes
* UPPER_CASE → constants

---

### File Naming

* `uniform_mask.py`
* `cdr_mask.py`
* `train_mlm.py`
* `evaluate_embeddings.py`

---

## 🎭 Masking System Design (CRITICAL)

### Rule: Each masking strategy MUST be isolated

Each masking method:

```python
class MaskingStrategy:
    def mask(self, tokens: list[int]) -> tuple[list[int], list[int]]:
        ...
```

---

### DO NOT:

* mix masking logic inside training loop
* hardcode masking in model
* combine multiple strategies in one file

---

### REQUIRED structure

```
masking/
    base.py
    uniform_mask.py
    cdr_mask.py
    span_mask.py
    structure_mask.py
```

---

## 🏋️ Training Rules

### NEVER change multiple variables at once

Allowed:

* change masking strategy

Not allowed:

* changing model + masking simultaneously

---

### Training script must:

* load config
* set seed
* log everything

---

### Required logging

Every run must log:

```json
{
  "masking_type": "uniform",
  "masking_ratio": 0.15,
  "model_config": {...},
  "dataset": "...",
  "seed": 42
}
```

---

## 🧪 Experiment Discipline

### Rule: Every experiment must be reproducible

* Fix seeds
* Save configs
* Save model checkpoints

---

### Config-driven experiments ONLY

Use:

```
configs/
    uniform.yaml
    cdr.yaml
    structure.yaml
```

---

### NEVER hardcode parameters in scripts

---

## 📊 Evaluation Rules

### MUST implement before new methods:

* MLM loss
* token accuracy
* embedding extraction

---

### For embeddings:

* mean pooling OR CLS (be consistent)
* same method across experiments

---

### Visualization:

* UMAP required
* same hyperparameters across runs

---

## 🧬 Data Handling Rules

* NEVER modify raw data
* always create processed versions

---

### All datasets must:

* be versioned
* have clear preprocessing scripts

---

## 🧩 Abstraction Rules

### Keep components independent

* masking → independent
* model → independent
* training → independent

---

### Example flow

```
dataset → tokenizer → masking → model → loss
```

---

## 🚨 Anti-Patterns (DO NOT DO)

### ❌ Bad

```python
if use_cdr_mask:
    ...
elif use_structure_mask:
    ...
```

### ✅ Good

```python
masking = MaskingFactory.create(config.masking_type)
```

---

### ❌ Bad

* mixing evaluation into training loop
* writing one giant script
* duplicating masking logic

---

## 🤖 Cursor / Agent Instructions

When assisting:

### ALWAYS:

* prioritize simple implementations
* preserve modular structure
* follow existing abstractions
* ask before changing architecture

---

### NEVER:

* rewrite large parts of code unnecessarily
* introduce new dependencies without reason
* combine multiple responsibilities in one function

---

### When generating code:

* keep functions small
* add docstrings
* include type hints
* make it runnable immediately

---

## 🔁 Iteration Strategy

### Phase 1 (current)

* uniform masking baseline

### Phase 2

* CDR masking
* span masking

### Phase 3

* structure-aware masking
* hotspot masking

---

## 🧠 Design Principles

### 1. Masking = Inductive Bias

All code should reflect this idea.

---

### 2. Comparisons must be fair

Same:

* model
* data
* training steps

Only change:

* masking

---

### 3. Keep experiments cheap

* start with small dataset
* scale later

---

## 📌 Code Quality Checklist

Before committing:

* [ ] modular code
* [ ] no duplicated logic
* [ ] masking isolated
* [ ] config-driven
* [ ] reproducible run
* [ ] logs saved

---

## 🔥 Final Reminder

> This is an ML research project, not a software project.

Optimize for:

* clarity
* correctness
* reproducibility

NOT:

* over-engineering
* premature optimization
