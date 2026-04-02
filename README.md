# Function- and Structure-Aware Masking for Antibody PLMs

Studying whether biologically-informed masking strategies improve antibody protein language models. We train RoFormer-based models from scratch on 500K VH sequences from OAS, comparing standard uniform masking against CDR-weighted masking that upweights complementarity-determining regions during MLM training.

Models are evaluated across zero-shot metrics (PLL, CDR infilling, perplexity, mutation effect correlation) and supervised downstream tasks (paratope prediction, binding specificity, developability).

**See [GUIDE.md](GUIDE.md) for the full walkthrough**: project structure, all commands, how to add new strategies/tasks, and how the evaluation framework works.

## Quick Start

```bash
# Setup
conda create -n abmlm python=3.10 -y && conda activate abmlm
pip install -r requirements.txt
pip install accelerate rjieba

# Download data
python scripts/download_data.py --config configs/medium.yaml
python scripts/annotate_cdrs.py --input data/processed/oas_vh_500k.jsonl

# Train (uniform masking)
python scripts/train.py --config configs/medium.yaml

# Train (CDR-weighted masking)
python scripts/train.py --config configs/cdr_medium.yaml

# Run all evaluations for all experiments
python scripts/run_all_evaluations.py --device cuda

# Generate comparison report
python scripts/generate_report.py
```

## Project Structure

```
configs/          YAML experiment and downstream task configurations
data/             Data downloading, preprocessing, benchmark datasets
masking/          Pluggable masking strategies (uniform, CDR-weighted)
models/           Model factory and checkpoint storage
training/         HuggingFace Trainer wrapper with early stopping
evaluation/       Zero-shot metrics, downstream tasks, comparison, reporting
scripts/          CLI entry points for every step of the pipeline
utils/            Tokenizer, seeding, CDR annotation
```

## Key Results Location

- Training summaries: `models/checkpoints/*/training_summary.json`
- Evaluation metrics: `evaluation_outputs/*/metrics*.json`
- Downstream results: `downstream_outputs/*/results.json`
- Comparison reports: `comparison_outputs/`
