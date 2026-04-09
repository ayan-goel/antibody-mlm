# Function- and Structure-Aware Masking for Antibody PLMs

Studying whether biologically-informed masking strategies improve antibody protein language models. We train RoFormer-based models from scratch on antibody sequences from OAS and compare a range of masking strategies that inject CDR, structural, paratope, germline, and paired-chain priors into the MLM objective.

| Strategy | What it biases the mask toward |
|----------|---------------------------------|
| `uniform` | Random positions (baseline) |
| `cdr` | CDR1 / CDR2 / CDR3 residues |
| `span` | Contiguous spans (SpanBERT-style) |
| `structure` | 3D-spatial neighborhoods via ESM2 contact-map kNN |
| `interface` | Predicted paratope (antigen-contacting) residues |
| `germline` | Somatic-hypermutation sites (residues differing from germline) |
| `multispecific` | Paired VH+VL: paratope, shared-light, VH-VL interface |
| `hybrid` | Stochastic mixture of the above with optional curriculum |

Models are evaluated across zero-shot metrics (PLL, CDR infilling, region-stratified perplexity, AB-Bind mutation effect correlation) and supervised downstream tasks (paratope prediction, binding specificity, developability).

**See [GUIDE.md](GUIDE.md) for the full walkthrough**: project structure, all commands, how to add new strategies/tasks, and how the evaluation framework works.

## Quick Start

```bash
# Setup
conda create -n abmlm python=3.10 -y && conda activate abmlm
pip install -r requirements.txt
pip install accelerate rjieba

# Download single-chain data
python scripts/download_data.py --config configs/medium.yaml
python scripts/annotate_cdrs.py --input data/processed/oas_vh_500k.jsonl

# (Optional) Precompute metadata sidecars for structure / interface / germline strategies
python scripts/compute_paratope_labels.py --input data/processed/oas_vh_500k.jsonl \
    --output data/structures/oas_vh_500k_paratope.pt
python scripts/compute_germline_labels.py --input data/processed/oas_vh_500k.jsonl \
    --output data/structures/oas_vh_500k_germline.pt
python scripts/predict_structures.py --input data/processed/oas_vh_500k.jsonl \
    --output data/structures/oas_vh_500k_coords.pt

# Train one or more models sequentially (logs go to logs/<name>.log)
python scripts/train.py --uniform --cdr --span --structure --interface --germline

# Run all evaluations for all experiments registered in configs/experiments.yaml
python scripts/run_all_evaluations.py --device cuda

# Generate comparison report
python scripts/generate_report.py
```

## Project Structure

```
configs/          YAML experiment and downstream task configurations
data/             Data downloading, preprocessing, sidecar metadata, benchmark datasets
masking/          Pluggable masking strategies (uniform, cdr, span, structure,
                  interface, germline, multispecific, hybrid)
models/           Model factory (single-chain + multispecific) and checkpoint storage
training/         HuggingFace Trainer wrapper with curriculum callback
evaluation/       Zero-shot metrics, downstream tasks, comparison, reporting
scripts/          CLI entry points for every step of the pipeline
utils/            Tokenizer, seeding, CDR annotation
```

## Key Results Location

- Training summaries: `models/checkpoints/*/training_summary.json`
- Per-experiment evaluation: `evaluation_outputs/*/all_metrics.json`
- Downstream results: `downstream_outputs/*/*/results.json`
- Comparison reports: `comparison_outputs/`
- Per-model training logs: `logs/*.log`
