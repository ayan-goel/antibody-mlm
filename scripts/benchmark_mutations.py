"""Zero-shot mutation-effect benchmark using AB-Bind.

Computes delta_PLL for each wildtype/mutant pair and correlates
with experimental ddG values. Reports per-complex and overall
Spearman rho, Pearson r, and binary AUROC.

Usage:
    python scripts/benchmark_mutations.py \
        --checkpoint models/checkpoints/uniform_medium \
        --data-dir data/ab_bind \
        --output-dir evaluation_outputs/uniform_medium \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot mutation benchmark (AB-Bind)")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--data-dir", default="data/ab_bind", help="AB-Bind data directory")
    parser.add_argument("--output-dir", default="evaluation_outputs", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=64, help="PLL internal batch size")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import torch
    from transformers import RoFormerForMaskedLM

    from data.benchmarks.ab_bind import download_ab_bind, load_ab_bind
    from evaluation.mutation_scoring import score_mutation
    from evaluation.pseudo_loglikelihood import compute_pll
    from utils.tokenizer import load_tokenizer

    logger.info("Loading model from %s", args.checkpoint)
    model = RoFormerForMaskedLM.from_pretrained(args.checkpoint)
    model.to(args.device)
    model.eval()
    tokenizer = load_tokenizer("alchemab/antiberta2")

    logger.info("Loading AB-Bind data from %s", args.data_dir)
    download_ab_bind(args.data_dir)
    records = load_ab_bind(args.data_dir)

    if not records:
        logger.error("No mutation records loaded. Exiting.")
        return

    complex_results: dict[str, list[dict]] = defaultdict(list)
    wt_pll_cache: dict[str, dict[str, float]] = {}

    for i, rec in enumerate(records):
        cache_key = f"{rec['pdb_id']}_{rec['chain_id']}"
        if cache_key not in wt_pll_cache:
            logger.info(
                "Computing wildtype PLL for %s chain %s (%d/%d)",
                rec["pdb_id"], rec["chain_id"], i + 1, len(records),
            )
            wt_pll_cache[cache_key] = compute_pll(
                model, tokenizer, rec["wildtype_seq"], args.device, args.batch_size,
            )

        wt_pll = wt_pll_cache[cache_key]
        mut_pll = compute_pll(
            model, tokenizer, rec["mutant_seq"], args.device, args.batch_size,
        )

        delta_pll = mut_pll["pll"] - wt_pll["pll"]
        complex_results[rec["pdb_id"]].append({
            "chain_id": rec["chain_id"],
            "mutation": rec["mutation_str"],
            "ddg": rec["ddg"],
            "delta_pll": delta_pll,
            "pll_wt": wt_pll["pll"],
            "pll_mut": mut_pll["pll"],
        })

        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d mutations", i + 1, len(records))

    all_ddg: list[float] = []
    all_delta_pll: list[float] = []
    per_complex: dict[str, dict] = {}

    for pdb_id, results in complex_results.items():
        ddgs = [r["ddg"] for r in results]
        delta_plls = [r["delta_pll"] for r in results]
        all_ddg.extend(ddgs)
        all_delta_pll.extend(delta_plls)

        entry: dict = {"n_mutants": len(results)}
        if len(results) >= 3 and len(set(ddgs)) > 1:
            rho, p_val = spearmanr(ddgs, delta_plls)
            entry["spearman_rho"] = float(rho)
            entry["spearman_pval"] = float(p_val)
            r, _ = pearsonr(ddgs, delta_plls)
            entry["pearson_r"] = float(r)
        per_complex[pdb_id] = entry

    summary: dict = {
        "n_complexes": len(complex_results),
        "n_mutants_total": len(all_ddg),
    }

    if len(all_ddg) >= 3:
        rho, _ = spearmanr(all_ddg, all_delta_pll)
        r, _ = pearsonr(all_ddg, all_delta_pll)
        summary["overall_spearman_rho"] = float(rho)
        summary["overall_pearson_r"] = float(r)

    binary_labels = [1 if d > 0 else 0 for d in all_ddg]
    if len(set(binary_labels)) == 2:
        neg_delta = [-d for d in all_delta_pll]
        summary["binary_auroc"] = float(roc_auc_score(binary_labels, neg_delta))

    summary["per_complex"] = per_complex

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "mutation_benchmark.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to %s", out_path)

    logger.info("=== Mutation Benchmark Summary ===")
    logger.info("Complexes: %d, Mutants: %d", summary["n_complexes"], summary["n_mutants_total"])
    if "overall_spearman_rho" in summary:
        logger.info("Overall Spearman rho: %.4f", summary["overall_spearman_rho"])
        logger.info("Overall Pearson r:    %.4f", summary["overall_pearson_r"])
    if "binary_auroc" in summary:
        logger.info("Binary AUROC:         %.4f", summary["binary_auroc"])


if __name__ == "__main__":
    main()
