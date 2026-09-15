"""Inspect what CDR3 infilling actually generates.

Reviewer #1 asked about "the novelty and variety of the recovered CDR3s",
i.e. whether a high exact-match score hides degenerate completions. This
dumps the raw generated CDR3s next to the ground truth and quantifies
degeneracy, so the answer is a table of real sequences rather than an
assertion.

Faithfully mirrors ``evaluation/infilling.py``: the *entire* CDR3 is
replaced with [MASK] and decoded in one parallel argmax pass (not
autoregressively), over the deterministic first-N prefix of the same
held-out split the reported metric uses.

Metrics:
  exact_match        reproduces infill_cdr3_exact_match as a sanity check
  distinct_frac      distinct generated CDR3s / N  (eval truth = ceiling)
  top1_share         share of the single most common generated CDR3
  homopolymer_frac   fraction of completions that are one residue repeated
  max_run_mean       mean longest single-residue run per completion
  position_entropy   mean per-position AA entropy across completions (bits)
  novel_frac         completions absent from the pretraining corpus

Usage:
    python scripts/inspect_cdr3_infilling.py --checkpoints models/checkpoints/cdr_medium/final
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import RoFormerForMaskedLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import AntibodyDataset
from data.splits import make_train_eval_split
from utils.tokenizer import load_tokenizer_for_checkpoint

logger = logging.getLogger(__name__)

_CDR3_LABEL = 3


@torch.no_grad()
def generate_cdr3(
    model, tokenizer, sample: dict, device: str,
) -> tuple[str, str] | None:
    """Return (true_cdr3, generated_cdr3) for one held-out sequence."""
    cdr_mask = sample.get("cdr_mask")
    if cdr_mask is None:
        return None
    positions = [i for i, v in enumerate(cdr_mask) if v == _CDR3_LABEL]
    if not positions:
        return None

    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)

    masked = input_ids.clone()
    for pos in positions:
        masked[pos] = tokenizer.mask_token_id

    logits = model(
        input_ids=masked.unsqueeze(0).to(device),
        attention_mask=attention_mask.unsqueeze(0).to(device),
    ).logits[0]
    pred = logits.argmax(dim=-1).cpu()

    true_aa = "".join(tokenizer.convert_ids_to_tokens([input_ids[p].item() for p in positions]))
    gen_aa = "".join(tokenizer.convert_ids_to_tokens([pred[p].item() for p in positions]))
    return true_aa.replace(" ", ""), gen_aa.replace(" ", "")


def longest_run(s: str) -> int:
    best = run = 1
    for a, b in zip(s, s[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if s else 0


def degeneracy_stats(
    generated: list[str], truths: list[str], corpus: set[str],
) -> dict[str, float]:
    n = len(generated)
    counts = Counter(generated)

    max_len = max((len(g) for g in generated), default=0)
    entropies = []
    for i in range(max_len):
        col = [g[i] for g in generated if len(g) > i]
        if len(col) < 2:
            continue
        freqs = np.array(list(Counter(col).values()), dtype=float)
        p = freqs / freqs.sum()
        entropies.append(float(-(p * np.log2(p)).sum()))

    return {
        "n": float(n),
        "exact_match": sum(g == t for g, t in zip(generated, truths)) / n,
        "distinct_frac": len(counts) / n,
        "truth_distinct_frac": len(set(truths)) / n,
        "top1_share": counts.most_common(1)[0][1] / n,
        "homopolymer_frac": sum(len(set(g)) == 1 for g in generated) / n,
        "max_run_mean": float(np.mean([longest_run(g) for g in generated])),
        "truth_max_run_mean": float(np.mean([longest_run(t) for t in truths])),
        "position_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "novel_frac": sum(g not in corpus for g in generated) / n,
    }


def load_corpus_cdr3s(path: str, limit: int) -> set[str]:
    out: set[str] = set()
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            c = json.loads(line).get("cdr3_aa")
            if c:
                out.add(c)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data", default="data/processed/oas_vh_500k.jsonl")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="comparison_outputs/cdr3_degeneracy.csv")
    parser.add_argument("--dump-dir", default="comparison_outputs/cdr3_generations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo = Path(__file__).resolve().parent.parent

    corpus = load_corpus_cdr3s(args.data, limit=200_000)
    logger.info("corpus CDR3 vocabulary: %d distinct", len(corpus))

    rows = []
    dump_dir = repo / args.dump_dir
    dump_dir.mkdir(parents=True, exist_ok=True)

    for ckpt in args.checkpoints:
        name = Path(ckpt).parent.name
        tokenizer = load_tokenizer_for_checkpoint(ckpt, "alchemab/antiberta2")
        model = RoFormerForMaskedLM.from_pretrained(ckpt).to(args.device).eval()

        # Same held-out split and same deterministic prefix as the metric.
        dataset = AntibodyDataset(
            data_path=args.data, tokenizer=tokenizer, max_length=160,
        )
        _, eval_ds = make_train_eval_split(dataset, 0.9)

        truths, generated = [], []
        for i in range(min(args.n_samples, len(eval_ds))):
            got = generate_cdr3(model, tokenizer, eval_ds[i], args.device)
            if got is None:
                continue
            truths.append(got[0])
            generated.append(got[1])

        stats = degeneracy_stats(generated, truths, corpus)
        rows.append({"experiment": name, **stats})

        with (dump_dir / f"{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true_cdr3", "generated_cdr3", "exact"])
            for t, g in zip(truths, generated):
                w.writerow([t, g, int(t == g)])

        logger.info(
            "\n[%s] em=%.3f distinct=%.3f (truth %.3f) top1=%.3f "
            "homopolymer=%.3f run=%.2f (truth %.2f) H=%.2f novel=%.3f",
            name, stats["exact_match"], stats["distinct_frac"],
            stats["truth_distinct_frac"], stats["top1_share"],
            stats["homopolymer_frac"], stats["max_run_mean"],
            stats["truth_max_run_mean"], stats["position_entropy"],
            stats["novel_frac"],
        )
        for t, g in list(zip(truths, generated))[: args.n_examples]:
            logger.info("    true %-22s gen %-22s %s", t, g, "MATCH" if t == g else "")

        del model
        torch.cuda.empty_cache()

    out = repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("\nwrote %s and per-model dumps to %s", out, dump_dir)


if __name__ == "__main__":
    main()
