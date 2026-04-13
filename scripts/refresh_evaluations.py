"""Refresh stale evaluations after the structure_probe / contact_map /
mutation_benchmark / attention_analysis fixes.

Four things to refresh:

  1. **structure_probe**: original results trained on the wrong PDB chain
     (longest chain in the AB-Bind complex, often the antigen). Fixed loader
     in data/benchmarks/structure_probe.py uses H/L antibody chains and
     augments with SAbDab Liberis. Old embedding caches are stale because
     the input sequences changed.

  2. **contact_map**: original results were on ESM-2-predicted kNN labels
     over OAS sequences (a circular metric: antibody LM imitates a general
     protein LM's contact guesses). Now uses **real X-ray Calpha contact
     maps** from SAbDab Liberis at the 8 Å threshold. The task also reports
     long-range AUROC and precision (|i-j| >= 24, >= 12) because the old
     short+medium range metric saturated at P@L ~0.997. Input sequences
     changed from OAS to SAbDab PDB chains, so the full embedding cache is
     rebuilt.

  3. **mutation_benchmark**: previously used delta-PLL difference scoring;
     now uses ESM-style wildtype-marginal scoring. The single-chain results
     are stale (old code gave AUROC ≈ 0.34 due to noisy multi-mutation
     accumulation); the two paired models were missing entirely because
     the old PLL path didn't add [MOD1][H] framing tokens.

  4. **attention_analysis**: the attention-contact correlation metric
     previously ran against ESM-2 kNN neighborhoods (same circularity
     problem as contact_map). Now runs against real X-ray Calpha contact
     maps (8 Å threshold) from SAbDab Liberis. Per-head entropy and head
     importance are unchanged; only the contact-correlation label source
     moves to real crystal structures.

Run this on a GPU node:

    python scripts/refresh_evaluations.py
    python scripts/refresh_evaluations.py --skip-structure-probe
    python scripts/refresh_evaluations.py --experiments uniform_medium cdr_medium
    python scripts/refresh_evaluations.py --device cuda:0

Each section can be skipped independently
(--skip-structure-probe / --skip-contact-map / --skip-mutations /
--skip-attention-analysis). Results are merged into
`evaluation_outputs/<exp>/all_metrics.json` per experiment.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from transformers import RoFormerForMaskedLM

from data.dataset import AntibodyDataset
from data.dataset_paired import PairedAntibodyDataset
from evaluation.attention_analysis import AttentionAnalyzer
from evaluation.downstream import DownstreamConfig, get_task
from training.config import load_config
from utils.seed import set_seed
from utils.tokenizer import load_tokenizer, load_tokenizer_multispecific

# Keep this in sync with EVAL_SPLIT_SEED in scripts/run_all_evaluations.py so
# the attention_analysis eval_dataset here uses the same held-out split the
# rest of the evaluation pipeline does.
EVAL_SPLIT_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _merge_into_all_metrics(
    name: str, key: str, results: dict, output_dir: Path,
) -> None:
    """Merge a new section into evaluation_outputs/<exp>/all_metrics.json."""
    am_path = output_dir / name / "all_metrics.json"
    if not am_path.exists():
        logger.warning("No all_metrics.json at %s — skipping merge", am_path)
        return
    with am_path.open() as f:
        am = json.load(f)
    if key == "downstream:structure_probe":
        ds = am.get("downstream") or {}
        if not isinstance(ds, dict) or "error" in ds:
            ds = {}
        ds["structure_probe"] = results
        am["downstream"] = ds
    elif key == "downstream:contact_map":
        ds = am.get("downstream") or {}
        if not isinstance(ds, dict) or "error" in ds:
            ds = {}
        ds["contact_map"] = results
        am["downstream"] = ds
    elif key == "mutation_benchmark":
        am["mutation_benchmark"] = results
    elif key == "attention_analysis":
        am["attention_analysis"] = results
    else:
        raise ValueError(f"Unknown merge key: {key}")
    with am_path.open("w") as f:
        json.dump(am, f, indent=2, default=str)
    logger.info("Merged %s into %s", key, am_path)


def _run_structure_probe(name: str, checkpoint: str, args) -> dict | None:
    """Re-run structure_probe with stale cache wiped."""
    task_dir = Path(f"downstream_outputs/{name}/structure_probe_probe")
    if task_dir.exists():
        logger.info("Removing stale structure_probe outputs at %s", task_dir)
        shutil.rmtree(task_dir)

    cfg = DownstreamConfig(
        task="structure_probe",
        checkpoint=checkpoint,
        mode="probe",
        learning_rate=1e-3,
        epochs=args.structure_probe_epochs,
        batch_size=16,
        early_stopping_patience=20,
        num_seeds=args.num_seeds,
        base_seed=42,
        output_dir=f"downstream_outputs/{name}",
        device=args.device,
    )
    try:
        task = get_task("structure_probe", cfg)
        return task.run()
    except Exception:
        logger.exception("structure_probe failed for %s", name)
        return None


def _run_contact_map(name: str, checkpoint: str, args) -> dict | None:
    """Re-run contact_map end-to-end.

    Input sequences changed when we switched from ESM-2 kNN labels on OAS
    sequences to real X-ray CA-CA distances on SAbDab Liberis PDB chains,
    so the full embedding cache is stale and has to be rebuilt. This is
    cheap (~20 s per model for ~600 chains) compared to head training.
    """
    task_dir = Path(f"downstream_outputs/{name}/contact_map_probe")
    if task_dir.exists():
        logger.info("Removing stale contact_map outputs at %s", task_dir)
        shutil.rmtree(task_dir)

    cfg = DownstreamConfig(
        task="contact_map",
        checkpoint=checkpoint,
        mode="probe",
        learning_rate=1e-3,
        epochs=50,
        batch_size=16,
        early_stopping_patience=10,
        num_seeds=args.num_seeds,
        base_seed=42,
        output_dir=f"downstream_outputs/{name}",
        device=args.device,
    )
    try:
        task = get_task("contact_map", cfg)
        return task.run()
    except Exception:
        logger.exception("contact_map failed for %s", name)
        return None


def _run_mutation_benchmark(
    name: str, config_path: str, checkpoint: str, args,
) -> dict | None:
    """Re-run mutation_benchmark with the new wildtype-marginal scorer."""
    # Local import keeps top-level imports cheap when this section is skipped.
    from scripts.run_all_evaluations import _run_mutations

    config = load_config(config_path)
    if config.data.paired:
        tokenizer = load_tokenizer_multispecific(config.model.model_name)
    else:
        tokenizer = load_tokenizer(config.model.model_name)

    model = RoFormerForMaskedLM.from_pretrained(checkpoint)
    model.to(args.device)
    model.eval()
    try:
        return _run_mutations(
            model, tokenizer, args.device, args.ab_bind_dir, args.pll_batch_size,
        )
    except Exception:
        logger.exception("mutation_benchmark failed for %s", name)
        return None
    finally:
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _run_attention_analysis(
    name: str, config_path: str, checkpoint: str, args,
) -> dict | None:
    """Re-run attention_analysis using REAL SAbDab X-ray contact maps.

    The attention-contact correlation metric previously ran against the
    ESM-2-predicted kNN neighborhoods (a circular metric). It now runs
    against real X-ray Calpha contact maps (8 Å threshold) from SAbDab
    Liberis. Per-head attention entropy and head-importance scores are
    computed on the eval split of the model's training dataset (same
    split used by MLM/infilling/PLL for consistency).
    """
    config = load_config(config_path)
    set_seed(config.seed)

    if config.data.paired:
        tokenizer = load_tokenizer_multispecific(config.model.model_name)
    else:
        tokenizer = load_tokenizer(config.model.model_name)

    if config.data.paired:
        full_dataset = PairedAntibodyDataset(
            data_path=config.data.processed_path,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            paratope_path=config.data.paratope_path or None,
            interface_path=config.data.interface_path or None,
            germline_path=config.data.germline_path or None,
            bispecific=config.data.bispecific,
        )
    else:
        full_dataset = AntibodyDataset(
            data_path=config.data.processed_path,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            coords_path=config.data.coords_path or None,
            paratope_path=config.data.paratope_path or None,
            germline_path=config.data.germline_path or None,
        )
    eval_size = int(len(full_dataset) * (1 - config.data.train_split))
    _, eval_dataset = torch.utils.data.random_split(
        full_dataset,
        [len(full_dataset) - eval_size, eval_size],
        generator=torch.Generator().manual_seed(EVAL_SPLIT_SEED),
    )

    # Prefer SAbDab real crystal coords for the attention-contact
    # correlation. Fall back to the dataset's ESM-2 kNN coords only if
    # the SAbDab file is missing (should not happen after Phase 1 of the
    # SAbDab plan).
    sabdab_real_coords = None
    sabdab_path = Path("data/structures/sabdab_liberis_coords.pt")
    if sabdab_path.exists():
        try:
            sabdab_real_coords = torch.load(sabdab_path, weights_only=False)
            logger.info(
                "  Loaded %d SAbDab real-coords entries for attention analysis",
                len(sabdab_real_coords),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("  Failed to load SAbDab coords: %s", e)
            sabdab_real_coords = None
    _base_dataset = getattr(eval_dataset, "dataset", eval_dataset)
    coords_data = getattr(_base_dataset, "coords", None)

    model = RoFormerForMaskedLM.from_pretrained(checkpoint)
    model.to(args.device)
    model.eval()
    try:
        set_seed(42)
        analyzer = AttentionAnalyzer(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            coords_data=coords_data,
            sabdab_real_coords=sabdab_real_coords,
            max_samples=args.attention_max_samples,
        )
        return analyzer.evaluate(dataset=eval_dataset)
    except Exception:
        logger.exception("attention_analysis failed for %s", name)
        return None
    finally:
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-yaml", default="configs/experiments.yaml")
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--structure-probe-epochs", type=int, default=100)
    parser.add_argument("--ab-bind-dir", default="data/ab_bind")
    parser.add_argument("--pll-batch-size", type=int, default=8)
    parser.add_argument("--attention-max-samples", type=int, default=100)
    parser.add_argument("--output-dir", default="evaluation_outputs")
    parser.add_argument("--skip-structure-probe", action="store_true")
    parser.add_argument("--skip-contact-map", action="store_true")
    parser.add_argument("--skip-mutations", action="store_true")
    parser.add_argument("--skip-attention-analysis", action="store_true")
    args = parser.parse_args()

    with open(args.experiments_yaml) as f:
        registry = yaml.safe_load(f)
    experiments = registry["experiments"]
    if args.experiments:
        experiments = {k: v for k, v in experiments.items() if k in args.experiments}
    if not experiments:
        logger.error("No experiments selected")
        return

    output_dir = Path(args.output_dir)

    summary: dict[str, dict] = {n: {} for n in experiments}

    for name, exp_def in experiments.items():
        config_path = exp_def["config"]
        checkpoint = exp_def["checkpoint"]
        if not Path(checkpoint).exists():
            logger.warning("Skipping %s: checkpoint not found at %s", name, checkpoint)
            continue

        logger.info("=" * 70)
        logger.info("Experiment: %s", name)
        logger.info("=" * 70)

        if not args.skip_structure_probe:
            logger.info("[1/4] structure_probe")
            r = _run_structure_probe(name, checkpoint, args)
            if r is not None:
                summary[name]["structure_probe"] = r
                _merge_into_all_metrics(
                    name, "downstream:structure_probe", r, output_dir,
                )

        if not args.skip_contact_map:
            logger.info("[2/4] contact_map (SAbDab real contacts, long-range metrics)")
            r = _run_contact_map(name, checkpoint, args)
            if r is not None:
                summary[name]["contact_map"] = r
                _merge_into_all_metrics(
                    name, "downstream:contact_map", r, output_dir,
                )

        if not args.skip_mutations:
            logger.info("[3/4] mutation_benchmark (wildtype-marginal scoring)")
            r = _run_mutation_benchmark(name, config_path, checkpoint, args)
            if r is not None:
                summary[name]["mutation_benchmark"] = r
                _merge_into_all_metrics(name, "mutation_benchmark", r, output_dir)

        if not args.skip_attention_analysis:
            logger.info("[4/4] attention_analysis (SAbDab real contacts)")
            r = _run_attention_analysis(name, config_path, checkpoint, args)
            if r is not None:
                summary[name]["attention_analysis"] = r
                _merge_into_all_metrics(name, "attention_analysis", r, output_dir)

    logger.info("=" * 70)
    logger.info("Refresh summary")
    logger.info("=" * 70)
    for name, sec in summary.items():
        sp = sec.get("structure_probe", {})
        cm = sec.get("contact_map", {})
        mb = sec.get("mutation_benchmark", {})
        aa = sec.get("attention_analysis", {})
        sp_s = sp.get("spearman_distance_mean", float("nan")) if sp else float("nan")
        sp_p = sp.get("contact_precision_at_L_mean", float("nan")) if sp else float("nan")
        cm_lr = cm.get("long_range_auroc_mean", float("nan")) if cm else float("nan")
        cm_lp = cm.get("long_range_precision_at_L_mean", float("nan")) if cm else float("nan")
        mu_s = mb.get("mean_per_complex_spearman_rho", float("nan")) if mb else float("nan")
        mu_a = mb.get("mean_per_complex_auroc", float("nan")) if mb else float("nan")
        aa_c = aa.get("attn_contact_correlation_mean", float("nan")) if aa else float("nan")
        logger.info(
            "  %-25s sp=[rho=%.3f, P@L=%.3f]  cm=[lrAUROC=%.3f, lrP@L=%.3f]  "
            "mut=[rho=%.3f, AUROC=%.3f]  attn=[contact_corr=%.3f]",
            name, sp_s, sp_p, cm_lr, cm_lp, mu_s, mu_a, aa_c,
        )


if __name__ == "__main__":
    main()
