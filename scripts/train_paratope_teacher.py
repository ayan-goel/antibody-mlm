"""Fine-tune alchemab/antiberta2 into a paratope teacher model.

Trains a small per-token classification head on top of the pretrained
AntiBERTa2 RoFormer encoder (200M params, 200M-OAS-sequence pretraining)
using TDC SAbDab_Liberis paratope labels. The resulting checkpoint is
used by ``compute_paratope_labels_v2.py`` to relabel all 500k training
OAS VH sequences with real per-sequence paratope probabilities.

Why: the existing ``oas_vh_500k_paratope.pt`` labels come from a marginal
P(paratope | aa, cdr_region) lookup table, which is position-agnostic
and has no per-sequence signal. Interface masking trained on those
labels is effectively AA-weighted CDR masking, which hurts paratope
downstream transfer. A real per-sequence teacher fixes that.

Usage:
    python scripts/train_paratope_teacher.py
    python scripts/train_paratope_teacher.py --epochs 30 --batch-size 16

Output: ``models/paratope_teacher/final/{encoder,head}.pt`` plus a
``test_metrics.json`` with AUROC, AUPRC, F1, MCC on the SAbDab test split.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import RoFormerForMaskedLM, get_cosine_schedule_with_warmup

from data.benchmarks.paratope import compute_class_weight, load_paratope_splits
from evaluation.downstream._metric_utils import find_youdens_threshold
from evaluation.downstream.collator import DownstreamCollator
from utils.seed import set_seed
from utils.tokenizer import ANTIBERTA2_MODEL_NAME, load_tokenizer

logger = logging.getLogger(__name__)


class ParatopeTeacher(nn.Module):
    """AntiBERTa2 encoder + per-token paratope head.

    The head is a small 2-layer MLP applied independently at every
    token position. Output is a single logit per token (sigmoid ->
    paratope probability).
    """

    def __init__(self, encoder: nn.Module, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.hidden_size = hidden_size

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-token logits (batch, seq_len)."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        return self.head(hidden).squeeze(-1)


def _masked_bce_loss(
    logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor,
) -> torch.Tensor:
    """BCE-with-logits over non-ignored positions (labels != -100)."""
    valid = labels != -100
    if not valid.any():
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        logits[valid], labels[valid].float(), pos_weight=pos_weight,
    )


def _evaluate(
    model: ParatopeTeacher,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Run the model over a loader and return metrics + raw preds/labels.

    The raw arrays are returned so the caller can fit a Youden's-J
    threshold on validation and re-score the test set with it. We
    flatten+filter inside the loop so variable-length batches can be
    concatenated without padding to a common length.
    """
    model.eval()
    pred_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attn)

            labels_flat = labels.view(-1)
            logits_flat = logits.view(-1)
            valid = labels_flat != -100
            pred_chunks.append(logits_flat[valid].float().cpu())
            label_chunks.append(labels_flat[valid].cpu())

    all_logits = torch.cat(pred_chunks).numpy()
    labels_v = torch.cat(label_chunks).numpy()
    preds_v = 1.0 / (1.0 + np.exp(-all_logits))

    if len(np.unique(labels_v)) < 2:
        return (
            {"auroc": 0.0, "auprc": 0.0, "f1": 0.0, "mcc": 0.0, "pos_rate": float(labels_v.mean())},
            preds_v, labels_v,
        )

    auroc = float(roc_auc_score(labels_v, preds_v))
    auprc = float(average_precision_score(labels_v, preds_v))
    binary = (preds_v >= threshold).astype(int)
    f1 = float(f1_score(labels_v, binary, zero_division=0))
    mcc = float(matthews_corrcoef(labels_v, binary))

    return (
        {"auroc": auroc, "auprc": auprc, "f1": f1, "mcc": mcc,
         "pos_rate": float(labels_v.mean())},
        preds_v, labels_v,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = args.device
    output_dir = Path(args.output_dir)
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading AntiBERTa2 tokenizer...")
    tokenizer = load_tokenizer(ANTIBERTA2_MODEL_NAME)

    logger.info("Loading SAbDab_Liberis splits...")
    train_ds, val_ds, test_ds = load_paratope_splits(tokenizer, max_length=args.max_length)
    pos_weight_val = compute_class_weight(train_ds)
    pos_weight = torch.tensor(pos_weight_val, device=device, dtype=torch.float)

    collator = DownstreamCollator(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collator, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collator,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collator,
    )

    logger.info("Loading pretrained AntiBERTa2 encoder (%s)...", ANTIBERTA2_MODEL_NAME)
    base = RoFormerForMaskedLM.from_pretrained(ANTIBERTA2_MODEL_NAME)
    encoder = base.roformer
    hidden_size = base.config.hidden_size
    del base

    model = ParatopeTeacher(encoder=encoder, hidden_size=hidden_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s (%.0f M)", f"{total_params:,}", total_params / 1e6)

    # Separate LRs: small LR for encoder, larger for head
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = max(1, args.epochs * len(train_loader))
    warmup = max(1, int(0.1 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    scaler = torch.amp.GradScaler(device="cuda", enabled=args.fp16)

    best_val_auprc = -1.0
    best_epoch = -1
    patience_counter = 0
    best_state: dict | None = None

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=args.fp16):
                logits = model(input_ids, attn)
                loss = _masked_bce_loss(logits, labels, pos_weight)

            if args.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        val_metrics, _, _ = _evaluate(model, val_loader, device)
        mean_loss = sum(train_losses) / max(len(train_losses), 1)
        logger.info(
            "Epoch %d/%d: train_loss=%.4f val_auroc=%.4f val_auprc=%.4f val_f1=%.4f",
            epoch + 1, args.epochs, mean_loss,
            val_metrics["auroc"], val_metrics["auprc"], val_metrics["f1"],
        )

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {
                "encoder": model.encoder.state_dict(),
                "head": model.head.state_dict(),
            }
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logger.info("Early stopping at epoch %d (best epoch: %d)", epoch + 1, best_epoch)
                break

    if best_state is not None:
        model.encoder.load_state_dict(best_state["encoder"])
        model.head.load_state_dict(best_state["head"])
        logger.info("Restored best model from epoch %d (val AUPRC=%.4f)", best_epoch, best_val_auprc)

    # Fit threshold on val, score test with it
    _, val_preds, val_labels = _evaluate(model, val_loader, device)
    threshold = 0.5
    if len(np.unique(val_labels)) >= 2:
        threshold = float(find_youdens_threshold(val_labels, val_preds))
    logger.info("Fitted Youden's-J threshold on val: %.4f", threshold)

    test_metrics, _, _ = _evaluate(model, test_loader, device, threshold=threshold)
    logger.info("=== Final test metrics ===")
    for k, v in test_metrics.items():
        logger.info("  %s: %.4f", k, v)

    # Save encoder + head + metadata
    torch.save(model.encoder.state_dict(), final_dir / "encoder.pt")
    torch.save(model.head.state_dict(), final_dir / "head.pt")
    tokenizer.save_pretrained(str(final_dir))
    metadata = {
        "base_model": ANTIBERTA2_MODEL_NAME,
        "hidden_size": hidden_size,
        "max_length": args.max_length,
        "pos_weight": pos_weight_val,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "best_val_auprc": best_val_auprc,
        "test_metrics": test_metrics,
    }
    with (final_dir / "teacher_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Teacher saved to %s", final_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="models/paratope_teacher")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train(args)


if __name__ == "__main__":
    main()
