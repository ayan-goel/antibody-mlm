"""Build the SAbDab real-Calpha-coordinates evaluation dataset.

Reads ``data/sabdab_liberis.pkl`` (the TDC SAbDab_Liberis dataset, ~1023
antibody chains across ~688 unique PDBs), downloads each PDB from RCSB
once, extracts Calpha coordinates for the requested chain, aligns the
PDB-extracted sequence with the TDC-supplied sequence, and saves a
single ``.pt`` file containing real X-ray crystal coordinates for use as
ground-truth labels in the contact_map and attention_analysis tasks.

This is the "no prediction bullshit" alternative to the previous ESM-2
contact-map labels: every label here comes from a deposited crystal
structure rather than a sequence-only LM's contact predictions.

Output format
-------------
``data/structures/sabdab_liberis_coords.pt`` is a list of dicts:

    {
        "id":         "2hh0_H",                # TDC chain id (PDB + chain)
        "pdb_id":     "2hh0",
        "chain":      "H",
        "sequence":   "EVQLVES...",            # AA from PDB ATOM records
        "coords_ca":  Tensor(L, 3, float32),   # real Calpha xyz
        "paratope":   List[int],               # 0-indexed paratope positions
                                               # from TDC Y, re-aligned to
                                               # the PDB sequence
    }

Failed chains (download error, parse error, length mismatch) are
omitted entirely — no None entries.

Usage
-----
    # Full build (~30-60 minutes on CPU, network-bound):
    python scripts/build_sabdab_real_coords.py

    # Quick smoke build (only the first 20 PDBs):
    python scripts/build_sabdab_real_coords.py --max_pdbs 20

    # Custom paths / overwrite:
    python scripts/build_sabdab_real_coords.py \\
        --liberis_path data/sabdab_liberis.pkl \\
        --pdb_cache_dir data/sabdab/pdbs \\
        --output data/structures/sabdab_liberis_coords.pt \\
        --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from data.benchmarks.structure_probe import extract_chain_seq_and_coords

logger = logging.getLogger(__name__)


_RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download_pdb(pdb_id: str, cache_dir: Path, retries: int = 3, sleep: float = 0.05) -> Path | None:
    """Download a PDB file from RCSB into ``cache_dir`` if not already present.

    Returns the local Path on success, None if all retries failed.
    Adds a small inter-request sleep so we don't hammer the RCSB endpoint.
    """
    pdb_id = pdb_id.lower()
    out_path = cache_dir / f"{pdb_id}.pdb"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url = _RCSB_PDB_URL.format(pdb_id=pdb_id)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "antibody-mlm/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            time.sleep(sleep)
            return out_path
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning("PDB %s not found on RCSB (404)", pdb_id)
                return None
            logger.warning("HTTP %s for %s (attempt %d/%d)", e.code, pdb_id, attempt + 1, retries)
        except Exception as e:  # noqa: BLE001
            logger.warning("Download error for %s (attempt %d/%d): %s",
                           pdb_id, attempt + 1, retries, e)
        time.sleep(2 ** attempt)
    return None


def _find_existing_local_pdb(pdb_id: str, search_dirs: list[Path]) -> Path | None:
    """Look for an already-downloaded copy of this PDB in any cache dir."""
    pdb_lower = pdb_id.lower()
    pdb_upper = pdb_id.upper()
    for d in search_dirs:
        for candidate in (
            d / f"{pdb_lower}.pdb",
            d / f"{pdb_upper}.pdb",
            d / f"{pdb_lower}.PDB",
        ):
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
    return None


def _align_paratope_to_pdb_seq(
    tdc_seq: str, pdb_seq: str, tdc_paratope: list[int],
) -> list[int]:
    """Re-anchor TDC paratope indices to the PDB-extracted sequence.

    The TDC sequence and the PDB ATOM-derived sequence usually agree
    (the TDC dataset is itself derived from PDB), but the PDB sequence
    can be missing residues (gaps in the crystal) so paratope indices
    must be remapped from TDC index space to PDB index space.

    Strategy: find the longest matching subsequence prefix and translate
    indices that fall inside it. Drop indices outside.
    """
    if pdb_seq == tdc_seq:
        return list(tdc_paratope)

    # Try to find pdb_seq as a substring of tdc_seq (PDB has missing
    # residues at the ends).
    if pdb_seq in tdc_seq:
        offset = tdc_seq.index(pdb_seq)
        return [
            i - offset for i in tdc_paratope
            if offset <= i < offset + len(pdb_seq)
        ]

    # Try the reverse (TDC has missing residues at the ends).
    if tdc_seq in pdb_seq:
        offset = pdb_seq.index(tdc_seq)
        return [i + offset for i in tdc_paratope if 0 <= i < len(tdc_seq)]

    # Last-resort: longest common substring at the start.
    common = 0
    for a, b in zip(tdc_seq, pdb_seq):
        if a != b:
            break
        common += 1
    if common >= 10:
        return [i for i in tdc_paratope if i < common]

    # Cannot align; return empty paratope (chain is still usable for
    # contact_map and structure_probe — we just lose paratope info).
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--liberis_path", default="data/sabdab_liberis.pkl",
                        help="Path to TDC SAbDab_Liberis pickle file")
    parser.add_argument("--pdb_cache_dir", default="data/sabdab/pdbs",
                        help="Directory to store downloaded PDB files")
    parser.add_argument("--abbind_pdb_dir", default="data/ab_bind/pdbs",
                        help="Existing AB-Bind PDB directory (reused as a secondary cache)")
    parser.add_argument("--output", default="data/structures/sabdab_liberis_coords.pt",
                        help="Output .pt file")
    parser.add_argument("--max_pdbs", type=int, default=0,
                        help="Cap on unique PDBs to process (0 = all)")
    parser.add_argument("--max_adj_gap_a", type=float, default=5.0,
                        help="Max allowed adjacent Calpha distance (Å). Chains with "
                        "any larger gap (= missing residues in the crystal) are dropped, "
                        "because the model sees a continuous sequence and can't predict "
                        "the missing-residue gap. 3.8 Å is normal CA-CA spacing.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        logger.warning("Output exists at %s — use --force to overwrite", out_path)
        return

    pdb_cache_dir = Path(args.pdb_cache_dir)
    pdb_cache_dir.mkdir(parents=True, exist_ok=True)
    abbind_dir = Path(args.abbind_pdb_dir)

    logger.info("Loading TDC SAbDab_Liberis from %s", args.liberis_path)
    df = pd.read_pickle(args.liberis_path)
    logger.info("  %d chains across %d unique PDBs",
                len(df), df["ID"].apply(lambda x: x.split("_")[0]).nunique())

    # Build (pdb_id, chain, tdc_seq, tdc_paratope) tuples
    requests: list[tuple[str, str, str, list[int]]] = []
    seen_pdbs: set[str] = set()
    for _, row in df.iterrows():
        chain_id = row["ID"]
        if "_" not in chain_id:
            continue
        pdb_id, chain = chain_id.split("_", 1)
        pdb_id = pdb_id.lower()
        chain = chain.upper()
        seq = row["X"]
        paratope = list(row["Y"]) if not isinstance(row["Y"], (int, float)) else []
        requests.append((pdb_id, chain, seq, paratope))
        seen_pdbs.add(pdb_id)

    unique_pdbs = sorted(seen_pdbs)
    if args.max_pdbs > 0 and args.max_pdbs < len(unique_pdbs):
        unique_pdbs_kept = set(unique_pdbs[: args.max_pdbs])
        requests = [r for r in requests if r[0] in unique_pdbs_kept]
        logger.info("  Capping to first %d unique PDBs (%d chains)",
                    args.max_pdbs, len(requests))

    # Download PDBs (use AB-Bind dir as a secondary cache)
    pdb_paths: dict[str, Path | None] = {}
    needed = sorted({r[0] for r in requests})
    logger.info("Downloading %d PDB files (cached → %s)", len(needed), pdb_cache_dir)
    for i, pdb_id in enumerate(needed, 1):
        existing = _find_existing_local_pdb(pdb_id, [pdb_cache_dir, abbind_dir])
        if existing is not None:
            pdb_paths[pdb_id] = existing
            continue
        pdb_paths[pdb_id] = download_pdb(pdb_id, pdb_cache_dir)
        if i % 25 == 0 or i == len(needed):
            n_ok = sum(1 for p in pdb_paths.values() if p is not None)
            logger.info("  Downloaded %d/%d (%d successful so far)", i, len(needed), n_ok)

    n_pdb_ok = sum(1 for p in pdb_paths.values() if p is not None)
    logger.info("Downloads complete: %d/%d PDBs available", n_pdb_ok, len(needed))

    # Extract chains
    logger.info("Extracting Calpha coordinates...")
    results: list[dict] = []
    n_skipped_no_pdb = 0
    n_skipped_no_chain = 0
    n_skipped_short = 0
    n_skipped_mismatch = 0
    n_skipped_gaps = 0
    for pdb_id, chain, tdc_seq, tdc_paratope in requests:
        pdb_path = pdb_paths.get(pdb_id)
        if pdb_path is None:
            n_skipped_no_pdb += 1
            continue
        extracted = extract_chain_seq_and_coords(pdb_path, chain)
        if extracted is None:
            n_skipped_no_chain += 1
            continue
        pdb_seq, coords_np = extracted
        if len(pdb_seq) < 10 or len(pdb_seq) != coords_np.shape[0]:
            n_skipped_short += 1
            continue

        coords_t = torch.tensor(coords_np, dtype=torch.float32)

        # Reject chains where any sequence-adjacent residue pair is
        # physically far apart in 3D — that means there's an unresolved
        # loop / missing residues in the crystal. The model would see a
        # continuous sequence with no gap markers, so the contact labels
        # built from these gappy coords would be wrong (i and i+1 would
        # appear non-contact even though sequence-adjacent).
        adj_gaps = torch.norm(coords_t[1:] - coords_t[:-1], dim=1)
        if adj_gaps.numel() > 0 and adj_gaps.max().item() > args.max_adj_gap_a:
            n_skipped_gaps += 1
            continue

        # Re-align paratope from TDC index space to PDB ATOM index space.
        # If alignment fails (no common subsequence), keep the chain
        # without paratope info — coords are still valid for contact_map
        # and structure_probe.
        paratope_aligned = _align_paratope_to_pdb_seq(tdc_seq, pdb_seq, tdc_paratope)

        results.append({
            "id":        f"{pdb_id}_{chain}",
            "pdb_id":    pdb_id,
            "chain":     chain,
            "sequence":  pdb_seq,
            "coords_ca": coords_t,
            "paratope":  paratope_aligned,
        })

    logger.info(
        "Extraction complete: %d chains kept | %d skipped (no PDB) | "
        "%d skipped (chain not in PDB) | %d skipped (too short / mismatch) | "
        "%d skipped (gappy crystal)",
        len(results), n_skipped_no_pdb, n_skipped_no_chain,
        n_skipped_short + n_skipped_mismatch, n_skipped_gaps,
    )

    if not results:
        logger.error("No chains extracted; aborting")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_path)
    file_mb = out_path.stat().st_size / 1e6
    logger.info("Saved %d entries to %s (%.1f MB)", len(results), out_path, file_mb)

    # Quick distribution summary
    lens = [len(e["sequence"]) for e in results]
    has_para = sum(1 for e in results if e["paratope"])
    logger.info(
        "  Sequence length: min=%d  median=%d  max=%d  mean=%.1f",
        min(lens), sorted(lens)[len(lens) // 2], max(lens),
        sum(lens) / len(lens),
    )
    logger.info("  Chains with paratope labels: %d/%d", has_para, len(results))


if __name__ == "__main__":
    main()
