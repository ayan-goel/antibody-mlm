#!/usr/bin/env bash
# Launch the remaining Table 2b seed replicates (hybrid reverse / perbatch /
# weighted, seeds 1 and 2) across 3 GPUs, one detached shell per shard.
#
#   bash scripts/launch_table2b.sh            # GPUs 0,1,2
#   GPUS="1 2 3" bash scripts/launch_table2b.sh
#   DRY_RUN=1 bash scripts/launch_table2b.sh  # print the plan only
#
# MANIFEST and TAG override the run list and the driver-log prefix, so the same
# launcher drives any sharded sweep. The 2026-08-14 restart used:
#
#   MANIFEST=configs/seeds/MANIFEST.rebuttal.tsv TAG=rebuttal \
#       bash scripts/launch_table2b.sh
#
# which is Table 2b plus the two experiments wiki/rebuttal.md §11 still owed:
# the interface label-permutation control and hybrid-adaptive at seeds 1,2.
#
# Safe to re-run: run_seed_replicates.sh skips any run whose final/ exists,
# so an interrupted sweep resumes at the first unfinished model.
#
# Context: the first attempt (2026-08-13 16:17) died at ~13% when the host
# rebooted with a mismatched NVIDIA driver. All six runs start from step 0 —
# scripts/train.py has no resume-from-checkpoint path.
#
# Budget: ~5.1 h/run, 2 runs per shard -> ~10.2 h wall clock, plus evaluation.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MANIFEST:-$REPO/configs/seeds/MANIFEST.table2b.tsv}"
GPUS="${GPUS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
TAG="${TAG:-table2b}"

read -r -a GPU_ARR <<<"$GPUS"
NUM_SHARDS="${#GPU_ARR[@]}"

[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST" >&2; exit 1; }

if [[ "$DRY_RUN" != "1" ]] && ! nvidia-smi -L >/dev/null 2>&1; then
    echo "REFUSING TO LAUNCH: nvidia-smi is not functional." >&2
    echo "  $(nvidia-smi -L 2>&1 | head -1)" >&2
    echo "  kernel module: $(sed -n 's/.*NVRM version: \(.*\) Release.*/\1/p' /proc/driver/nvidia/version)" >&2
    echo "  userspace lib: $(basename "$(readlink -f /usr/lib64/libnvidia-ml.so.1)" 2>/dev/null)" >&2
    echo "  fix (root): rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && modprobe nvidia" >&2
    exit 1
fi

cd "$REPO"
mkdir -p logs

for i in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$i]}"
    log="$REPO/logs/${TAG}_shard${i}.log"
    echo "shard $i/$NUM_SHARDS -> GPU $gpu  (driver log: $log)"
    if [[ "$DRY_RUN" == "1" ]]; then
        DRY_RUN=1 MANIFEST="$MANIFEST" bash scripts/run_seed_replicates.sh "$i" "$NUM_SHARDS" "$gpu"
    else
        RUN_EVAL=1 MANIFEST="$MANIFEST" \
            nohup bash scripts/run_seed_replicates.sh "$i" "$NUM_SHARDS" "$gpu" \
            >"$log" 2>&1 &
        echo "  pid $!"
    fi
done

[[ "$DRY_RUN" == "1" ]] && exit 0

wait_note="tail -f $REPO/logs/${TAG}_shard*.log"
cat <<EOF

launched $NUM_SHARDS shard(s). watch with:
  $wait_note

when all shards report 0 failures:
  python scripts/compare.py && python scripts/aggregate_seeds.py --paper-metrics-only
EOF
