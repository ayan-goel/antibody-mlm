#!/usr/bin/env bash
# Launch the pretraining seed replicates, sharded across GPUs.
#
#   run_seed_replicates.sh <shard_index> <num_shards> [gpu_id]
#
# Runs every num_shards-th entry of configs/seeds/MANIFEST.tsv, so each shard
# gets a disjoint slice. `gpu_id` is the PHYSICAL device passed to
# CUDA_VISIBLE_DEVICES and defaults to shard_index — pass it explicitly when
# the free GPUs are not 0..N-1.
#
#   # two shards on physical GPUs 3 and 7:
#   bash scripts/run_seed_replicates.sh 0 2 3
#   bash scripts/run_seed_replicates.sh 1 2 7
#
# Training is ~5.27 h/run (measured). 16 training runs:
#   2 shards -> 8 runs each -> ~42 h wall clock
#   4 shards -> 4 runs each -> ~21 h
#
# Idempotent: a run whose checkpoint already has final/ is skipped, so you can
# re-launch after an interruption without redoing finished work.
#
# Env overrides:
#   PYTHON=/path/to/python      (default: the abmlm env)
#   MANIFEST=path/to/MANIFEST.tsv
#   RUN_EVAL=1                  run the full evaluation suite after each model
#   DRY_RUN=1                   print the plan without running anything
#
# Evaluation deliberately passes NO sample-count overrides, so the replicates
# use the identical protocol behind the published tables: 1000 infilling
# samples and 500 PLL sequences (run_all_evaluations.py defaults, confirmed
# against infill_cdr3_exact_match_count / pll_num_sequences in the existing
# evaluation_outputs). Error bars are only meaningful if they attach to the
# numbers actually in the paper. INFILL_SAMPLES / PLL_SEQUENCES exist for a
# separate higher-precision sweep and must be left unset here.

set -uo pipefail

SHARD="${1:?usage: run_seed_replicates.sh <shard_index> <num_shards> [gpu_id]}"
NUM_SHARDS="${2:?usage: run_seed_replicates.sh <shard_index> <num_shards> [gpu_id]}"
GPU_ID="${3:-$SHARD}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/usr/scratch/agoel320/miniconda3/envs/abmlm/bin/python}"
MANIFEST="${MANIFEST:-$REPO/configs/seeds/MANIFEST.tsv}"
LOG_DIR="$REPO/logs"
DRY_RUN="${DRY_RUN:-0}"
RUN_EVAL="${RUN_EVAL:-0}"
# Empty by default = use run_all_evaluations.py's own defaults = the paper's
# protocol. Only set these for a deliberate, separately-reported sweep.
INFILL_SAMPLES="${INFILL_SAMPLES:-}"
PLL_SEQUENCES="${PLL_SEQUENCES:-}"

EVAL_FLAGS=()
[[ -n "$INFILL_SAMPLES" ]] && EVAL_FLAGS+=(--max-infilling-samples "$INFILL_SAMPLES")
[[ -n "$PLL_SEQUENCES"  ]] && EVAL_FLAGS+=(--max-pll-sequences  "$PLL_SEQUENCES")

[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST (run scripts/make_seed_configs.py)" >&2; exit 1; }
[[ -x "$PYTHON"   ]] || { echo "python not executable: $PYTHON" >&2; exit 1; }
(( SHARD < NUM_SHARDS )) || { echo "shard_index ($SHARD) must be < num_shards ($NUM_SHARDS)" >&2; exit 1; }
mkdir -p "$LOG_DIR"

cd "$REPO"
echo "shard $SHARD/$NUM_SHARDS on GPU $GPU_ID | eval=$RUN_EVAL | python=$PYTHON"

i=0
failed=0
while IFS=$'\t' read -r config name runner; do
    [[ -z "${config:-}" ]] && continue
    # Round-robin: this shard takes entries where idx % NUM_SHARDS == SHARD.
    if (( i % NUM_SHARDS != SHARD )); then i=$((i+1)); continue; fi
    i=$((i+1))

    log="$LOG_DIR/$name.log"

    case "$runner" in
        train)     script="scripts/train.py" ;;
        untrained) script="scripts/create_untrained_baseline.py" ;;
        *)         echo "SKIP  $name (unknown runner '$runner')" >&2; continue ;;
    esac

    if [[ -d "$REPO/models/checkpoints/$name/final" ]]; then
        echo "SKIP  $name (already has final/)"
    else
        echo "RUN   $name  ($script)  -> $log"
        if [[ "$DRY_RUN" != "1" ]]; then
            if CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$script" --config "$config" >>"$log" 2>&1; then
                echo "DONE  $name"
            else
                echo "FAIL  $name (train, exit $?) — see $log" >&2
                failed=$((failed+1))
                continue   # don't evaluate a model that failed to train
            fi
        fi
    fi

    if [[ "$RUN_EVAL" == "1" ]]; then
        # Registered in configs/experiments.yaml by make_seed_configs.py.
        # No sample-count overrides by default — same protocol as the paper.
        echo "EVAL  $name  ${EVAL_FLAGS[*]:-(paper defaults)}  -> $log"
        if [[ "$DRY_RUN" != "1" ]]; then
            if CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/run_all_evaluations.py \
                    --experiments "$name" \
                    --device cuda \
                    "${EVAL_FLAGS[@]}" \
                    >>"$log" 2>&1; then
                echo "DONE  $name (eval)"
            else
                echo "FAIL  $name (eval, exit $?) — see $log" >&2
                failed=$((failed+1))
            fi
        fi
    fi
done < "$MANIFEST"

echo "shard $SHARD/$NUM_SHARDS finished with $failed failure(s)"
if [[ "$RUN_EVAL" == "1" && "$failed" == "0" ]]; then
    echo "next: python scripts/compare.py && python scripts/aggregate_seeds.py --paper-metrics-only"
fi
exit $(( failed > 0 ))
