#!/bin/bash
# Launch IgFold structure prediction across multiple GPUs in parallel.
#
# Each GPU gets a contiguous slice of the input JSONL. After all shards
# finish, run merge_igfold_shards.py to assemble a single .pt file in
# the same format as data/structures/oas_vh_500k_coords.pt.
#
# Usage:
#   bash scripts/predict_structures_igfold_launch.sh \
#       data/processed/oas_vh_500k.jsonl \
#       data/structures/oas_vh_500k_igfold \
#       0,1,2                # comma-separated GPU IDs to use
#       1                    # IgFold ensemble size (1-4)
#       32                   # k_neighbors

set -euo pipefail

INPUT="${1:?usage: launch.sh INPUT OUTPUT_PREFIX GPU_IDS [NUM_MODELS=1] [K=32]}"
OUTPUT_PREFIX="${2:?missing OUTPUT_PREFIX}"
GPU_IDS="${3:?missing GPU_IDS (e.g. 0,1,2)}"
NUM_MODELS="${4:-1}"
K_NEIGHBORS="${5:-32}"

PYTHON="${PYTHON:-/usr/scratch/agoel320/miniconda3/envs/abmlm/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse GPU_IDS into an array
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

N_TOTAL=$(wc -l < "$INPUT")
echo "Total sequences: $N_TOTAL"
echo "Sharding across $NUM_GPUS GPUs: ${GPU_ARRAY[*]}"
echo "IgFold ensemble: $NUM_MODELS models, k=$K_NEIGHBORS"

CHUNK=$(( (N_TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
echo "Chunk size: $CHUNK"

LOG_DIR="$(dirname "$OUTPUT_PREFIX")/igfold_logs"
mkdir -p "$LOG_DIR"

PIDS=()
for ((shard=0; shard<NUM_GPUS; shard++)); do
    GPU_ID="${GPU_ARRAY[$shard]}"
    START=$(( shard * CHUNK ))
    END=$(( START + CHUNK ))
    if [[ $END -gt $N_TOTAL ]]; then END=$N_TOTAL; fi

    SHARD_OUT="${OUTPUT_PREFIX}_shard${shard}.pt"
    LOG_FILE="${LOG_DIR}/shard${shard}_gpu${GPU_ID}.log"
    echo "Shard $shard on GPU $GPU_ID: sequences [$START, $END) -> $SHARD_OUT  (log: $LOG_FILE)"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    nohup "$PYTHON" "$SCRIPT_DIR/predict_structures_igfold.py" \
        --input "$INPUT" \
        --output "$SHARD_OUT" \
        --k_neighbors "$K_NEIGHBORS" \
        --num_models "$NUM_MODELS" \
        --start_idx "$START" \
        --end_idx "$END" \
        --checkpoint_every 1000 \
        --resume \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} workers: ${PIDS[*]}"
echo "Tail any log with: tail -f ${LOG_DIR}/shard0_gpu${GPU_ARRAY[0]}.log"
echo "Waiting for all shards to complete..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "Worker PID $pid failed"
        FAIL=1
    fi
done

if [[ $FAIL -eq 0 ]]; then
    echo "All shards finished successfully."
    echo "Run: $PYTHON $SCRIPT_DIR/merge_igfold_shards.py \\"
    echo "    --shard_prefix $OUTPUT_PREFIX --num_shards $NUM_GPUS \\"
    echo "    --output ${OUTPUT_PREFIX}.pt"
else
    echo "One or more shards failed; check logs in $LOG_DIR/"
    exit 1
fi
