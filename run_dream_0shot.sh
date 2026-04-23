#!/usr/bin/env bash
# run_0shot.sh — Evaluate LLaDA-8B-Instruct on Sudoku, zero-shot.
#
# Streams 20 000 easy + 20 000 hard problems from HuggingFace with no
# in-context examples.  No dataset download required.
#
# Usage:
#   bash run_0shot.sh
#   MODEL=GSAI-ML/LLaDA-8B-Instruct bash run_0shot.sh
#
# Environment variable overrides (all optional):
#   MODEL        HuggingFace model ID or path    (default: GSAI-ML/LLaDA-8B-Instruct)
#   DEVICE       torch device                    (default: cuda)
#   BATCH_SIZE   generation batch size           (default: 64)
#   N_SAMPLES    max problems per difficulty     (default: 20000)
#   OUTPUT_DIR   results directory               (default: results/0shot)
#   GEN_LENGTH   tokens to generate             (default: 256)
#   STEPS        diffusion steps                (default: 128)
#   BLOCK_LENGTH block length for semi-AR        (default: 256)

set -euo pipefail

# MODEL="${MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
MODEL="${MODEL:-Dream-org/Dream-v0-Instruct-7B}"
DEVICE="${DEVICE:-cuda}"
# BATCH_SIZE="${BATCH_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_SAMPLES="${N_SAMPLES:-10000}"
# N_SAMPLES="${N_SAMPLES:-20}"
# OUTPUT_DIR="${OUTPUT_DIR:-results/0shot}"
OUTPUT_DIR="${OUTPUT_DIR:-results_sudoku/0shot_dream}"
GEN_LENGTH="${GEN_LENGTH:-256}"
STEPS="${STEPS:-128}"
BLOCK_LENGTH="${BLOCK_LENGTH:-256}"
DIFFICULTY="hard"

echo "=================================================="
echo "  Dream Sudoku Evaluation — 0-shot"
echo "  Dataset: beta3/GridCorpus_9M_Sudoku_Puzzles_Enriched (HuggingFace)"
echo "=================================================="
echo "  Model       : $MODEL"
echo "  Device      : $DEVICE"
echo "  Batch size  : $BATCH_SIZE"
echo "  N samples   : $N_SAMPLES per difficulty"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Gen length  : $GEN_LENGTH"
echo "  Steps       : $STEPS"
echo "  Block length: $BLOCK_LENGTH"
echo "=================================================="

mkdir -p "$OUTPUT_DIR"

python eval_sudoku.py \
    --model         "$MODEL" \
    --device        "$DEVICE" \
    --difficulty    "$DIFFICULTY" \
    --n-samples     "$N_SAMPLES" \
    --batch-size    "$BATCH_SIZE" \
    --few-shot      0 \
    --gen-length    "$GEN_LENGTH" \
    --steps         "$STEPS" \
    --block-length  "$BLOCK_LENGTH" \
    --output-dir    "$OUTPUT_DIR" \
    --n-success-samples 5 \
    --n-failure-samples 10 \
    --seed          42      2>&1 | tee "$OUTPUT_DIR/run_${MODEL##*/}_${DIFFICULTY}_output.log"

echo ""
echo "Done. Results saved to $OUTPUT_DIR/"
