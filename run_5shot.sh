#!/usr/bin/env bash
# run_5shot.sh — Evaluate LLaDA-8B-Instruct on Sudoku, 5-shot.
#
# Streams 20 000 easy + 20 000 hard problems from HuggingFace with 5
# in-context examples prepended to each prompt.  No dataset download required.
#
# Usage:
#   bash run_5shot.sh
#   MODEL=GSAI-ML/LLaDA-8B-Instruct bash run_5shot.sh
#
# Environment variable overrides (all optional):
#   MODEL        HuggingFace model ID or path    (default: GSAI-ML/LLaDA-8B-Instruct)
#   DEVICE       torch device                    (default: cuda)
#   BATCH_SIZE   generation batch size           (default: 64)
#   N_SAMPLES    max problems per difficulty     (default: 20000)
#   OUTPUT_DIR   results directory               (default: results/5shot)
#   GEN_LENGTH   tokens to generate             (default: 256)
#   STEPS        diffusion steps                (default: 128)
#   BLOCK_LENGTH block length for semi-AR        (default: 256)
#
# NOTE: 5-shot prompts are ~5x longer than 0-shot prompts.
#       Reduce BATCH_SIZE (e.g. to 16 or 32) if you hit OOM errors.

set -euo pipefail

MODEL="${MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_SAMPLES="${N_SAMPLES:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-results/5shot}"
GEN_LENGTH="${GEN_LENGTH:-256}"
STEPS="${STEPS:-128}"
BLOCK_LENGTH="${BLOCK_LENGTH:-256}"

echo "=================================================="
echo "  LLaDA Sudoku Evaluation — 5-shot"
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
echo ""
echo "  NOTE: 5-shot prompts are ~5x longer than 0-shot."
echo "        Reduce BATCH_SIZE if you hit OOM errors."
echo ""

mkdir -p "$OUTPUT_DIR"

python eval_sudoku.py \
    --model         "$MODEL" \
    --device        "$DEVICE" \
    --difficulty    easy hard \
    --n-samples     "$N_SAMPLES" \
    --batch-size    "$BATCH_SIZE" \
    --few-shot      3 \
    --gen-length    "$GEN_LENGTH" \
    --steps         "$STEPS" \
    --block-length  "$BLOCK_LENGTH" \
    --output-dir    "$OUTPUT_DIR" \
    --n-success-samples 5 \
    --n-failure-samples 10 \
    --seed          42

echo ""
echo "Done. Results saved to $OUTPUT_DIR/"
