#!/usr/bin/env bash
# run_deepseek_0shot.sh — DeepSeek-Math-7B-Instruct, zero-shot, 20k per difficulty.
#
# Overridable env vars:
#   MODEL          (default: deepseek-ai/deepseek-math-7b-instruct)
#   DEVICE         (default: cuda)
#   BATCH_SIZE     (default: 32)
#   N_SAMPLES      (default: 20000)
#   MAX_NEW_TOKENS (default: 512)
#   OUTPUT_DIR     (default: results/deepseek_0shot)

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/deepseek-math-7b-instruct}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-128}"
N_SAMPLES="${N_SAMPLES:-10000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-results/deepseek_0shot}"

echo "=================================================="
echo "  DeepSeek-Math-7B-Instruct — 0-shot Sudoku Eval"
echo "=================================================="
echo "  Model          : $MODEL"
echo "  Device         : $DEVICE"
echo "  Batch size     : $BATCH_SIZE"
echo "  N samples      : $N_SAMPLES per difficulty"
echo "  Max new tokens : $MAX_NEW_TOKENS"
echo "  Output dir     : $OUTPUT_DIR"
echo "=================================================="

mkdir -p "$OUTPUT_DIR"

python eval_sudoku_ar.py \
    --model          "$MODEL" \
    --device         "$DEVICE" \
    --difficulty     medium \
    --n-samples      "$N_SAMPLES" \
    --batch-size     "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --few-shot       0 \
    --output-dir     "$OUTPUT_DIR" \
    --n-success-samples 5 \
    --n-failure-samples 10 \
    --seed           42

echo ""
echo "Done. Results saved to $OUTPUT_DIR/"
