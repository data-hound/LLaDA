#!/usr/bin/env bash
# run_deepseek_5shot.sh — DeepSeek-Math-7B-Instruct, 5-shot, 20k per difficulty.
#
# NOTE: 5-shot prompts are ~5x longer. Reduce BATCH_SIZE if you hit OOM.
#
# Overridable env vars:
#   MODEL          (default: deepseek-ai/deepseek-math-7b-instruct)
#   DEVICE         (default: cuda)
#   BATCH_SIZE     (default: 16)
#   N_SAMPLES      (default: 20000)
#   MAX_NEW_TOKENS (default: 512)
#   OUTPUT_DIR     (default: results/deepseek_5shot)

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/deepseek-math-7b-instruct}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_SAMPLES="${N_SAMPLES:-10000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-results/deepseek_5shot}"

echo "=================================================="
echo "  DeepSeek-Math-7B-Instruct — 5-shot Sudoku Eval"
echo "=================================================="
echo "  Model          : $MODEL"
echo "  Device         : $DEVICE"
echo "  Batch size     : $BATCH_SIZE"
echo "  N samples      : $N_SAMPLES per difficulty"
echo "  Max new tokens : $MAX_NEW_TOKENS"
echo "  Output dir     : $OUTPUT_DIR"
echo "=================================================="
echo ""
echo "  NOTE: 5-shot prompts are much longer. Lower BATCH_SIZE if OOM."
echo ""

mkdir -p "$OUTPUT_DIR"

python eval_sudoku_ar.py \
    --model          "$MODEL" \
    --device         "$DEVICE" \
    --difficulty     easy hard \
    --n-samples      "$N_SAMPLES" \
    --batch-size     "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --few-shot       3 \
    --output-dir     "$OUTPUT_DIR" \
    --n-success-samples 5 \
    --n-failure-samples 10 \
    --seed           42

echo ""
echo "Done. Results saved to $OUTPUT_DIR/"
