#!/usr/bin/env bash
# Thorough holistic prefill BO (task #88): 9-dim search (attn QK/PV tiles, HC,
# softmax TPG, dense-GEMM tile, MoE-sparse N-tile) against the faithful
# super-chunk proxy, at multiple kv brackets — to see if the optimum shifts with
# kv (which would justify a kv-adaptive config). One-process-at-a-time.
set -u
cd "$(dirname "$0")/.."          # -> python/
BIN=../target/release/diffgemma-mps
M=../model/diffusiongemma-q4emb
DB="sqlite:///$(pwd)/holistic.db"
TRIALS=${TRIALS:-80}
ITERS=${ITERS:-3}

for KV in 2000 8000 30000 100000; do
  echo "==================== KV=$KV ===================="
  echo -n "### baseline (default 64x64 / moe_bn 128): "
  $BIN -m $M bench-prefill-super --kv-len "$KV" --iters "$ITERS" 2>/dev/null | grep -oE '"ms": [0-9.]+'
  uv run python scripts/tune_prefill_attn.py --proxy --kv-len "$KV" \
    --trials "$TRIALS" --iters "$ITERS" --model "$M" \
    --storage "$DB" --study-name "kv$KV" --seed 1
done
echo "==================== SWEEP DONE ===================="
