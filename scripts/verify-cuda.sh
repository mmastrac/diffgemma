#!/usr/bin/env bash
# One-command verification of the whole CUDA path on an NVIDIA box.
#
#   scripts/verify-cuda.sh admin@host
#   DGQ_CUDA_HOST=admin@host scripts/verify-cuda.sh
#
# Mirrors the working tree, then runs, in order:
#   1. gpukit runtime compile + launch (NVRTC -> PTX -> cuLaunchKernel -> result)
#   2. dgemm + dgops tier-1 parity: every CUDA body vs its CPU oracle
#   3. nanogpt forward vs the independent CPU reference
#   4. nanogpt finite-difference gradient check of the backward pass
#   5. a short training run and a sample
set -eo pipefail

HOST="$1"
if [ -z "$HOST" ]; then HOST="$DGQ_CUDA_HOST"; fi
if [ -z "$HOST" ]; then
    echo "usage: $0 <user@host>" >&2
    echo "   or: DGQ_CUDA_HOST=user@host $0" >&2
    exit 2
fi

REMOTE_DIR="$DGQ_CUDA_DIR"
if [ -z "$REMOTE_DIR" ]; then REMOTE_DIR=diffgemma-cuda; fi

HERE="$(cd "$(dirname "$0")" && pwd)"
"$HERE/sync-cuda.sh" "$HOST" --no-test

ssh -o BatchMode=yes "$HOST" "cd $REMOTE_DIR && bash -s" <<'EOS'
set -e
export PATH=$HOME/.cargo/bin:$PATH
echo '=== 1/5 gpukit CUDA driver smoke ==='
cargo test -p gpukit --features cuda --test cuda_smoke -- --nocapture
echo '=== 2/5 dgemm + dgops tier-1 parity on CUDA ==='
cargo test --release -p dgemm --features cuda
cargo test --release -p dgops --features cuda
echo '=== 3/5 nanogpt forward vs CPU reference ==='
cargo run --release -p nanogpt --features cuda -- --check
echo '=== 4/5 nanogpt gradient check ==='
cargo run --release -p nanogpt --features cuda -- --gradcheck
echo '=== 5/5 short train + sample ==='
cargo run --release -p nanogpt --features cuda -- --train --steps 200 --batch 4 --seed 42 --sample --tokens 160
EOS
