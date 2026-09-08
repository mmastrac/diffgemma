#!/usr/bin/env bash
# Mirror the working tree to the CUDA box and (by default) run the CUDA smoke
# test there. The host comes from the first argument or DGQ_CUDA_HOST.
#
#   scripts/sync-cuda.sh admin@host            # sync + smoke test
#   scripts/sync-cuda.sh admin@host --no-test  # sync only
#
# Heavy, non-source directories (target/, .git/, model packs, python venv) are
# excluded so a sync is a few MB.
set -eo pipefail

HOST="$1"
if [ -z "$HOST" ]; then HOST="$DGQ_CUDA_HOST"; fi
if [ -z "$HOST" ]; then
    echo "usage: $0 <user@host> [--no-test]" >&2
    echo "   or: DGQ_CUDA_HOST=user@host $0" >&2
    exit 2
fi
RUN_TEST=1
if [ "$#" -ge 2 ] && [ "$2" = "--no-test" ]; then RUN_TEST=0; fi

REMOTE_DIR="$DGQ_CUDA_DIR"
if [ -z "$REMOTE_DIR" ]; then REMOTE_DIR=diffgemma-cuda; fi

rsync -az --delete -e "ssh -o BatchMode=yes" \
    --exclude 'target/' \
    --exclude '.git/' \
    --exclude '.claude/' \
    --exclude 'model/' \
    --exclude 'python/.venv/' \
    --exclude '.venv/' \
    --exclude 'debug/' \
    --exclude 'runs/' \
    --exclude '*.dgq' \
    --exclude '.DS_Store' \
    ./ "$HOST:$REMOTE_DIR/"

echo "synced to $HOST:$REMOTE_DIR"

if [ "$RUN_TEST" -eq 1 ]; then
    ssh -o BatchMode=yes "$HOST" "cd $REMOTE_DIR && ~/.cargo/bin/cargo test -p gpukit --features cuda --test cuda_smoke -- --nocapture"
fi
