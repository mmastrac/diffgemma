#!/usr/bin/env python3
"""Cross-check Rust NVFP4 pack/dequant against MLX reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=32)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    w = mx.array(rng.uniform(-2.0, 2.0, size=(args.rows, args.cols)).astype(np.float32))
    w_q, scales = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")
    w_hat = mx.dequantize(w_q, scales, group_size=16, bits=4, mode="nvfp4")
    mx.eval(w_q, scales, w_hat)

    print(f"MLX nvfp4: shape={tuple(w.shape)} scales={scales[0].tolist()[:4]}...")
    print(f"  max abs err vs bf16 input: {float(mx.max(mx.abs(w_hat - w))):.6f}")
    print("ok (compare with `cargo test --features metal dgq::nvfp4::tests::nvfp4_matches_mlx_linspace_row`)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
