#!/usr/bin/env python3
"""Compare MLX nvfp4 round-trip vs .dgq dequant on real q_proj bf16 weights."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffgemma_parity.vec_stats import vec_stats

TENSOR = "model.decoder.layers.0.self_attn.q_proj.weight"


def fp8_e4m3_to_f32(bits: int) -> float:
    b = bits & 0xFF
    s = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        val = m * (2.0**-9)
    elif e == 15 and m == 7:
        val = float("nan")
    else:
        val = (1.0 + m * 0.125) * (2.0 ** (e - 7))
    return -val if s else val


def e2m1_to_f32(q: int) -> float:
    mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    mag = mags[q & 7]
    return -mag if q & 8 else mag


def dequant_nvfp4_flat(blob: bytes, out_dim: int, in_dim: int) -> np.ndarray:
    gscale = struct.unpack("<f", blob[:4])[0]
    body = memoryview(blob)[4:]
    row_bytes = (in_dim + 1) // 2 + (in_dim + 15) // 16
    data_len = (in_dim + 1) // 2
    out = np.empty(out_dim * in_dim, dtype=np.float32)
    for row in range(out_dim):
        row_base = body[row * row_bytes : (row + 1) * row_bytes]
        data = row_base[:data_len]
        scales = row_base[data_len:]
        base = row * in_dim
        for col in range(in_dim):
            byte = data[col // 2]
            q = (byte & 0x0F) if col % 2 == 0 else (byte >> 4)
            scale = fp8_e4m3_to_f32(scales[col // 16])
            out[base + col] = e2m1_to_f32(q) * scale * gscale
    return out


def load_bf16_q_proj(model_dir: Path) -> tuple[np.ndarray, int, int]:
    import torch

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard = model_dir / index["weight_map"][TENSOR]
    with safe_open(shard, framework="pt") as f:
        w = f.get_tensor(TENSOR)
    out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
    flat = w.float().reshape(-1).numpy().astype(np.float32)
    return flat, out_dim, in_dim


def load_dgq_dequant(dgq_dir: Path, out_dim: int, in_dim: int) -> np.ndarray:
    manifest = json.loads((dgq_dir / "model.dgq.json").read_text())
    entry = next(t for t in manifest["tensors"] if t["name"] == TENSOR)
    blob_path = dgq_dir / manifest["blob_file"]
    off = int(entry["offset"])
    n = int(entry["byte_len"])
    blob = blob_path.read_bytes()[off : off + n]
    return dequant_nvfp4_flat(blob, out_dim, in_dim)


def report(label: str, ref: np.ndarray, cand: np.ndarray) -> None:
    s = vec_stats(ref.tolist(), cand.tolist())
    print(
        f"  {label:28s}: cos={s['cosine']:.6f}  rel_l2={s['rel_l2']:.6f}  "
        f"max_abs={s['max_abs']:.6f}  mean_abs={np.mean(np.abs(cand - ref)):.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=root / "model" / "transformer",
        help="bf16 safetensors source",
    )
    parser.add_argument(
        "--dgq-dir",
        type=Path,
        default=Path("/tmp/nvfp4-weights"),
        help="nvfp4 .dgq directory",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=0,
        help="limit to first N rows (0 = full matrix)",
    )
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        print(f"error: missing model dir {args.model_dir}", file=sys.stderr)
        return 2
    if not (args.dgq_dir / "model.dgq.json").exists():
        print(f"error: missing dgq manifest in {args.dgq_dir}", file=sys.stderr)
        return 2

    orig_flat, out_dim, in_dim = load_bf16_q_proj(args.model_dir)
    if args.rows > 0:
        rows = min(args.rows, out_dim)
        orig_flat = orig_flat[: rows * in_dim].copy()
        out_dim = rows
        print(f"Using first {rows} rows of q_proj ({out_dim}x{in_dim})")
    else:
        print(f"Full q_proj {out_dim}x{in_dim} ({orig_flat.size} weights)")

    orig = mx.array(orig_flat.reshape(out_dim, in_dim))
    mx.eval(orig)

    w_q, scales = mx.quantize(orig, group_size=16, bits=4, mode="nvfp4")
    mlx_hat = mx.dequantize(w_q, scales, group_size=16, bits=4, mode="nvfp4")
    mx.eval(w_q, scales, mlx_hat)
    mlx_flat = np.array(mx.array(mlx_hat, dtype=mx.float32).reshape(-1))

    dgq_flat = load_dgq_dequant(args.dgq_dir, out_dim, in_dim)
    if dgq_flat.size != orig_flat.size:
        dgq_flat = dgq_flat[: orig_flat.size]

    print(f"\n{TENSOR}  bf16 source: {args.model_dir}")
    print(f"  dgq: {args.dgq_dir}\n")
    report("bf16 vs MLX nvfp4 roundtrip", orig_flat, mlx_flat)
    report("bf16 vs .dgq dequant (ours)", orig_flat, dgq_flat)
    report("MLX roundtrip vs .dgq dequant", mlx_flat, dgq_flat)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
