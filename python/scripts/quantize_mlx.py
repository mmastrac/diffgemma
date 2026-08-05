#!/usr/bin/env python3
"""
Convert cached HuggingFace DiffusionGemma weights to MLX for low-RAM inference on Apple Silicon.

This does NOT produce `.dgq` (Rust mmap format). Use `diffgemma quantize` for that.
MLX output (~15–22 GiB on disk) loads in Python via mlx-vlm without materializing 48 GiB bf16.

Example (requires `uv sync --extra mlx` and a local HF cache):

  cd python
  uv sync --extra mlx
  uv run python scripts/quantize_mlx.py \\
    --hf-path google/diffusiongemma-26B-A4B-it \\
    -o ../model/mlx-mxfp4

Then generate (mlx-vlm CLI; denoise trace hookup is separate):

  uv run python -m mlx_vlm.generate \\
    --model ../model/mlx-mxfp4 \\
    --prompt "Hello" --max-tokens 64 --temp 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_HF_ID = "google/diffusiongemma-26B-A4B-it"
# mlx mxfp4/nvfp4 modes require group size 32; affine defaults to 64.
MODE_DEFAULT_GROUP: dict[str, int] = {
    "mxfp4": 32,
    "nvfp4": 32,
    "mxfp8": 32,
    "affine": 64,
}


def resolve_group_size(q_mode: str, explicit: int | None) -> int:
    required = MODE_DEFAULT_GROUP.get(q_mode, 64)
    if explicit is None:
        return required
    if q_mode in ("mxfp4", "nvfp4", "mxfp8") and explicit != 32:
        raise SystemExit(
            f"error: {q_mode} quantization requires group size 32 (got {explicit})"
        )
    return explicit


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_hf_path() -> str:
    """Prefer repo symlink `model/transformer` when present."""
    link = repo_root() / "model" / "transformer"
    if link.is_dir():
        return str(link.resolve())
    return DEFAULT_HF_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-path",
        default=None,
        help=f"HF hub id or local snapshot dir (default: {DEFAULT_HF_ID} or ../model/transformer)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output directory for MLX weights",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        choices=(4, 5, 6, 8),
        help="weight bits (default: 4)",
    )
    parser.add_argument(
        "--q-mode",
        default="mxfp4",
        choices=("affine", "mxfp4", "nvfp4", "mxfp8"),
        help="quantization mode (mxfp4 ~15 GiB; affine 4-bit ~19 GiB)",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=None,
        help="quant group size (default: 32 for mxfp4/nvfp4/mxfp8, 64 for affine)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="non-quantized tensor dtype",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="HF revision/branch when pulling from the Hub",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hf_path = args.hf_path or default_hf_path()
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    group_size = resolve_group_size(args.q_mode, args.q_group_size)

    try:
        from mlx_vlm.convert import convert
    except ImportError:
        print(
            "error: mlx-vlm not installed. Run: cd python && uv sync --extra mlx",
            file=sys.stderr,
        )
        return 2

    print(f"convert: {hf_path} -> {out}")
    print(f"  quantize: {args.q_bits}-bit mode={args.q_mode} group={group_size}")
    print("  note: streams weights lazily; peak RAM is much lower than 48 GiB bf16")
    print("  note: output is MLX format, not .dgq — Rust inference still uses diffgemma quantize")

    convert(
        hf_path=hf_path,
        mlx_path=str(out),
        quantize=True,
        q_bits=args.q_bits,
        q_group_size=group_size,
        q_mode=args.q_mode,
        dtype=args.dtype,
        revision=args.revision,
        trust_remote_code=True,
    )

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"done: {out} ({total / (1024**3):.2f} GiB on disk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
