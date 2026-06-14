#!/usr/bin/env python3
"""Dump one embed table row from MLX (scaled) for parity vs rust embed-row-dump."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _vec_l2(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _embed_row_weight(embed_tokens, token: int):
    import mlx.core as mx
    import mlx.nn as nn

    if isinstance(embed_tokens, nn.QuantizedEmbedding):
        kwargs = {"group_size": embed_tokens.group_size, "bits": embed_tokens.bits}
        if embed_tokens.biases is None:
            kwargs["mode"] = "mxfp4"
            return mx.dequantize(embed_tokens.weight, embed_tokens.scales, **kwargs)[token]
        return mx.dequantize(
            embed_tokens.weight,
            embed_tokens.scales,
            embed_tokens.biases,
            **kwargs,
        )[token]
    return embed_tokens.weight[token]


def dump_mlx_embed_row(args: argparse.Namespace) -> dict:
    import mlx.core as mx
    from mlx_vlm.utils import load

    model, _processor = load(args.model, lazy=False)
    decoder = model.model.decoder
    token = int(args.token)
    row = _embed_row_weight(decoder.embed_tokens, token).astype(mx.float32)
    scaled = row * decoder.embed_scale
    mx.eval(row, scaled)
    raw = [float(x) for x in row.tolist()]
    scaled_list = [float(x) for x in scaled.tolist()]
    return {
        "schema_version": 1,
        "source": "mlx-python",
        "token": token,
        "hidden_dim": len(raw),
        "embed_scale": float(decoder.embed_scale),
        "dequant_raw": raw,
        "dequant_scaled": scaled_list,
        "dequant_raw_l2": _vec_l2(raw),
        "dequant_scaled_l2": _vec_l2(scaled_list),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--token", type=int, default=71153)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model is None:
        root = Path(__file__).resolve().parents[2]
        local = root / "model" / "mlx-mxfp4"
        args.model = str(local if local.is_dir() else "mlx-community/diffusiongemma-26B-A4B-it-mxfp4")

    try:
        payload = dump_mlx_embed_row(args)
    except ImportError as exc:
        print("error: mlx-vlm required (cd python && uv sync --extra mlx)", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output} (token={payload['token']}, l2={payload['dequant_scaled_l2']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
