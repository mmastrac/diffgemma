#!/usr/bin/env python3
"""Dump step-1 preamble hidden (canvas embed) from MLX before layer 0."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffgemma_parity.canvas_rng import initialize_canvas

SCHEMA_VERSION = 1


def _vec_l2(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _vec_max_abs(v: list[float]) -> float:
    return max(abs(x) for x in v) if v else 0.0


def dump_mlx_preamble(args: argparse.Namespace) -> dict:
    import mlx.core as mx
    from mlx_vlm.generate.diffusion import _diffusion_prefill_cache
    from mlx_vlm.utils import load

    mx.random.seed(args.seed)
    model, processor = load(args.model, lazy=False)
    messages = [{"role": "user", "content": args.prompt}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True
    )
    prompt_ids = inputs["input_ids"]
    prompt_ids_list = (
        prompt_ids[0].tolist() if getattr(prompt_ids, "ndim", 1) > 1 else prompt_ids.tolist()
    )
    input_ids = mx.array([prompt_ids_list])
    text_config = model.config.text_config
    vocab_size = int(text_config.vocab_size)
    canvas_length = min(int(model.config.canvas_length), int(args.max_new_tokens))

    kv_cache = model.make_cache()
    kv_cache = _diffusion_prefill_cache(
        model,
        input_ids,
        attention_mask=None,
        kv_cache=kv_cache,
        pixel_values=None,
        mm_token_type_ids=None,
        prefill_step_size=None,
        chunk_prefill=False,
    )
    mx.eval([c.state for c in kv_cache])

    canvas_list = initialize_canvas(args.seed, canvas_length, vocab_size, rng=args.canvas_rng)
    current_canvas = mx.array([canvas_list], dtype=input_ids.dtype)
    decoder = model.model.decoder
    pos = int(args.position)

    embed_scaled = decoder.embed_tokens(current_canvas) * decoder.embed_scale
    after_preamble = decoder._embed_canvas(current_canvas, None, None)
    mx.eval(embed_scaled, after_preamble)

    embed_row = [float(x) for x in embed_scaled[0, pos].tolist()]
    preamble_row = [float(x) for x in after_preamble[0, pos].tolist()]

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "mlx-python",
        "prompt": args.prompt,
        "prompt_token_ids": prompt_ids_list,
        "initial_canvas_ids": canvas_list,
        "seed": args.seed,
        "position": pos,
        "canvas_token": int(canvas_list[pos]),
        "kv_len": len(prompt_ids_list),
        "embed_scaled": embed_row,
        "embed_scaled_l2": _vec_l2(embed_row),
        "embed_scaled_max_abs": _vec_max_abs(embed_row),
        "after_preamble": preamble_row,
        "after_preamble_l2": _vec_l2(preamble_row),
        "after_preamble_max_abs": _vec_max_abs(preamble_row),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("-p", "--prompt", default="Hello")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--canvas-rng", choices=("rust", "mlx"), default="rust")
    parser.add_argument("--position", type=int, default=129)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model is None:
        root = Path(__file__).resolve().parents[2]
        local = root / "model" / "mlx-mxfp4"
        args.model = str(local if local.is_dir() else "mlx-community/diffusiongemma-26B-A4B-it-mxfp4")

    try:
        payload = dump_mlx_preamble(args)
    except ImportError as exc:
        print("error: mlx-vlm required (cd python && uv sync --extra mlx)", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output} (pos={payload['position']}, token={payload['canvas_token']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
