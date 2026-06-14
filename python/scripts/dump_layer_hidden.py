#!/usr/bin/env python3
"""Dump per-layer hidden states at one canvas row (MLX reference for monolithic parity)."""

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


def dump_mlx_layer_hidden(args: argparse.Namespace) -> dict:
    import mlx.core as mx
    from mlx_vlm.generate.diffusion import _diffusion_prefill_cache
    from mlx_vlm.models.diffusion_gemma.language import _cache_offset
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
    mask_mapping = decoder._make_decoder_masks(current_canvas[..., None], kv_cache, None)
    offset = _cache_offset(kv_cache[0]) if kv_cache else 0

    h = decoder._embed_canvas(current_canvas, None, None)
    mx.eval(h)
    pos = int(args.position)
    checkpoints: list[dict] = []

    def push(label: str, layer: int | None, hidden: mx.array) -> None:
        row = [float(x) for x in hidden[0, pos].tolist()]
        checkpoints.append(
            {
                "label": label,
                "layer": layer,
                "hidden": row,
                "hidden_l2": _vec_l2(row),
                "hidden_max_abs": _vec_max_abs(row),
            }
        )

    push("after_preamble", None, h)

    for layer_idx, (layer, cache) in enumerate(zip(decoder.layers, kv_cache)):
        h = layer(
            h,
            mask_mapping.get(layer.layer_type),
            cache,
            decoder=True,
            offset=offset,
        )
        mx.eval(h)
        push(f"after_layer_{layer_idx}", layer_idx, h)

    h_norm = decoder.norm(h)
    mx.eval(h_norm)
    n_layers = len(decoder.layers)
    push("after_final_norm", n_layers, h_norm)

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "mlx-python",
        "prompt": args.prompt,
        "prompt_token_ids": prompt_ids_list,
        "initial_canvas_ids": canvas_list,
        "seed": args.seed,
        "layers": n_layers,
        "position": pos,
        "canvas_token": int(canvas_list[pos]),
        "checkpoints": checkpoints,
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
        payload = dump_mlx_layer_hidden(args)
    except ImportError as exc:
        print("error: mlx-vlm required (cd python && uv sync --extra mlx)", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output} (pos={payload['position']}, {len(payload['checkpoints'])} checkpoints)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
