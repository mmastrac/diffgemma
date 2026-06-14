#!/usr/bin/env python3
"""Dump single-expert MoE output from MLX (router bypassed) for step-moe-single-dump parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffgemma_parity.canvas_rng import initialize_canvas

SCHEMA_VERSION = 1


def dump_mlx_single_expert(args: argparse.Namespace) -> dict:
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
    kv_len = int(offset)
    pos = int(args.position)
    layer_idx = int(args.layer)
    expert_id = int(args.expert)

    h = decoder._embed_canvas(current_canvas, None, None)
    mx.eval(h)

    for i in range(layer_idx):
        layer = decoder.layers[i]
        h = layer(
            h,
            mask_mapping.get(layer.layer_type),
            kv_cache[i],
            decoder=True,
            offset=offset,
        )
        mx.eval(h)

    target = decoder.layers[layer_idx]
    residual = h
    h_ln = target.input_layernorm(h)
    attn = target.self_attn(
        h_ln,
        mask_mapping.get(target.layer_type),
        kv_cache[layer_idx],
        decoder=True,
        offset=offset,
    )
    attn = target.post_attention_layernorm(attn)
    post_attn = residual + attn
    mx.eval(post_attn)

    flat = post_attn.reshape(-1, post_attn.shape[-1])
    h2_in = target.pre_feedforward_layernorm_2(flat)
    mx.eval(h2_in)

    row_in = h2_in[pos : pos + 1]
    indices = mx.array([[expert_id]], dtype=mx.uint32)
    weights = mx.array([[1.0]], dtype=mx.float32)
    expert_out = target.experts(row_in, indices, weights)
    mx.eval(expert_out)

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "mlx-python",
        "prompt": args.prompt,
        "prompt_token_ids": prompt_ids_list,
        "initial_canvas_ids": canvas_list,
        "seed": args.seed,
        "layer": layer_idx,
        "position": pos,
        "expert_id": expert_id,
        "canvas_token": int(canvas_list[pos]),
        "kv_len": kv_len,
        "moe_in": [float(x) for x in h2_in[pos].tolist()],
        "expert_out": [float(x) for x in expert_out[0].tolist()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("-p", "--prompt", default="Hello")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--canvas-rng", choices=("rust", "mlx"), default="rust")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--position", type=int, default=129)
    parser.add_argument("--expert", type=int, default=18)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model is None:
        root = Path(__file__).resolve().parents[2]
        local = root / "model" / "mlx-mxfp4"
        args.model = str(local if local.is_dir() else "mlx-community/diffusiongemma-26B-A4B-it-mxfp4")

    try:
        payload = dump_mlx_single_expert(args)
    except ImportError as exc:
        print("error: mlx-vlm required (cd python && uv sync --extra mlx)", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output} (layer={payload['layer']}, pos={payload['position']}, "
        f"expert={payload['expert_id']}, kv_len={payload['kv_len']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
