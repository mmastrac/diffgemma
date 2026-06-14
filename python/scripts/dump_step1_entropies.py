#!/usr/bin/env python3
"""Dump all 256 step-1 canvas entropies + argmax from MLX (reference) or read Rust log."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffgemma_parity.canvas_rng import initialize_canvas
from diffgemma_parity.denoise_stats import ENTROPY_BOUND, accept_count_from_entropies


def dump_mlx_step1(args: argparse.Namespace) -> dict:
    import mlx.core as mx
    from mlx_vlm.generate.diffusion import (
        DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH,
        _diffusion_config_dict,
        _diffusion_linear_temperature,
        _diffusion_prefill_cache,
        _diffusion_token_entropy,
        _make_diffusion_decoder_logits_fns,
    )
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
    generation_config = _diffusion_config_dict(getattr(model.config, "generation_config", None))
    temperature_config = _diffusion_config_dict(
        generation_config.get("linear_temperature_schedule_config")
    ) or {"t_min": 0.4, "t_max": 0.8}
    text_config = model.config.text_config
    vocab_size = int(text_config.vocab_size)
    max_denoising_steps = int(args.steps)
    min_canvas_length = min(
        int(model.config.canvas_length),
        int(args.min_canvas_length or DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH),
    )
    canvas_length = min(int(model.config.canvas_length), max(int(args.max_new_tokens), min_canvas_length))

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
    mask_mapping = model.model.decoder._make_decoder_masks(
        current_canvas[..., None], kv_cache, None
    )
    decoder_logits_without_sc, _ = _make_diffusion_decoder_logits_fns(
        model, kv_cache, mask_mapping, compile_graph=False
    )
    cur_step = max_denoising_steps
    processed_logits = decoder_logits_without_sc(current_canvas)
    schedule_temperature = _diffusion_linear_temperature(
        cur_step, max_denoising_steps, temperature_config
    )
    if schedule_temperature is not None:
        processed_logits = processed_logits / schedule_temperature
    argmax_canvas = mx.argmax(processed_logits, axis=-1).astype(input_ids.dtype)
    token_entropy = _diffusion_token_entropy(processed_logits)
    mx.eval(token_entropy, argmax_canvas)
    ent = [float(x) for x in token_entropy[0].tolist()]
    argmax = [int(x) for x in argmax_canvas[0].tolist()]
    bound = float(args.entropy_bound)
    accept = accept_count_from_entropies(ent, bound)
    return {
        "source": "mlx",
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": max_denoising_steps,
        "cur_step": cur_step,
        "canvas_rng": args.canvas_rng,
        "entropy_bound": bound,
        "initial_canvas_ids": canvas_list,
        "prompt_token_ids": prompt_ids_list,
        "entropies": ent,
        "argmax": argmax,
        "accept_count": accept,
        "mean_entropy": sum(ent) / len(ent),
        "min_entropy": min(ent),
        "low_entropy_positions": sum(1 for e in ent if e < bound),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("-p", "--prompt", default="Hello")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--entropy-bound", type=float, default=ENTROPY_BOUND)
    parser.add_argument("--canvas-rng", choices=("rust", "mlx"), default="rust")
    parser.add_argument("--min-canvas-length", type=int, default=None)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model is None:
        root = Path(__file__).resolve().parents[2]
        local = root / "model" / "mlx-mxfp4"
        args.model = str(local if local.is_dir() else "mlx-community/diffusiongemma-26B-A4B-it-mxfp4")

    try:
        payload = dump_mlx_step1(args)
    except ImportError as exc:
        print("error: mlx-vlm required (cd python && uv sync --extra mlx)", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output}: accept={payload['accept_count']} "
        f"mean_H={payload['mean_entropy']:.4f} min_H={payload['min_entropy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
