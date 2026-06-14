#!/usr/bin/env python3
"""Dump step-1 logit rows from MLX (post-softcap, tempered) for parity vs monolithic."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffgemma_parity.canvas_rng import initialize_canvas
from diffgemma_parity.denoise_stats import ENTROPY_BOUND

SCHEMA_VERSION = 1


def _entropy_nats(row: list[float]) -> float:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    z = sum(exps)
    probs = [e / z for e in exps]
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def _argmax_row(row: list[float]) -> tuple[int, float]:
    idx = max(range(len(row)), key=lambda i: row[i])
    return idx, row[idx]


def _top_k(row: list[float], k: int) -> list[dict]:
    scored = sorted(enumerate(row), key=lambda t: (-t[1], t[0]))
    return [{"token": int(i), "logit_tempered": float(v)} for i, v in scored[:k]]


def dump_mlx_step1(args: argparse.Namespace) -> dict:
    import mlx.core as mx
    from mlx_vlm.generate.diffusion import (
        DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH,
        _diffusion_config_dict,
        _diffusion_linear_temperature,
        _diffusion_prefill_cache,
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
    cur_step = max_denoising_steps
    out = model(
        cache=kv_cache,
        canvas_ids=current_canvas,
        self_conditioning_logits=None,
        decoder_attention_mask=mask_mapping,
    )
    raw_logits = out.logits
    hidden_states = out.hidden_states[-1]
    schedule_temperature = _diffusion_linear_temperature(
        cur_step, max_denoising_steps, temperature_config
    )
    temperature = float(schedule_temperature) if schedule_temperature is not None else 1.0
    processed = raw_logits
    if schedule_temperature is not None:
        processed = raw_logits / schedule_temperature
    mx.eval(processed, raw_logits, hidden_states)
    raw_rows = raw_logits[0].tolist()
    proc_rows = processed[0].tolist()
    hidden_rows = hidden_states[0].tolist()

    def _vec_l2(v: list[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    def _vec_max_abs(v: list[float]) -> float:
        return max(abs(x) for x in v) if v else 0.0

    positions = args.positions
    rows = []
    for pos in positions:
        raw_row = [float(x) for x in raw_rows[pos]]
        proc_row = [float(x) for x in proc_rows[pos]]
        hidden = [float(x) for x in hidden_rows[pos]]
        argmax_raw, _ = _argmax_row(raw_row)
        argmax_t, logit_t_at_argmax_t = _argmax_row(proc_row)
        token_set = {1, argmax_raw, argmax_t}
        for e in _top_k(proc_row, args.top_k):
            token_set.add(e["token"])
        token_logits = [
            {
                "token": int(tok),
                "logit_raw": raw_row[tok],
                "logit_tempered": proc_row[tok],
            }
            for tok in sorted(token_set)
        ]
        rows.append(
            {
                "position": pos,
                "canvas_token": int(canvas_list[pos]),
                "hidden": hidden,
                "hidden_l2": _vec_l2(hidden),
                "hidden_max_abs": _vec_max_abs(hidden),
                "argmax_raw": int(argmax_raw),
                "argmax_tempered": int(argmax_t),
                "logit_raw_at_argmax_tempered": raw_row[argmax_t],
                "logit_tempered_at_argmax_tempered": logit_t_at_argmax_t,
                "entropy_raw_nats": _entropy_nats(raw_row),
                "entropy_tempered_nats": _entropy_nats(proc_row),
                "top_k_tempered": [
                    {
                        "token": e["token"],
                        "logit_raw": raw_row[e["token"]],
                        "logit_tempered": e["logit_tempered"],
                    }
                    for e in _top_k(proc_row, args.top_k)
                ],
                "token_logits": token_logits,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "mlx-python",
        "prompt": args.prompt,
        "prompt_token_ids": prompt_ids_list,
        "initial_canvas_ids": canvas_list,
        "seed": args.seed,
        "max_denoise_steps": max_denoising_steps,
        "layers": int(getattr(text_config, "num_hidden_layers", 30)),
        "temperature": temperature,
        "entropy_bound": float(args.entropy_bound),
        "rows": rows,
    }


def parse_positions(spec: str) -> list[int]:
    if not spec.strip():
        return [0, 43, 58]
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


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
    parser.add_argument("--logit-positions", default="", help="comma-separated canvas rows")
    parser.add_argument("--logit-top-k", type=int, default=10)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    args.positions = parse_positions(args.logit_positions)
    args.top_k = max(1, args.logit_top_k)
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
        f"wrote {args.output} (T={payload['temperature']:.4f}, {len(payload['rows'])} rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
