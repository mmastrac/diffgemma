#!/usr/bin/env python3
"""
Ground-truth MLX generation for DiffusionGemma, for output-level equivalence
checks against the Rust engine's full-message chat.

Uses mlx-vlm's canonical generate path (the same one `mlx_vlm.generate` drives),
applies the Gemma-4 chat template, and runs realistic full-message generation
(real entropy-bound sampler + early stop, stops at the model's end-of-turn).

Example:
  cd python && uv run python scripts/mlx_generate.py \\
    -p "Why is the sky blue? Answer in one sentence." \\
    --max-tokens 256 -o /tmp/mlx_reply.json

Defaults to the cached 4-bit MLX weights (mlx-community/diffusiongemma-26B-A4B-it-4bit).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    mx.random.seed(args.seed)

    t0 = time.time()
    model, processor = load(args.model, lazy=False)
    load_s = time.time() - t0

    # Template the user turn the same way the Rust chat path does
    # (add_generation_prompt=True; thinking left at the template default).
    templated = apply_chat_template(
        processor,
        model.config,
        args.prompt,
        add_generation_prompt=True,
    )

    diffusion_kwargs: dict[str, Any] = {
        "temperature": args.temperature,
    }
    if args.steps is not None:
        diffusion_kwargs["max_denoising_steps"] = args.steps

    t1 = time.time()
    result = generate(
        model,
        processor,
        templated,
        max_tokens=args.max_tokens,
        verbose=False,
        **diffusion_kwargs,
    )
    gen_s = time.time() - t1

    text = result.text if hasattr(result, "text") else str(result)
    return {
        "engine": "mlx",
        "model": args.model,
        "prompt": args.prompt,
        "templated_prompt": templated,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "max_denoising_steps": args.steps,
        "reply": text,
        "finish_reason": getattr(result, "finish_reason", None),
        "prompt_tokens": getattr(result, "prompt_tokens", None),
        "generation_tokens": getattr(result, "generation_tokens", None),
        "diffusion_denoising_steps": getattr(result, "diffusion_denoising_steps", None),
        "diffusion_canvas_tokens": getattr(result, "diffusion_canvas_tokens", None),
        "generation_tps": getattr(result, "generation_tps", None),
        "peak_memory_gb": getattr(result, "peak_memory", None),
        "load_seconds": round(load_s, 2),
        "generate_seconds": round(gen_s, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MLX model dir or HF repo id")
    parser.add_argument("-p", "--prompt", default="Why is the sky blue? Answer in one sentence.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (argmax)")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="max denoising steps (default: from generation_config)",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    result = run(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    print("=" * 72)
    print(f"MLX reply ({result['generation_tokens']} tokens, "
          f"finish={result['finish_reason']}, "
          f"{result['diffusion_denoising_steps']} denoise steps, "
          f"{result['generate_seconds']}s):")
    print("-" * 72)
    print(result["reply"])
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
