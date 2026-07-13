#!/usr/bin/env python3
"""
Prefill-throughput bench for MLX DiffusionGemma, to A/B against the Rust engine's
`ask --prompt-len N` prefill timing.

Prefill compute is content-independent (same GEMMs regardless of token values), so
we size by token count: build filler text that tokenizes to ~N tokens, run generate
with a tiny max_tokens, and read mlx-vlm's reported prompt-processing throughput
(prompt_tps / prompt_tokens). Falls back to wall-timing if prompt_tps is absent.

Example:
  cd python && uv run python scripts/mlx_prefill_bench.py --target-tokens 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

DEFAULT_MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"


def build_prompt(processor, target_tokens: int) -> tuple[str, int]:
    """Repeat a filler chunk until the tokenized length reaches ~target_tokens."""
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    chunk = "The quick brown fox jumps over the lazy dog. "
    # Estimate chunk token count once, then scale.
    per_chunk = len(tok.encode(chunk))
    reps = max(1, target_tokens // max(1, per_chunk))
    text = chunk * reps
    # Nudge to land close to target.
    n = len(tok.encode(text))
    while n < target_tokens:
        text += chunk
        n = len(tok.encode(text))
    return text, n


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    mx.random.seed(args.seed)

    t0 = time.time()
    model, processor = load(args.model, lazy=False)
    load_s = time.time() - t0

    prompt_text, actual_tokens = build_prompt(processor, args.target_tokens)
    templated = apply_chat_template(
        processor, model.config, prompt_text, add_generation_prompt=True
    )

    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass

    gen_kwargs = dict(max_tokens=args.max_tokens, verbose=False, temperature=0.0)
    if args.prefill_step_size:
        # Chunked prefill: process the prompt in windows so the attention
        # intermediate stays [step, N] instead of a single [N, N] quadratic
        # buffer (the thing that OOMs single-shot past ~22k on a 36GB Mac).
        gen_kwargs["prefill_step_size"] = args.prefill_step_size

    t1 = time.time()
    result = generate(
        model,
        processor,
        templated,
        **gen_kwargs,
    )
    gen_s = time.time() - t1

    prompt_tokens = getattr(result, "prompt_tokens", None)
    prompt_tps = getattr(result, "prompt_tps", None)
    prefill_s = None
    if prompt_tokens and prompt_tps:
        prefill_s = prompt_tokens / prompt_tps

    return {
        "engine": "mlx",
        "model": args.model,
        "prefill_step_size": args.prefill_step_size or None,
        "target_tokens": args.target_tokens,
        "actual_prompt_tokens": prompt_tokens or actual_tokens,
        "prompt_tps": prompt_tps,
        "prefill_seconds": round(prefill_s, 2) if prefill_s is not None else None,
        "generation_tokens": getattr(result, "generation_tokens", None),
        "generation_tps": getattr(result, "generation_tps", None),
        "generate_seconds_total": round(gen_s, 2),
        "peak_memory_gb": getattr(result, "peak_memory", None),
        "load_seconds": round(load_s, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target-tokens", type=int, default=10000)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--prefill-step-size", type=int, default=0,
                        help="0 = single-shot (default); >0 enables chunked prefill")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run(args)
    print(json.dumps(result, indent=2), file=sys.stderr)
    print("PREFILL_RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
