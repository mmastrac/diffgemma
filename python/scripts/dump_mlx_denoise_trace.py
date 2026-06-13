#!/usr/bin/env python3
"""
Dump a compact denoise trace from an MLX DiffusionGemma checkpoint.

Uses mlx-vlm's block-diffusion engine (same path as `mlx_vlm.generate`).

Example:
  cd python && uv sync --extra mlx
  uv run python scripts/dump_mlx_denoise_trace.py \\
    --model ../model/mlx-mxfp4 -p Hello --seed 42 --steps 8 \\
    -o /tmp/mlx_30L_8.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffgemma_parity.canvas_rng import initialize_canvas
from diffgemma_parity.denoise_stats import (  # noqa: E402
    ENTROPY_BOUND,
    step_stats_from_entropies,
    trace_step_record,
)

SCHEMA_VERSION = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_path() -> str:
    local = repo_root() / "model" / "mlx-mxfp4"
    if local.is_dir():
        return str(local)
    return "mlx-community/diffusiongemma-26B-A4B-it-mxfp4"


def _soft_embedding_weight(embed_tokens) -> mx.array:
    import mlx.nn as nn

    if isinstance(embed_tokens, nn.QuantizedEmbedding):
        kwargs = {
            "group_size": embed_tokens.group_size,
            "bits": embed_tokens.bits,
        }
        if embed_tokens.biases is None:
            kwargs["mode"] = "mxfp4"
            return mx.dequantize(embed_tokens.weight, embed_tokens.scales, **kwargs)
        return mx.dequantize(
            embed_tokens.weight,
            embed_tokens.scales,
            embed_tokens.biases,
            **kwargs,
        )
    return embed_tokens.weight


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    from mlx_vlm.generate.diffusion import (
        DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH,
        _diffusion_config_dict,
        _diffusion_entropy_and_soft_embeddings,
        _diffusion_entropy_transfer_mask,
        _diffusion_initialize_canvas,
        _diffusion_linear_temperature,
        _diffusion_prefill_cache,
        _diffusion_sample_canvas,
        _diffusion_token_entropy,
        _make_diffusion_decoder_logits_fns,
        _diffusion_stable_and_confident,
    )
    from mlx_vlm.utils import load

    mx.random.seed(args.seed)

    model, processor = load(args.model, lazy=False)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    messages = [{"role": "user", "content": args.prompt}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    prompt_ids = inputs["input_ids"]
    if hasattr(prompt_ids, "tolist"):
        prompt_ids_list = (
            prompt_ids[0].tolist() if getattr(prompt_ids, "ndim", 1) > 1 else prompt_ids.tolist()
        )
    else:
        prompt_ids_list = list(prompt_ids[0])

    input_ids = mx.array([prompt_ids_list])
    attention_mask = mx.ones((1, len(prompt_ids_list)), dtype=mx.bool_)

    generation_config = _diffusion_config_dict(getattr(model.config, "generation_config", None))
    sampler_config = _diffusion_config_dict(generation_config.get("sampler_config"))
    entropy_bound = float(
        args.entropy_bound if args.entropy_bound is not None else sampler_config.get("entropy_bound", ENTROPY_BOUND)
    )
    temperature_config = _diffusion_config_dict(
        generation_config.get("linear_temperature_schedule_config")
    )
    if not temperature_config:
        temperature_config = {"t_min": 0.4, "t_max": 0.8}

    text_config = model.config.text_config
    batch_size, prompt_length = input_ids.shape
    model_canvas_length = int(model.config.canvas_length)
    max_canvas_length = model_canvas_length
    min_canvas_length = min(
        max_canvas_length,
        int(args.min_canvas_length or DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH),
    )
    vocab_size = int(text_config.vocab_size)
    max_new_tokens = int(args.max_new_tokens)
    max_denoising_steps = int(args.steps)

    diffusion_stopping_config = _diffusion_config_dict(
        generation_config.get("diffusion_stopping_config")
    )
    if not diffusion_stopping_config:
        diffusion_stopping_config = {
            key: generation_config[key]
            for key in ("confidence_threshold", "stability_threshold")
            if key in generation_config
        } or None
    if args.no_early_stop and diffusion_stopping_config is not None:
        diffusion_stopping_config = {
            **diffusion_stopping_config,
            "confidence_threshold": 1e9,
            "stability_threshold": 10**9,
        }

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

    canvas_length = min(max_canvas_length, max(max_new_tokens, min_canvas_length))
    if args.canvas_ids is not None:
        canvas_list = json.loads(args.canvas_ids.read_text())
        if len(canvas_list) != canvas_length:
            raise SystemExit(
                f"error: canvas_ids length {len(canvas_list)} != canvas_length {canvas_length}"
            )
        canvas_rng_label = "file"
    else:
        canvas_list = initialize_canvas(
            args.seed, canvas_length, vocab_size, rng=args.canvas_rng
        )
        canvas_rng_label = f"{args.canvas_rng}-default"
    initial_canvas_ids = list(canvas_list)
    current_canvas = mx.array([canvas_list], dtype=input_ids.dtype)
    accepted_canvas = current_canvas
    argmax_canvas = current_canvas
    self_conditioning_embeddings = None
    mask_mapping = model.model.decoder._make_decoder_masks(
        current_canvas[..., None], kv_cache, None
    )
    decoder_logits_without_sc, decoder_logits_with_sc = _make_diffusion_decoder_logits_fns(
        model, kv_cache, mask_mapping, compile_graph=False
    )
    soft_embedding_weight = _soft_embedding_weight(model.model.decoder.embed_tokens)

    step_records: list[dict[str, Any]] = []
    diffusion_history: list[mx.array] = []
    steps_run = 0

    for cur_step in reversed(range(1, max_denoising_steps + 1)):
        if self_conditioning_embeddings is None:
            processed_logits = decoder_logits_without_sc(current_canvas)
        else:
            processed_logits = decoder_logits_with_sc(
                current_canvas, self_conditioning_embeddings
            )

        schedule_temperature = _diffusion_linear_temperature(
            cur_step, max_denoising_steps, temperature_config
        )
        if schedule_temperature is not None:
            processed_logits = processed_logits / schedule_temperature

        argmax_canvas = mx.argmax(processed_logits, axis=-1).astype(input_ids.dtype)
        steps_run += 1
        step_index = max_denoising_steps - cur_step + 1
        next_self_conditioning_embeddings = None

        if cur_step == 1:
            token_entropy = _diffusion_token_entropy(processed_logits)
            acceptance_mask = mx.zeros(current_canvas.shape, dtype=mx.bool_)
        else:
            token_entropy, next_self_conditioning_embeddings = _diffusion_entropy_and_soft_embeddings(
                processed_logits,
                soft_embedding_weight,
                model.model.decoder.embed_scale,
            )
            acceptance_mask = _diffusion_entropy_transfer_mask(token_entropy, entropy_bound)
            denoiser_canvas = (
                argmax_canvas
                if args.temperature <= 0
                else _diffusion_sample_canvas(
                    processed_logits, input_ids.dtype, args.temperature
                )
            )
            accepted_canvas = mx.where(acceptance_mask, denoiser_canvas, current_canvas)
            current_canvas = mx.where(
                acceptance_mask,
                accepted_canvas,
                _diffusion_initialize_canvas(
                    batch_size, canvas_length, vocab_size, input_ids.dtype
                ),
            )

        mx.eval(token_entropy, acceptance_mask, argmax_canvas)
        ent_list = token_entropy[0].tolist()
        mask_list = [bool(v) for v in acceptance_mask[0].tolist()]
        stats = step_stats_from_entropies(ent_list, mask_list, entropy_bound)
        step_records.append(
            trace_step_record(
                block=1,
                step_index=step_index,
                max_denoise_steps=max_denoising_steps,
                stats=stats,
                argmax_prefix=[int(t) for t in argmax_canvas[0].tolist()],
                early_stop=False,
                entropy_prefix=[float(x) for x in ent_list[:16]],
            )
        )

        if cur_step == 1:
            break

        if _diffusion_stable_and_confident(
            argmax_canvas, processed_logits, diffusion_history, diffusion_stopping_config
        ):
            step_records[-1]["early_stop"] = True
            break

        if next_self_conditioning_embeddings is not None:
            self_conditioning_embeddings = next_self_conditioning_embeddings

    current_canvas = argmax_canvas
    mx.eval(current_canvas)
    output_ids = prompt_ids_list + [int(t) for t in current_canvas[0].tolist()]
    if step_records and len(step_records) < max_denoising_steps:
        step_records[-1]["early_stop"] = True

    layers = int(getattr(text_config, "num_hidden_layers", 30))
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "mlx-python",
        "prompt": args.prompt,
        "prompt_token_ids": prompt_ids_list,
        "seed": args.seed,
        "max_denoise_steps": max_denoising_steps,
        "layers": layers,
        "max_new_tokens": max_new_tokens,
        "weights_profile": "mlx-mxfp4",
        "entropy_bound": entropy_bound,
        "step_traces": step_records,
        "denoise_steps_run": steps_run,
        "blocks_committed": 1,
        "output_token_ids": output_ids,
        "initial_canvas_ids": initial_canvas_ids,
        "canvas_rng": canvas_rng_label,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="MLX model directory")
    parser.add_argument("-p", "--prompt", default="Hello")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--entropy-bound", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--min-canvas-length", type=int, default=None)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument(
        "--canvas-rng",
        choices=("mlx", "rust"),
        default="mlx",
        help="canvas RNG (use rust to match monolithic seed=42 LCG)",
    )
    parser.add_argument(
        "--canvas-ids",
        type=Path,
        default=None,
        help="JSON array of canvas token ids (overrides --canvas-rng)",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    args.model = args.model or default_model_path()

    try:
        trace = run_trace(args)
    except ImportError as exc:
        print("error: mlx-vlm not installed. Run: cd python && uv sync --extra mlx", file=sys.stderr)
        print(f"detail: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2) + "\n")
    accepts = [s["accept_count"] for s in trace["step_traces"]]
    print(
        f"wrote {args.output} ({len(trace['step_traces'])} steps, "
        f"accept={accepts}, {len(trace['output_token_ids'])} tokens)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
