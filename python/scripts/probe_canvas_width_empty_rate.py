#!/usr/bin/env python3
"""E3<->E6 probe: does a NARROWER MLX canvas reduce the empty/degenerate rate
on the short-answer prompts that trigger our E6 retry?

Loads the MLX model ONCE and sweeps prompts x canvas widths x seeds, reporting
the degenerate-reply rate per width. Degenerate = the answer region (canvas up
to first eos) is empty OR leads with a control/special token (the E6 attractor:
eos-empty or <|channel>thought ceremony).

  cd python && uv run python scripts/probe_canvas_width_empty_rate.py \
    --model mlx-community/diffusiongemma-26B-A4B-it-4bit --steps 48
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffgemma_parity.canvas_rng import initialize_canvas  # noqa: E402

# Short-answer prompts that empty at seed 123 in our engine (the E6 retry set).
PROMPTS = {
    "one_plus_one": "Print the result of 1 + 1 as a single English word in lowercase, with no other text.",
    "blue_plus_yellow": "What color do you get by mixing blue and yellow paint? Reply with one lowercase word.",
    "opposite_of_hot": "The opposite of the word 'hot' is what? Reply with one lowercase word.",
    "romeo_author_first": "What is the first name of the author of Romeo and Juliet? Reply with only the first name.",
}
# eos + bos + channel/turn open+close: a reply leading with any is degenerate.
CONTROL_IDS = {1, 2, 100, 101, 105, 106}


def denoise_final_canvas(mods, model, tokenizer, prompt_ids, canvas_length, seed, steps):
    (
        _diffusion_entropy_and_soft_embeddings,
        _diffusion_entropy_transfer_mask,
        _diffusion_initialize_canvas,
        _diffusion_linear_temperature,
        _diffusion_prefill_cache,
        _diffusion_token_entropy,
        _make_diffusion_decoder_logits_fns,
        _diffusion_stable_and_confident,
        _diffusion_config_dict,
        soft_embed_weight_fn,
    ) = mods

    mx.random.seed(seed)
    input_ids = mx.array([prompt_ids])
    gen = _diffusion_config_dict(getattr(model.config, "generation_config", None))
    sampler_cfg = _diffusion_config_dict(gen.get("sampler_config"))
    entropy_bound = float(sampler_cfg.get("entropy_bound", 0.0) or 0.0)
    temp_cfg = _diffusion_config_dict(gen.get("linear_temperature_schedule_config")) or {
        "t_min": 0.4,
        "t_max": 0.8,
    }
    stop_cfg = _diffusion_config_dict(gen.get("diffusion_stopping_config")) or {
        k: gen[k] for k in ("confidence_threshold", "stability_threshold") if k in gen
    } or None

    text_cfg = model.config.text_config
    vocab_size = int(text_cfg.vocab_size)
    batch_size = 1

    kv_cache = model.make_cache()
    kv_cache = _diffusion_prefill_cache(
        model, input_ids, attention_mask=None, kv_cache=kv_cache,
        pixel_values=None, mm_token_type_ids=None, prefill_step_size=None, chunk_prefill=False,
    )
    mx.eval([c.state for c in kv_cache])

    canvas_list = initialize_canvas(seed, canvas_length, vocab_size, rng="mlx")
    current_canvas = mx.array([canvas_list], dtype=input_ids.dtype)
    argmax_canvas = current_canvas
    sc_emb = None
    mask_mapping = model.model.decoder._make_decoder_masks(current_canvas[..., None], kv_cache, None)
    logits_wo, logits_w = _make_diffusion_decoder_logits_fns(model, kv_cache, mask_mapping, compile_graph=False)
    soft_w = soft_embed_weight_fn(model.model.decoder.embed_tokens)
    history = []

    for cur_step in reversed(range(1, steps + 1)):
        proc = logits_wo(current_canvas) if sc_emb is None else logits_w(current_canvas, sc_emb)
        t = _diffusion_linear_temperature(cur_step, steps, temp_cfg)
        if t is not None:
            proc = proc / t
        argmax_canvas = mx.argmax(proc, axis=-1).astype(input_ids.dtype)
        nxt = None
        if cur_step == 1:
            break
        _, nxt = _diffusion_entropy_and_soft_embeddings(proc, soft_w, model.model.decoder.embed_scale)
        ent = _diffusion_token_entropy(proc)
        accept = _diffusion_entropy_transfer_mask(ent, entropy_bound)
        accepted = mx.where(accept, argmax_canvas, current_canvas)
        current_canvas = mx.where(
            accept, accepted,
            _diffusion_initialize_canvas(batch_size, canvas_length, vocab_size, input_ids.dtype),
        )
        mx.eval(argmax_canvas)
        if _diffusion_stable_and_confident(argmax_canvas, proc, history, stop_cfg):
            break
        if nxt is not None:
            sc_emb = nxt
    mx.eval(argmax_canvas)
    return [int(t) for t in argmax_canvas[0].tolist()]


def is_degenerate(canvas_ids, eos_id=1):
    # answer region = up to first eos
    end = next((i for i, t in enumerate(canvas_ids) if t == eos_id), len(canvas_ids))
    region = canvas_ids[:end]
    if not region:
        return True  # empty
    return region[0] in CONTROL_IDS  # ceremony / control-leading


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/diffusiongemma-26B-A4B-it-4bit")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123, 5, 11, 99, 200, 314])
    args = ap.parse_args()

    from mlx_vlm.generate.diffusion import (
        _diffusion_config_dict,
        _diffusion_entropy_and_soft_embeddings,
        _diffusion_entropy_transfer_mask,
        _diffusion_initialize_canvas,
        _diffusion_linear_temperature,
        _diffusion_prefill_cache,
        _diffusion_stable_and_confident,
        _diffusion_token_entropy,
        _make_diffusion_decoder_logits_fns,
    )
    from mlx_vlm.utils import load

    def soft_embed_weight_fn(embed_tokens):
        import mlx.nn as nn
        if isinstance(embed_tokens, nn.QuantizedEmbedding):
            kw = {"group_size": embed_tokens.group_size, "bits": embed_tokens.bits}
            if embed_tokens.biases is None:
                return mx.dequantize(embed_tokens.weight, embed_tokens.scales, mode="mxfp4", **kw)
            return mx.dequantize(embed_tokens.weight, embed_tokens.scales, embed_tokens.biases, **kw)
        return embed_tokens.weight

    mods = (
        _diffusion_entropy_and_soft_embeddings, _diffusion_entropy_transfer_mask,
        _diffusion_initialize_canvas, _diffusion_linear_temperature, _diffusion_prefill_cache,
        _diffusion_token_entropy, _make_diffusion_decoder_logits_fns, _diffusion_stable_and_confident,
        _diffusion_config_dict, soft_embed_weight_fn,
    )

    print(f"loading {args.model} ...", flush=True)
    model, processor = load(args.model, lazy=False)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    prompt_ids = {}
    for name, text in PROMPTS.items():
        inp = processor.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, return_dict=True
        )
        ids = inp["input_ids"]
        ids = ids[0].tolist() if getattr(ids, "ndim", 1) > 1 else list(ids[0]) if not hasattr(ids, "tolist") else ids.tolist()
        prompt_ids[name] = ids

    # width -> (degenerate, total)
    tally = {w: [0, 0] for w in args.widths}
    print(f"\nseeds={args.seeds}  widths={args.widths}  prompts={list(PROMPTS)}\n")
    for w in args.widths:
        for name in PROMPTS:
            row = []
            for s in args.seeds:
                canvas = denoise_final_canvas(mods, model, tok, prompt_ids[name], w, s, args.steps)
                deg = is_degenerate(canvas)
                tally[w][0] += int(deg)
                tally[w][1] += 1
                row.append("X" if deg else ".")
            print(f"  width={w:3}  {name:20} {''.join(row)}  ({row.count('X')}/{len(row)} degenerate)")
        d, t = tally[w]
        print(f"  ==> width={w:3}: {d}/{t} degenerate ({100*d/t:.0f}%)\n")

    print("SUMMARY (degenerate rate by canvas width):")
    for w in args.widths:
        d, t = tally[w]
        print(f"  canvas {w:3}: {d:2}/{t}  {100*d/t:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
