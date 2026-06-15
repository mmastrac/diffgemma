#!/usr/bin/env python3
"""Dump layer attention state (Q/K/scores/attn_out) from MLX for monolithic parity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffgemma_parity.canvas_rng import initialize_canvas

SCHEMA_VERSION = 1
STEP_NQ_HEADS = 16


def _vec_l2(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _row_raw_scores(
    q: list[float],
    k: list[float],
    qi: int,
    total_kv: int,
    n_heads: int,
    n_kv: int,
    hd: int,
) -> list[float]:
    groups = max(n_heads // max(n_kv, 1), 1)
    scores = [0.0] * (n_heads * total_kv)
    for h in range(n_heads):
        kvh = h // groups
        q_off = qi * n_heads * hd + h * hd
        for t in range(total_kv):
            k_off = t * n_kv * hd + kvh * hd
            dot = sum(q[q_off + d] * k[k_off + d] for d in range(hd))
            scores[h * total_kv + t] = dot
    return scores


def _softmax_rows(scores: list[float], n_heads: int, total_kv: int) -> list[float]:
    probs = [0.0] * len(scores)
    for h in range(n_heads):
        base = h * total_kv
        row = scores[base : base + total_kv]
        mx = max(row)
        exps = [math.exp(s - mx) for s in row]
        z = sum(exps) or 1.0
        probs[base : base + total_kv] = [e / z for e in exps]
    return probs


def _top_key_positions(probs: list[float], n_heads: int, total_kv: int, k: int) -> list[int]:
    mass = [0.0] * total_kv
    for h in range(n_heads):
        base = h * total_kv
        for t in range(total_kv):
            mass[t] += probs[base + t]
    order = sorted(range(total_kv), key=lambda t: (-mass[t], t))
    return order[:k]


def _k_vec_at(k: list[float], t: int, n_kv: int, hd: int) -> list[float]:
    off = t * n_kv * hd
    return k[off : off + n_kv * hd]


def _capture_layer_attention(
    attn,
    x,
    mask,
    cache,
    offset: int,
    *,
    decoder: bool = True,
):
    """Return (attn_out_pre_o_proj, q_flat, k_flat, raw_scores_row, probs_row, k_samples)."""
    import mlx.core as mx
    from mlx_vlm.models.base import scaled_dot_product_attention
    from mlx_vlm.models.diffusion_gemma.language import _cache_offset, _cache_state

    B, L, _ = x.shape
    if offset is None:
        offset = _cache_offset(cache)

    queries = attn.q_proj(x).reshape(B, L, attn.n_heads, attn.head_dim)
    q_raw = queries
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    q_pre_rope = queries
    queries = attn.rope(queries, offset=offset)

    keys = attn.k_proj(x).reshape(B, L, attn.n_kv_heads, attn.head_dim)
    values = (
        attn.v_proj(x).reshape(B, L, attn.n_kv_heads, attn.head_dim)
        if attn.v_proj is not None
        else keys
    )
    keys = attn.k_norm(keys).transpose(0, 2, 1, 3)
    keys = attn.rope(keys, offset=offset)
    values = attn.v_norm(values).transpose(0, 2, 1, 3)

    if decoder:
        state = _cache_state(cache)
        if state is not None:
            encoder_keys, encoder_values = state
            if attn.is_sliding:
                window = max(attn.config.sliding_window - 1, 0)
                encoder_len = encoder_keys.shape[2]
                if window and encoder_len > window and offset >= encoder_len:
                    encoder_keys = encoder_keys[:, :, -window:, :]
                    encoder_values = encoder_values[:, :, -window:, :]
                    if mask is not None and not isinstance(mask, str):
                        mask = mask[..., -(window + L) :]
            keys = mx.concatenate([encoder_keys, keys], axis=2)
            values = mx.concatenate([encoder_values, values], axis=2)
        attn_cache = None
    else:
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        attn_cache = cache

    output = scaled_dot_product_attention(
        queries, keys, values, cache=attn_cache, scale=attn.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    mx.eval(q_raw, q_pre_rope, queries, keys, output)
    return q_raw, q_pre_rope, queries, keys, output


def dump_mlx_layer_attn(args: argparse.Namespace) -> dict:
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

    hidden_in = [float(x) for x in h[0, pos].tolist()]

    target = decoder.layers[layer_idx]
    h_ln = target.input_layernorm(h)
    mx.eval(h_ln)
    hidden_ln = [float(x) for x in h_ln[0, pos].tolist()]

    queries_raw, queries_pre, queries, keys, attn_pre_o = _capture_layer_attention(
        target.self_attn,
        h_ln,
        mask_mapping.get(target.layer_type),
        kv_cache[layer_idx],
        offset,
    )
    mx.eval(queries_raw, queries_pre, queries, keys, attn_pre_o)

    n_heads = int(target.self_attn.n_heads)
    n_kv = int(target.self_attn.n_kv_heads)
    hd = int(target.self_attn.head_dim)
    total_kv = int(keys.shape[2])

    q_raw_row = queries_raw[0, pos].reshape(n_heads * hd)
    q_raw_flat = [float(x) for x in q_raw_row.tolist()]
    q_pre_row = queries_pre[0].transpose(1, 0, 2).reshape(canvas_length, n_heads * hd)[pos]
    q_pre_flat = [float(x) for x in q_pre_row.tolist()]
    q_row = queries[0].transpose(1, 0, 2).reshape(canvas_length, n_heads * hd)[pos]
    q_flat = [float(x) for x in q_row.tolist()]
    k_flat = [float(x) for x in keys[0].reshape(n_kv, total_kv, hd).transpose(1, 0, 2).reshape(-1).tolist()]
    attn_out = [float(x) for x in attn_pre_o[0, pos].tolist()]

    q_all = [float(x) for x in queries[0].transpose(1, 0, 2).reshape(canvas_length, n_heads * hd).reshape(-1).tolist()]
    raw_scores = _row_raw_scores(q_all, k_flat, pos, total_kv, n_heads, n_kv, hd)
    attn_probs = _softmax_rows(raw_scores, n_heads, total_kv)

    canvas_abs = kv_len + pos
    sample_pos = sorted(
        {
            0,
            max(kv_len - 1, 0),
            kv_len,
            canvas_abs,
            *_top_key_positions(attn_probs, n_heads, total_kv, 8),
        }
    )
    k_samples = [
        {"position": int(t), "k": _k_vec_at(k_flat, t, n_kv, hd)} for t in sample_pos
    ]

    is_full = target.layer_type == "full_attention"
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "mlx-python",
        "prompt": args.prompt,
        "prompt_token_ids": prompt_ids_list,
        "initial_canvas_ids": canvas_list,
        "seed": args.seed,
        "layer": layer_idx,
        "position": pos,
        "canvas_token": int(canvas_list[pos]),
        "kv_len": kv_len,
        "total_kv": total_kv,
        "head_dim": hd,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "is_full": is_full,
        "hidden_in": hidden_in,
        "hidden_ln": hidden_ln,
        "q_raw_proj": q_raw_flat,
        "q_pre_rope": q_pre_flat,
        "q_post_rope": q_flat,
        "attn_out": attn_out,
        "raw_scores": raw_scores,
        "attn_probs": attn_probs,
        "k_samples": k_samples,
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
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model is None:
        root = Path(__file__).resolve().parents[2]
        local = root / "model" / "mlx-mxfp4"
        args.model = str(local if local.is_dir() else "mlx-community/diffusiongemma-26B-A4B-it-mxfp4")

    try:
        payload = dump_mlx_layer_attn(args)
    except ImportError as exc:
        print("error: mlx-vlm required (cd python && uv sync --extra mlx)", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output} (layer={payload['layer']}, pos={payload['position']}, kv_len={payload['kv_len']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
