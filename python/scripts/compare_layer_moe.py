#!/usr/bin/env python3
"""Compare layer MoE/FFN dumps (MLX reference vs rust-monolithic step-moe-dump)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffgemma_parity.vec_stats import fp16_bits_to_f32, vec_l2, vec_stats


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def decode_rust_weights(raw: list[int]) -> list[float]:
    return [fp16_bits_to_f32(int(x)) for x in raw]


def expert_match(ref: list[int], cand: list[int]) -> tuple[int, int, list[int], list[int]]:
    ref_set = set(ref)
    cand_set = set(cand)
    overlap = sorted(ref_set & cand_set)
    only_ref = sorted(ref_set - cand_set)
    only_cand = sorted(cand_set - ref_set)
    ordered = sum(1 for a, b in zip(ref, cand) if a == b)
    return ordered, len(overlap), only_ref, only_cand


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="MLX dump JSON")
    parser.add_argument("candidate", type=Path, help="rust step-moe-dump JSON")
    args = parser.parse_args()

    ref = load(args.reference)
    cand = load(args.candidate)
    pos = int(ref.get("position", -1))
    layer = int(ref.get("layer", -1))

    ic_r = ref.get("initial_canvas_ids")
    ic_c = cand.get("initial_canvas_ids")
    if ic_r and ic_c and ic_r != ic_c:
        print("warning: initial_canvas_ids differ", file=sys.stderr)

    print(
        f"layer={layer} pos={pos}  ref={ref.get('source')} cand={cand.get('source')}  "
        f"kv_len {ref.get('kv_len')} vs {cand.get('kv_len')}  "
        f"canvas {ref.get('canvas_token')} vs {cand.get('canvas_token')}"
    )

    ref_experts = [int(x) for x in ref.get("experts", [])]
    cand_experts = [int(x) for x in cand.get("experts", [])]
    ordered, overlap, only_ref, only_cand = expert_match(ref_experts, cand_experts)
    print(f"  experts: ref={ref_experts}")
    print(f"  experts: cand={cand_experts}")
    print(
        f"  expert match: ordered={ordered}/{len(ref_experts)}  "
        f"set_overlap={overlap}  only_ref={only_ref}  only_cand={only_cand}"
    )

    ref_w = [float(x) for x in ref.get("expert_weights", [])]
    cand_w_raw = cand.get("expert_weights", [])
    cand_w = decode_rust_weights(cand_w_raw) if cand_w_raw else []
    if ref_w and cand_w:
        print("  router weights (slot order; ref vs cand):")
        for i, (re, ce, rw, cw) in enumerate(
            zip(ref_experts, cand_experts, ref_w, cand_w)
        ):
            print(f"    slot {i}: expert ref={re} cand={ce}  w ref={rw:.6f} cand={cw:.6f}  d={cw-rw:+.6f}")
        ref_by_e = dict(zip(ref_experts, ref_w))
        cand_by_e = dict(zip(cand_experts, cand_w))
        shared = sorted(set(ref_experts) & set(cand_experts))
        if shared:
            diffs = [abs(cand_by_e[e] - ref_by_e[e]) for e in shared]
            print(
                f"  shared-expert |dW|: mean={sum(diffs)/len(diffs):.6f}  "
                f"max={max(diffs):.6f}  n={len(shared)}"
            )

    fields = [
        ("post_attn", "post_attn"),
        ("dense_out", "dense_out"),
        ("moe_out", "moe_out"),
        ("layer_out", "layer_out"),
    ]
    cliff = None
    for label, key in fields:
        a = ref.get(key) or []
        b = cand.get(key) or []
        if not a or not b:
            print(f"  {label:12s}: missing")
            continue
        s = vec_stats(a, b)
        mark = ""
        if cliff is None and s["cosine"] < 0.99:
            cliff = label
            mark = "  <-- divergence"
        print(
            f"  {label:12s}: cos={s['cosine']:.6f}  rel_l2={s['rel_l2']:.4f}  "
            f"max_abs={s['max_abs']:.4f}  l2_ref={s['l2_ref']:.2f}  l2_cand={s['l2_cand']:.2f}{mark}"
        )

    ref_ln = ref.get("moe_out_ln") or []
    cand_moe = cand.get("moe_out") or []
    if ref_ln and cand_moe:
        s = vec_stats(ref_ln, cand_moe)
        print(
            f"  moe_out_ln vs cand.moe_out (pre-norm): cos={s['cosine']:.6f}  "
            f"(MLX post_ff_ln_2 vs Rust pre-norm raw)"
        )

    for label, key in fields:
        v = cand.get(key) or ref.get(key)
        if v:
            print(f"  {label}_l2: ref={vec_l2(ref.get(key) or []):.2f}  cand={vec_l2(cand.get(key) or []):.2f}")

    if cliff:
        print(f"\nfirst checkpoint with cos < 0.99: {cliff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
