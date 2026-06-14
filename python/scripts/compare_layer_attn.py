#!/usr/bin/env python3
"""Compare layer attention dumps (MLX reference vs rust-monolithic step-attn-dump)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def vec_stats(a: list[float], b: list[float]) -> dict[str, float]:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    cos = dot / (na * nb) if na > 1e-12 and nb > 1e-12 else float("nan")
    diff = [b[i] - a[i] for i in range(n)]
    l2 = math.sqrt(sum(d * d for d in diff))
    max_abs = max(abs(d) for d in diff) if diff else 0.0
    rel = l2 / na if na > 1e-12 else float("nan")
    return {"cosine": cos, "l2": l2, "max_abs": max_abs, "rel_l2": rel}


def k_by_pos(doc: dict) -> dict[int, list[float]]:
    return {int(e["position"]): [float(x) for x in e["k"]] for e in doc.get("k_samples", [])}


def top_keys(probs: list[float], n_heads: int, total_kv: int, k: int = 5) -> list[tuple[int, float]]:
    mass = [0.0] * total_kv
    for h in range(n_heads):
        base = h * total_kv
        for t in range(total_kv):
            mass[t] += probs[base + t]
    order = sorted(range(total_kv), key=lambda t: (-mass[t], t))
    return [(t, mass[t]) for t in order[:k]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="MLX dump JSON")
    parser.add_argument("candidate", type=Path, help="rust step-attn-dump JSON")
    args = parser.parse_args()

    ref = load(args.reference)
    cand = load(args.candidate)
    pos = int(ref.get("position", -1))
    layer = int(ref.get("layer", -1))
    n_heads = int(ref.get("n_heads", 16))
    total_kv = int(ref.get("total_kv", 0))

    print(
        f"layer={layer} pos={pos}  ref={ref.get('source')} cand={cand.get('source')}  "
        f"kv_len {ref.get('kv_len')} vs {cand.get('kv_len')}  total_kv {total_kv}"
    )

    for label, key in [
        ("hidden_in", "hidden_in"),
        ("hidden_ln", "hidden_ln"),
        ("q_post_rope", "q_post_rope"),
        ("attn_out", "attn_out"),
    ]:
        a = ref.get(key) or []
        b = cand.get(key) or []
        if not a or not b:
            print(f"  {label}: missing")
            continue
        s = vec_stats(a, b)
        print(
            f"  {label:12s}: cos={s['cosine']:.6f}  rel_l2={s['rel_l2']:.4f}  "
            f"max_abs={s['max_abs']:.4f}"
        )

    ref_k = k_by_pos(ref)
    cand_k = k_by_pos(cand)
    print("  K samples:")
    for t in sorted(set(ref_k) & set(cand_k)):
        s = vec_stats(ref_k[t], cand_k[t])
        print(
            f"    key@{t:3d}: cos={s['cosine']:.6f}  rel_l2={s['rel_l2']:.4f}  "
            f"max_abs={s['max_abs']:.4f}"
        )

    ref_probs = ref.get("attn_probs") or []
    cand_probs = cand.get("attn_probs") or []
    if ref_probs and cand_probs and len(ref_probs) == len(cand_probs):
        s = vec_stats(ref_probs, cand_probs)
        print(
            f"  attn_probs  : cos={s['cosine']:.6f}  rel_l2={s['rel_l2']:.4f}  "
            f"max_abs={s['max_abs']:.4f}"
        )
        print("  top attended keys (ref mass -> cand mass):")
        ref_top = top_keys(ref_probs, n_heads, total_kv)
        cand_mass = [0.0] * total_kv
        for h in range(n_heads):
            base = h * total_kv
            for t in range(total_kv):
                cand_mass[t] += cand_probs[base + t]
        for t, m in ref_top:
            print(f"    key@{t:3d}: ref={m:.6f}  cand={cand_mass[t]:.6f}  d={cand_mass[t]-m:+.6f}")

    ref_scores = ref.get("raw_scores") or []
    cand_scores = cand.get("raw_scores") or []
    if ref_scores and cand_scores and len(ref_scores) == len(cand_scores):
        s = vec_stats(ref_scores, cand_scores)
        print(
            f"  raw_scores  : cos={s['cosine']:.6f}  rel_l2={s['rel_l2']:.4f}  "
            f"max_abs={s['max_abs']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
