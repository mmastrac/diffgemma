#!/usr/bin/env python3
"""Compare embed row dumps (MLX vs rust embed-row-dump)."""

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


def report(label: str, ref: list[float], cand: list[float]) -> None:
    if not ref or not cand:
        print(f"  {label}: missing")
        return
    s = vec_stats(ref, cand)
    print(
        f"  {label:16s}: cos={s['cosine']:.6f}  rel_l2={s['rel_l2']:.4f}  "
        f"max_abs={s['max_abs']:.4f}"
    )
    worst_i = max(range(len(ref)), key=lambda i: abs(cand[i] - ref[i]))
    print(
        f"    worst dim {worst_i}: ref {ref[worst_i]:.6f} vs cand {cand[worst_i]:.6f} "
        f"(d={cand[worst_i]-ref[worst_i]:+.6f})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="MLX dump JSON")
    parser.add_argument("candidate", type=Path, help="rust embed-row-dump JSON")
    args = parser.parse_args()

    ref = load(args.reference)
    cand = load(args.candidate)
    print(
        f"token={ref.get('token')}  ref={ref.get('source')} cand={cand.get('source')}  "
        f"scale ref={ref.get('embed_scale')} cand={cand.get('embed_scale')}"
    )
    if "scale_f32" in cand:
        print(
            f"  q8 scale: bf16_bits={cand.get('scale_bf16_bits')} "
            f"scale_f32={cand.get('scale_f32'):.8f}"
        )

    report("dequant_raw", ref.get("dequant_raw") or [], cand.get("dequant_raw") or [])
    report("dequant_scaled", ref.get("dequant_scaled") or [], cand.get("dequant_scaled") or [])

    gpu = cand.get("gpu_scaled")
    if gpu:
        report("gpu_scaled", ref.get("dequant_scaled") or [], gpu)

    bf16 = cand.get("bf16_ref_scaled")
    if bf16:
        report("bf16_ref_scaled", ref.get("dequant_scaled") or [], bf16)
        report("cpu_vs_bf16", bf16, cand.get("dequant_scaled") or [])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
