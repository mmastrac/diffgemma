#!/usr/bin/env python3
"""Compare preamble hidden dumps (MLX vs rust-monolithic step-preamble-dump)."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="MLX dump JSON")
    parser.add_argument("candidate", type=Path, help="rust step-preamble-dump JSON")
    args = parser.parse_args()

    ref = load(args.reference)
    cand = load(args.candidate)

    print(
        f"pos={ref.get('position')} token={ref.get('canvas_token')}  "
        f"ref={ref.get('source')} cand={cand.get('source')}"
    )
    if ref.get("canvas_token") != cand.get("canvas_token"):
        print(
            f"warning: canvas_token ref={ref.get('canvas_token')} "
            f"cand={cand.get('canvas_token')}",
            file=sys.stderr,
        )

    for label in ("embed_scaled", "after_preamble"):
        a = ref.get(label) or []
        b = cand.get(label) or []
        if not a or not b:
            print(f"  {label}: missing")
            continue
        s = vec_stats(a, b)
        print(
            f"  {label:16s}: cos={s['cosine']:.6f}  rel_l2={s['rel_l2']:.4f}  "
            f"max_abs={s['max_abs']:.4f}  "
            f"l2 ref={ref.get(label + '_l2', float('nan')):.2f} "
            f"cand={cand.get(label + '_l2', float('nan')):.2f}"
        )
        worst_i = max(range(len(a)), key=lambda i: abs(b[i] - a[i]))
        print(
            f"    worst dim {worst_i}: ref {a[worst_i]:.4f} vs cand {b[worst_i]:.4f} "
            f"(d={b[worst_i]-a[worst_i]:+.4f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
