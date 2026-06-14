#!/usr/bin/env python3
"""Compare full step-1 entropy vectors (MLX vs monolithic trace JSON)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffgemma_parity.denoise_stats import ENTROPY_BOUND, accept_count_from_entropies


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def entropies_from_trace(trace: dict, step: int) -> list[float]:
    row = next(s for s in trace["step_traces"] if s["step_index"] == step)
    ent = row.get("entropy_prefix")
    if not ent:
        raise SystemExit(f"trace missing entropy_prefix (use DGQ_TRACE_ENTROPY=full)")
    return [float(x) for x in ent]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="MLX step1 JSON or denoise trace")
    parser.add_argument("candidate", type=Path, help="monolithic trace with full entropy")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--bound", type=float, default=ENTROPY_BOUND)
    parser.add_argument("--top", type=int, default=20, help="show largest |dH| positions")
    args = parser.parse_args()

    ref = load(args.reference)
    cand = load(args.candidate)

    if "entropies" in ref:
        re = ref["entropies"]
        ra = ref["argmax"]
    else:
        re = entropies_from_trace(ref, args.step)
        row = next(s for s in ref["step_traces"] if s["step_index"] == args.step)
        ra = row["argmax_prefix"]

    ce = entropies_from_trace(cand, args.step)
    row_c = next(s for s in cand["step_traces"] if s["step_index"] == args.step)
    ca = row_c["argmax_prefix"]

    n = min(len(re), len(ce))
    re, ce, ra, ca = re[:n], ce[:n], ra[:n], ca[:n]

    ref_accept = accept_count_from_entropies(re, args.bound)
    cand_accept = accept_count_from_entropies(ce, args.bound)
    diffs = [(i, re[i], ce[i], ce[i] - re[i], ra[i], ca[i]) for i in range(n)]
    diffs.sort(key=lambda t: abs(t[3]), reverse=True)

    print(
        f"step {args.step}: n={n}  accept {ref_accept} vs {cand_accept} (bound={args.bound})"
    )
    print(
        f"  mean_H {sum(re)/n:.4f} vs {sum(ce)/n:.4f}  "
        f"min_H {min(re):.4f} vs {min(ce):.4f}  "
        f"low_H {sum(1 for e in re if e < args.bound)} vs {sum(1 for e in ce if e < args.bound)}"
    )
    argmax_match = sum(1 for i in range(n) if ra[i] == ca[i])
    print(f"  argmax match: {argmax_match}/{n}")

    print(f"\ntop {args.top} |dH| positions:")
    print(f"{'pos':>4} {'H_ref':>8} {'H_cand':>8} {'dH':>8} {'argmax_ref':>10} {'argmax_cand':>10}")
    for i, hr, hc, dh, ar, ac in diffs[: args.top]:
        mark = "" if ar == ac else " *"
        print(f"{i:4d} {hr:8.4f} {hc:8.4f} {dh:+8.4f} {ar:10d} {ac:10d}{mark}")

    # Histogram: how many positions have H below various thresholds
    print("\npositions with H < threshold:")
    for thr in [0.05, 0.1, 0.5, 1.0, 2.0]:
        print(
            f"  <{thr:4.2f}: ref {sum(1 for e in re if e < thr):3d}  "
            f"cand {sum(1 for e in ce if e < thr):3d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
