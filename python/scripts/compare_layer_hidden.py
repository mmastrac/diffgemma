#!/usr/bin/env python3
"""Compare per-layer hidden dumps (MLX reference vs rust-monolithic step-layer-probe)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def checkpoint_by_label(doc: dict, label: str) -> dict | None:
    for cp in doc.get("checkpoints", []):
        if cp.get("label") == label:
            return cp
    return None


def hidden_stats(a: list[float], b: list[float]) -> dict[str, float]:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    cos = dot / (na * nb) if na > 1e-12 and nb > 1e-12 else float("nan")
    diff = [b[i] - a[i] for i in range(n)]
    l2 = math.sqrt(sum(d * d for d in diff))
    max_abs = max(abs(d) for d in diff) if diff else 0.0
    rel = l2 / na if na > 1e-12 else float("nan")
    return {"cosine": cos, "l2": l2, "max_abs": max_abs, "rel_l2": rel}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="MLX dump JSON")
    parser.add_argument("candidate", type=Path, help="rust step-layer-probe JSON")
    parser.add_argument(
        "--cos-threshold",
        type=float,
        default=0.99,
        help="report first layer with cosine below this (default 0.99)",
    )
    args = parser.parse_args()

    ref = load(args.reference)
    cand = load(args.candidate)
    pos_r = int(ref.get("position", -1))
    pos_c = int(cand.get("position", -1))
    if pos_r != pos_c:
        print(f"warning: position ref={pos_r} cand={pos_c}", file=sys.stderr)

    ic_r = ref.get("initial_canvas_ids")
    ic_c = cand.get("initial_canvas_ids")
    if ic_r and ic_c and ic_r != ic_c:
        print("warning: initial_canvas_ids differ", file=sys.stderr)

    print(
        f"ref={ref.get('source')} cand={cand.get('source')}  "
        f"pos={pos_r} canvas={ref.get('canvas_token')}"
    )
    print(f"{'label':>20}  {'cos':>8}  {'rel_l2':>8}  {'max_abs':>8}  {'l2_ref':>8}  {'l2_cand':>8}")

    labels = [cp["label"] for cp in ref.get("checkpoints", [])]
    first_bad = None
    for label in labels:
        lr = checkpoint_by_label(ref, label)
        lc = checkpoint_by_label(cand, label)
        if lr is None or lc is None:
            print(f"{label:>20}  missing ref={lr is not None} cand={lc is not None}")
            continue
        ha = [float(x) for x in lr["hidden"]]
        hb = [float(x) for x in lc["hidden"]]
        hs = hidden_stats(ha, hb)
        mark = ""
        if hs["cosine"] < args.cos_threshold and first_bad is None:
            first_bad = label
            mark = "  <-- first divergence"
        print(
            f"{label:>20}  {hs['cosine']:8.4f}  {hs['rel_l2']:8.4f}  "
            f"{hs['max_abs']:8.4f}  {lr.get('hidden_l2', float('nan')):8.2f}  "
            f"{lc.get('hidden_l2', float('nan')):8.2f}{mark}"
        )

    if first_bad:
        print(f"\nfirst layer with cos < {args.cos_threshold}: {first_bad}")
    else:
        print(f"\nall layers cos >= {args.cos_threshold}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
