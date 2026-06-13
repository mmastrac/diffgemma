#!/usr/bin/env python3
"""Print per-position entropy/argmax diff from two denoise trace JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="typically MLX trace")
    parser.add_argument("candidate", type=Path, help="typically monolithic trace")
    parser.add_argument("--step", type=int, default=1, help="1-based step_index")
    parser.add_argument("--positions", type=int, default=16)
    args = parser.parse_args()

    ref = load(args.reference)
    cand = load(args.candidate)
    rs = next(s for s in ref["step_traces"] if s["step_index"] == args.step)
    cs = next(s for s in cand["step_traces"] if s["step_index"] == args.step)

    ref_canvas = ref.get("initial_canvas_ids")
    cand_canvas = cand.get("initial_canvas_ids")
    if ref_canvas and cand_canvas and ref_canvas != cand_canvas:
        print("warning: initial_canvas_ids differ", file=sys.stderr)

    re = rs.get("entropy_prefix")
    ce = cs.get("entropy_prefix")
    if not re or not ce:
        print("error: traces need entropy_prefix (DGQ_TRACE_ENTROPY=1 on Rust side)", file=sys.stderr)
        return 2

    n = min(args.positions, len(re), len(ce))
    ra = rs["argmax_prefix"]
    ca = cs["argmax_prefix"]
    print(
        f"step {args.step}: accept {rs['accept_count']} vs {cs['accept_count']}  "
        f"mean_H {rs['mean_entropy']:.4f} vs {cs['mean_entropy']:.4f}"
    )
    print(f"{'pos':>3}  {'H_ref':>8}  {'H_cand':>8}  {'dH':>8}  {'argmax_ref':>10}  {'argmax_cand':>10}  match")
    for i in range(n):
        dh = float(ce[i]) - float(re[i])
        match = "yes" if ra[i] == ca[i] else "NO"
        print(
            f"{i:3d}  {float(re[i]):8.4f}  {float(ce[i]):8.4f}  {dh:+8.4f}  "
            f"{ra[i]:10d}  {ca[i]:10d}  {match}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
