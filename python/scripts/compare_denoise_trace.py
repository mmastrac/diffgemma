#!/usr/bin/env python3
"""Compare two denoise trace JSON files (Rust monolithic vs HuggingFace reference)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `uv run python/scripts/compare_denoise_trace.py` without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffgemma_parity.denoise_stats import ENTROPY_BOUND  # noqa: E402


def load_trace(path: Path) -> dict:
    return json.loads(path.read_text())


def compare_traces(
    left: dict,
    right: dict,
    *,
    entropy_tol: float,
    mean_entropy_tol: float,
) -> list[str]:
    errors: list[str] = []

    for key in ("seed", "max_denoise_steps", "entropy_bound"):
        if left.get(key) != right.get(key):
            errors.append(f"config {key}: {left.get(key)!r} vs {right.get(key)!r}")

    left_canvas = left.get("initial_canvas_ids")
    right_canvas = right.get("initial_canvas_ids")
    if left_canvas is not None and right_canvas is not None and left_canvas != right_canvas:
        errors.append(
            f"initial_canvas_ids differ (first 8: {left_canvas[:8]!r} vs {right_canvas[:8]!r})"
        )

    left_steps = left.get("step_traces") or []
    right_steps = right.get("step_traces") or []
    if len(left_steps) != len(right_steps):
        errors.append(
            f"step count: {len(left_steps)} vs {len(right_steps)} "
            f"(denoise_steps_run {left.get('denoise_steps_run')} vs {right.get('denoise_steps_run')})"
        )

    n = min(len(left_steps), len(right_steps))
    for i in range(n):
        ls = left_steps[i]
        rs = right_steps[i]
        label = (
            f"block={ls.get('block')} step_index={ls.get('step_index')} "
            f"cur_step={ls.get('cur_step')}"
        )

        if ls.get("accept_count") != rs.get("accept_count"):
            errors.append(
                f"{label}: accept_count {ls.get('accept_count')} vs {rs.get('accept_count')}"
            )

        if ls.get("low_entropy_positions") != rs.get("low_entropy_positions"):
            errors.append(
                f"{label}: low_entropy_positions "
                f"{ls.get('low_entropy_positions')} vs {rs.get('low_entropy_positions')}"
            )

        for field, tol in (
            ("min_entropy", entropy_tol),
            ("mean_entropy", mean_entropy_tol),
            ("max_entropy", entropy_tol),
        ):
            lv = float(ls.get(field, 0.0))
            rv = float(rs.get(field, 0.0))
            if abs(lv - rv) > tol:
                errors.append(f"{label}: {field} {lv:.6f} vs {rv:.6f} (tol={tol})")

        if ls.get("argmax_prefix") != rs.get("argmax_prefix"):
            errors.append(
                f"{label}: argmax_prefix diverged "
                f"{ls.get('argmax_prefix')!r} vs {rs.get('argmax_prefix')!r}"
            )

        le = ls.get("entropy_prefix")
        re = rs.get("entropy_prefix")
        if le is not None and re is not None:
            n = min(len(le), len(re), 16)
            for pos in range(n):
                if abs(float(le[pos]) - float(re[pos])) > entropy_tol:
                    errors.append(
                        f"{label}: entropy_prefix[{pos}] {float(le[pos]):.6f} vs "
                        f"{float(re[pos]):.6f} (tol={entropy_tol})"
                    )
                    break

        if ls.get("early_stop") != rs.get("early_stop"):
            errors.append(
                f"{label}: early_stop {ls.get('early_stop')} vs {rs.get('early_stop')}"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="reference trace (typically HF)")
    parser.add_argument("right", type=Path, help="candidate trace (typically Rust)")
    parser.add_argument(
        "--entropy-tol",
        type=float,
        default=0.02,
        help="abs tolerance for min/max entropy (default 0.02 nats)",
    )
    parser.add_argument(
        "--mean-entropy-tol",
        type=float,
        default=0.05,
        help="abs tolerance for mean entropy (default 0.05 nats)",
    )
    args = parser.parse_args()

    left = load_trace(args.left)
    right = load_trace(args.right)
    print(f"left:  {args.left} source={left.get('source')!r} steps={len(left.get('step_traces', []))}")
    print(
        f"right: {args.right} source={right.get('source')!r} "
        f"steps={len(right.get('step_traces', []))}"
    )
    print(f"entropy_bound={left.get('entropy_bound', ENTROPY_BOUND)}")

    errors = compare_traces(
        left,
        right,
        entropy_tol=args.entropy_tol,
        mean_entropy_tol=args.mean_entropy_tol,
    )
    if errors:
        print(f"compare: FAIL ({len(errors)} mismatches)")
        for err in errors[:40]:
            print(f"  - {err}")
        if len(errors) > 40:
            print(f"  ... and {len(errors) - 40} more")
        return 1

    print("compare: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
