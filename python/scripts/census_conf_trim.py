#!/usr/bin/env python3
"""Confidence-trim census: objective wart counts from p_max traces.

The commit-time defense stack (`DGQ_COMMIT_CONF_TRIM`) is judged on how many
CONTESTED ROWS reach committed KV, not on reading prose. Every `block_commit`
record carries the block's `pmax`/`argmax` arrays and `kept` (post-trim
length), so the two wart classes are countable directly:

  hard  — a committed row with p_max < 0.5. The insertion/omission class
          (`ownBut` 0.405, `("."`, dropped clauses at 0.296-0.49).
  dup   — a committed row with p_max < tau that argmax-copies a neighbour.
          The duplication micro-stutter (`,,`, doubled indent, `the the`).

Only rows actually COMMITTED (index < kept) count: a row the trim cut never
reached KV. Usage:

    census_conf_trim.py <trace-dir> [--tau 0.9]

Reads every *.jsonl in the directory; the arm is taken from the filename
stem before the first '.', e.g. `on.seed42.jsonl` -> arm `on`.
"""

import argparse
import collections
import json
import pathlib
import sys


def scan(path, tau):
    """(blocks, committed_rows, hard, dup, trims, steps) for one trace file."""
    blocks = committed = hard = dup = trims = steps = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") != "block_commit":
            continue
        pmax = rec.get("pmax") or []
        argmax = rec.get("argmax") or []
        kept = rec.get("kept")
        if kept is None:
            kept = len(pmax)
        kept = min(kept, len(pmax), len(argmax))
        blocks += 1
        committed += kept
        steps += rec.get("steps") or 0
        if rec.get("conf_trim_row") is not None:
            trims += 1
        for i in range(kept):
            p = pmax[i]
            if p < 0.5:
                hard += 1
            elif p < tau and (
                (i > 0 and argmax[i] == argmax[i - 1])
                or (i + 1 < kept and argmax[i] == argmax[i + 1])
            ):
                dup += 1
    return blocks, committed, hard, dup, trims, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--tau", type=float, default=0.9)
    args = ap.parse_args()

    arms = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    files = sorted(pathlib.Path(args.trace_dir).glob("*.jsonl"))
    if not files:
        sys.exit(f"no *.jsonl in {args.trace_dir}")
    for f in files:
        arm = f.name.split(".")[0]
        for i, v in enumerate(scan(f, args.tau)):
            arms[arm][i] += v

    print(f"{'arm':<10} {'blocks':>7} {'commit_rows':>12} "
          f"{'hard<0.5':>9} {'dup<tau':>8} {'trims':>6} {'steps':>7}")
    for arm in sorted(arms):
        b, c, h, d, t, s = arms[arm]
        print(f"{arm:<10} {b:>7} {c:>12} {h:>9} {d:>8} {t:>6} {s:>7}")
    print()
    # The decision line: contested rows that reached KV, per 1k committed rows.
    for arm in sorted(arms):
        b, c, h, d, t, s = arms[arm]
        rate = 1000.0 * (h + d) / c if c else 0.0
        print(f"{arm:<10} contested-committed per 1k rows: {rate:6.2f} "
              f"(hard {h}, dup {d}); mean steps/block "
              f"{(s / b if b else 0):.1f}")


if __name__ == "__main__":
    main()
