#!/usr/bin/env python3
"""Compare step-1 logit row dumps (MLX reference vs rust-monolithic)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def row_by_pos(doc: dict, pos: int) -> dict | None:
    for row in doc.get("rows", []):
        if int(row["position"]) == pos:
            return row
    return None


def token_logit(row: dict, token: int) -> tuple[float | None, float | None]:
    for e in row.get("token_logits", []):
        if int(e["token"]) == token:
            return float(e.get("logit_raw", 0)), float(e.get("logit_tempered", 0))
    return None, None


def hidden_vec(row: dict) -> list[float] | None:
    h = row.get("hidden")
    if not h:
        return None
    return [float(x) for x in h]


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
    parser.add_argument("candidate", type=Path, help="rust step-logits-dump JSON")
    args = parser.parse_args()

    ref = load(args.reference)
    cand = load(args.candidate)
    ic_r = ref.get("initial_canvas_ids")
    ic_c = cand.get("initial_canvas_ids")
    if ic_r and ic_c and ic_r != ic_c:
        print("warning: initial_canvas_ids differ", file=sys.stderr)

    print(
        f"ref={ref.get('source')} cand={cand.get('source')}  "
        f"T {ref.get('temperature'):.4f} vs {cand.get('temperature'):.4f}"
    )
    positions = sorted(
        {int(r["position"]) for r in ref.get("rows", [])}
        | {int(r["position"]) for r in cand.get("rows", [])}
    )

    for pos in positions:
        lr = row_by_pos(ref, pos)
        lc = row_by_pos(cand, pos)
        if lr is None or lc is None:
            print(f"\npos {pos}: missing row ref={lr is not None} cand={lc is not None}")
            continue
        print(f"\n=== pos {pos} canvas={lr.get('canvas_token')} ===")
        ha = hidden_vec(lr)
        hb = hidden_vec(lc)
        if ha and hb:
            hs = hidden_stats(ha, hb)
            print(
                f"  hidden: cos={hs['cosine']:.6f}  l2={hs['l2']:.4f}  "
                f"max_abs={hs['max_abs']:.4f}  rel_l2={hs['rel_l2']:.4f}"
            )
            print(
                f"  hidden_l2: ref {lr.get('hidden_l2', float('nan')):.4f} vs "
                f"cand {lc.get('hidden_l2', float('nan')):.4f}"
            )
            worst_i = max(range(len(ha)), key=lambda i: abs(hb[i] - ha[i]))
            print(
                f"  worst dim {worst_i}: ref {ha[worst_i]:.4f} vs cand {hb[worst_i]:.4f} "
                f"(d={hb[worst_i]-ha[worst_i]:+.4f})"
            )
        else:
            print("  hidden: missing (re-dump with updated step-logits-dump / dump_step1_logits.py)")
        print(
            f"  argmax_t: ref {lr['argmax_tempered']} vs cand {lc['argmax_tempered']}  "
            f"match={'yes' if lr['argmax_tempered'] == lc['argmax_tempered'] else 'NO'}"
        )
        print(
            f"  H_t: ref {lr['entropy_tempered_nats']:.4f} vs cand {lc['entropy_tempered_nats']:.4f}  "
            f"dH={lc['entropy_tempered_nats'] - lr['entropy_tempered_nats']:+.4f}"
        )
        print(
            f"  logit_t@argmax_t: ref {lr['logit_tempered_at_argmax_tempered']:.4f} vs "
            f"cand {lc['logit_tempered_at_argmax_tempered']:.4f}  "
            f"d={lc['logit_tempered_at_argmax_tempered'] - lr['logit_tempered_at_argmax_tempered']:+.4f}"
        )
        watch = {1, int(lr["argmax_tempered"]), int(lc["argmax_tempered"])}
        for tok in sorted(watch):
            rr, rt = token_logit(lr, tok)
            cr, ct = token_logit(lc, tok)
            if rr is None or cr is None:
                continue
            print(
                f"  token {tok:6d}: raw {rr:8.3f} vs {cr:8.3f} (d={cr-rr:+.3f})  "
                f"t {rt:8.3f} vs {ct:8.3f} (d={ct-rt:+.3f})"
            )
        print("  top-k tempered (ref -> cand):")
        ref_top = lr.get("top_k_tempered", [])
        cand_top = {int(e["token"]): e for e in lc.get("top_k_tempered", [])}
        for i, e in enumerate(ref_top[:8]):
            tok = int(e["token"])
            lt_r = float(e["logit_tempered"])
            lt_c = float(cand_top[tok]["logit_tempered"]) if tok in cand_top else float("nan")
            mark = " *" if tok not in cand_top else ""
            print(f"    #{i+1} tok={tok:6d}  ref={lt_r:8.3f}  cand={lt_c:8.3f}  d={lt_c-lt_r:+.3f}{mark}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
