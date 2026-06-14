#!/usr/bin/env python3
"""Compare single-expert MoE dumps (MLX vs rust step-moe-single-dump)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffgemma_parity.vec_stats import vec_stats


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="MLX dump JSON")
    parser.add_argument("candidate", type=Path, help="rust step-moe-single-dump JSON")
    args = parser.parse_args()

    ref = load(args.reference)
    cand = load(args.candidate)
    pos = int(ref.get("position", -1))
    layer = int(ref.get("layer", -1))
    expert = int(ref.get("expert_id", -1))

    print(
        f"layer={layer} pos={pos} expert={expert}  "
        f"ref={ref.get('source')} cand={cand.get('source')}  "
        f"kv_len {ref.get('kv_len')} vs {cand.get('kv_len')}"
    )

    pairs = [
        ("moe_in", "moe_in", "moe_in"),
        ("mlx expert_out vs rust moe", "expert_out", "gpu_out"),
        ("mlx expert_out vs rust cpu", "expert_out", "cpu_out"),
        ("rust moe vs bf16 oracle", "gpu_out", "bf16_out"),
        ("rust moe vs grouped mirror", "gpu_out", "grouped_mirror_out"),
    ]
    for label, ref_key, cand_key in pairs:
        a = ref.get(ref_key) or []
        b = cand.get(cand_key) or []
        if not a or not b:
            print(f"  {label:32s}: missing")
            continue
        s = vec_stats(a, b)
        mark = "  <-- divergence" if s["cosine"] < 0.99 else ""
        print(
            f"  {label:32s}: cos={s['cosine']:.6f}  rel_l2={s['rel_l2']:.4f}  "
            f"max_abs={s['max_abs']:.4f}{mark}"
        )

    gpu = cand.get("gpu_out") or []
    cpu = cand.get("cpu_out") or []
    if gpu and cpu:
        s = vec_stats(gpu, cpu)
        mark = "  <-- divergence" if s["cosine"] < 0.99 else ""
        print(
            f"  {'rust moe vs cpu ref':32s}: cos={s['cosine']:.6f}  rel_l2={s['rel_l2']:.4f}  "
            f"max_abs={s['max_abs']:.4f}{mark}"
        )
    else:
        print(f"  {'rust moe vs cpu ref':32s}: missing")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
