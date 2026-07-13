#!/usr/bin/env python3
"""
Bayesian optimization of the E17 prefill-attention kernel tile config (task #87).

Drives the Rust `bench-prefill-attn` subcommand (which compiles the kernels for a
{qk tile, pv tile, HC, softmax TPG} config and prints a machine `RESULT {json}`
line with ms/layer + TF/s), and uses Optuna's TPE sampler to minimize ms/layer.
The two GEMMs (QK: M=canvas,K=hd,N=T; PV: M=canvas,K=T,N=hd) tune independently.

Search space (all points compile-valid by construction — BN divides 128, BM is a
multiple of 16, TPG a power of two):
  qk_bm/pv_bm in {16,32,48,64}   qk_bn/pv_bn in {32,64,128}
  hc in {1,2,4,8,16}             sm_tpg in {64,128,256,512,1024}
Compile failures (e.g. register spill on a big tile) return ok:false and are
scored as a large penalty so TPE learns to avoid that region.

Example:
  cd python && uv run python scripts/tune_prefill_attn.py --kv-len 30000 \
      --trials 80 --iters 10 --side
Persisting/resuming a study:
  ... --storage sqlite:///tune.db --study-name kv30k_side
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna

PENALTY = 1.0e6  # ms; assigned to configs that fail to compile/dispatch.

BM_CHOICES = [16, 32, 48, 64]
BN_CHOICES = [32, 64, 128]
HC_CHOICES = [1, 2, 4, 8, 16]
TPG_CHOICES = [64, 128, 256, 512, 1024]


def run_bench(binary: Path, kv_len: int, iters: int, side: bool, cfg: dict) -> dict:
    """Invoke bench-prefill-attn for one config; return the parsed RESULT dict."""
    args = [
        str(binary), "bench-prefill-attn",
        "--kv-len", str(kv_len),
        "--iters", str(iters),
        "--qk-bm", str(cfg["qk_bm"]),
        "--qk-bn", str(cfg["qk_bn"]),
        "--pv-bm", str(cfg["pv_bm"]),
        "--pv-bn", str(cfg["pv_bn"]),
        "--hc", str(cfg["hc"]),
        "--sm-tpg", str(cfg["sm_tpg"]),
    ]
    if side:
        args.append("--side")
    proc = subprocess.run(args, capture_output=True, text=True, timeout=600)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return {"ok": False, "reason": f"no RESULT (rc={proc.returncode})"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kv-len", type=int, default=30000)
    p.add_argument("--trials", type=int, default=80)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--side", action="store_true",
                   help="tune the f32-side-KV path (production default); else f16")
    p.add_argument("--binary", default="../target/release/diffgemma-mps")
    p.add_argument("--storage", default=None, help="e.g. sqlite:///tune.db to persist/resume")
    p.add_argument("--study-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    binary = Path(args.binary).resolve()
    if not binary.exists():
        print(f"error: binary not found: {binary} (cargo build --release first)", file=sys.stderr)
        return 2

    results: list[tuple[float, dict, dict]] = []  # (ms, cfg, result)

    def objective(trial: optuna.Trial) -> float:
        cfg = {
            "qk_bm": trial.suggest_categorical("qk_bm", BM_CHOICES),
            "qk_bn": trial.suggest_categorical("qk_bn", BN_CHOICES),
            "pv_bm": trial.suggest_categorical("pv_bm", BM_CHOICES),
            "pv_bn": trial.suggest_categorical("pv_bn", BN_CHOICES),
            "hc": trial.suggest_categorical("hc", HC_CHOICES),
            "sm_tpg": trial.suggest_categorical("sm_tpg", TPG_CHOICES),
        }
        res = run_bench(binary, args.kv_len, args.iters, args.side, cfg)
        if not res.get("ok"):
            trial.set_user_attr("failed", res.get("reason", "?"))
            return PENALTY
        ms = float(res["ms"])
        trial.set_user_attr("tf_s", res.get("tf_s"))
        results.append((ms, cfg, res))
        return ms

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    path = "f32-side" if args.side else "f16"
    print(f"tuning prefill-attn ({path}, kv={args.kv_len}, {args.trials} trials, "
          f"iters={args.iters})...", flush=True)

    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        best = study.best_value
        tf = trial.user_attrs.get("tf_s")
        tag = f"{trial.value:8.2f}ms" if trial.value < PENALTY else "  FAIL   "
        tfs = f"{tf:5.2f}TF/s" if tf else "  --     "
        print(f"  trial {trial.number:3d}: {tag} {tfs}  best={best:8.2f}ms  {trial.params}",
              flush=True)

    study.optimize(objective, n_trials=args.trials, callbacks=[cb])

    print("\n=== top 8 configs ===")
    for ms, cfg, res in sorted(results)[:8]:
        print(f"  {ms:8.3f} ms/layer  {res.get('tf_s', 0):5.2f} TF/s  "
              f"qk={cfg['qk_bm']}x{cfg['qk_bn']} pv={cfg['pv_bm']}x{cfg['pv_bn']} "
              f"hc={cfg['hc']} tpg={cfg['sm_tpg']}")
    if results:
        best_ms, best_cfg, best_res = min(results)
        print(f"\nBEST {path} kv={args.kv_len}: {best_ms:.3f} ms/layer "
              f"({best_res.get('tf_s', 0):.2f} TF/s)")
        print(f"  flags: --qk-bm {best_cfg['qk_bm']} --qk-bn {best_cfg['qk_bn']} "
              f"--pv-bm {best_cfg['pv_bm']} --pv-bn {best_cfg['pv_bn']} "
              f"--hc {best_cfg['hc']} --sm-tpg {best_cfg['sm_tpg']}")
        # Default 64x64/hc4/256 baseline for reference (if it was sampled).
        base = next((r for r in results
                     if r[1] == {"qk_bm": 64, "qk_bn": 64, "pv_bm": 64,
                                 "pv_bn": 64, "hc": 4, "sm_tpg": 256}), None)
        if base:
            print(f"  vs default 64x64/hc4/256: {base[0]:.3f} ms -> "
                  f"{best_ms:.3f} ms ({base[0]/best_ms:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
