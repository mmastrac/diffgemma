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
# Sparse MoE expert-GEMM block HEIGHT (DGQ_MOE_PREFILL_BM). 32 = shipped
# default; 64/128 = weight-stationary wide blocks. Correct at every height
# since the pipeline-cache-label fix (the source-baked TUNE_BM now disambiguates
# the cache; pre-fix, bm>32 silently ran a bm=32 kernel = half the rows zeroed).
# Proxy-only (needs real weights). Wide path pins its N-tile at 64 today, so the
# moe_bn axis is inert when this is >32 (a known, harmless redundancy).
MOE_PREFILL_BM_CHOICES = [32, 64, 128]

# E17 GEMM-attention (DGQ_GEMM_ATTN): default-on production path (full-layer
# attention via the GEMM decomposition). =0 falls back to attention_mma_full
# (the flash-style kernel ILP2 grafts onto). Sweeping this lets TPE test
# whether ILP2 — or any other mma_full-only lever — wins when E17 is off, which
# a single-axis A/B at default settings cannot see (E17-on makes mma_full inert
# for full layers). Both paths run prefill; only the full-layer kernel differs.
GEMM_ATTN_CHOICES = [0, 1]
# E5 QK-ILP2 chain-split on attention_mma_full (DGQ_ATTN_MMA_FULL_QK_ILP2).
# Only meaningful when DGQ_GEMM_ATTN=0 (otherwise mma_full doesn't run for full
# layers). Non-bit-identical; quality-gated separately.
MMA_FULL_QK_ILP2_CHOICES = [0, 1]
# Super-chunk size (n_subs): how many 256-token sub-chunks per prefill
# super-chunk (M = n_subs*256). The single-axis sweep found n_subs=4 optimal at
# default settings, but attention cost grows with M (more queries = more QK/PV
# work), so the optimal attention tile config — and the {E17, ILP2} tradeoff —
# could shift with n_subs. Joint axis so TPE can test the interaction.
N_SUBS_CHOICES = [1, 2, 4]


def run_bench(binary: Path, kv_len: int, iters: int, side: bool, cfg: dict,
              proxy: str | None, model: str | None) -> dict:
    """Invoke the bench for one config; return the parsed RESULT dict.

    Two objectives:
      isolated (default): `bench-prefill-attn` — the attention kernel alone
        (fast, but over-weights attention — does NOT track real prefill).
      proxy (--proxy, needs --model): `bench-prefill-super` — one real M=1024
        super-chunk (all stages, real weights), the FAITHFUL holistic objective.
    Tile config is passed via env (the production pipelines read the flags).
    """
    import os
    env = dict(os.environ)
    env.update({
        "DGQ_GEMM_ATTN_QK_BM": str(cfg["qk_bm"]),
        "DGQ_GEMM_ATTN_QK_BN": str(cfg["qk_bn"]),
        "DGQ_GEMM_ATTN_PV_BM": str(cfg["pv_bm"]),
        "DGQ_GEMM_ATTN_PV_BN": str(cfg["pv_bn"]),
        "DGQ_GEMM_ATTN_HC": str(cfg["hc"]),
        "DGQ_GEMM_ATTN_SM_TPG": str(cfg["sm_tpg"]),
    })
    if proxy:  # holistic: also the dense-GEMM + MoE-sparse tiles (task #88)
        env.update({
            "DGQ_GEMM_TUNE_BM": str(cfg["gemm_bm"]),
            "DGQ_GEMM_TUNE_BN": str(cfg["gemm_bn"]),
            "DGQ_MOE_SPARSE_BN": str(cfg["moe_bn"]),
            "DGQ_MOE_PREFILL_BM": str(cfg["moe_prefill_bm"]),
            "DGQ_GEMM_W32": str(cfg["gemm_w32"]),
        })
    # E17 path selection + E5 ILP2 (proxy only — these are full-prefill knobs,
    # not isolated-attention-kernel params). mma_full_qk_ilp2 is only active
    # when gemm_attn=0; TPE learns the correlation.
    if proxy:
        env["DGQ_GEMM_ATTN"] = str(cfg["gemm_attn"])
        env["DGQ_ATTN_MMA_FULL_QK_ILP2"] = str(cfg["mma_full_qk_ilp2"])
    if proxy:  # holistic: one real super-chunk at kv=proxy, needs the model
        args = [str(binary), "-m", model, "bench-prefill-super",
                "--kv-len", str(kv_len), "--iters", str(iters),
                "--n-subs", str(cfg["n_subs"])]
    else:  # isolated attention kernel (no weights needed)
        args = [
            str(binary), "bench-prefill-attn",
            "--kv-len", str(kv_len), "--iters", str(iters),
            "--qk-bm", str(cfg["qk_bm"]), "--qk-bn", str(cfg["qk_bn"]),
            "--pv-bm", str(cfg["pv_bm"]), "--pv-bn", str(cfg["pv_bn"]),
            "--hc", str(cfg["hc"]), "--sm-tpg", str(cfg["sm_tpg"]),
        ]
        if side:
            args.append("--side")
    proc = subprocess.run(args, capture_output=True, text=True, timeout=1200, env=env)
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
    p.add_argument("--binary", default="../target/release/diffgemma")
    p.add_argument("--proxy", action="store_true",
                   help="objective = real M=1024 super-chunk (bench-prefill-super, FAITHFUL); "
                        "else the isolated attention kernel (fast but not real-prefill-tracking)")
    p.add_argument("--model", default="../model/diffusiongemma-q4emb",
                   help="model dir for --proxy")
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
        if args.proxy:  # co-optimize the dominant dense-GEMM + MoE tiles too
            cfg["gemm_bm"] = trial.suggest_categorical("gemm_bm", BM_CHOICES)
            cfg["gemm_bn"] = trial.suggest_categorical("gemm_bn", BN_CHOICES)
            cfg["moe_bn"] = trial.suggest_categorical("moe_bn", BN_CHOICES)
            cfg["moe_prefill_bm"] = trial.suggest_categorical(
                "moe_prefill_bm", MOE_PREFILL_BM_CHOICES)
            # Aligned u32 weight-byte loads (bit-identical; DGQ_GEMM_W32).
            # Flat at the default tiles in the single-axis A/B; joint axis so
            # TPE can test whether it shifts the tile optimum.
            cfg["gemm_w32"] = trial.suggest_categorical("gemm_w32", [0, 1])
            # E17 path + E5 ILP2 — only meaningful together (ILP2 grafts onto
            # mma_full, which is the E17-off path). Categorical so TPE can
            # explore the joint {gemm_attn, ilp2} space, not just default × ILP2.
            cfg["gemm_attn"] = trial.suggest_categorical(
                "gemm_attn", GEMM_ATTN_CHOICES)
            cfg["mma_full_qk_ilp2"] = trial.suggest_categorical(
                "mma_full_qk_ilp2", MMA_FULL_QK_ILP2_CHOICES)
            # Super-chunk size — attention cost grows with M (n_subs*256), so
            # the optimal attention config + {E17, ILP2} tradeoff may shift.
            cfg["n_subs"] = trial.suggest_categorical("n_subs", N_SUBS_CHOICES)
        model = str(Path(args.model).resolve()) if args.proxy else None
        res = run_bench(binary, args.kv_len, args.iters, args.side, cfg,
                        "proxy" if args.proxy else None, model)
        if not res.get("ok"):
            trial.set_user_attr("failed", res.get("reason", "?"))
            return PENALTY
        ms = float(res["ms"])
        trial.set_user_attr("tf_s", res.get("tf_s"))
        # When n_subs is swept, the raw ms/super-chunk objective is WRONG — it
        # rewards smaller super-chunks (n_subs=1 does 1/4 the work of n_subs=4).
        # Normalize to ms/token (ms / M, M = n_subs * CANVAS=256) so the BO
        # optimizes prefill THROUGHPUT, not super-chunk wall-clock. Store the
        # raw ms as a user_attr for display; minimize ms/token.
        if args.proxy and "n_subs" in cfg:
            m_tokens = cfg["n_subs"] * 256
            ms_per_tok = ms / m_tokens
            res["ms_per_tok"] = ms_per_tok  # for display
            trial.set_user_attr("ms_super_chunk", ms)
            trial.set_user_attr("ms_per_tok", ms_per_tok)
            results.append((ms_per_tok, cfg, res))
            return ms_per_tok
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

    # Pin the shipped default as trial 0 so best-vs-default is measured
    # apples-to-apples within THIS run (not vs a separately-benched baseline).
    # HC=16 and moe_prefill_bm=32 are the SHIPPED production defaults, so trial 0
    # measures best-vs-production apples-to-apples within this run.
    default_cfg = {"qk_bm": 64, "qk_bn": 64, "pv_bm": 64, "pv_bn": 64,
                   "hc": 16, "sm_tpg": 256}
    if args.proxy:
        default_cfg.update({"gemm_bm": 64, "gemm_bn": 64, "moe_bn": 128,
                            "moe_prefill_bm": 32, "gemm_w32": 0,
                            "gemm_attn": 1, "mma_full_qk_ilp2": 0,
                            "n_subs": 4})
    if not (args.storage and any(
            t.params == default_cfg for t in study.get_trials(deepcopy=False))):
        study.enqueue_trial(default_cfg)
    # Also pin default-tiles + W32 so the single-axis effect is measured
    # in-study alongside the joint search.
    if args.proxy:
        w32_cfg = dict(default_cfg)
        w32_cfg["gemm_w32"] = 1
        if not (args.storage and any(
                t.params == w32_cfg for t in study.get_trials(deepcopy=False))):
            study.enqueue_trial(w32_cfg)

    path = "PROXY:super-chunk" if args.proxy else ("f32-side" if args.side else "f16")
    print(f"tuning prefill-attn ({path}, kv={args.kv_len}, {args.trials} trials, "
          f"iters={args.iters})...", flush=True)

    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        best = study.best_value
        tf = trial.user_attrs.get("tf_s")
        per_tok = trial.user_attrs.get("ms_per_tok")
        if per_tok is not None:
            sc = trial.user_attrs.get("ms_super_chunk", 0.0)
            tag = (f"{per_tok:6.3f}ms/t ({sc:7.1f}ms/sc)"
                   if trial.value < PENALTY else "  FAIL   ")
            unit = "ms/t"
        else:
            tag = f"{trial.value:8.2f}ms" if trial.value < PENALTY else "  FAIL   "
            unit = "ms"
        tfs = f"{tf:5.2f}TF/s" if tf else "  --     "
        print(f"  trial {trial.number:3d}: {tag} {tfs}  best={best:8.3f}{unit}  "
              f"{trial.params}", flush=True)

    study.optimize(objective, n_trials=args.trials, callbacks=[cb])

    def gemm_moe_tag(cfg: dict) -> str:
        if "gemm_bm" not in cfg:
            return ""
        ilp = " on" if cfg.get("mma_full_qk_ilp2") else " off"
        e17 = "on" if cfg.get("gemm_attn") else "off"
        ns = cfg.get("n_subs", "?")
        return (f" gemm={cfg['gemm_bm']}x{cfg['gemm_bn']} moe_bn={cfg['moe_bn']} "
                f"moe_bm={cfg['moe_prefill_bm']} E17={e17} ilp2={ilp} n_subs={ns}")

    per_tok_mode = args.proxy and "n_subs" in (default_cfg if args.proxy else {})
    unit = "ms/tok" if per_tok_mode else "ms/layer"
    print("\n=== top 8 configs ===")
    for ms, cfg, res in sorted(results)[:8]:
        sc = res.get("ms") if per_tok_mode else None
        sc_tag = f" ({sc:.1f}ms/sc)" if sc is not None else ""
        print(f"  {ms:8.4f} {unit} {sc_tag}{res.get('tf_s', 0):5.2f} TF/s  "
              f"qk={cfg['qk_bm']}x{cfg['qk_bn']} pv={cfg['pv_bm']}x{cfg['pv_bn']} "
              f"hc={cfg['hc']} tpg={cfg['sm_tpg']}{gemm_moe_tag(cfg)}")
    if results:
        best_ms, best_cfg, best_res = min(results)
        print(f"\nBEST {path} kv={args.kv_len}: {best_ms:.4f} {unit} "
              f"({best_res.get('tf_s', 0):.2f} TF/s)")
        print(f"  flags: --qk-bm {best_cfg['qk_bm']} --qk-bn {best_cfg['qk_bn']} "
              f"--pv-bm {best_cfg['pv_bm']} --pv-bn {best_cfg['pv_bn']} "
              f"--hc {best_cfg['hc']} --sm-tpg {best_cfg['sm_tpg']}")
        if "gemm_bm" in best_cfg:
            print(f"  env:   DGQ_GEMM_TUNE_BM={best_cfg['gemm_bm']} "
                  f"DGQ_GEMM_TUNE_BN={best_cfg['gemm_bn']} "
                  f"DGQ_MOE_SPARSE_BN={best_cfg['moe_bn']} "
                  f"DGQ_MOE_PREFILL_BM={best_cfg['moe_prefill_bm']} "
                  f"DGQ_GEMM_ATTN={best_cfg['gemm_attn']} "
                  f"DGQ_ATTN_MMA_FULL_QK_ILP2={best_cfg['mma_full_qk_ilp2']}")
            print(f"  cli:   --n-subs {best_cfg['n_subs']}")
        # Shipped-default baseline for reference (enqueued as trial 0).
        base = next((r for r in results if r[1] == default_cfg), None)
        if base:
            print(f"  vs shipped default: {base[0]:.4f} {unit} -> "
                  f"{best_ms:.4f} {unit} ({base[0]/best_ms:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
