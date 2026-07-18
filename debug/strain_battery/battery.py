#!/usr/bin/env python3
"""Strain-rate battery: measure the collapse/strain RATE across flag arms.

Single-trajectory A/Bs are meaningless here — any perturbation (flag, seed,
three prompt tokens) forks the whole trajectory (see
memory: opencode-collapse-repro, the invalid top-k bisect). This harness
drives the SAME matched prompt pair (the 6,900-token clean-history and
6,903-token collapse-history OpenCode requests, recovered from serve turn
dumps) across N seeds per arm and scores each turn from the per-block
commit stats serve flushes to --log-dir:

  - nonconverged blocks: denoise_stop == "max_steps" with late_mean_ent > 0.2
    (the commit-guard detector, measured WITHOUT the guard active so commits
    happen naturally, matching collapse conditions)
  - flood score: top-unigram fraction of the reply's words (the `the the the`
    / `}\n` degenerate attractor)

Usage: battery.py <serve_binary> <model_dir> <prompts_dir> <outdir>
  prompts_dir must hold clean.json + collapse.json (serve turn dumps with
  "messages" + "tools").

Arms are defined in ARMS below. Each arm restarts serve with its env/args.
"""
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

SERVE_BIN, MODEL_DIR, PROMPTS_DIR, OUTDIR = sys.argv[1:5]
CTX = sys.argv[5] if len(sys.argv) > 5 else "16384"
PORT = 8121
SEEDS = [42, 7, 123, 1234, 9999]
MAX_TOKENS = 3584  # 14 blocks — onset is ~block 6; leave runway to observe the flood
NC_LATE_ENT = 0.2

ARMS = [
    ("base", {}, []),
    ("topk_decode_off", {"DGQ_ATTN_TOPK_DECODE": "0"}, []),
    ("steps96", {}, ["--steps", "96"]),
    ("earlystop_off", {"DGQ_EARLY_STOP_MEAN_ENT": "0"}, []),
]


def wait_health(deadline=180):
    t0 = time.time()
    while time.time() - t0 < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def flood_score(text):
    words = [w for w in text.split() if w]
    if len(words) < 20:
        return 0.0
    top = max(words.count(w) for w in set(words))
    return top / len(words)


def run_turn(prompt, seed):
    body = json.dumps(
        {
            "model": "m",
            "stream": False,
            "seed": seed,
            "max_tokens": MAX_TOKENS,
            "messages": prompt["messages"],
            "tools": prompt["tools"],
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        json.load(r)
    return time.time() - t0


def collect_blocks(log_dir, seq):
    stats = []
    for f in sorted(pathlib.Path(log_dir).glob(f"model-{seq:05}-*.json")):
        stats.append(json.load(open(f)))
    return stats


def main():
    prompts = {
        name: json.load(open(f"{PROMPTS_DIR}/{name}.json"))
        for name in ("clean", "collapse")
    }
    os.makedirs(OUTDIR, exist_ok=True)
    results = open(f"{OUTDIR}/results.jsonl", "a")

    for arm, env_extra, args_extra in ARMS:
        arm_dir = f"{OUTDIR}/{arm}"
        shutil.rmtree(arm_dir, ignore_errors=True)
        os.makedirs(arm_dir)
        env = dict(os.environ, **env_extra)
        serve = subprocess.Popen(
            [SERVE_BIN, "serve", "-m", MODEL_DIR, "--addr", f"127.0.0.1:{PORT}",
             "--ctx", CTX, "--log-dir", arm_dir, *args_extra],
            env=env,
            stdout=open(f"{arm_dir}/serve.log", "w"),
            stderr=subprocess.STDOUT,
        )
        try:
            if not wait_health():
                print(f"[{arm}] serve failed to start", flush=True)
                continue
            seq = 0
            for pname, prompt in prompts.items():
                for seed in SEEDS:
                    seq += 1
                    try:
                        secs = run_turn(prompt, seed)
                    except Exception as e:
                        print(f"[{arm}] {pname} seed={seed} ERROR {e}", flush=True)
                        continue
                    # serve flushes turn dump serve-{seq}.json + block stats
                    time.sleep(2)
                    turn = {}
                    tf = pathlib.Path(arm_dir) / f"serve-{seq:05}.json"
                    if tf.exists():
                        turn = json.load(open(tf))
                    blocks = collect_blocks(arm_dir, seq)

                    # The guard's detector, recomputed from the flushed per-step
                    # series (the dumps carry mean_ent_per_step, not a scalar).
                    def late_min_ent(b):
                        ents = b.get("mean_ent_per_step") or []
                        return min(ents[-8:]) if ents else -1.0

                    nc = sum(
                        1
                        for b in blocks
                        if b.get("denoise_stop") == "max_steps"
                        and late_min_ent(b) > NC_LATE_ENT
                    )
                    rec = {
                        "arm": arm,
                        "prompt": pname,
                        "seed": seed,
                        "secs": round(secs, 1),
                        "blocks": len(blocks),
                        "nonconverged": nc,
                        "steps_eff": [b.get("steps_eff") for b in blocks],
                        "late_min_ent": [round(late_min_ent(b), 3) for b in blocks],
                        "flood": round(
                            flood_score(turn.get("raw_ids_decode", "")), 3
                        ),
                        "completion_tokens": turn.get("completion_tokens"),
                        "stopped": turn.get("stopped"),
                    }
                    results.write(json.dumps(rec) + "\n")
                    results.flush()
                    print(
                        f"[{arm}] {pname} seed={seed} blocks={rec['blocks']} "
                        f"nc={nc} flood={rec['flood']} {rec['secs']}s",
                        flush=True,
                    )
        finally:
            serve.send_signal(signal.SIGTERM)
            try:
                serve.wait(timeout=15)
            except subprocess.TimeoutExpired:
                serve.kill()
            time.sleep(3)
    print("battery done", flush=True)


if __name__ == "__main__":
    main()
