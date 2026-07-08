#!/usr/bin/env python3
"""
Side-by-side output-level generation comparison: Rust .dgq engine vs MLX reference.

Runs the same templated prompt through both engines with realistic full-message
generation (each as its own subprocess so only one ~15 GB model is resident at a
time), then prints a side-by-side of the decoded replies plus token / denoise-step
/ latency telemetry. Intended for spotting quality and convergence gaps.

Example:
  cd python && uv run python scripts/compare_generation.py \\
    -p "Why is the sky blue? Answer in one sentence." \\
    --rust-model ../model/diffusiongemma-q4 \\
    --mlx-model mlx-community/diffusiongemma-26B-A4B-it-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_mlx(args: argparse.Namespace, out_path: Path) -> dict[str, Any]:
    script = Path(__file__).resolve().parent / "mlx_generate.py"
    cmd = [
        sys.executable, str(script),
        "-p", args.prompt,
        "--model", args.mlx_model,
        "--seed", str(args.seed),
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "-o", str(out_path),
    ]
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    subprocess.run(cmd, check=True, env=env)
    return json.loads(out_path.read_text())


# Rust chat telemetry line:
#   turn: 30 new tokens, 1 blocks, 22 denoise steps, 48.83s (turn ended)
_TURN_RE = re.compile(
    r"turn:\s*(\d+)\s*new tokens,\s*(\d+)\s*blocks,\s*(\d+)\s*denoise steps,\s*([\d.]+)(m?s)\s*\(([^)]*)\)"
)


def run_rust(args: argparse.Namespace) -> dict[str, Any]:
    binary = (repo_root() / "target" / "release" / "diffgemma-mps").resolve()
    if not binary.exists():
        raise SystemExit(
            f"error: {binary} not found. Build it: cargo build --release"
        )
    cmd = [
        str(binary),
        "-m", args.rust_model,
        "-p", args.prompt,
        "chat",
        "--seed", str(args.seed),
        "--max-new-tokens", str(args.max_tokens),
    ]
    env = {**os.environ, "DGQ_QUIET": "1"}
    proc = subprocess.run(
        cmd, input="", capture_output=True, text=True, env=env, check=False
    )
    reply = ""
    for line in proc.stdout.splitlines():
        idx = line.find("model> ")
        if idx != -1:
            reply = line[idx + len("model> "):].strip()
            break
    telemetry: dict[str, Any] = {}
    for line in proc.stderr.splitlines():
        m = _TURN_RE.search(line)
        if m:
            secs = float(m.group(4))
            if m.group(5) == "ms":
                secs /= 1000.0
            telemetry = {
                "generation_tokens": int(m.group(1)),
                "blocks": int(m.group(2)),
                "denoise_steps": int(m.group(3)),
                "generate_seconds": round(secs, 2),
                "finish_reason": m.group(6),
            }
            break
    if proc.returncode != 0 and not reply:
        sys.stderr.write(proc.stderr[-2000:])
        raise SystemExit(f"error: rust engine exited {proc.returncode}")
    return {"engine": "rust", "reply": reply, **telemetry}


def fmt(v: Any) -> str:
    return "?" if v is None else str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--prompt", default="Why is the sky blue? Answer in one sentence.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rust-model", default=str(repo_root() / "model" / "diffusiongemma-q4"))
    parser.add_argument("--mlx-model", default="mlx-community/diffusiongemma-26B-A4B-it-4bit")
    parser.add_argument("-o", "--output", type=Path, default=None, help="write combined JSON")
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument("--skip-mlx", action="store_true")
    args = parser.parse_args()

    results: dict[str, Any] = {"prompt": args.prompt, "seed": args.seed}

    if not args.skip_rust:
        print("running rust engine...", file=sys.stderr)
        results["rust"] = run_rust(args)
    if not args.skip_mlx:
        print("running mlx reference...", file=sys.stderr)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            mlx_out = Path(tf.name)
        results["mlx"] = run_mlx(args, mlx_out)

    rust = results.get("rust", {})
    mlx = results.get("mlx", {})

    print("\n" + "=" * 78)
    print(f"PROMPT: {args.prompt}")
    print("=" * 78)
    rows = [
        ("tokens", "generation_tokens", "generation_tokens"),
        ("denoise steps", "denoise_steps", "diffusion_denoising_steps"),
        ("seconds", "generate_seconds", "generate_seconds"),
        ("finish", "finish_reason", "finish_reason"),
    ]
    print(f"{'metric':<16}{'rust (.dgq q4)':<24}{'mlx (4-bit ref)':<24}")
    print("-" * 64)
    for label, rk, mk in rows:
        print(f"{label:<16}{fmt(rust.get(rk)):<24}{fmt(mlx.get(mk)):<24}")
    print()
    print("--- RUST reply ---")
    print(rust.get("reply", "(skipped)"))
    print()
    print("--- MLX reply ---")
    print(mlx.get("reply", "(skipped)"))
    print("=" * 78)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
