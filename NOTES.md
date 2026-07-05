# diffgemma-mps — notes

Trimmed 2026-07-05. The generation contract is **SPEC.md**; kernel verdicts +
precision policy are **KERNELS.md**; env flags are **src/flags.rs**; the plan
is **PLAN.md**. Bug archaeology and measured-data history live in git history
and the agent memory — this file keeps only the standing rationale and the
auxiliary command reference.

## Lineage

Consolidated 2026-06 from PLAN.md (v1), PLAN2.md, PLAN_MONOLITHIC.md; trimmed
again 2026-07-05 after the near-done milestone (MLX-exact sampler default,
0/10 wart census, suite green, flags centralized).

## Standing decisions & rationale

- **Checkpoint-orientation quant layout** (row-major `[out,in]`, not
  pre-transposed): streaming converter, shared CPU/GPU layout, parity indices
  match the checkpoint. The historical "transpose cost" was a
  kernel-orientation artifact.
- **Mixed precision: bf16 where it's cheap, q4 only for the bulk.** Attention,
  dense FFN, and `embed_tokens` are bf16 (lossless into the half GEMM tiles;
  q8 embed flattened hard-tail logits and stalled convergence). Only the MoE
  experts (the bulk bytes) are 4-bit.
- **Experts stay ~4-bit — hard memory constraint** (bf16 experts ≈ 4× the
  bytes). Quantization was fully exonerated as a quality lever (q6 at 2% error
  changed nothing); close any future gap memory-neutrally.
- **Router stays bf16 weights / f32 logits** — routing is control flow;
  near-boundary noise flips experts discretely.
- **Compute-bound regime** (M3-class, canvas=256): smaller quant formats buy
  ~nothing in speed (everything dequantizes to f16 for the MMA units); MFU is
  the lever, not bandwidth.
- **Two KV layouts coexist** — monolithic b4 (half, unified per-layer region)
  is canonical; the engine keeps its legacy f32 KV. Unify only if the engine
  ever stops being validation-only.

## Auxiliary commands

Everyday commands (ask/chat/smoketest/bench/quantize) are in PLAN.md.

```bash
# Single monolithic forward with per-stage activation checkpoints
diffgemma-mps step-probe -m $WEIGHTS --layers 3 --kv-len 64 --seed 42
# Engine (f32, validation-only) generate
diffgemma-mps generate-gpu -m $WEIGHTS -p "Hello" --seed 42 --steps 2
# Engine-vs-monolith per-step logits oracle
diffgemma-mps step-parity -m $WEIGHTS
# Engine prefill bench (prompt -> KV only)
diffgemma-mps bench-prefill -m $WEIGHTS --prefill-len 64 --layers 30 --iters 5
# GEMM micro-bench vs MPS oracle
diffgemma-mps bench-gemm --shapes 256x2816x2816 --oracle mps --iters 10
# Metal device info / weight introspection (no GPU forward)
diffgemma-mps probe-device
diffgemma-mps summary; diffgemma-mps config
```

Debug/probe env flags (DGQ_TRACE_*, DGQ_LOG_*, DGQ_DUMP_KV, ...) are all
documented in `src/flags.rs`.

## MLX parity tooling (python/)

`python/scripts/`: `mlx_generate.py` (ground-truth generation),
`compare_generation.py`, layer-cos + denoise-trace dump/compare pairs.
Rules: ALWAYS prompt-match layer-cos comparisons; NEVER run MLX and our
engine in parallel (unified-memory crash — serialize every model-loading
process).
