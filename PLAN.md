# diffgemma-mps — production plan

Low-dependency Rust + Metal inference engine for [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it) (Gemma-4 26B-A4B MoE, discrete block diffusion) on Apple Silicon.

Single forward-looking plan: **open work only.** Resolved work, measured data, and bug archaeology live in `NOTES.md`. How to work on the code: `STRATEGY.md`. Model semantics: `ARCHITECTURE.md`.

---

## Where we are (2026-06)

`generate-monolithic` (single-encoder step kernel) is the production path. **Mixed-precision `.dgq`** (`model/diffusiongemma-q4emb`): bf16 attention + dense FFN, bf16 embed (tied lm_head + SC), q8 SC-MLP, **q4 experts** (only bulk-quantized tensor — memory constraint). Blob ~18.9 GiB.

**Quality vs MLX-4bit** (`mlx-community/diffusiongemma-26B-A4B-it-4bit`, matched-canvas "sky blue"): output-level equivalent — coherent, correct, ~8-step convergence; smoketest 16/16. Residual: a few steps behind MLX-13 on the hard tail; minor long-chat doubling.

**Latency (M3 Pro, 30L, sparse SC default-on):** **~1.26 s/step**, **1.34× slower/step than MLX-4bit (0.94 s)** — down from 1.71× at the start of this cycle. Step is GPU-compute-bound across many kernels already near peak MFU; no single dominant bucket.

### Target
Sustained **≥ 8 tok/s e2e on 24 GiB base; 25+ on Pro/Max 36 GiB**, MLX-quality chat. Per-step ≤ 1.2 s stretch.

### Non-goals (this cycle)
Training/LoRA, multi-user serving, CUDA/Linux, vision (deferred; `--skip-vision`).

---

## Shipped this cycle (perf, all validated)
block-sparse MoE GEMM · GQA matrix-unit attention (`DGQ_ATTN_MMA`) · bf16 stacked QKV+gate/up · **MoE gather-fusion + scatter rewrite** (~100 ms, bit-identical) · **sparse SC softembed** (`DGQ_SC_SPARSE`, default-on, ~16%/step, smoketest 16/16, MLX-equivalent — *approximate*, drops prob tail < e⁻¹⁰ of row max). Cumulative this session: 1.61 → 1.26 s/step.

## Disproven non-levers (do NOT re-attempt without new info — see `NOTES.md`/memory)
2D dense-GEMM register-blocking (production 1D wide-N already beats it at M=256 fat-N) · SC softembed tiling (occupancy-bound; wide-N/bigger-chunk/coalesce all worse) · full-layer (hd=512) 1-head attention MMA (staging/occupancy-bound, ties scalar) · partial bf16 lm_head (frozen rows' hidden states still evolve → breaks convergence) · partial-forward (K/V staleness) · dispatch/ICB (~0.2 ms/step encode) · bounds-check removal · load_unsafe · nvfp4 experts.

---

## Q — MLX equivalence (active)

| # | Task | Exit |
|---|------|------|
| Q4 | Close last few steps to MLX-4bit (hard-tail convergence) + residual long-chat doubling | matched-canvas ≤ 14 steps; no visible doubling |
| Q5 | Confirm equivalence on long multi-block generations (not just single-block) | side-by-side long-gen quality table |

**Q4 leads (memory-neutral only — no resident bf16 experts):** bf16 (7-mantissa) vs MLX f16 (10-mantissa) **activation** precision is the leading hypothesis — per-layer cos drift accumulates 0–29 (NOTES §10), memory-neutral to test (bf16→f16 arena, check f16 overflow vs ~100s-magnitude residual stream). NOT experts (nvfp4 didn't help) and NOT embed (bf16-embed fix holds, after_preamble cos 1.0). Tooling: `--write-trace` + `DGQ_TRACE_ENTROPY=1` → `dump_mlx_denoise_trace.py --canvas-ids` → `compare_denoise_trace.py`.

---

## P2 — Latency

**Largely closed this cycle (1.71×→1.34×).** Remaining levers are low-EV; the step is compute-bound at good MFU.

**Honest per-step split (30L, `--profile-steps`, post-sparse-SC):** pre_moe (qkv+attn+o_proj+dense) ~43% · moe_grouped (experts) ~22% (was ~26%, gather/scatter shipped) · finish (lm_head) ~15% · preamble (SC) ~10% (was ~16%, sparse shipped) · moe_post ~0.3%.
> Use `--profile-steps N`, NOT `--layer-profile` (per-stage sync flattens the breakdown).

| # | Task | Status / note |
|---|------|------|
| P2.6 | lm_head MPS/precision sweep | open — finish ~15%, partial-lm-head is q8-only (bf16 partial breaks convergence) |
| P2.8 | KV windowing for full layers as committed context grows | open — full/global layers (hd=512, 5 of 30) dominate attention in long context |
| P2.9 | Flash full-attention kernel (register-resident O → group-8 K/V sharing) | open, **meaty + non-bit-identical + uncertain**; only real attention lever left, bigger payoff in long-context |

**P2 exit:** ≤ 1.2 s/step @ 30L; ≥ 8 tok/s e2e.

---

## P3 — Harden & ship

| # | Task | Exit |
|---|------|------|
| P3.1 | Multi-block extend + `kv>0` golden parity | multi-block matches engine on fixed seed |
| P3.3 | 24 GiB memory budget enforcement | `--skip-vision` + q4 documented; `iogpu.wired_limit_mb` guidance |
| P3.5 | CI default monolithic | `step-ci` + templated gate on monolithic path |
| P3.6 | Declarative step dispatch schedule | `build_step_schedule()` sole source; arena-liveness unit tests; probe=mode-not-fork. (`step_kernel.rs` ~4k lines imperative + a duplicate comment-schedule → drift risk) |
| P3.7 | GPU debug status / invariant flag | debug-gated `DebugStatus` buffer + shared error codes; ≥3 hot kernels wired; zero prod regression |
| P3.8 | Subkernel extraction completion | all monolithic stage bodies in `shaders/kernels/` + Tier-1 oracles; retire legacy `qgemm.metal` |

**Resolved loose ends** (now in NOTES): NONDET-SC-1, COLD-START-1 (warm-up workaround). **Still open:** `gemm_q8`/`gemm_q8_rowk` standalone oracle tests fail (32-tile vs `dispatch_shape` n_tile=128; production dispatch correct) — migrate to 128-tile or fix test dispatch (7 pre-existing failures incl. 1 sampler).

**Smoketest gate** (`smoketest` subcommand, `fixtures/smoketest/prompts.json`): convergence + adherence prompts; pre-commit gate, ratchet `max_steps` down. `diffgemma-mps -m <model.dgq> smoketest --layers 30 --seed 42`.

**P3 exit:** ship-quality chat @ 30L on 24 GiB; schedule asserts prevent drift; `STRATEGY.md` §6 invariants enforced in debug.

---

## Risk register

| Risk | Mitigation |
|------|------------|
| Sparse SC softembed quality drift on flat distributions | MAXK=8192 overflow keeps first-found not top-K (doesn't trigger at 16/16); `DGQ_SC_SPARSE=0` opt-out; monitor |
| Accept/entropy changes shift token goldens | synthetic-entropy fixtures; goldens keep `--raw` + fixed `--steps` |
| 24 GiB cap tight (blob ~18.9 GiB) | q4 experts + `--skip-vision`; `iogpu.wired_limit_mb` |
| MoE scatter nondeterminism | `moe_scatter_weighted` TG-reduce (no float atomics); determinism golden |

---

## Command reference

`WEIGHTS=model/diffusiongemma-q4emb`. Chat template by default; `--raw` for parity goldens.

```bash
# Generate / chat
cargo run --release --features metal -- -m $WEIGHTS generate-monolithic -p "Hello" --layers 30 --seed 42
cargo run --release --features metal -- -m $WEIGHTS chat -p "Hello" --layers 30 --seed 42
# Gate / bench
cargo run --release --features metal -- -m $WEIGHTS smoketest --layers 30 --seed 42
cargo run --release --features metal -- -m $WEIGHTS bench-step-kernel --layers 30 --iters 8 --profile-steps 8
cargo run --release --features metal -- bench-gemm --shapes 256x2816x8192 --iters 5
# MLX equivalence
HF_HUB_OFFLINE=1 python/.venv/bin/python python/scripts/compare_generation.py -p "Why is the sky blue?" --rust-model $WEIGHTS
```

Auxiliary commands (`step-probe`, `bench-prefill`, `convert-model`, `probe-device`) documented in `NOTES.md`.
