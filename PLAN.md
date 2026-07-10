# diffgemma-mps — plan

Low-dependency Rust + Metal inference engine for
[DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
(Gemma-4 26B-A4B MoE, discrete block diffusion) on Apple Silicon.

Docs: **SPEC.md** = the implemented generation contract. **KERNELS.md** =
kernel verdicts + precision policy. **STRATEGY.md** = how to work here.
**ROADMAP.md** = the 4-week production plan + experiment playbooks.
**src/flags.rs** = every env flag. History lives in git + agent memory,
not here.

## Where we are (2026-07-05 — near done)

The engine is in ship shape:

- **Perf**: ~0.93–0.97 s/step (M3 Pro, 30L) — under MLX-4bit's 0.94 s/step
  pace. Tunable GEMM, block-sparse MoE, MMA attention, sparse SC all
  default-on and validated. **Convergence steps: parity-class** (matched-canvas
  multi-seed 2026-07-05: ~1.15× vs MLX's best config = end-to-end denoise
  parity; ~1.6× FASTER than the mxfp4 checkpoint; the old "2× step gap" was a
  mixed-config single-seed artifact — quant format alone swings MLX's own
  convergence 12↔27 steps on one canvas). Chat: cross-turn KV reuse (turn-3
  prefill −71%), entropy early stop default-on (0.05, signed off), fast
  between-block extend (~10s → 0.85s/block), attention occupancy fix
  (2.4-2.8×, kv-scaling halved). **Wall-clock vs MLX-4bit (their fastest
  config, temp 0, natural finish): ours WINS on every probe** — capital
  2.9 vs 3.7s; sky ~410tok 22.0 vs 27.6s (18.6 tok/s); transformer ~840tok
  50.2 vs 59.2s (16.7 tok/s).
- **Quality**: MLX-exact sampler semantics default (no-freeze + argmax commit,
  signed off 2026-07-05). Wart census 0/10 (was 4/10). Smoketest 17/17 at the
  spec seed. Fast-prefill degenerate class fixed as a side effect.
- **Correctness**: full test suite green (704). Step-parity oracle valid.
  Quantization exonerated as a quality lever (q6 experiment).
- **Model**: `model/diffusiongemma-q4emb` (bf16 attn/dense/embed, q8 SC, q4
  experts, ~18.9 GiB blob). q6/nvfp4 profiles + split-blob supported.

**Standing rule: quality never ratchets without a human sign-off.**
Bit-identical changes ship on identity evidence; anything else needs the
multi-seed gate aggregate + wart census + explicit user approval.

## Long context (2026-07-06 — works to 100k+, speed work remains)

Model supports 262k positions natively. Shipped: **ring-buffer sliding KV**
(~20 KB/token + ~410 MiB fixed; 100k ≈ 2.4 GiB, was an impossible 22 GiB),
**MMA prefill attention** (6.5k prefill 98.6→31.4s), **f16 KV +
direct-device simdgroup_load** (staging deleted; mma_full 22.6→7.8 ms/layer
@kv1024; gate IMPROVED to 47/51), `chat --ctx N`. Needle retrieval EXACT at
6.5k, 33.4k (prefill 166s, ~2s/step) and **105k (prefill 1085s, ~4.5s/step,
KV 2.4 GiB)**. Flash-decode sequential blocking was built (bit-identical,
`DGQ_ATTN_KV_BLOCK`) and DISPROVEN — the lockstep threadgroup sweep already
gets SLC service (171 > 150 GB/s DRAM at 32k); default off. q8 KV was then also built
(group-32, quality-neutral, halves KV memory, `DGQ_KV_Q8` opt-in) and
DISPROVEN as a speed lever at ≤33k: +9% prefill / +54% denoise — the
f16 direct-load kernels are ISSUE-bound at SLC speed, not byte-starved,
which rules out TurboQuant-class low-bit KV for speed by the same physics.
E4 (ROADMAP 2.4) settled the last cell 2026-07-07: at kv=105k the sign
flips but only to −6% denoise (still SLC-served, ~580 GB/s effective) —
under the 15% adaptive gate, so f16 stays the default at every length;
q8 is the KV MEMORY lever (+6% bonus past 100k). q8 105k needle exact.
Batched multi-chunk prefill SHIPPED
(bit-identical, `DGQ_PREFILL_BATCH`): 4x256 causal sub-chunks as one M=1024
forward — 33k prefill 165.7→142.4s (−14%). The weight-stationary expert
GEMM (ROADMAP E1) was then built and DISPROVEN 2026-07-07
(`DGQ_MOE_PREFILL_BM`, default off): taller expert blocks are bit-identical
but a perf WASH at 64 and 3.6x slower at 128 (register spill) — the expert
GEMM at M=1024 is COMPUTE-bound at the ~2.3 TF/s kernel wall (~6x margin
over weight bytes even with per-block re-reads), so byte-cutting can't win;
further prefill MoE gains need higher GEMM TF/s (fragment-tile class). See
agent memory `long-context-100k`.

## Long-prompt correctness (2026-07-10 — fixed by cap; fp16 stream is the real fix)

Field incident: long agentic turns at 100k ctx collapsed into newline soup.
Root-caused (task #64) to the FAST QUANTIZED PREFILL, two stacked defects:

1. **Pad-row ring clobber (FIXED, 3285ebe).** The zero-padded tail chunk
   wrote pad K/V; pad positions wrap onto `pos & ring_mask` and clobber the
   OLDEST LIVE window slots on all 25 sliding layers (186 slots at a 4.4k
   prompt). Inert below ring size (2048) — why every short-context gate was
   green. Fix: `StepParams.kv_write_end` suppresses cache stores past the
   prompt end. KV-verified (layer-0 fast-vs-engine: 186 broken slots → 0).
   The old "padding is provably overwritten" claim was true only for linear
   full-attention regions.
2. **bf16 activation-stream accumulation (CAPPED, e8de7d1; fix = E11).**
   With (1) fixed and every kernel family A/B-exonerated (GEMM/MoE/rope
   swaps byte-identical; scalar and MMA attention both degrade), the
   remaining delta vs the correct f32 engine is activation precision. Fast
   prefill answers real-document questions exactly to ~2.2k tokens, degrades
   at ~4.2k, hallucinates fluently at 6.6k; the engine is exact at every
   probed length and MLX (fp16 stream) is exact at 6.6k. Default now:
   `DGQ_FAST_PREFILL_MAX=2048` — prompts AND cross-turn deltas above it take
   the f32 engine (deltas via canvas-block engine extends at the reuse
   offset). Serve stays fast in steady state (reuse + tool-output compactor
   keep deltas small); cold long-document prefill is slow until E11/E12.

**Validation lesson (institutionalize as E13):** needle probes stayed EXACT
through all of this — literal retrieval survives on a few sharp attention
edges while grounded comprehension (many medium-weight edges) dies. Long-ctx
claims must be gated on real-document Q&A ladders, not needles.

Also fixed en route: shader edits were invisible to cargo (`include_metal!`
registers no deps) and the pipeline cache hashed only 60/93 shader files —
stale metallibs served after kernel edits. Both now keyed on a build.rs
whole-tree hash. Distrust pre-3285ebe "byte-identical" kernel A/Bs on
previously-unlisted files.

## Open items

| Item | Note |
|---|---|
| **Rolling window KV (E14)** | THE cap-lifter after the 2026-07-10 disproofs: E11 fp16-arena BUILT (opt-in `DGQ_PREFILL_F16`) but 4.2k probe still hallucinates; ring-uncap disproven too → failure is chunk-boundary f16 KV rounding compounding ~p/256 hops (engine/MLX correct = unchunked). Fix = f32 side window K/V (~1.3 GB constant) for sliding layers during fast prefill (ROADMAP §6.4) |
| **Engine prefill (E12)** | PARTIAL 2026-07-10: ring-correct GPU hydrate (was O(n²) CPU + wrong past wrap), hydrate-once chunked extend, resident extend (wash — kernel-bound), bit-identical gqa clamp (−19%). Engine ≈55-78 ms/tok, kernel-bound (scalar 3-pass GQA ~70%, f32 MoE); further surgery poor ROI vs E14; linear-f32 engine KV = memory wall past ~10k |
| Long-context speed | GEMM ledger CLOSED 2026-07-07: tunable = MPS wall, sparse 92-96% of dense at prefill distribution, SPARSE_BN=128 shipped (+6-8% kernel); MLX-qmm gap ~10-15% = pipelined-loader port worth ≤2-3% (non-lever). Remaining: attention fragment-tile (ROADMAP E5) only |
| Seed-123 empty-reply artifact | short factual prompts, both prefill paths (engine 5 / fast 2 of 17 at that seed); trajectory-level, pre-existing |
| Legacy GEMM retirement | `gemm_block*` legacy pipelines after a stable tunable cycle (KERNELS.md deprecation list; needs user nod) |
| Mechanical kernel merges | embed_gather / gather_rows / f32_to_half families (KERNELS.md) |

## Command reference

`WEIGHTS=model/diffusiongemma-q4emb`; binary at `target/release/diffgemma-mps`
(build: `cargo build --release`).

```bash
# Generate / chat
diffgemma-mps ask  -m $WEIGHTS -p "Hello" --seed 42
diffgemma-mps chat -m $WEIGHTS
# Gate / bench / tests
diffgemma-mps smoketest -m $WEIGHTS            # 17/17 required before commit
diffgemma-mps bench-step-kernel -m $WEIGHTS --profile-steps 8
cargo test --release
# Requantize from HF safetensors
diffgemma-mps quantize -m model/transformer -o model/diffusiongemma-q4emb --profile q4
# MLX reference comparison (SERIALIZE with our runs — never in parallel)
python/.venv/bin/python python/scripts/mlx_generate.py -p "..." -o /tmp/mlx.json
```

Wart census: `bash <scratchpad>/wart_census.sh $WEIGHTS out.txt` (10-seed
greentext; the sensitive sampler probe — 0/10 is the baseline).
