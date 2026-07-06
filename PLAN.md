# diffgemma-mps — plan

Low-dependency Rust + Metal inference engine for
[DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
(Gemma-4 26B-A4B MoE, discrete block diffusion) on Apple Silicon.

Docs: **SPEC.md** = the implemented generation contract. **KERNELS.md** =
kernel verdicts + precision policy. **STRATEGY.md** = how to work here.
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
gets SLC service (171 > 150 GB/s DRAM at 32k); default off. Remaining
levers: KV quantization (q8 = 2× bytes; TurboQuant-style rotated low-bit =
4-5×, arXiv 2504.19874) and batched multi-chunk prefill (the M=256
chunk-serial floor). See agent memory `long-context-100k`.

## Open items

| Item | Note |
|---|---|
| Long-context speed | KV quantization (q8 / TurboQuant) + batched prefill (above) |
| Seed-123 empty-reply artifact | short factual prompts, both prefill paths (engine 5 / fast 2 of 17 at that seed); trajectory-level, pre-existing |
| Legacy GEMM retirement | `gemm_block*` legacy pipelines after a stable tunable cycle (KERNELS.md deprecation list; needs user nod) |
| Mechanical kernel merges | embed_gather / gather_rows / f32_to_half families (KERNELS.md) |
| Inert padding-KV write | fast prefill writes garbage KV at [n..256), provably overwritten (SPEC.md §3) |

## Command reference

`WEIGHTS=model/diffusiongemma-q4emb`; binary at `target/release/diffgemma-mps`
(build: `cargo build --release --features metal,blas`).

```bash
# Generate / chat
diffgemma-mps ask  -m $WEIGHTS -p "Hello" --seed 42
diffgemma-mps chat -m $WEIGHTS
# Gate / bench / tests
diffgemma-mps smoketest -m $WEIGHTS            # 17/17 required before commit
diffgemma-mps bench-step-kernel -m $WEIGHTS --profile-steps 8
cargo test --release --features metal,blas
# Requantize from HF safetensors
diffgemma-mps quantize -m model/transformer -o model/diffusiongemma-q4emb --profile q4
# MLX reference comparison (SERIALIZE with our runs — never in parallel)
python/.venv/bin/python python/scripts/mlx_generate.py -p "..." -o /tmp/mlx.json
```

Wart census: `bash <scratchpad>/wart_census.sh $WEIGHTS out.txt` (10-seed
greentext; the sensitive sampler probe — 0/10 is the baseline).
