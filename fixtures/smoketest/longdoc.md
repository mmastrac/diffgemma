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

## Long-prompt correctness (2026-07-10 — ROOT CAUSE FIXED: spurious encoder RmsNormHidden; cap raise pending gate)

**FINAL RESOLUTION (later on 2026-07-10, supersedes item 2 below):** the
"bf16 accumulation" theory was wrong. The fast prefill applied the DENOISE
preamble's no-scale RMSNorm to the embedded hidden; the reference encoder
has no such norm. Scale-invariance through input_layernorm kept layer-0
K/V engine-exact (which hid it), while the residual stream carried a
per-token rescale → systematic MoE route flips at every length → collapse
past ~2.5k. Fixed by removing `RmsNormHidden` from the two prefill
encoders (denoise keeps it — parity-validated there). Doc-QA grounded at
3.2k/4.2k/6.6k on the plain fast path; 5.9k tokens prefill in 17 s. En
route, DISPROVEN with machinery kept opt-in: fp16 arena (E11), f32 side
KV (E14), ring uncap; plus the `DGQ_KV_NOISE` sensitivity anchor (engine
+1% KV noise stays correct). Historical account below.

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
| **Fast-prefill ROOT CAUSE FIXED (E15) — cap LIFTED** | 2026-07-10: the fast prefill wrongly applied the denoise preamble's no-scale RMSNorm to the embedded hidden (encoder reference has none) — scale-invariance through input_layernorm kept layer-0 K/V engine-exact while the residual stream carried a per-token rescale → systematic route flips → length-dependent collapse. Fixed (RmsNormHidden removed from both prefill encoders); doc-QA grounded 3.2k/4.2k/6.6k, 5.9k tokens in 17 s. GATE GREEN 2026-07-10: smoketest 17/17 at ALL {7,42,123} + 13k prompt_b grounded → `DGQ_FAST_PREFILL_MAX` default now **0 (uncapped)**. Remaining: E13 doc-QA smoketest tier; needle 33k/105k re-validation; 100k field-incident repro re-run |
| **Engine prefill (E12)** | PARTIAL 2026-07-10: ring-correct GPU hydrate (was O(n²) CPU + wrong past wrap), hydrate-once chunked extend, resident extend (wash — kernel-bound), bit-identical gqa clamp (−19%). Engine ≈55-78 ms/tok, kernel-bound (scalar 3-pass GQA ~70%, f32 MoE); further surgery poor ROI vs E14; linear-f32 engine KV = memory wall past ~10k |
| Long-context speed | GEMM ledger CLOSED 2026-07-07: tunable = MPS wall, sparse 92-96% of dense at prefill distribution, SPARSE_BN=128 shipped (+6-8% kernel); MLX-qmm gap ~10-15% = pipelined-loader port worth ≤2-3% (non-lever). Remaining: attention fragment-tile (ROADMAP E5) only |
| Seed-123 empty-reply artifact | short factual prompts, both prefill paths (engine 5 / fast 2 of 17 at that seed); trajectory-level, pre-existing |
| Legacy GEMM retirement | `gemm_block*` legacy pipelines after a stable tunable cycle (KERNELS.md deprecation list; needs user nod) |
| Mechanical kernel merges | embed_gather / gather_rows / f32_to_half families (KERNELS.md) |
| **"Un-RoPE" the KV for TurboQuant (E8 revisit)** | Idea (2026-07-10): E8's blocker was that RoPE sits between W_k and the stored K, so the Hadamard rotation can't fold offline into W_k. If we store K PRE-RoPE (or rotate stored K back by its position) and apply RoPE at attention-read time instead, the K-side rotation folds offline like V's — unlocking rotated low-bit KV without the runtime-rotation cost. Trade: RoPE moves into the attention read path (per-key trig per read vs once per write — the engine already reads-time-RoPEs Q, and cos/sin can come precomputed per position). Worth an experiment: quant-error stats on pre-RoPE vs post-RoPE K (pre-RoPE may also quantize better — RoPE mixes dims and can widen per-group ranges), then a read-time-RoPE attention prototype. Same idea may apply to other rotation-blocked weights |

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
# DiffusionGemma generation spec — as implemented

The precise, validated behavior of this port's generation pipeline, down to the
sampler details. ARCHITECTURE.md is the conceptual overview; this file is the
contract. Every deliberate divergence from the MLX/HF reference is listed in
§9 with its evidence and sign-off. Flags: every `DGQ_*` env toggle lives in
`src/flags.rs` (the single registry). Kernel-level verdicts: KERNELS.md.

Reference implementations compared against:
- **MLX**: `mlx_vlm/generate/diffusion.py` + `mlx_vlm/models/diffusion_gemma/`
  (in `python/.venv`) — the generation-loop reference.
- **HF**: `EntropyBoundSampler` / `LinearTemperatureScheduleLogitsProcessor`
  semantics (mirrored by our CPU sampler in `src/sample.rs`).

## 1. Model constants (google/diffusiongemma-26B-A4B-it)

| Constant | Value | Notes |
|---|---|---|
| layers | 30 | 25 sliding + 5 full (`FULL_LAYERS = [5,11,17,23,29]`) |
| hidden | 2816 | |
| vocab | 262144 | tied embeddings (lm_head = embed^T) |
| canvas | 256 | `canvas_length`; compile-time `CANVAS` in the monolith |
| attention | 16 Q heads; sliding: hd=256, 8 KV heads; full/global: hd=512, 2 KV heads | GQA |
| sliding window | 1024 | sliding layers attend only the last window−1 context positions |
| dense FFN | 2112 | gelu_pytorch_tanh |
| MoE | 128 experts, top-8, expert FF 704 | every layer; router in bf16 (Raw) |
| final softcap | 30.0 | `logits = 30·tanh(logits/30)` |
| RMS eps | 1e-6 | |
| embed scale | sqrt(2816) | on gather AND on SC soft-embeddings |
| special ids | pad=0, bos=2, eos/turn = {1, 106, 50} | filler sentinel 262143 (ours, invalid-logits guard) |

Sampler config (generation_config.json — we implement these defaults):
`entropy_bound=0.1` (NATS, not bits), `max_denoising_steps=48`,
`t_min=0.4`, `t_max=0.8`, `stability_threshold=1`,
`confidence_threshold=0.005`, `max_new_tokens=256`.

## 2. Two attention phases

- **Prefill (encoder role)**: causal attention over prompt (and later, each
  committed block) → KV cache. Sliding layers: query q attends [q−(window−1), q].
- **Denoise (decoder role)**: canvas attends bidirectionally to itself and to
  the KV cache. On sliding layers the canvas sees only the **last window−1
  encoder positions** plus all canvas (matches MLX `_make_decoder_masks`;
  `DGQ_ATTN_WINDOW=0` restores unwindowed for A/B). Canvas position ids are
  `kv_len .. kv_len+255` (continuation after prefill). No timestep/noise-level
  embedding exists.

## 2.1 Forward-pass details (checkpoint-specific; several are counterintuitive)

Settled against the reference implementations and the CPU oracle — do NOT
"correct" these from general Gemma knowledge:

- **Norms**: classic Gemma RMSNorm (`x/sqrt(mean(x²)+eps) · w`, eps 1e-6) —
  not `(1+w)`. Pre-norm everywhere. QK-norm = per-head RMSNorm with learned
  weights; V gets `rms_norm_no_scale`. The FF sandwich has SEVEN norms
  (input, post_attn, pre_ff, post_ff_1, pre_ff_2, post_ff_2, post_ff) — drop
  none. A per-layer bf16 `layer_scalar` multiplies the whole layer output.
- **RoPE**: rotation pairs are **split-half** `(d, d+rotary_dim/2)`, NOT
  interleaved. Sliding layers: full rotary, theta 1e4. Full layers: theta 1e6,
  `partial_rotary_factor 0.25` (128 of 512 dims), rope_type "proportional" —
  and the frequency exponent uses the FULL head_dim (512) as denominator, not
  the rotated subset (128).
- **No explicit 1/sqrt(d) attention scale** — raw dots → mask → softmax (the
  QK-norm folds the scale).
- **Full-attention layers have no v_proj.** V aliases the RAW k_proj output
  (`rms_norm_no_scale`, no RoPE) — `values = keys` before k_norm/rope. The
  kernel must not mutate the K buffer in place before V reads it.
- **MoE order**: rms_norm_no_scale(stream) → `router_scale[i]·hidden^-0.5` →
  linear → softmax over 128 → top-8 → renormalize the 8 to sum 1 → multiply
  by `per_expert_scale[expert]`. Tie-break: higher prob wins; equal prob →
  lower expert index (CPU and GPU must match). No separate shared expert —
  the dense MLP runs in parallel as the always-on path. `gate_up_proj`
  trailing dim 1408 = fused [gate‖up], 704 each. Activation gelu_pytorch_tanh,
  NOT SiLU.
- **SC injection**: before layer 0 only —
  `hidden = embeds + down(gelu_tanh(gate(x))·up(x))` where
  `x = RMSNorm_w(soft_signal)`; then rms_norm_no_scale. The soft mix gets the
  same sqrt(hidden) scale as token embeds.
- **RNG**: LCG `state = state·6966169279 + 1039523323`, uniform from the high
  32 bits; `Rng::new(seed)` → state = seed+1; GPU resumes the post-canvas-init
  state.

## 3. Prefill path selection

`should_fast_prefill(prompt_len)` (src/flags.rs): prompts ≤ 256 tokens use
the **f32 engine** prefill; longer prompts use the **fast quantized (bf16
activation) prefill** (~20× faster, ~3 ms/token). `DGQ_FAST_PREFILL_MAX`
(default **0** = uncapped) can reinstate an upper band above which prompts
return to the engine. Cross-turn delta reuse follows the same rule on the
DELTA length: short deltas fast-resume at the reuse offset, long deltas
extend via the engine in canvas-sized blocks. `DGQ_FAST_PREFILL=1|0` still
forces either path for all lengths.

Cap history (task #64/#68): from 2026-07-09..10 the default cap was 2048 as
a mitigation for a length-dependent comprehension collapse (grounded to
~2.2k, hallucinating at 6.6k). The root cause (fixed 2b0d12b, 2026-07-10)
was a spurious `RmsNormHidden` on the encoder pass — the DENOISE preamble's
no-scale norm applied to the prefill hidden. Scale-invariance through
`input_layernorm` kept layer-0 K/V engine-exact (hiding it) while the
residual stream carried a per-token rescale that systematically flipped MoE
routes (L1 KV 33% off at every length; 0.013 post-fix). Post-fix the fast
path is doc-QA-grounded at 3.2k/4.2k/6.6k/13k and the cap default is 0.
Validation lesson: needle probes DO NOT catch this class (retrieval rides a
few sharp attention edges and stayed exact throughout) — long-context
validation must use document-comprehension ladders (ROADMAP E13).

History of the short-prompt floor (see KERNELS.md for the full data): fast
prefill's bf16 activations perturb outlier KV channels enough to flip MoE
expert routing on borderline tokens. Under the legacy freezing sampler this
produced LOUD degenerate outputs on 2/16 short prompts. **Since the no-freeze
sampler (2026-07-05) the degenerate class is gone** — the perturbation costs
extra denoise steps instead — so the ≤256 floor is kept purely on wall-clock.

Pad-row contract (3285ebe): the fast prefill pads its last chunk to CANVAS,
but pad rows MUST NOT write KV — `StepParams.kv_write_end` (set to the
prompt end during prefill, `u32::MAX` otherwise) suppresses their cache
stores in `qk_rope_kv`. The previous "garbage KV at [n..256) is provably
overwritten" claim was WRONG on sliding layers: pad positions wrap onto
`pos & kv_ring_mask` and clobber the oldest live window slots (186 slots at
a 4.4k prompt, silently corrupting every sliding layer's window start).

## 4. Denoise step (one forward = one canvas refinement)

Order of operations per step (monolithic GPU path, `interpret_step`):

1. **Preamble**
   - Step 1 (no prior prediction): **deterministic first-step SC seed** — run
     the SC MLP on the *initial canvas's own embedding* (ScPreNorm reads
     `hidden` post-EmbedGather). OURS, deliberate (696ef2e): SC=0 makes step-1
     logits degenerate (cold-start empty reply), and leaving the previous
     generation's SC residual made reused sessions nondeterministic. MLX runs
     step 1 with no SC at all; our seed is prompt-independent + reproducible.
   - Steps ≥ 2: SC soft-embeddings from the PREVIOUS step's logits (§5), then
     SC MLP (q8 gate/up/down, GLU), added to the token embedding, RMS-normed.
2. **30 decoder layers**: QKV (stacked GEMM) → RoPE → attention (§2) →
   o_proj + residual → dense FFN (stacked gate|up, GLU, down) → MoE
   (router GEMM + top-8, block-sparse expert GEMMs, weighted scatter) →
   post-norms + residuals. All activations bf16 (see KERNELS.md precision
   policy); MoE scratch f32.
3. **Finish**: final norm → lm_head (tied embed; q8 or bf16 rows per model) →
   softcap 30 → sampler (§6) → SC logits pre-scaled by 1/T for the next step's
   soft-embed (matches MLX: schedule temperature shapes SC).

## 5. Self-conditioning (SC)

`probs = softmax(logits / T_schedule)` per position;
`soft_embed = probs @ embed × sqrt(hidden)`; fed to the next step.

Production paths (`src/flags.rs`):
- **Sparse (default on bf16-embed models, `DGQ_SC_SPARSE=0` opts out)**:
  gather only survivor tokens with prob ≥ e^-10 of the row max. APPROXIMATE
  (drops the tail); signed off — ~16%/step, output-level equivalent to MLX.
- **Chunked (the exact path)**: vocab-chunked prob GEMM with f32 accumulate;
  bit-matches the full prob-matrix GEMM (which was removed as dead).

## 6. Sampler — the entropy-bound loop (the details that matter)

Per step, given softcapped logits `L[256, 262144]`:

1. **Schedule temperature** `T(step) = t_min + (t_max − t_min)·(cur/max)`
   where `cur` counts DOWN from max_steps (first step T=0.8 → last ~0.4).
2. **Row stats** on `L/T`: per-position entropy `H` (nats), argmax, softmax
   normalizers.
3. **Accept mask (re-decided from scratch EVERY step)**: sort positions by
   ascending H; accept while the prefix-sum of accepted H ≤ 0.1 nats. The
   first (lowest-H) position is always accepted. **No accumulation and no
   freezing** — a position accepted last step gets no special treatment this
   step. (Exception: the final scheduled step accepts nothing; the loop exits
   on argmax.)
4. **Committed token at accepted positions = the row argmax** of this step
   (`DGQ_DENOISER_ARGMAX=0` restores HF's tempered categorical draw). This is
   MLX's default (user temperature 0): the schedule T shapes H and SC probs,
   never the token choice.
5. **Renoise**: every NON-accepted position gets a fresh uniform-random token
   id (seeded LCG) — including positions that were accepted in earlier steps
   but fell out of this step's accept set.
6. **Early stop** — any of:
   - **confident**: full-canvas argmax identical to the previous step
     (`stability_threshold=1` history ring) AND mean full-canvas H < 0.005;
   - **plateau (OURS, superset)**: accept mask bit-identical for ≥ 8
     consecutive steps AND mean H < 0.05 — catches stuck-but-stable canvases
     the confident rule misses;
   - **pad gate (OURS, superset)**: an all-pad/filler argmax NEVER stops
     (degenerate forward, not a converged answer) — enforced in both
     `sample_commit.metal` and the CPU stopper;
   - **entropy-only stop (OURS, DEFAULT ON at 0.05 nats)**: after the
     min-early-stop floor (12 steps), stop when full-canvas mean H <
     `DGQ_EARLY_STOP_MEAN_ENT` WITHOUT waiting for argmax stability — the
     answer text settles ~10 steps before the argmax fully locks; the tail
     only micro-flips near-tie positions. `=0` disables. Signed off
     2026-07-06: probe answers byte-identical, gate aggregate 43→45/51
     (sky_blue converges under budget), census-neutral, −15% steps on the
     gate set / −25-35% on long replies;
   - max_steps = 48.
7. **Final commit = the full-canvas argmax of the last executed step** — not
   the accepted canvas. Accepted tokens only ever shape intermediate
   conditioning.

### 6.1 "Unfreezing" — why there is no freeze

Steps 3–5+7 mean every canvas position can be revised at every step: the
model re-argmaxes accepted positions, drops them back to noise if their
entropy rises, and the final answer is whatever the last forward believes.
Any apparent "freezing" is emergent self-reinforcement (an accepted token
conditions its own next-step logits), not a mechanism.

This is the MLX/HF reference semantics. **Our port originally diverged**: a
`frozen` bitmask pinned each row's token permanently at first accept (feeding
a partial-lm_head row skip and faking frozen rows' entropy to 0 / argmax to
the frozen token, which also gamed early-stop). That freeze was PROVEN to be
the flat-row wart driver ("be be me", "偶然", "\>"): census 4/10 warty →
0/10 without it, adherence unchanged, and it also fixed the fast-prefill
degenerate class (§3). Removed as default 2026-07-05 with user sign-off
(a259d8c); `DGQ_FREEZE=1` restores the legacy behavior (and reactivates
partial lm_head), kept for A/B only.

## 7. Block-autoregressive chaining

The canvas is always the full 256 positions (see §9 for the MLX difference).
After a block converges: scan the committed argmax for a stop token
({1, 106, 50}); if found, trim there and end the turn. Otherwise append all
256 tokens, causally prefill them to extend the KV cache, re-seed the canvas
RNG, and denoise the next block. Blocks are immutable once committed.

Determinism: generation is deterministic at fixed seed (nondeterminism bugs
NONDET-SC-1/2 fixed at the root). Under no-freeze + argmax the trajectory is
additionally near-seed-invariant on non-flat prompts: once the accept set
saturates, the iteration is a deterministic argmax fixed-point and the initial
canvas washes out (seeds 7 and 42 produce identical gate outputs).

## 8. Quantization (.dgq)

`classify_tensor` policy (src/dgq/layout.rs): MoE experts → q4 group-32
affine (q6 group-32 under `--profile q6`; nvfp4 under `--profile nvfp4`);
SC MLP → q8 per-row; embed, attention, dense FFN, router, norms → Raw bf16
(lossless). Blobs above Metal's single-buffer cap (20.25 GiB on M3 Pro/36 GB)
are split at `expert_split` into two no-copy MTLBuffer regions (experts
written last, 16 KB-aligned). Measured: expert quant error (q4 7.9% → q6 2%
rel-RMS) does NOT affect the wart classes — quantization is exonerated as a
quality lever (task #20).

## 9. Deliberate divergences from the MLX/HF reference

| # | Divergence | Direction | Why / evidence |
|---|---|---|---|
| 1 | Committed token = argmax (not HF's categorical draw) | matches MLX default, diverges from HF | census 0/10 warty, fewer steps than categorical no-freeze; sign-off 2026-07-05 |
| 2 | First-step SC seed (SC MLP on initial-canvas embedding) | ours only (MLX: no SC at step 1) | fixes cold-start empty reply + session-carryover nondeterminism (696ef2e); smoketest evidence in commit |
| 3 | Plateau early-stop backstop | ours only | stuck-but-stable canvases; part of gate baseline |
| 4 | Pad-aware stop gate | ours only | never "converge" on an all-pad forward; inert on real prompts |
| 5 | Fixed 256-token canvas every block | MLX shrinks to max(remaining, 64) | structural simplification (compile-time CANVAS); only matters near max_tokens; trailing positions converge to eos/pad |
| 6 | Sparse SC soft-embed (survivor gather) | ours only (approximate) | ~16%/step; output-level equivalent to MLX-4bit; exact chunked path is one flag away |
| 7 | Length-heuristic prefill (engine f32 ≤ 256 tokens) | ours only | §3 — wall-clock decomposition; MLX prefills one way |
| 8 | Canvas init RNG (seeded LCG) | different RNG than mx.random | noise is noise; parity tooling pins exact canvases via `initial_canvas_ids` |
| 9 | Entropy accept + stop thresholds in nats | same as MLX (natural log) | noted because prose sometimes says "bits" — the code is nats everywhere |
| 10 | Entropy-only early stop (mean H < 0.05, §6) | ours only | tail steps only micro-flip near-ties; sign-off 2026-07-06, gate 45/51 ≥ baseline 43/51, census-neutral |

Historical divergences that were BUGS, since fixed to match the reference:
hard-freeze of accepted rows (§6.1, the wart driver); SC logits not
pre-tempered (fixed; legacy flag removed); early-stop gamed by frozen-row
fake entropy (gone with the freeze).

## 10. Validation harnesses

- **Smoketest gate** (`smoketest`, fixtures/smoketest/prompts.json): 12
  adherence + 5 convergence prompts with step budgets = spec-seed actuals + 2;
  spec seed 7, re-baselined 2026-07-05 for the MLX-exact sampler (17/17).
  Multi-seed aggregate {7,42,123} judges trajectory-reshuffling changes;
  single-seed results are arbitrary for those.
- **Wart census**: 10-seed greentext generation (flat/creative canvas), the
  sensitive probe for sampler-semantics regressions (baseline 0/10 warty).
- **Step-parity oracle**: engine (f32) vs monolith per-step logits (mean|Δ|,
  softcap-aware).
- **Kernel oracle matrix**: per-kernel CPU-mirror tests (`cargo test`); the
  recurring failure class is stale test dispatch grids after kernel rewrites —
  check the harness grid before suspecting kernel math.
- **MLX parity tooling** (python/scripts): layer-cos, denoise traces,
  generation comparison. ALWAYS prompt-match layer-cos comparisons.
# Kernel audit (2026-07-02)

Full pass over the 72 Metal kernels: aggregation, genericization, perf, and
per-step precision. Verdicts below are the source of truth for "should this
kernel exist / merge / change dtype"; re-audit when a family gains members.

## Hot-path budget (steady denoise step ≈ 0.97 s since tunable GEMM, kv=25)

> **2026-07-02 REBASE: tunable GEMM default-ON** (phases 1-4, all BIT-EXACT +
> token-identical): pre_moe ~565→449 ms, MoE ~320→234, finish ~150→106.
> Step 1.22 → 0.97 s — at/under MLX's 0.94 per-step pace with fewer steps.
> The table below predates tunable; treat its ms as the LEGACY baseline and
> its "wall" verdicts as legacy-kernel walls.

## Legacy hot-path budget (pre-tunable, kv=25, all-GPU)

> **2026-07-02 CORRECTION: the GEMM "walls" below are OUR-KERNEL walls, not
> machine walls.** At our exact production shapes on this M3 Pro: MPS matmul
> 3.4–3.7 TF/s (f32!) / 3.7–4.2 (f16); MLX steel/qmm 4.0–4.4 (f16 and q4g32,
> verified CSE-defeated + per-eval). Our best: 2.4–2.9. ~1.5–1.7× kernel
> headroom across ~1.0s of the 1.22s step. See "GEMM headroom investigation".

| Bucket | ms | Rate vs roofline | Verdict |
|---|---|---|---|
| dense GEMMs (qkv / o_proj / FFN, `gemm_block*`) | ~380 | 1.8–2.3 TF/s ≈ our-kernel wall | ~1.5x headroom vs MPS/MLX (see correction above) |
| attention (`attention_mma2` / `attention_mma_full`) | ~89 @kv64, ~372 @kv1024 (was ~212/~894) | 2.4–2.8x over prior | OCCUPANCY was the wall, not barriers/tuning (2026-07-06): mma2's 16 KiB tgmem O tile → 1 tg/core → kv-scaling latencies never overlapped. Fixes: mma2 O → registers (bit-identical, 2.7x); mma_full d-SPLIT across the 2 simdgroups (oreg 128→64/lane, ~10 KiB tgmem; QK needs a cross-half partial-sum → NOT bit-identical, gate-neutral 45/51, 1.6x). Full-tile K/V staging + fewer-barrier variants were TESTED AND SLOWER (wide-stride simdgroup_load + serial MMA chain); QG=1 mma_full also slower (register-bound, halved staging lanes). Window-clipped ≥1024 |
| MoE experts (`gemm_block_sparse` + scatter) | ~315 | ~2.3 TF/s on USEFUL flops ≈ dense wall | at the wall; M-tile levers disproven (see below) |
| SC preamble (`sc_sparse_*`, SC MLP q8, rowstats) | ~165 | occupancy/gather-bound | tiling levers disproven; NOT the vocab GEMM (sparse-SC active at default on bf16-embed; rowk chunked only runs with DGQ_SC_SPARSE=0) — rowk tunable port has NO production surface (2026-07-02) |
| finish (lm_head `gemm_block` Raw, softcap, sampler) | ~150 | at flops bound | at the wall |
| router (`gemm_block` + `moe_router_topk`) | ~22 | occupancy-limited N=128 | accepted (was 69 ms serial-dot) |

## Precision policy (settled empirically — see working notes for the data)

- **Weights**: bf16 lossless (attention / dense FFN / embed), q4 group-32
  experts (memory constraint), q8 SC-MLP. No changes.
- **Activation planes**: bf16 (`arena_load/store`) — **verdict SCOPE
  CORRECTED 2026-07-10**: valid for DENOISE and for prefill ≤~2k tokens
  ONLY. The 2026-07-05 disproofs of K_ACT_F16 / DGQ_HIDDEN_F32 were
  short-context claims; at long prefill the bf16 stream's 2⁻⁸ per-op step
  compounds across (position × layer) through the causal K/V recurrence and
  destroys document comprehension (exact ≤2.2k, degraded 4.2k, fluent
  hallucination 6.6k; engine-f32 exact everywhere; MLX-fp16 exact at 6.6k).
  Every discrete kernel was A/B-exonerated first. New evidence = the
  re-litigation bar is met: ROADMAP E11 rebuilds the arena-dtype FC as
  fp16 for PREFILL pipelines only (denoise stays bf16). Until it lands,
  `DGQ_FAST_PREFILL_MAX=2048` routes longer prompts/deltas to the engine.
- **f16 where values are bounded [0,1]**: `sc_probs` (fp16 + GEMM prob scale),
  attention P tiles (half). Already done; nothing else qualifies.
- **Always-bf16 buffers** (range or cross-path layout): logits (`FC29` forced),
  RouteScratch weights.
- **KV cache: f16 (2026-07-06, was bf16).** Range-checked (max|KV| = 21.9 on a
  real prompt; f16 max 65504) — f16's 10 mantissa bits beat bf16's 7
  everywhere in the live range (gate went 45/51 -> 47/51 on the flip). The
  real motive: f16 lets the MMA attention kernels `simdgroup_load` K/V tiles
  straight from device memory, deleting the whole bf16->half staging pass —
  the long-context attention enabler. All writers (qk_rope_kv,
  pack_encoder_kv, CPU pack) and readers converted together; the
  producer/consumer-mismatch rule below applies doubly here.
- **Always-f32**: moeout plane, rowstats planes, MoE grouped scratch.
- **The real precision hazard is producer/consumer dtype mismatch**, not the
  choice of dtype. Audit found exactly one (latent): `sc_probs` /
  `sc_softembed` read the always-bf16 logits via the *toggleable* loader —
  misread under `K_ACT_F16`. Fixed (always-bf16 reads, bit-identical at
  default). All other buffers verified consistent, including the previously
  bitten RouteScratch-weight path.

## Consolidation verdicts

Done already (no action): `gemm_bf16`→`gemm_block` (QUANT_RAW), moe grouped
q4/nvfp4 (QUANT_FORMAT), sc_softembed bf16 variant.

**Merge (mechanical, bit-identical, low priority)**
- `embed_gather` + `embed_gather_bf16` → one shader, quant FC (q8/raw) —
  halves the pipeline variants (the hf32 variants are gone as of 2026-07-05).
- `gather_rows` / `gather_rows_bf16` / `gather_rows_bf16_to_f32` → one shader,
  in/out dtype FCs.
- `f32_to_half` + `f32_to_half_scale` → scale FC (plain variant currently has
  no production dispatch).

**Keep separate (justified)**
- Attention family (scalar oracle / mma2 sliding / mma_full full): different
  algorithms and geometries; scalar is the causal-prefill + oracle path.
- `rms_norm_rows` vs `rms_norm_rows_tiled`: engine-vs-monolith dispatch
  patterns differ.
- `residual_half` (monolith, scalar axis) vs `vec_add_inplace`
  (engine): different paths, both production.

**Deleted (2026-07-02 cleanup after tunable landed)**
- `attention_mma` (1-head): superseded twice (mma2, mma_full); was test-only.
- `f32_to_half` plain: had no production dispatch (scale variant remains).
- `gemm_block_sq`: bench research prototype, superseded by `gemm_tunable`.

**Kept (correction: NOT dead)**
- `scatter_vocab_chunk`: the ENGINE-validation lm_head (lm_head.rs via
  sampler_kernels) dispatches it — engine-validation-only, not a placeholder.

**Deprecation candidates (need a stable tunable cycle first — do NOT delete yet)**
- Legacy `gemm_block` / `gemm_block_stacked` / `gemm_block_sparse`: still the
  DGQ_GEMM_TUNABLE=0 fallback, the non-bf16-checkpoint stacked-q4 path, the
  nvfp4 expert path, AND the bit-exactness ORACLE the tunable bench rows
  verify against. Revisit after a few stable weeks.
- Legacy adaptive-M machinery (`DGQ_MOE_TILE_ADAPT` + m16/m8 helpers in
  gemm_block_tile + classed pipelines): superseded by gemm_tunable_sparse at
  default (only live at DGQ_GEMM_TUNABLE=0). User signed it default-on
  2026-07-02; removal needs their nod.
- `gemm_block_grouped` (pre-sparse MoE): DGQ_MOE_BLOCK_SPARSE=0 fallback only.

**Graveyard (disproven machinery DELETED in the 2026-07-05 flag cleanup):**
DGQ_HIDDEN_F32 (f32 hidden planes + hf32 kernel variants), K_ACT_F16 (f16
arena FC), ICB record/replay (step_icb), DGQ_ACCEPT_ROW_CAP (superseded by the
no-freeze fix), DGQ_SC_PRE_TEMP, the slow + full-prob SC softembed paths
(sc_softembed.metal deleted; chunked + sparse remain), scalar-per-expert MoE
env toggle, DGQ_GEMM_N_TILE (fixed 128), DGQ_MONOLITHIC, DGQ_TIME_DISPATCH,
DGQ_TRACE_BISECT. The partial-lm trio stays (live under DGQ_FREEZE=1).
**All surviving flags live in `src/flags.rs` — the single registry.**

## MoE tiny-M investigation (2026-07-02) — CLOSED, no lever

The old "1.3 TF/s tiny-M-bound" verdict was stale (isolated `bench-gemm` M=33
numbers, not the production step). Measured on the production step: useful MoE
flops = 24.5 GFLOP/layer at ~10.5 ms/layer = **~2.3 TF/s useful ≈ the
dense-GEMM wall** — there was never a 60-90 ms chunk to recover.

Two implementations built and measured (routing IS heavily skewed — step-1
sample: 38-71 active experts of 128, median m_e≈6, +59% padded rows at 32-row
tiles — the padding is real, it just isn't on the critical path):

- **M-tile classes** (3 pipelines, 32/16/8-row tails, class-partitioned block
  list): ~4% SLOWER. Cause isolated with an all-C32 diagnostic: the extra
  indirect dispatches cost ~0.12 ms each even at ZERO height (~0.5 ms/layer
  for 4 empty dispatches) — Metal hazard tracking serializes same-buffer
  dispatches. Machinery removed. **Rule: never split a hot GEMM into multiple
  same-encoder dispatches on this path.**
- **Adaptive-M** (`DGQ_MOE_TILE_ADAPT`, **default ON** since 2026-07-02 —
  user call; `=0` opts out): single dispatch, per-block runtime simdgroup
  remap (8/16/32 rows) — bit-identical (subkernel bitwise tests q4+nvfp4,
  E2E token-identical on/off, gate 17/17 at default). Perf: wash (adjacent
  A/Bs ±1.5%, overlapping). Removing 50-75% of the MMA work from ~1/3 of
  threadgroups moved wall time ~0% → per-TG cost is pipeline/fixed-dominated
  (see GEMM headroom investigation), not MMA-bound. Enabled anyway as a
  LATENT win: once the tunable GEMM port (task #19) makes per-TG cost
  compute-bound, the padding savings activate — carry the adaptive M-mapping
  into that port.
- **Weight-stationary prefill blocks** (ROADMAP E1, 2026-07-07 —
  `DGQ_MOE_PREFILL_BM=64|128` opt-in, default off): batched-prefill block
  list built at a taller height, same tunable sparse kernel at that TUNE_BM,
  one weight stream per expert instead of ~2.4. BIT-IDENTICAL (6.5k needle
  KV dump byte-equal at 32/64/128) but DISPROVEN as perf: 64 = wash, 128 =
  3.6x slower (TM=8 → 64 f32 acc/lane register spill). Roofline: at M=1024
  the expert GEMM is COMPUTE-bound (~48 ms/layer MMA at 2.3 TF/s vs ~7.6
  ms/layer W bytes incl. re-reads) — byte-cutting is a non-lever here; the
  prefill MoE lever, if any, is GEMM TF/s (fragment-tile class).

## GEMM TF/s ceiling — SETTLED 2026-07-07 (bench: `bench-gemm --shapes sparse`)

Measured at the PREFILL shapes (M=8192-slot MoE, M=1024 dense), M3 Pro:
- **dense tunable 64x64 = 3.69-3.88 TF/s = AT the MPS matmul wall
  (3.65-3.69)** — the 2026-07-02 "1.5-1.7x headroom" is CLOSED; the tunable
  port captured it.
- **sparse @ prefill route distribution (128 experts x 64 rows): 3.33-3.64
  useful = 92-96% of its dense twin.** SHIPPED from the sweep: SPARSE_BN
  64→128 (gate_up +5.6%, down +8% within-run; both N dims divide 128; same
  block list; bit-identical — KV dump byte-equal, gate 17/17).
- **sparse @ denoise distribution (x16 rows): 1.73-1.87 useful** — the old
  "~2.3 TF/s useful wall" figure was THIS regime (per-TG fixed cost +
  padding at tiny M_tile), i.e. the CLOSED tiny-M family, not a kernel
  quality gap.
- **MLX steel qmm: 3.96-4.15 at the same shapes** (f16 dense 4.29-4.45).
  The remaining ~10-15% vs MLX is their software-pipelined loader/MMA;
  porting it is a real GEMM project worth <= ~2-3% end-to-end prefill →
  NON-LEVER at current ROI. GEMM TF/s is now a closed ledger; prefill
  gains must come from non-GEMM stages or fewer FLOPs.

## GEMM headroom investigation (2026-07-02) — CLOSED by the above (history)

MPS/MLX prove ~1.5–1.7× GEMM headroom at our shapes (numbers above). Root
cause hunted with a bench-only prototype (`gemm_block_sq`, wired into
`bench-gemm` as `gemm_block_sq/*` rows, correctness-checked vs gemm_block):

**Exonerated (each measured null on this machine):**
- Occupancy alone: 32x32x32 @ ~9KB tgmem (3 TGs/core vs production's 1 at
  20KB) TIES gemm_block (~2.3). But N_TILE=32 at UNCHANGED 20KB = 1.47 —
  narrow tiles need small tgmem to break even; occupancy matters, it's just
  not the MLX delta.
- bf16 A-load conversion: x_fp16 reinterpret variant = no change.
- W-dequant lane utilization (32→128 threads): no change.
- Transposed tgmem B loads (store-W-transposed variant): no change.
- f16 accumulation: MLX BlockMMA AccumType defaults to FLOAT — they
  accumulate f32 like us.

**Partial win, shippable:** 64x64x32 tile (4 simdgroups, 32x32 quadrant, 16
accs) at ~10KB via aliasing the f32 store tile over the dead load buffers:
+7% over production at MoE shapes (3.07 vs 2.88 @ 2048x704x2816), tie at
M=256, and **bit-exact vs gemm_block (max|d| = 0.0)** — the K-chain per
output is unchanged by tiling.

**TUNABLE GEMM SHIPPING (task #19, phase 1 landed 2026-07-02,
`DGQ_GEMM_TUNABLE=1` bring-up flag):** `gemm_tunable.metal` — fragment-level
(per-lane thread_elements() loads, mem_none hints, per-lane C store) +
vectorized loaders (4-wide bf16 converts, 8-nibble q4 decode), TUNE_BM/BN via
#define prepend; 64x64 won every production shape. Bench: q4 3.5-3.8 TF/s,
Raw 3.61 (production kernels: 2.2-2.9) — ALL configs BIT-EXACT vs gemm_block
per element. Phase 1 wires the plain Raw path (o_proj / dense down / router /
lm_head-bf16): pre_moe 564-567 → 524-530 ms/step, token-identical, gate
17/17. Remaining phases: stacked (qkv, dense gate/up), q8 (lm_head on q4emb
is q8-embed), rowk (SC softembed), block-sparse MoE (carry adaptive-M).

**Where the rest lives:** MLX steel's fragment-level machinery — per-lane
`vec<T,2>` register-tile loads with compile-time strides (never
`simdgroup_load` from tgmem), `simdgroup_barrier(mem_none)` scheduling hints,
software-pipelined loader/MMA overlap (BlockLoader/BlockMMA in the mlx wheel
headers, MIT). Capturing it = a real GEMM-engineering project: port
steel-style loaders+MMA tiles into our FC/dequant/bf16 framework. Payoff is
measured, not speculative: up to ~300ms/step (1.22 → ~0.9s, past MLX).
Bit-identity is plausibly preservable (steel's kk order is ascending 8-chunks
like ours; keep our dequant math + bf16 I/O rounding) — verify per element.

## Test-harness debt

- `gemm_q8_rowk` oracle tests dispatched 128-wide n-tiles against the 32×32
  kernel (cols 32–127 unwritten → the long-standing cos≈0.34 failures). Fixed:
  harness now dispatches the production 32-tile grid.
- `moe_grouped(+nvfp4)` oracle tests (8): FIXED 2026-07-05. Root cause was the
  SAME class as the gemm_q8_rowk item above — stale test dispatch, not expert
  math: `moe_scatter_weighted` was rewritten to a (d-tile, token)×256-thread
  layout (`d = tgid.x*256 + tid`) but the sub-tests still dispatched the old
  (hidden, canvas)×8 grid, so only d<8 was ever written (hence cos≈0.34 at
  weight=1). Harnesses now dispatch the production grid, and the CPU side
  mirrors the full production chain (unweighted per-slot rows →
  `moe_scatter_weighted` CPU mirror). The scatter's own sub-test had the same
  stale shape but passed by fixture luck (hidden ≤ 8) — also fixed.
- `sample::stopper_blocks_degenerate_all_pad_argmax`: FIXED 2026-07-05. The
  pad-aware degenerate gate documented on `early_stop_allowed` was never wired
  into the stoppers; now enforced in BOTH `StableConfidentStopper` (CPU) and
  `sample_commit.metal` (GPU: all-pad/filler argmax suppresses confident/
  plateau stops; inert on normal prompts — gate 17/17 verified after).

## Ring-write hazard + shader staleness (postmortems, 2026-07-10)

- **Any kernel that writes KV by absolute position must respect
  `StepParams.kv_write_end`.** The fast prefill's zero-padded tail chunk
  wrote pad K/V; pad positions wrap onto `pos & kv_ring_mask` and CLOBBER
  the oldest LIVE window slots on sliding layers (the "padding is causally
  masked / overwritten" reasoning only holds for linear full-attention
  regions). 186 slots corrupted at a 4.4k prompt; inert below ring size —
  invisible to every short-context gate. Fixed in `qk_rope_kv` (3285ebe);
  the rule generalizes: past-the-end ≠ dead on a ring.
- **Shader-edit staleness (two independent holes, both fixed 3285ebe).**
  (1) `include_metal!` registers no cargo file dependencies → shader edits
  did not trigger rebuilds; (2) the pipeline `MTLBinaryArchive` cache keyed
  on a hand-listed 60/93 subset of shader files → edits to unlisted kernels
  (qk_rope_kv, attention_device, gemm_tunable, sample_commit, …) were served
  STALE from `~/.cache` metallibs. Both now keyed on a build.rs whole-tree
  hash (`DGQ_SHADER_TREE_HASH`, `cargo:rerun-if-changed=shaders`).
  **Trust rule:** a "byte-identical" kernel-edit A/B is only evidence if the
  build postdates 3285ebe (or the pipeline demonstrably recompiled);
  historical identical-output A/Bs on previously-unlisted files are suspect.

## Notes

- The old `gemm_q8` 32-tile migration flagged in earlier notes is DONE
  (`gemm_q8.rs` compiles `gemm_block` with `QuantFormat::Q8`); stale note
  corrected.
- Every non-bit-identical change must be judged by MULTI-SEED gate aggregate +
  wart census, not a single-seed smoketest (see prompts.json seed comment).
- Long-context claims must be judged by real-document Q&A ladders, not
  needle probes: needles ride a few sharp attention edges and stayed EXACT
  while 6.6k comprehension was fully hallucinated (ROADMAP E13).
# diffgemma-mps — 4-week road to production-ready

Written 2026-07-07 from a deep pass over PLAN.md / SPEC.md / KERNELS.md /
STRATEGY.md / NOTES.md, the agent-memory disproof ledger, the mlx_vlm
reference source, and the TurboQuant literature. Everything cited here is
measured on this machine (M3 Pro 36 GB) unless marked *estimate*.

---

## 0. Where the engine actually is (evidence snapshot)

| Axis | State | Evidence |
|---|---|---|
| Short/medium chat wall-clock | **Beats MLX-4bit (their fastest config) on every probe** | capital 2.9 vs 3.7s; sky ~410tok 22.0 vs 27.6s; transformer ~840tok 50.2 vs 59.2s |
| Denoise convergence | Parity-class: ~1.15× steps vs MLX best config, ~1.6× faster than mxfp4 | 7 matched-canvas pairs, multi-seed |
| Long context | **Works to 105k, exact needle retrieval**; KV 2.4 GiB; ~4.5 s/step at 100k | needle probes 6.5k/33k/105k |
| 100k prefill | ~15.5 min (batched; was 18) | 33k = 142 s measured, linear-ish extrapolation |
| Quality gates | spec-seed 17/17; genuine multi-seed 47/51; wart census 0/10 | gate re-run after every landing |
| Test suite | 708 unit + oracle matrix green; 7 ignored (model-dir) | `cargo test` |
| Known quality debt | seed-123 empty-reply artifact (5-6 short factual prompts at that seed) | pre-existing, trajectory-level |
| Product surface | CLI `ask`/`chat` (+`--ctx`, JSONL events); no server, no vision, single session | — |
| Ops surface | no CI, no packaged release, quantize pipeline is a CLI subcommand, first run pays Metal JIT | — |

**Perf levers already BUILT AND DISPROVEN — do not re-chase** (each has
machinery + a recorded verdict in flags.rs/memory):
sequential-kv-block "flash decode" (SLC already lockstep-served, 171 > 150
GB/s); q8 KV for speed (+9% prefill / +54% denoise at 33k — kernels are
issue-bound, not byte-starved; memory lever only, `DGQ_KV_Q8=1`);
TurboQuant-class low-bit KV *for speed* (same physics); MoE tiny-M tiles;
partial-forward; f32 hidden; f16 arena; ICB.

---

## 1. What "production-ready" means here (proposed acceptance)

1. **Install-to-first-token < 30 min on a clean 36 GB Apple Silicon Mac**
   (documented model fetch + quantize + warm), < 10 s model load thereafter.
2. **A local HTTP server** (OpenAI-ish chat completions + streaming) in
   addition to the CLI; concurrent-request safe (serialized generation,
   queued requests — the GPU is single-tenant by hard rule).
3. **No panics reachable from user input.** Context overflow, over-long
   prompts, bad UTF-8, missing files → typed errors with actionable messages.
4. **Quality**: spec-seed gate 17/17 in CI nightly; multi-seed aggregate ≥
   the 47/51 baseline; needle exact at 33k in nightly.
5. **Perf floor** (regression-gated numbers, not aspirations): ≥ 15 tok/s
   mid-length replies; ≤ 1.1 s/step at chat lengths; 33k prefill ≤ 140 s;
   100k functional ≤ 16 min prefill, ≤ 5 s/step.
6. **Docs**: README quickstart, SPEC/KERNELS current, model card + license
   notes, a benchmark page with the MLX head-to-head methodology.
7. **Scope decision recorded**: vision (VLM) in or out for v1 (recommend OUT,
   see §5.3 — it's the single biggest unscoped item).

---

## 2. Week 1 — finish the perf story (kernel work, all gate-bounded)

### 2.1 Weight-stationary expert GEMM — DONE 2026-07-07: DISPROVEN
Built (`DGQ_MOE_PREFILL_BM=64|128`, default 32/off, machinery kept): the
batched-prefill block list is built at a taller block height and consumed by
the same tunable sparse kernel compiled at that TUNE_BM, so one threadgroup
owns all (most) of an expert's ~64 rows = one dequantized weight stream.
Bit-identical PROVEN (6.5k needle: full 30-layer KV dump byte-equal and
tokens identical at 32/64/128); suite 708/708; spec gate 17/17.
- **Result**: BM=64 is a WASH (21.1s vs 21.1s 6.5k prefill, alternating warm
  runs); BM=128 is 3.6x SLOWER (76s — TM=8 means 64 f32 accumulators/lane,
  the known register-spill regime).
- **Why (roofline, real dims)**: expert W per layer = 128 x 3.72 MB q4 =
  476 MB → ~7.6 ms/layer at DRAM even WITH the 2.4x per-block re-reads; MoE
  MMA at M=1024 ≈ 110 GFLOP/layer ÷ ~2.3 TF/s ≈ 48 ms/layer. The expert
  GEMM is COMPUTE-bound by ~6x — cutting W bytes can't help. The premise
  ("win capped by weight re-reads", eb0edba) was a wrong attribution; the
  cap is kernel TF/s. Raising prefill MoE throughput now means raising the
  GEMM's TF/s (E5-class fragment work), not byte reduction.

### 2.2 Retire the f32 engine prefill — DONE 2026-07-07: DISPROVEN, keep the engine
Forced `DGQ_FAST_PREFILL=1` for ALL lengths, multi-seed gate {7,42,123}:
**41/51 (16/15/10) vs engine baseline 47/51 (17/17/13)** — a 6-point
regression, all in the EMPTY-REPLY / non-convergence class on short factual
prompts (blue_plus_yellow, opposite_of_hot, one_plus_one … "answer BAD",
blank output). Fast prefill's bf16-perturbed prompt KV aggravates the
seed-123 empty-reply artifact and spreads it to seed 42. Fails the ≥47/51
ship gate on quality alone (wall-clock moot). **VERDICT: keep the f32 engine
for short prompts — the length heuristic is a QUALITY mitigation, not just
wall-clock. Strong mechanistic clue for E6: the empty-reply artifact is
prefill-KV-precision-sensitive (engine f32 forward = partial cure).**

### 2.2b (superseded) Retire the f32 engine prefill for short prompts — 1 day
Short prompts (≤256) still run the legacy f32 engine at session open
(~0.85 s steady + one-time shader JIT; ~2 s cold). Post-no-freeze evidence
says fast prefill is adherence-clean (17/17 forced at seeds 7/42); the
length heuristic survives on wall-clock grounds only (+convergence steps on
some prompts).
- **Experiment**: multi-seed gate + census with `DGQ_FAST_PREFILL` forced for
  ALL lengths; measure convergence-step delta across the gate set and total
  turn wall-clock (fast prefill ~0.8 s floor vs engine 0.85 s + occasional
  cold JIT).
- **Ship if**: aggregate ≥ 47/51 and mean turn time not worse. Payoff: delete
  an entire engine dependency from the hot path (engine becomes pure
  validation), kill the cold-start JIT wart, simplify session open.

### 2.3 Canvas shrink near max_tokens (MLX parity, divergence #5) — 1 day
MLX shrinks the canvas to `max(remaining, 64)`; we always run 256-wide.
Matters at reply tails and tight `--max-new-tokens`: the finish stages
(lm_head + SC softembed, ~0.25 s/step) scale with canvas rows.
- **Experiment**: implement denoise-side M=`min(CANVAS, remaining)` rounded
  to 64; gate multi-seed (trajectory-affecting: canvas width changes the
  entropy pool → NOT bit-identical).
- **Expected**: minor wall win on capped replies; main value is closing a
  deliberate divergence before v1 and matching MLX's `min_canvas_length`
  semantics. Low risk, bounded.

### 2.4 q8-KV at 100k A/B — DONE 2026-07-07: table CLOSED, no default change
Measured per the timing methodology (isolated `bench-step-kernel
--layers 30 --kv-len 105000 --profile-steps 4`, full KV allocated, 3
interleaved rounds per arm — NOT full-prefill wall-clock):
- **The sign flips at 100k but only to −6%**: pre_moe (attention)
  4.14 → 3.87 s/step, total 4.66 → 4.39 (every q8 step beat every f16 step
  across all rounds). Under the ≥15% adaptive-flip gate → default stays f16
  at every length; q8 remains opt-in.
- **Why so small**: even at kv=105k the full-layer sweep reads ~430 MB/layer
  in ~740 ms ≈ 580 GB/s effective — still SLC-served lockstep, still
  issue-bound; unique DRAM traffic (~430 MB once) is milliseconds. The
  33k physics holds at 105k.
- Updated guidance: q8 = the KV MEMORY lever (halves KV), and at 100k+ it's
  additionally ~6% faster — win-win for very long contexts.
- Retrieval at 105k under q8 run separately (untimed, correctness only).
- **SHIPPED 2026-07-07: q8 auto-enables at long context** (`kv_q8(max_seq)`
  scales with the captured working-set cap; `DGQ_KV_Q8=1/0` overrides). f16
  @262k SWAPS (91% + 704 MB, pressure WARN via DGQ_MEM_WATCH); auto-q8 keeps
  it at 82%. Threshold ~178k on 36 GB, lower on smaller Macs. Short prompts
  stay f16 → gate 17/17 unchanged.

### 2.5 Attention issue-bound follow-up (timeboxed research, ≤ 1 day)
Full-layer attention at 100k ≈ 3 s/step is instruction-issue-bound. The one
untried shape: fewer, larger MMA fragments per stream (steel-style per-lane
`vec<T,2>` register tiles instead of `simdgroup_load`, as KERNELS.md maps
for GEMMs). Timebox a prototype on `attention_mma_full` only; if the bench
row doesn't show ≥ 1.5× at kv 32k, record and move on. Do NOT ship without
the full multi-seed gate (f16 math order changes).

---

## 3. Week 2 — correctness debt, robustness, CI

### 3.1 Seed-123 empty-reply artifact — 2 days, investigation playbook
The one standing quality bug: at seed 123, 5-6 short factual prompts emit
empty replies (both prefill paths; pre-existing across many changes).
- Localize per STRATEGY §2: trace a failing prompt (`--write-trace`,
  `DGQ_TRACE_ENTROPY=full`) and diff the first-step entropy/accept pattern
  vs a passing seed. Hypotheses to bisect: (a) initial-canvas collision with
  the stop-token region (canvas RNG at that seed seeds an `<eos>`-heavy
  region → stop-token trim fires on a garbage block), (b) first-block
  early-stop misfire (stop conditions on a canvas that converged to pad+eos),
  (c) chat-template/stop-token interaction.
  The stop-token TRIM path (`sequences.truncate(block_base + rel)`) is the
  prime suspect: an `<eos>` at offset 0 of block 1 yields an empty reply by
  construction.
- **Fix bar**: whatever changes is trajectory-affecting → multi-seed gate +
  census; target aggregate > 47/51 with seed-123 specifically improving.
- If root cause is "model genuinely emits eos-first on these canvases", the
  fix is a retry-with-reseed on empty first block (bounded, deterministic
  seed bump) — a product-level guard, gated the same way.

### 3.2 Panic-to-error audit — 1 day
Grep-driven: every `assert!`/`panic!`/`unwrap()` reachable from user input
becomes a typed error (the `set_kv_len` overflow assert, prompt >
context, tokenizer file missing, model dir malformed, `--ctx` beyond
memory). Add a `--ctx` memory estimator (we already print the budget table)
that refuses configs that exceed physical RAM with a clear message.

### 3.3 CI: three tiers wired to actual runners — 1-2 days
- **Tier 1 (every push, GitHub-runner-safe, no model)**: `cargo test`
  (708) minus `metal`-gated GPU tests… on a mac runner if available;
  otherwise compile + CPU-oracle subset. Clippy + fmt.
- **Tier 2 (self-hosted mac nightly)**: full suite + spec-seed smoketest
  17/17 + needle_3k exact + `--profile-steps` perf floor assertions
  (step ≤ 1.1 s at kv64, prefill 6.5k ≤ 21 s — 10% margins over current).
- **Tier 3 (weekly)**: multi-seed gate aggregate ≥ 47/51, wart census 0/10,
  33k needle, MLX head-to-head refresh (serialized!).
- Add `smoketest --json` output for CI parsing if not already parseable.

### 3.4 Session/robustness tests — 1 day
Multi-turn chat property tests: cross-turn KV reuse vs reset equivalence
(exists as ignored test — promote to Tier 2), context-full behavior, repeat
`--repeat` session-carryover loop (exists), kill/resume, empty prompt,
maximum prompt.

---

## 4. Week 3 — product surface

### 4.1 HTTP server mode — 3 days
`diffgemma-mps serve -m <model> [--ctx N] [--port]`:
- OpenAI-compatible `/v1/chat/completions` (stream + non-stream). The chat
  JSONL event protocol (`ChatEvent`) already models streaming with rewinds;
  map `Text{committed,..}` to SSE deltas (only emit committed-monotonic
  prefixes; diffusion rewinds stay internal — emit on commit, or expose a
  `x-diffusion-draft` extension event for UIs that want the shimmer).
- Single generation queue (GPU is single-tenant — hard rule); per-request
  seeds; cancellation (client disconnect → abort block loop between steps —
  needs a cancellation check in `generate_with_session`'s step loop, ~1 line
  per step).
- Reuse cross-turn KV: key sessions by conversation prefix (the
  `kv_valid_tokens` diff already handles arbitrary prefix reuse).

### 4.2 Sampler/feature parity with MLX — 1-2 days
From the mlx_vlm source diff (measured surface, not guesses):
- **`temperature` user knob**: expose categorical sampling (machinery exists
  behind `DGQ_DENOISER_ARGMAX=0`) as a per-request parameter; default stays
  argmax (their default is temperature 0 too).
- **`confidence-threshold` sampler** (their alternate accept rule,
  `diffusion_threshold=0.9`): implement in the GPU sampler as a variant of
  the accept kernel; compare convergence on the gate set. Ship only if
  neutral-or-better; otherwise document as unsupported.
- **`min/max_canvas_length`** — covered by §2.3.
- **NOT porting**: `diffusion_compile`, `static_cache`, unmasking display
  (we have inline dimmed streaming), `prefill_step_size` (ours is batched
  differently and faster per token).

### 4.3 Install & model pipeline UX — 1-2 days
- `diffgemma-mps fetch` (or documented `huggingface-cli` recipe) +
  `quantize --profile q4` one-liner; verify the 15 GiB disk headroom check
  and resume-safety of the converter.
- Warm-start: persist the Metal pipeline archive (exists —
  `DGQ_METAL_PIPELINE_CACHE`) by default; document first-run JIT.
- Homebrew formula / notarized binary decision (stretch).

### 4.4 Docs — 1 day
README quickstart (install → chat → serve → long-context flags), perf page
with the MLX methodology (matched canvas, temp 0, natural finish,
serialized runs), flag reference generated from `flags.rs` doc comments.

---

## 5. Week 4 — release engineering + the two big scope calls

### 5.1 Release hardening — 2 days
Version tagging, CHANGELOG from the commit narrative, binary size pass,
`--version` with model-format compatibility check (`.dgq` manifest version
gate — old q8-embed blobs still load, verify), final full-matrix run:
suite + gate 3 seeds + census + needle {6.5k, 33k, 105k} + MLX head-to-head.

### 5.2 TurboQuant / rotated-KV — IN PROGRESS (moved up from Week 4, 2026-07-07)
TurboQuant (arXiv 2504.19874) / QuaRot: an orthogonal rotation → near-uniform
coordinates → outlier-free low-bit quant. Our measured physics says low-bit
KV is a MEMORY lever only (issue-bound kernels), so this targets **262k
contexts** (f16 KV 5.3 GiB → ~1.4 GiB at q4) / smaller Macs. **Do not do
this for speed** — the disproof stands (E4: q8@100k only −6%).

**PREMISE CORRECTED 2026-07-07 (the old text below the line was wrong).**
The earlier claim "fold R into (W_q,W_k) and (W_v,W_o) offline, zero runtime
cost" is FALSE for K, and the V half needs rework too:
- **K cannot fold offline.** RoPE sits between W_k and the KV store
  (`qk_rope_kv.metal` applies RoPE, then writes K), and a Hadamard doesn't
  commute with position-dependent RoPE. K needs a RUNTIME Walsh-Hadamard on
  `head[]` AFTER RoPE, plus the same H on Q after its RoPE — `(Hq)·(Hk)=q·k`
  keeps scores exact. Cheap: O(hd·log hd), hd=256/512 both pow2.
- **The offline W_v fold is nearly worthless.** Full layers (5,11,17,23,29)
  have NO separate `v_proj` — V aliases the raw k_proj output — and those
  full layers store KV LINEARLY (sliding layers ring-cap at 2048), so they
  hold ~2.1 of the 2.4 GB at 105k. The memory that matters is exactly the
  layers with no W_v to fold.
- **Winning design = uniform runtime rotation, only W_o folded offline.** In
  `qk_rope_kv`, after RoPE apply H to q / k / v (v has no RoPE); store q4 K+V.
  Attention reads rotated q (arena) + rotated q4 K/V — scores and output
  `Σattn·(Hv)=H·o` come out right with NO rotation logic in the attention
  kernels (they need only q4 dequant). Unrotate V via W_o offline: fold Hᵀ
  into each q-head hd-block of W_o. Q rotation self-cancels in the QK dot.
- **Milestones**: M1 q4-KV format end-to-end, no rotation (mechanical, mirror
  the KV_Q8 lattice → KV_Q4; establishes the degraded baseline) → M2 runtime
  Hadamard (q/k/v) + offline W_o Hᵀ fold, prove rotated-q4 recovers quality →
  M3 non-expert bf16 weights → M4 rotated experts (§5.4, E9, v2+). Full
  corrected math + code map: memory `turboquant-rotated-kv`. WHT helper
  shipped (`src/kernels/sub/hadamard.rs`, 834cc1d). Gate as usual.

### 5.3 Vision (VLM) — decision, not implementation
The checkpoint ships a vision tower (mlx_vlm serves images; our config.rs
parses `vision_config` but the engine is decoder-only). Porting = SigLIP
encoder + image token splicing + mm masks — realistically 2+ weeks alone.
**Recommendation: v1 is text-only, stated in the README; vision is the v2
headline.** If overruled, this displaces §4.1-4.3 and the month becomes
perf-freeze + vision.

### 5.4 Rotated experts (E9) — far-future / v2+, experts LAST
The same Hadamard rotation applied to the MoE expert weights. Value is
**near-bf16 fidelity within the mandatory 4-bit budget** (bf16 experts don't
fit; the q6 experiment showed spending bits barely moves warts, so rotation
attacks CONDITIONING — a different axis — not size). Only start after the
KV rotation infra (E8) is proven bit-exact; build experts last; quality
change → full multi-seed gate + wart census + sign-off.

Two sub-cases, different cost — and the "gate/up is free" framing needs a
correction (verified against the code 2026-07-07):
- **gate/up.** Both read the pre_ff_ln_2-normed hidden (`moein`, one shared
  buffer). The rotation folds into the WEIGHTS offline (W_gate·Rᵀ, W_up·Rᵀ),
  but feeding the rotated input `R·moein` is NOT zero-cost: it needs either
  (a) ONE runtime WHT on `moein` per token — cheap, shared across all 128
  experts (moein is expert-branch-local; the router uses a *separate* norm
  `router_scale`, so it's not a shared fold) — or (b) a global
  residual-stream rotation (QuaRot computational invariance: rotate once at
  embed, unrotate at lm_head) which makes it truly free but is a much bigger
  change. So gate/up = "one cheap shared WHT," not "free." Covers 2/3 of
  expert params; the whole bet until proven insufficient.
- **down_proj.** `out = h·W_downᵀ`, `h = silu(gate)·up` — the nonlinearity
  blocks any offline fold; needs a runtime WHT on `h` (moe_ff 704 → pad
  1024) every denoise step. Contingent, hot-path.

Milestones: M1 gate/up-only (shared WHT + offline weight fold, gated vs a
bf16-expert reference — STOP here if it reaches near-bf16) → M2 down_proj
WHT (only if M1 short; measure with `DGQ_MEM_WATCH` + `bench-step-kernel`
A/B first, abort if it regresses the short-context, non-memory-bound case) →
M3 optional sub-q4 gate/up for memory (separate gate). Headroom evidence
(q6 = warts unchanged) shows expert-quant error isn't today's wart driver
but does NOT prove rotated-q4 recovers bf16 or that q3 is safe.

**Prior-art specifics + build-order correction (2026-07-07).** "TurboQuant
for weights" is NOT novel: TurboQuant (2504.19874) never quantizes weights
(KV + nearest-neighbor only), and its stage-2 QJL-residual trick only buys
*inner-product* unbiasedness — an attention concern, irrelevant to weights
(we want MSE). The real weight-domain twins are **PolarQuant** (2603.29078:
128-block L2-norm → Walsh–Hadamard → Lloyd–Max centroids fit to 𝒩(0,1),
calibration-free) and **QAM-W** (2605.26339: Hadamard + 2D codebook +
activation-aware scaling); lineage QuIP/QuIP#/QuaRot/SpinQuant. So E9 is a
*port*, not research — benchmark to beat is PolarQuant/QuIP#, not "is it new."
The one actionable result they surface: **PolarQuant measured Hadamard
rotation = ~98% of the quality gain (PPL 6.90→6.40 at Q5); optimal Lloyd–Max
centroids add only ~1%.** This matches our q6 finding (bits ≈ no wart
movement) — the lever is CONDITIONING, not the codebook. So **M1 should prove
the rotation with PLAIN absmax/uniform q4 first** and treat the good
quantizer as a near-non-lever; only revisit optimal 𝒩(0,1)/Beta centroids
for the M3 sub-q4 case, where the centroid share grows (TurboQuant's whole
point is centroids dominate at low bit — 98% is a Q5 number, not a q4/q3
one). PolarQuant's 128-elt intra-block matches our existing WHT helper block
size; all of it is calibration-free (no new data pipeline).

### 5.5 Slack (2-3 days)
History says every week here surfaces one unplanned wall (this week it was
the sampler-struct growth ripple). The slack is the plan.

---

## 6. Experiment playbook index (single table)

| # | Experiment | Hypothesis | Cost | Ship gate | Abort criterion |
|---|---|---|---|---|---|
| E1 | Weight-stationary expert GEMM | ~~33k prefill −20%~~ **DISPROVEN 2026-07-07**: BM=64 wash, BM=128 3.6x slower; expert GEMM at M=1024 is compute-bound (~6x margin over W bytes) | done | bit-identity held (KV byte-equal), gate 17/17 | hit: < 5% → machinery kept, default off |
| E2 | Fast prefill for short prompts | turn time neutral, engine retired | 1 d | multi-seed ≥ 47/51 + census | adherence regression |
| E3 | Canvas shrink at tail | MLX parity, small tail win | 1 d | multi-seed gate | quality regression |
| E4 | q8 KV @ 100k | ~~q8 wins where DRAM-bound~~ **CLOSED 2026-07-07**: −6% at 105k (issue-bound); but f16 @262k SWAPS (91%) → **q8 now AUTO-enables at long ctx** so long sessions stay resident | done | q8 = memory lever; auto above ~178k (scales with cap); gate 17/17 (short prompts stay f16) | hit: table closed |
| E5 | Fragment-tile attention | ≥ 1.5× at kv 32k | 1 d timebox | bench row, then full gate | < 1.5× on bench |
| E6 | seed-123 artifact root-cause | ~~stop-token trim~~ **CLOSED 2026-07-07**: NOT a forward bug — MLX-faithful (empties on the identical canvas too); intrinsic (prompt,canvas)→eos/ceremony attractor. **FIX SHIPPED default-on**: `DGQ_EMPTY_REPLY_RETRY=3` re-rolls the first block's canvas on a degenerate pos-0 (eos/stop/control token) | done | seed-123 answers 13→17; seeds 7/42 unchanged 17/17 (no-op on good replies); suite 714/0; deterministic per --seed | hit: fix landed |
| E7 | confidence-threshold sampler | parity feature, maybe faster stops | 1 d | gate neutral | worse convergence |
| E8 | Hadamard-rotated q4 KV (§5.2) | **IN PROGRESS 2026-07-07**: 262k KV 5.3→~1.4 GiB, quality-neutral. Premise corrected — K rotates at RUNTIME after RoPE (can't fold), V rotates at runtime too (full layers alias V/no W_v), only W_o folds offline | ~3 d | rotated-q4 passes multi-seed gate + needle 33k/100k exact + DGQ_MEM_WATCH < 90% @ 262k | rotated-q4 fails gate (rotation doesn't recover q4 quality) |
| E9 | Rotated experts (near-bf16 fidelity @ q4) (§5.4) | M1 gate/up (shared WHT + offline fold) recovers ~bf16 forward; M2 down_proj WHT only if needed | v2+ | full gate vs bf16-expert ref + census + sign-off | M1 no better than plain q4, or M2 WHT regresses short-context |
| E10 | Precision-decay KV (recent f16, aged rotated-q4) | segmented full-layer KV: f16 recent window + q4 aged bulk, demoted+rotated as tokens age past the window; better quality/memory than uniform q4 | v2+ | needle 262k exact + DGQ_MEM_WATCH under budget + gate | on 36 GB marginal over q8-auto (which already fits 262k) — value is 18-24 GB / >262k / long-range quality |
| E11 | ~~fp16 prefill activation stream~~ **BUILT + DISPROVEN 2026-07-10** (§6.1): K_ARENA_F16 machinery shipped opt-in (`DGQ_PREFILL_F16`; `DGQ_ARENA_F16_ALL` diagnostic; M0 ranges all ≤129, ~500× headroom; all-f16 session generates correctly) but the 4.2k doc probe STILL hallucinates — stream dtype is NOT the driver. Ring-uncap (`DGQ_KV_RING_UNCAPPED`) also disproven. Failure is STRUCTURAL: chunk-boundary f16 KV rounding compounds ~p/256 causal hops (8 hops exact / 17 degraded / 26 gone); engine-f32 and MLX are correct because they prefill full-M UNCHUNKED | done (disproof) | — | hit: see E14 |
| E12 | Engine prefill (§6.2) — **PARTIAL 2026-07-10** (7c621b7, b55128b): GPU ring-correct hydrate (13-32 ms, was O(n²) CPU + wrong past the ring wrap), hydrate-once chunked extend, resident extend (perf WASH — engine is KERNEL-bound, not sync-bound), bit-identical gqa masked-key clamp (−19% full prefill). Engine ≈55-78 ms/tok; remaining levers (attention rewrite = quality-gated, GEMM/MoE f32 ports) are days for 2-4× — POOR ROI vs E14; engine KV (linear f32) is also a memory wall past ~10k | done (bridge) | hydrate ring-exactness gate + finite-KV gate + fingerprint bit-identity | further engine surgery deprioritized for E14 |
| E14 | ~~Rolling f32 side KV for chunked prefill~~ **BUILT + DISPROVEN 2026-07-10** (§6.4): f32 side K/V for sliding layers, then ALL 30 layers, then + fp16 arena — 4.2k doc probe STILL hallucinates in every combination (1.7k stays grounded, machinery verified). THEN the sensitivity anchor: ENGINE prefill + 1% random relative noise on all 255M KV values answers 4.2k CORRECTLY (`DGQ_KV_NOISE` probe) → the model tolerates ~100× more KV noise than any precision delta in play → the fast path's failure is a REAL COMPUTATIONAL DEFECT, not accumulation physics. Onset bracketed 2.2k–3.2k. Machinery kept opt-in (`DGQ_PREFILL_KV_F32`) | done (disproof) | — | hit: see E15 |
| E15 | **ROOT CAUSE FOUND + FIXED 2026-07-10**: the fast prefill applied the DENOISE preamble's `RmsNormHidden` (no-scale RMSNorm of the embedded hidden) to the ENCODER pass — the reference encoder feeds embed·√H straight into layer 0 (engine == CPU oracle). RMSNorm is per-row scale-invariant through input_layernorm, so layer-0 K/V matched the engine exactly (rel 0.0025) and hid it for weeks; the RESIDUAL stream carried a per-token rescale → L1 KV 33% off at EVERY length → systematic MoE route flips → short prompts tolerated the warped trajectory, long ones collapsed. Caught by the e15_layer_kv_bisect (L1 0.33 uniform across bands even at 700 tokens = systematic, not length-dependent) after e15_causality_check exonerated chunk structure (bit-exact prefixes). Fix: drop RmsNormHidden from encode_prefill_chunk/_super (denoise keeps it — parity-validated there). POST-FIX: L1 KV 0.33→0.013 (25×); doc-QA grounded at 3.2k/4.2k/6.6k on the plain fast path; 5.9k tokens prefilled in 17 s (~2.9 ms/tok ≈ 20× the engine) | done | doc ladder grounded (done) + suite + multi-seed smoketest + census before raising the cap | hit |
| E13 | Comprehension probes in the gate (§6.3) | needle probes are blind to comprehension loss (stayed EXACT while 6.6k answers were hallucinated); a real-doc Q&A ladder catches the whole class | ~0.5 d | model-gated long-ctx tier in smoketest (planted-fact real-doc Q at 2k/4.2k/6.6k + classic needle); wired into nightly CI tier | n/a (pure validation) |

Every experiment observes the standing rules: bit-identical ships on
identity evidence; anything else needs multi-seed gate + census + explicit
sign-off; serialize all model-loading runs; check the disproof ledger first.

### 6.4 E14 playbook — rolling window KV — DISPROVEN (machinery kept; see E15)

The 2026-07-10 differential: every CHUNKED configuration fails the 4.2k doc
probe identically (bf16 arena, fp16 arena, ring capped/uncapped, every
kernel A/B) while both UNCHUNKED implementations (our f32 engine full-M,
MLX fp16 full-M) answer correctly. The only mechanism left standing:
each 256-token chunk reads the whole prefix's K/V through the f16
monolithic cache, so position p's context signal passes through ~p/256
compounding rounding hops (~8 at 2.2k = exact, ~17 at 4.2k = degraded,
~26 at 6.6k = hallucination — matches the smooth position-wise layer-5 KV
drift measured earlier; batched super-chunks don't help, attention/KV stay
256-granular inside them; fp16 arena can't help, K/V were already f16).

Design: during fast prefill only, sliding layers keep a SIDE f32 K/V ring
of window+chunk (~1280) positions; qk_rope_kv writes both (f32 side +
f16 monolithic), prefill attention reads the f32 side for sliding layers.
Memory ~1.3 GB constant in kv_len. Full layers stay on f16 monolithic
(their long-range edges provably survive f16 — needle exact at 105k; and
an f32 full-layer store would be O(kv) memory again). Denoise unchanged.
Validation = the E13 doc-probe ladder, then raise the cap.

### 6.1 E11 playbook — fp16 prefill activation stream (task #65) — DISPROVEN, machinery kept

**Failure mechanism (measured 2026-07-10).** The causal chunk forward
re-ingests every prior position's already-noisy K/V at each of 30 layers, so
per-op activation noise compounds across (position × layer). Attention-logit
noise from a bf16 Q row against f16 K is ≈ |q||k|·2⁻⁸ ≈ 0.1–0.3 nats — small
enough that sharp single-edge heads (needle retrieval) survive, large enough
that the many medium-weight edges carrying document comprehension decohere as
the window fills. Empirical quality-vs-length: exact ≤2.2k, degraded 4.2k,
fluent hallucination 6.6k. All discrete candidates were eliminated first: pad
ring clobber fixed (3285ebe, layer-0 KV now engine-clean), GEMM/MoE/rope
swaps byte-identical, scalar vs MMA attention both degrade (scalar less),
batching bit-identical, q8 off, rope precise-trig A/B byte-identical.

**M0 — ranges before conclusions** (check-ranges rule): `DGQ_TRACE_RANGES`
over a real 6.6k prefill; per-plane max|x| for hidden / q / k / v / attn-out
/ dense / gate-up. fp16 range 65504; Gemma-class residual OUTLIER CHANNELS
can reach 1e3–1e4 — if within range with ≥8× headroom, direct flip is safe;
else per-plane scaled-fp16 (store x·2⁻ˢ, fold 2ˢ into the consumer) for the
offending planes only, or the M2b trunk fallback.

**M1 — arena dtype function-constant, prefill-scoped.** Rebuild the deleted
K_ACT_F16 infra as `K_ARENA_F16`: `arena_load/arena_store` already funnel
every plane access, so the kernel-side change is one macro pair + FC plumb.
Compile SECOND variants only for pipelines dispatched under
`prefill_causal` (embed_gather, rms_norm, qk_rope_kv, attention*, GEMM/MoE
stages, residual) — denoise keeps bf16 (gate-validated; do not re-litigate).
Same bytes/bandwidth as bf16 → perf-neutral expectation; pipeline count grows
only for the prefill stage set.

**M2 fallbacks, in order.** (a) scaled-fp16 for outlier planes (M0 data);
(b) f32 residual TRUNK only (hidden plane f32, branches stay bf16 — halves
the FC surface; the accumulator lives in the trunk, and old DGQ_HIDDEN_F32
evidence showed branch storage dominates self-noise at SHORT ctx, so this
may under-deliver at long ctx — test only if fp16 fails); (c) stochastic
rounding on prefill arena stores (unbiased noise accumulates as √N instead
of N — cheap FC, marginal alone, useful as a 100k+ topper).

**Prior-disproof note.** K_ACT_F16 and DGQ_HIDDEN_F32 were built, measured,
DISPROVEN and deleted 2026-07-05 — for SHORT-context quality/speed claims.
This is a different claim (long-context comprehension stability) with new
evidence (engine-vs-fast behavioral split + MLX fp16 correctness at 6.6k),
which is exactly the re-litigation bar the disproof ledger sets.

**Ship sequence.** M0 ranges → M1 build → ladder+gate → raise
`DGQ_FAST_PREFILL_MAX` default (2048 → 32k probe → uncapped) → keep the cap
flag for A/B triage.

### 6.2 E12 playbook — engine prefill throughput (the correctness bridge)

Engine prefill ≈ 10 s/256-token chunk (~40 ms/token; 6.6k ≈ 4.5 min, 100k ≈
65 min). It was deliberately left unoptimized when fast prefill shipped. The
three levers, all bit-identity-preserving (pure perf, byte-equal output
gate): (1) **tunable-GEMM f32 variants** — the fragment kernels are
dtype-templated at the loader level; the engine's legacy f32 GEMMs are the
bulk of the 40 ms; (2) **chunk batching** — mirror the fast path's M=1024
super-chunk for the row-independent stages (embed/norm/GEMM/MoE), keeping
per-chunk causal attention; (3) **sync/readback trim** — the engine still
pays a ~1.3 s fixed sync per call plus per-stage readbacks the P2 arc
removed from the step path. Target ≥3×; even 2× makes >2k-token engine
fallbacks routine while E11 cooks.

### 6.3 E13 — long-context validation that can actually fail

Add a model-gated `longctx` smoketest tier: a real technical document
(fixtures: KERNELS/SPEC excerpts) with a planted fact, questioned at 2k /
4.2k / 6.6k prompt tokens, judged on the FACT (substring), plus the classic
synthetic needle as the retrieval control. The pair separates the two
capabilities the incident conflated: needle-EXACT + doc-WRONG is precisely
the fast-prefill failure signature and must page, not pass.

---

## 7. Risk register

1. **Single-machine evidence.** All numbers are one M3 Pro. Prod claims need
   at least one M-series variant re-run (M1/M2/M4 differ in SLC + core
   counts — the SLC-locality disproof might not hold on M1). Mitigation:
   recruit one external config in week 3; gate perf floors per-machine.
2. **36 GB floor.** 19 GiB weights + scratch. 18 GB Macs need q6/q4 embed +
   rotated-q4 KV (E8) — currently out of scope; state minimum RAM in README.
3. **Upstream drift.** mlx_vlm's mxfp4 production bug got fixed upstream? Our
   head-to-head cites their 4bit; refresh before publishing benchmarks.
4. **Gate is 17 prompts.** It has been sensitive so far, but v1 confidence
   wants a broader eval (e.g. 100-prompt adherence set, run weekly, not
   blocking). Cheap to add in Tier 3.
5. **The seed-123 class** may be a model property, not a bug — the retry
   guard (§3.1) is the containment plan either way.
# diffgemma-mps — engineering strategy for agents

Read this before writing kernels, tests, or chasing bugs. It is not a task list (that's `PLAN.md`) or the behavior contract (that's `SPEC.md`). It is **how to work on this codebase without repeating the mistakes that have already cost days.**

The project: a Rust + Metal inference engine for DiffusionGemma (Gemma-4 26B-A4B MoE, discrete block diffusion) on Apple Silicon. Model semantics: `ARCHITECTURE.md` (concept) + `SPEC.md` (implemented contract); authoritative numeric behavior is in the CPU reference (`src/kernels/cpu/`, `sample.rs`) and the manifest (`model.dgq.json`).

---

## 1. The one thing to internalize

**Every serious bug in this project has lived in a fused or accelerated GPU path that a slower reference path got right.** MPS-Q4 producing uniform logits, the SC GEMM transpose, the softmax grid collapse, the MoE route-garbage from a last-expert `n_tok` bug — all the same shape: the optimized path computes a different function than the reference, parity is green because the optimized path has no golden, and the symptom only appears downstream (layer 2+, entropy collapse, pad output) far from the cause.

The corollary that governs everything below: **an untested path is where the next bug is.** Speed without a per-path correctness gate is how bugs ship. So the strategy is not "go fast"; it is "make divergence impossible to introduce silently, then go fast."

---

## 2. Diagnostic discipline (how to chase a bug)

When output is wrong, follow this order. Do not skip to optimization or to rewriting a kernel you can't explain.

1. **Localize before you theorize.** Find the *smallest* unit that diverges from the reference. A 30-layer entropy collapse is not a bug location; it's a symptom. Bisect: which layer, which kernel, which stage within the kernel. The MoE hunt took ten turns because we theorized about kernel math for several rounds before dumping the one value (`x`/`tok` as the kernel actually read it) that localized it in one read.

2. **Same input, same weights, two paths.** The fastest localization is always: run the suspect GPU path and the CPU reference on *byte-identical* input and weights, compare cosine. cos > 0.999 = that stage is fine. cos ~0.5 = correlated-but-wrong (often a swap, a scale, or partial corruption). cos ~0 = orthogonal, reading the wrong data entirely. The cosine *magnitude* is a diagnostic, not just pass/fail — read it.

3. **Dump the actual bytes/values the kernel reads, not what you think it reads.** Repeatedly, the bug was "kernel reads a different thing than the reference" — wrong row, wrong tensor, wrong activation. A CPU transliteration of the *source* can match the reference while the *GPU execution* diverges, because the transliteration can't reproduce threadgroup semantics, arena bindings, or route resolution. When in doubt, write the kernel's actual inputs to a scratch buffer and read them back.

4. **Impossible numbers mean wrong N or wrong normalization.** Entropy > ln(N), Z=0, cos > 1, values at ~1e38 — these are never "the model is just bad." They are indexing, normalization, or precision-overflow bugs with a specific cause. Chase them as such. (Softmax entropy 3.84 > ln(14) turned out to be the reader confusing 14 prompt tokens with 270 attention keys — a presentation bug, but the "impossible" was the tell.)

5. **Two paths failing differently is a gift, not noise.** When the engine and monolithic paths diverge on the same input, the *difference* localizes the bug to the path-specific code. One exploding (inf), one inert (zero) meant two different bugs in the same conceptual spot — and finding one explained the other.

6. **Don't let a workaround end the investigation.** Swapping a broken fused kernel for a slow reference path unblocks convergence but leaves the latent bug in every *other* kernel that shares the flawed pattern. Root-cause it, then decide whether to fix or replace. (The fused-MoE bug, had it been swapped-not-fixed, would have left the same `dequant_q4_group` usage suspect everywhere it appears.)

7. **A contradiction is a second bug, not an anomaly.** "act went 0.015 → 1.0 but gpu_out is bit-identical" cannot happen in a correct pipeline. When a fix changes an intermediate but not the output, you are measuring two different code paths (probe vs production) or reading a stale buffer. Resolve contradictions; do not note-and-move-on.

8. **Reproduce a gap across inputs before attributing it to a bug.** A difference measured on one prompt/seed can be *chaos*, not a defect: in a sensitive iterative system (the denoise loop), a sub-1e-4 per-step difference can flip an entropy-boundary accept decision and cascade into a different — but equally valid — trajectory. The chunked-vs-slow SC "10-step regression" looked like a bug on "sky blue" (chunked 31 vs slow 22) and reversed on the next prompt (chunked 26 vs slow 30) — it was trajectory chaos, and per-step parity was cos 0.999994. Before chasing a single-input delta, check that it's *systematic* (same direction across prompts/seeds) and that per-unit parity is actually broken. Otherwise you'll "fix" noise.

9. **Localize the gap to the right axis, then change one thing.** A quality gap can live in any of {sampler, forward precision, a specific quantized tensor}. Bisect to the axis before rebuilding: the MLX convergence gap was localized to the *tail* (step ≥9) and to *embed* quant specifically by (a) matching steps 1–8, (b) ruling out experts with a memory-neutral nvfp4 rebuild that changed nothing, (c) measuring embed quant relerr directly (1.76× worse than MLX). Each step changed exactly one variable. A "rebuild with everything different" tells you nothing.

---

## 3. Measure before optimizing — always

Several regressions came from optimizing against an unmeasured or instrumented baseline:
- The SC GEMM "fast path" regressed to ~130 s/step because it reorganized a 262144-long contraction into a cache-hostile orientation.
- A "correct now" step measured 12 s — worse than the prior 4.8 s — because debug probes with full-buffer readbacks were still live, the slow decomposed MoE fallback was active, and native-Q4 dense (0.18 TFLOP/s) had replaced MPS dense (2.22 TFLOP/s).

**Rules:**
- **Get a clean measurement first.** Compile out probes/readbacks before timing anything. A readback is a GPU pipeline stall; instrumentation can dominate a step.
- **Attribute the time before reducing it.** Per-dispatch timing on one clean step. Do not guess which stage dominates; the guesses have been wrong.
- **Know the regime.** On M3-class at canvas=256 the step is **compute-bound on f16 matmul**, not bandwidth-bound. Weight read is ~70 ms; the GEMMs are the cost. This means: (a) smaller quant formats (fp4/fp8) buy ~nothing in speed here — they're dequant-to-f16 anyway, no native low-bit compute on any Mac; (b) MFU and dispatch/round-trip overhead are the levers, not bandwidth tricks.
- **Sequence optimizations to the bottleneck.** ICB, step-distillation, canvas-128 are *sub-second-step* optimizations. At multi-second steps the win is "stop doing the slow/temporary thing" (remove probes, use the fused kernel not the reference, re-enable MPS dense if its correctness bug is fixed). Don't reach for architecture when the cost is a left-on debug path.

---

## 4. Kernel variants: one body, compile-time specialization

Never fork a kernel into `k_foo`, `k_foo_fp8`, `k_foo_debug`, `k_foo_mps`. Forks drift, and drift is the bug source. Instead:

- **One source body per logical kernel.** Variant axes (dequant format q4/mxfp4/nvfp4/q8, accumulation dtype, dense backend, dump-depth) are **function constants** selected at pipeline-compile time. A "variant" is a tuple of constant values, not a file. The matmul loop exists once; a format bug cannot exist in one variant and not another.
- **Intermediate dumps are a compile-time mode of the production kernel,** not a separate probe kernel. A `DUMP_STAGE` function constant writes a chosen intermediate to a scratch buffer; production compiles it out (writes vanish). This prevents probe-vs-production drift — the exact failure behind the "act fixed, output unchanged" contradiction.
- **Fold hunt-time probe kernels back into the main bodies** behind the dump flag once a bug is found. Do not leave parallel probe kernels in the tree.

---

## 5. Testing: three tiers, push assertions down

The tests were slow *and* missed the bugs because they ran whole pipelines to exercise small logic. Fix both at once by pushing assertions to the smallest unit with the smallest fixture.

**Tier 1 — per-kernel unit tests. Synthetic, blob-free, milliseconds. Run on every save.**
One test per kernel against a CPU transliteration of *that kernel*, on a tiny hand-built fixture (e.g. 2 experts, M up to 100 routed rows, 64 hidden) — **never the 15 GiB blob**. The moment a "unit" test mmaps the real model it leaves the inner loop. This tier catches the entire class of bugs that took multi-turn hunts (route resolution, decode K-order, transpose, grid collapse, **grouped M>32 truncation**) in milliseconds. Every kernel has a permanent CPU twin; the twin is the oracle forever. Promote ad-hoc hunt transliterations (e.g. the MoE mirror) into permanent Tier-1 references. For grouped MoE: `rows_per_expert` must include **>32** for `gemm_block_grouped` (M striped in-kernel); `gemm_linear_grouped` tiles M via `grid.height` but still benefits from realistic counts for cross-kernel parity tests.

**Tier 2 — staged comparison. Real weights, reduced stages, flagged dump depth. Seconds. On demand / pre-push.**
2–3 layers, 1 step, intermediate-dump ON, comparing GPU vs CPU (vs MLX where available) at each stage. Catches *integration* bugs between correct kernels — wrong wiring, wrong buffer, stale read — that unit tests can't. Bounded (few layers/steps), not the full matrix.

**Tier 3 — end-to-end goldens. Full stack. Minutes. CI only, not the inner loop.**
Full 30L, real prompt, token-id match; the ship gates. These are regression gates, **not** debugging tools. Using Tier 3 to localize a bug is the slow path that caused the hunts.

**Transitivity is the speed win:** if monolithic `k_moe_grouped` and engine `f32_q4_linear_grouped` are each pinned to the *same* CPU oracle in Tier 1, "do the two engines agree" is automatic — you never need a slow engine-vs-engine end-to-end comparison to find a divergence. The 476 s engine-vs-monolithic trace becomes unnecessary.

**Variant matrix:** Tier-1 + Tier-2 run as a cross product of variant tuples ({format} × {accum} × {dense backend}) against the oracle, each cell with a characterized tolerance (fp4 looser than q4). A new variant is not "done" until it has a passing matrix row. `bench-matrix` mirrors this for perf. This makes "should we use fp4" a measured table cell, not a debate — and makes a silently-wrong fast variant impossible to ship.

---

## 6. Non-negotiable invariants (cheap checks that catch catastrophes)

Property assertions need no oracle and catch the "catastrophic but novel" class that no fixture anticipated. Wire these into the cheapest tier that can run them:

- **Finite:** no NaN/Inf in logits, activations, attention output, SC signal. (Caught: NaN-from-unzeroed-buffers, inf-from-bad-GEMM.)
- **Softmax rows sum to 1.0** over their actual support, every softmax kernel. (Would have caught the grid-collapse Z=0.)
- **Entropy ≤ ln(N)** over N keys/classes. (Would have flagged the impossible-entropy presentation bug.)
- **SC signal finite and non-zero on step ≥ 2.** (Would have caught both SC bugs — inf and inert.)
- **Determinism:** same seed → same tokens across runs on the deterministic path. (Caught the original MPS nondeterminism.)
- **Not all-pad / all-filler** on a converged block before early-stop fires. (The premature-commit quality bug.)
- **Offsets in `ulong`** for all blob addressing — the blob exceeds uint32; a uint intermediate truncates silently.
- **Quant K-order:** sequential `dequant_q4_group[m]` == `q4_weight_at(row, base+m)` in K-order, not just as a set (VERIFY-K — the blind spot that hid behind VERIFY-N).
- **Tile-bound dimensions:** For every kernel, list each compile-time tile (32, 64, 128, 65535 grid width) and ask whether a **data-dependent** dimension can exceed it. If yes, there must be either **grid tiling** (`gemm_block`: `m0 = tgid.y * 32`), an **in-kernel striping loop** (`gemm_block_grouped`: `m_base += 32`), or **`dispatch_1d_ranged`** for grid overflow. Tier-1 fixtures must exceed the worst tile (e.g. `rows_per_expert ≥ 33`). Audit mechanically (grep below).

**Three states of this bug class** (all historical "firing" instances are fixed; the taxonomy is what to check for in NEW kernels):
1. **Firing** — fixed array/tile vs data index overflows.
2. **Exact-fit fragile** — correct at current config, zero margin (add a `K_SHAPE_ASSERT`).
3. **Ranged correctly** — dimension can grow unbounded via loop or ranged dispatch.

**Mechanical audit:** grep for fixed-size threadgroup/private arrays (`float acc[N]`, `threadgroup float x[N]`) and check whether any index is runtime-bounded by a value that can exceed `N`. That catches `acc[8]`, `red[8]`, and siblings in one pass.

---

## 7. Authority & sources of truth

- **Numeric behavior:** the CPU reference (`sample.rs`, `kernels/cpu.rs`) is the oracle. When GPU and CPU disagree and parity-vs-HF historically passed, the CPU is right and the GPU path is the suspect.
- **Weight layout:** `model.dgq.json` manifest is authoritative for shapes, offsets, and tensor orientation. When a kernel's addressing assumption (stride, transpose, fused-vs-separate tensors) is in question, the manifest decides — not memory, not the comment. (The SC GEMM `A@W` vs `A@W^T` and the MoE fused-1408 questions were both settled by the manifest.)
- **Model semantics:** `SPEC.md` (esp. §2.1 forward-pass details: RoPE split-half pairing, proportional-RoPE full-head-dim denominator, temperature count-down, prefix-sum accept rule, QK-norm folding the attention scale, V aliased from raw k_proj on full layers). These are checkpoint-specific and several are counterintuitive — do not "correct" them from general Gemma knowledge without checking the reference.
- **Do not trust comments over code/manifest.** A stale comment ("entropy before temperature") has already misled. Verify against the authoritative source.

---

## 8. Working rules for agents

- **State assumptions as checks, not beliefs.** "The stride is K+2" → assert it against the manifest and a byte dump, don't assume it.
- **One bisecting measurement beats three rounds of theory.** When stuck, find the cheapest dump that splits the hypothesis space in half, and take it before proposing fixes.
- **Every bug fixed gets a regression test at the lowest tier that would have caught it.** This is how the test suite stops missing the bug class it keeps missing.
- **No path ships without a golden.** If you add a kernel variant, dense backend, or code path, it gets a matrix row before it's "done." Untested path = next bug.
- **Don't optimize against a dirty baseline.** Probes out, path confirmed, time attributed — then optimize.
- **Keep `PLAN.md` (open work) and `SPEC.md` (behavior contract) current.** Resolved-bug history lives in git + agent memory; settled kernel verdicts in `KERNELS.md`. Update SPEC.md whenever generation behavior changes.
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
