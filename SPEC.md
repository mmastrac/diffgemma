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

`should_fast_prefill(prompt_len)` (src/flags.rs): prompts ≤ 256 tokens use the
**f32 engine** prefill; longer prompts use the **fast quantized (bf16
activation) prefill** (~10× faster where it matters). `DGQ_FAST_PREFILL=1|0`
forces either for all lengths.

History of the heuristic (see memory/KERNELS.md for the full data): fast
prefill's bf16 activations perturb outlier KV channels enough to flip MoE
expert routing on borderline tokens (a discrete cliff — no precision lever
fixes it; f16 just shuffles which prompts flip). Under the legacy freezing
sampler this produced LOUD degenerate outputs (empty / thought-mode) on 2/16
short prompts. **Since the no-freeze sampler (2026-07-05) the degenerate class
is gone** — forced fast prefill scores 17/17 adherence at seeds 7/42 — the
perturbation now costs extra denoise steps instead (e.g. capital_france 10 vs
3). The heuristic is kept purely on wall-clock: engine prefill of a short
prompt (~0.85 s) is cheaper than the extra steps fast prefill induces.

Known inert bug: fast prefill pads its last chunk to CANVAS and writes garbage
KV at [n..256); provably overwritten by the first denoise step before any
read. Documented, not load-bearing.

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
