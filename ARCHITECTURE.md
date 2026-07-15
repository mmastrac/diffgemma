# diffgemma-mps — architecture & implemented contract

A low-dependency Rust + Metal inference engine for
[DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
(Gemma-4 26B-A4B MoE, discrete block diffusion) on Apple Silicon.

Part I is the conceptual model. Part II is the precise implemented
generation contract — every deliberate divergence from the MLX/HF reference
is in §9 with its evidence. Part III is the engineering design and its
rationale. **Negative Knowledge** at the end records approaches that were
built, measured, and disproven on this hardware — with the standing caveat
that any of them can be re-tested if the math/physics can be made to work.

How to work here: AGENTS.md. Open work: PLAN.md. History: git log.

Reference implementations compared against:
- **MLX**: `mlx_vlm/generate/diffusion.py` + `mlx_vlm/models/diffusion_gemma/`
  (in `python/.venv`) — the generation-loop reference.
- **HF**: `EntropyBoundSampler` / `LinearTemperatureScheduleLogitsProcessor`
  semantics (mirrored by our CPU sampler in `src/sample.rs`).

---

# Part I — the model, conceptually

## The core trade: memory bandwidth → compute

Standard autoregressive decode is memory-bandwidth-bound — each forward pass
produces one token while sweeping all weights, leaving tensor cores idle.
DiffusionGemma shifts the bottleneck to compute by generating and refining a
256-token canvas in parallel: each denoising forward operates on 256
positions simultaneously, dense enough to saturate the compute pipeline.

Corollary for this port: the win depends on being compute-bound. On
M3-class Apple Silicon at canvas 256 the denoise step **is** compute-bound
on f16 matmul (see Part III), so the trade holds — but levers that assume a
bandwidth-bound regime do not (see Negative Knowledge).

## Backbone

26B A4B Gemma-4 MoE: 25.2B total / 3.8B active parameters, 30 layers, 8 of
128 experts active (+ the always-on dense path), 262K vocabulary, up to
256K context, and a ~550M vision tower (not ported; text-only). Effective
per-forward cost ≈ a dense ~4B model — which is how it fits in ~19 GiB
quantized.

## Discrete diffusion, not continuous

Text diffusion operates over a discrete vocabulary: "noise" means uniformly
random token IDs, not Gaussian perturbation. A fully noised canvas is 256
random tokens; denoising iteratively re-predicts every position in parallel
until the sequence snaps into focus. Distinct from masked LMs: uncertain
positions get fresh random samples each step, not a stable `[MASK]`.

## Two attention phases, one set of weights

- **Causal prefill (encoder role)**: the prompt — and later each committed
  block — is processed with standard causal attention to build the KV cache.
- **Bidirectional denoise (decoder role)**: canvas positions attend causally
  to the KV cache and fully/symmetrically to each other.

## The entropy-bound sampler

After each denoising forward, per-position prediction entropy decides which
positions are kept; the rest are re-noised. Low entropy = the model has made
up its mind. Production config (generation_config.json): entropy bound 0.1
nats, max 48 steps, linear temperature decay 0.8 → 0.4, early stop on
stable-and-confident canvases. ~15–20 tokens commit per forward in practice.

## Block-autoregressive chaining

Once a canvas converges it is committed, causally prefilled to extend the KV
cache, and a fresh canvas begins. Left-to-right at block level; blocks are
immutable once committed.

## Quality tradeoffs

DiffusionGemma scores below standard Gemma-4 on most benchmarks (e.g. MMLU
Pro 77.6 vs 82.6). The exception is constraint-propagation tasks, where
bidirectional in-canvas attention lets constraints resolve symmetrically.

---

# Part II — the implemented generation contract

## 1. Model constants (google/diffusiongemma-26B-A4B-it)

| Constant | Value | Notes |
|---|---|---|
| layers | 30 | 25 sliding + 5 full (`FULL_LAYERS = [5,11,17,23,29]`) |
| hidden | 2816 | |
| vocab | 262144 | tied embeddings (lm_head = embed^T) |
| canvas | 256 | `canvas_length`; compile-time `CANVAS` in the monolith |
| attention | 16 Q heads; sliding: hd=256, 8 KV heads; full: hd=512, 2 KV heads | GQA |
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
  committed block) → KV cache. Sliding layers: query q attends
  [q−(window−1), q]. The encoder feeds `embed·√hidden` straight into layer 0
  — there is NO pre-layer norm on the encoder pass (the denoise preamble's
  no-scale RMSNorm is denoise-only; see Negative Knowledge).
- **Denoise (decoder role)**: canvas attends bidirectionally to itself and to
  the KV cache. On sliding layers the canvas sees only the **last window−1
  encoder positions** plus all canvas (matches MLX `_make_decoder_masks`;
  `DGQ_ATTN_WINDOW=0` restores unwindowed for A/B). Canvas position ids are
  `kv_len .. kv_len+255`. No timestep/noise-level embedding exists.

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
activation) prefill** (~3 ms/token, ~20× the engine, doc-QA-grounded to 13k+
and needle-exact to 105k). `DGQ_FAST_PREFILL_MAX` (default 0 = uncapped) can
reinstate an upper band routed back to the engine; `DGQ_FAST_PREFILL=1|0`
forces either path. Cross-turn delta reuse follows the same rule on the
DELTA length.

The ≤256 engine floor is a QUALITY choice, not just wall-clock: fast prefill
for all lengths regresses the multi-seed gate on short factual prompts
(empty-reply class; see Negative Knowledge).

Pad-row contract: the fast prefill pads its last chunk to CANVAS, but pad
rows MUST NOT write KV — `StepParams.kv_write_end` (the prompt end during
prefill, `u32::MAX` otherwise) suppresses their cache stores in
`qk_rope_kv`. On a ring, past-the-end positions are NOT dead: they wrap onto
`pos & kv_ring_mask` and clobber the oldest live window slots.

## 4. Denoise step (one forward = one canvas refinement)

Order of operations per step (monolithic GPU path, `interpret_step`):

1. **Preamble**
   - Step 1 (no prior prediction): **deterministic first-step SC seed** — run
     the SC MLP on the *initial canvas's own embedding* (ScPreNorm reads
     `hidden` post-EmbedGather). OURS, deliberate: SC=0 makes step-1 logits
     degenerate (cold-start empty reply), and leaving the previous
     generation's SC residual made reused sessions nondeterministic. MLX runs
     step 1 with no SC at all; our seed is prompt-independent + reproducible.
   - Steps ≥ 2: SC soft-embeddings from the PREVIOUS step's logits (§5), then
     SC MLP (q8 gate/up/down, GLU), added to the token embedding, RMS-normed.
2. **30 decoder layers**: QKV (stacked GEMM) → RoPE → attention (§2) →
   o_proj + residual → dense FFN (stacked gate|up, GLU, down) → MoE
   (router GEMM + top-8, block-sparse expert GEMMs, weighted scatter) →
   post-norms + residuals. All activations bf16 (Part III precision policy);
   MoE scratch f32.
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
  bit-matches the full prob-matrix GEMM.

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
     answer text settles ~10 steps before the argmax fully locks. `=0`
     disables. Known cost: minor tail warts on flat/creative canvases only
     (census 6/10 vs 2/10 intrinsic floor with it off); factual/doc-QA
     unaffected; signed off;
   - max_steps = 48.
7. **Final commit = the full-canvas argmax of the last executed step** — not
   the accepted canvas. Accepted tokens only ever shape intermediate
   conditioning.

Degenerate-first-block guard (`DGQ_EMPTY_REPLY_RETRY=3`, default on): a
first block whose position 0 is an eos/stop/control token re-rolls the
canvas (deterministic per --seed), shrinking 256→128→64 on successive
retries. The empty-reply attractor is intrinsic model behavior (MLX empties
on the identical canvas), not a forward bug.

### 6.1 Why there is no freeze

Steps 3–5+7 mean every canvas position can be revised at every step: the
model re-argmaxes accepted positions, drops them back to noise if their
entropy rises, and the final answer is whatever the last forward believes.
Any apparent "freezing" is emergent self-reinforcement (an accepted token
conditions its own next-step logits), not a mechanism. This is the MLX/HF
reference semantics; a hard freeze is PROVEN to cause flat-row warts (see
Negative Knowledge). `DGQ_FREEZE=1` restores the legacy freeze for A/B only.

## 7. Block-autoregressive chaining

The canvas is always the full 256 positions (see §9 for the MLX difference).
After a block converges: scan the committed argmax for a stop token
({1, 106, 50}); if found, trim there and end the turn. Otherwise append all
256 tokens, causally prefill them to extend the KV cache, re-seed the canvas
RNG, and denoise the next block. Blocks are immutable once committed.

Determinism: generation is deterministic at fixed seed. Under no-freeze +
argmax the trajectory is additionally near-seed-invariant on non-flat
prompts: once the accept set saturates, the iteration is a deterministic
argmax fixed-point and the initial canvas washes out.

## 8. Quantization (.dgq)

`classify_tensor` policy (src/dgq/layout.rs): MoE experts → q4 group-32
affine (q6 / nvfp4 under `--profile`); SC MLP → q8 per-row; embed,
attention, dense FFN, router, norms → Raw bf16 (lossless). Blobs above
Metal's single-buffer cap (20.25 GiB on M3 Pro/36 GB) are split at
`expert_split` into two no-copy MTLBuffer regions.

## 9. Deliberate divergences from the MLX/HF reference

| # | Divergence | Direction | Why / evidence |
|---|---|---|---|
| 1 | Committed token = argmax (not HF's categorical draw) | matches MLX default, diverges from HF | census-clean, fewer steps; signed off |
| 2 | First-step SC seed (SC MLP on initial-canvas embedding) | ours only (MLX: no SC at step 1) | fixes cold-start empty reply + session-carryover nondeterminism |
| 3 | Plateau early-stop backstop | ours only | stuck-but-stable canvases; part of gate baseline |
| 4 | Pad-aware stop gate | ours only | never "converge" on an all-pad forward; inert on real prompts |
| 5 | Fixed 256-token canvas every block | MLX shrinks to max(remaining, 64) | structural simplification (compile-time CANVAS); only matters near max_tokens |
| 6 | Sparse SC soft-embed (survivor gather) | ours only (approximate) | ~16%/step; output-level equivalent to MLX-4bit; exact chunked path is one flag away |
| 7 | Length-heuristic prefill (engine f32 ≤ 256 tokens) | ours only | §3 — quality + wall-clock; MLX prefills one way |
| 8 | Canvas init RNG (seeded LCG) | different RNG than mx.random | noise is noise; parity tooling pins exact canvases via `initial_canvas_ids` |
| 9 | Entropy accept + stop thresholds in nats | same as MLX (natural log) | noted because prose sometimes says "bits" — the code is nats everywhere |
| 10 | Entropy-only early stop (mean H < 0.05, §6) | ours only | tail steps only micro-flip near-ties; signed off, census cost = creative-tail only |
| 11 | Degenerate-first-block re-roll + shrink-on-retry | ours only | empty-reply attractor is intrinsic; deterministic per seed; no-op on good replies |

## 10. Validation harnesses

- **Smoketest gate** (`smoketest`, fixtures/smoketest/prompts.json): 12
  adherence + 5 convergence prompts; spec seed 7. Multi-seed aggregate
  {7,42,123} judges trajectory-reshuffling changes.
- **Long-context ladder** (`smoketest --longctx`): real-document Q&A at
  3.3k/8.2k/13.3k/20.6k (frozen fixture) + needle control. Needle-EXACT +
  doc-WRONG is the fast-prefill failure signature and must page, not pass.
- **Golden byte-identity** (`golden` / `golden --bless`): 8-case path matrix
  through the production session; token ids + KV hash. The refactor gate.
- **Wart census**: 10-seed greentext (flat/creative canvas), the sensitive
  sampler-semantics probe.
- **Step-parity oracle**: engine (f32) vs monolith per-step logits (mean|Δ|,
  softcap-aware). The engine is validation-only.
- **Kernel oracle matrix**: per-kernel CPU-mirror tests (`cargo test`).
- **MLX parity tooling** (python/scripts): layer-cos, denoise traces,
  generation comparison. ALWAYS prompt-match layer-cos comparisons.

---

# Part III — engineering design

## Runtime shape

- `src/shaders/<group>/<kernel>/` — every kernel's Rust wrapper, Metal
  source, CPU oracle, and manifest `SPEC` colocated (see AGENTS.md §4).
- `src/metal/` — the runtime: device/queue, pipeline binary-archive cache
  (keyed on the whole shader-tree hash), buffer arenas, KV cache, the
  production step dispatch (`step_kernel.rs`), and the f32 validation engine.
- One production step path (the "monolith"); the f32 engine survives for
  ≤256-token prefill and as the step-parity oracle.

## Standing design decisions

- **Checkpoint-orientation quant layout** (row-major `[out,in]`, not
  pre-transposed): streaming converter, shared CPU/GPU layout, parity indices
  match the checkpoint.
- **Mixed precision: bf16 where it's cheap, q4 only for the bulk.**
  Attention, dense FFN, and `embed_tokens` are bf16 (lossless into the half
  GEMM tiles; q8 embed flattened hard-tail logits and stalled convergence).
  Only the MoE experts (the bulk bytes) are 4-bit — a hard memory
  constraint (bf16 experts ≈ 4× the bytes).
- **Router stays bf16 weights / f32 logits** — routing is control flow;
  near-boundary noise flips experts discretely.
- **Two KV layouts coexist** — monolithic (f16, unified per-layer region,
  ring-buffer sliding windows) is canonical; the engine keeps its legacy
  f32 KV. Unify only if the engine ever stops being validation-only.
- **Tunable GEMM is the sole production GEMM path** (dense / stacked /
  block-sparse MoE × raw/q8/q4/q6/nvfp4 via function constants); the legacy
  block-GEMM family exists only as bit-exact validation oracles
  (`src/shaders/oracle/`).

## Precision policy

- **Weights**: bf16 lossless (attention / dense FFN / embed / router), q4
  group-32 experts, q8 SC-MLP.
- **Activation planes**: bf16 (`arena_load/store`) — denoise AND prefill.
  (The long-prompt comprehension collapse once blamed on bf16 accumulation
  was a computational defect — a denoise-only norm running on the encoder
  pass — not precision; see Negative Knowledge.)
- **f16 where values are provably bounded [0,1]**: `sc_prob_cols`, attention
  P tiles. Nothing else qualifies.
- **Always-bf16**: logits (FC29 forced), RouteScratch weights.
- **KV cache: f16.** Range-checked (max|KV| ≈ 22 vs f16 max 65504); f16's 10
  mantissa bits beat bf16's 7 across the live range, and f16 lets the MMA
  attention kernels `simdgroup_load` K/V straight from device memory — the
  long-context attention enabler. q8 KV (group-32) auto-enables at very long
  context as a MEMORY lever (`kv_q8(max_seq)`, `DGQ_KV_Q8` overrides).
- **Always-f32**: moeout plane, rowstats planes, MoE grouped scratch, router
  logits.
- The real hazard is producer/consumer dtype mismatch, not dtype choice
  (AGENTS.md §6).

## Performance regime (M3 Pro, measured)

- **Denoise GEMMs: compute-bound at the MPS matmul wall** (~3.7-3.9 TF/s
  tunable dense = MPS 3.65-3.69; MLX steel qmm 3.96-4.15 — the residual
  ~10-15% is their software-pipelined loader, worth ≤2-3% end-to-end:
  non-lever). Smaller quant formats buy ~nothing in speed — everything
  dequantizes to f16 for the MMA units; there is no native low-bit compute
  on any Mac.
- **Attention: instruction-issue-bound at SLC service rates** (~171 GB/s
  effective > 150 GB/s DRAM at 32k; ~580 GB/s effective at 105k) — the
  lockstep threadgroup sweep is already SLC-served, so byte-cutting KV does
  not speed it up.
- **Expert GEMM at prefill M=1024: compute-bound** with ~6× margin over
  weight bytes even with per-block re-reads.
- **Occupancy is the recurring attention wall**: threadgroup-memory footprint
  gates residency; register tiles beat tgmem tiles.
- Wall-clock: beats MLX-4bit (their fastest config) on short/medium chat;
  needle-exact to 105k (KV 2.4 GiB); denoise convergence parity-class
  (~1.15× steps vs MLX's best config, matched-canvas multi-seed).
- Memory: single 36 GB machine is NOT swap-bound to 105k; the cliff is
  ~262k f16 KV (q8-auto covers it) or running beside another model.

---

# Negative Knowledge

Approaches that were **built, measured, and disproven on this hardware**
(M3 Pro 36 GB unless noted). Each entry records the physics that blocked
it. Standing caveat: **any of these can be re-tested if the math/physics can
be made to work** — new hardware, new evidence, or a corrected premise is
exactly the re-litigation bar. Check here (and agent memory) before
planning perf or quality work; the machinery for several survives behind
opt-in flags for A/B.

**Speed levers, disproven by regime:**
- **int8/int4 dot-product MoE expert GEMM (E18 redirect)** — built a real
  int8-accumulated block-sparse MoE expert GEMM (`gemm_int_sparse`) and benched
  it head-to-head vs the production q4→half-MMA path at real MoE shapes
  (128 experts, gate_up 1408×2816 + down 2816×704, rpe 64/16). Correctness
  gated FIRST (cos=1.000000 vs CPU int8 oracle at every tile). RESULT: int8 =
  ~0.43 TFLOP/s vs production = ~3.78 TFLOP/s → **~9× SLOWER**. The synthetic
  `int_mma_probe` microbench reported int8 dot ≈ f32 FMA ≈ 14× half-MMA in
  per-instruction GMAC/s and could NOT settle this — the trap is that
  per-instruction throughput is misleading for GEMM shapes: the half-simdgroup
  -MMA does 512 MAC/simdgroup-inst (16 MAC/lane) while the int8 char4 dot does
  4 MAC/lane/inst, AND the int8 path pays 32 sequential dependent int32 adds
  per fragment element (no MMA register-array parallelism). On M3's
  SIMD-ALU-emulated matrix unit the half-MMA's per-instruction width dominates
  the int8 dot's per-lane scalar chain by ~9×, exactly the measured gap. The
  MLX-dequants-to-half "tell" was PROOF, not oversight. Tile sweep (BM=32/64,
  BN=64/128) did not help — the loss is in the inner accumulation, not tiling.
  int4 cannot close a 9× compute gap (and has no dot instruction on M3 — would
  unpack to int8). `gemm_int_sparse` + `bench-gemm --shapes sparse` kept as
  documented negative; not wired to production. Lesson: a per-instruction
  GMAC/s microbench cannot settle a GEMM-shape question — only a real prototype
  on real shapes can.
- **Flash-decode / sequential KV blocking** — attention is issue-bound at
  SLC service; the lockstep sweep already gets SLC locality. Restructuring
  the sweep bought nothing (`DGQ_ATTN_KV_BLOCK`, default off).
- **Low-bit KV for speed** (q8, and TurboQuant-class by the same physics) —
  +9% prefill / +54% denoise at 33k; only −6% even at 105k. Kernels are
  issue-bound, not byte-starved. q8 KV survives as the MEMORY lever
  (auto-on at long context).
- **int4/int8 KV cache quantization at long context (Gemma attn_scale)** —
  disproven for Gemma architectures *by physics, not just measurement*: Gemma's
  undampened attention scale (attn_scale = 1.0, vs Llama's 1/√d ≈ 0.088) means
  a per-element affine error of ε is amplified 25-100× *per layer* through the
  softmax. Across 60+ layers this compounds into incoherent output. Open-TQ-Metal
  (2026-04, ensue.dev/blog/introducing-open-tq-metal) independently confirmed:
  int4 group-32 KV works on Llama (cosine 0.998, identical output) but on Gemma 4
  degrades past ~950 tokens (int4) / ~1k tokens (int8). PolarQuant/QJL angular
  methods produce outright gibberish on Gemma (cosine 0.621). This is why the
  project uses f16 main cache + f32 side ring (E14), not int4 KV. The attn_scale
  finding explains the ceiling: no per-group scheme can recover the 25-100×
  amplification at α=1.0. Re-test only if the architecture changes (dampened
  scale) or for sub-1k-token sessions (where it's a memory lever, not speed).
- **Weight-stationary expert GEMM** (taller prefill MoE blocks) — the expert
  GEMM at M=1024 is compute-bound ~6× over weight bytes; BM=64 wash, BM=128
  3.6× slower (register spill). Byte-cutting can't win; the lever is GEMM
  TF/s (fragment-tile class).
- **MoE tiny-M tiling** (M-tile classes, adaptive-M) — per-TG cost is
  fixed/dequant-dominated; an indirect dispatch costs ~0.12 ms even at zero
  height (Metal hazard serialization). Rule: never split a hot GEMM into
  multiple same-encoder dispatches.
- **Partial forward** (denoise only active rows) — frozen-K/V staleness
  breaks convergence even at N=2; MoE is bandwidth-bound not row-bound.
- **Dispatch/sync micro-optimization** — encode is ~0.2 ms/step; non-lever.
  Same for MLX-style `load_unsafe`/bounds-check tricks and ICB record/replay.
- **Steel-loader GEMM port (software-pipelined double-buffering)** — DISPROVEN
  2026-07-14 (the parked ≤2-3% ROI estimate was optimistic). Built the real
  prototype: `gemm_tunable_db` (sibling entry in `gemm_tunable.metal`) doubles
  the tgmem tiles `Xs[2][BM][PAD]` + `Ws[2][BN][PAD]`, runs a prologue load
  tile 0, then in the K-loop overlaps the device→tgmem load of tile N+1 with
  the MMA of tile N (one barrier/K-tile vs two in the single-buffered kernel).
  Bit-exact vs the single-buffered `gemm_tunable` (Tier-1 test green; the
  K-accumulation chain, dequant, and store rounding are unchanged — only the
  tgmem buffering schedule differs). Benched head-to-head at the prefill-
  relevant dense shapes (256×2816×2816 + 1024×2816×2816, production tile
  64×64, q4 weights): double-buf = **3.377 / 3.566 TF/s** vs single-buf =
  **3.611 / 3.919 TF/s** → **0.93× / 0.91× (7-9% SLOWER)**. PHYSICS: the
  device→tgmem load is already fully hidden behind the ~6× compute margin
  (compute >> load, so single-buffered load already overlaps with the next
  K-tile's compute via the GPU's natural instruction issue). The extra
  barrier sync + doubled tgmem footprint (which hurts tile occupancy / SLC
  pressure) costs more than the explicit load/MMA overlap gains. There is no
  async-copy engine on Apple GPU — the "overlap" is just re-issuing the load
  instructions before the MMA, which the single-buffered version already does
  implicitly because the GPU issues load/store and matrix-unit instructions
  concurrently. This is the same shape as the int8 dot disproof: the physics
  (compute-bound regime, load already hidden) predicted the result, and a real
  prototype on real shapes confirmed it. `gemm_tunable_db` + the `bench-gemm
  --shapes db` harness kept as a documented negative; not wired to production.
  Lesson: when compute >> load (the 6× margin regime), software-pipelined
  double-buffering is a regression, not a lever — the doubled tgmem footprint
  + extra sync costs more than the already-hidden load overlap returns.
- **E5 QK-ILP2 chain-split (full-layer attention)** — PENDING BO 2026-07-14
  (a single-axis A/B was INCONCLUSIVE — it didn't exercise the path). The
  production `attention_mma_full` QK dot runs a single 32-deep serial
  `simdgroup_multiply_accumulate` chain (one accumulator, NCH_H=32 chunks).
  The ILP2 prototype (FC31 `DGQ_ATTN_MMA_FULL_QK_ILP2`, opt-in) splits it
  into two independent 16-deep chains (even/odd chunks) so the GPU can issue
  both MMAs concurrently, halving the QK serial-dependency depth. Each chain
  stores to its own tgmem slot (`st` + `st_ilp`); the cross-half softmax
  sums all four partials. Built as a function-constant variant of the
  production kernel (one body, FC31-selected) — non-bit-identical (different
  FP-associativity), parity checked via step-probe (identical max_abs at
  every stage). A warm 3-trial A/B at kv=15000 with E17 ON (default) showed
  no measurable difference (3365 vs 3370 ms) — **but this was an invalid
  test**: with E17 default-on, full layers route to `attention_gemm`, so
  `attention_mma_full` (and ILP2) is inert for the dominant attention cost.
  The earlier E5 sweep's ~5% kernel / ~3% prefill was measured pre-E17, on
  the path ILP2 actually runs. PHYSICS (unresolved): ILP2 helps when the
  bottleneck is *dependency latency* in the MMA pipeline; the full-attn
  layers may be instruction-issue-bound (per the perf-regime note + the
  existing "Flash-decode / sequential KV blocking" negative), in which case
  splitting one chain into two independent chains buys nothing (the same
  total MMAs compete for the same issue slots). But that regime call was
  also made on the E17 path, not on mma_full+ILP2. **The lever is added as
  a categorical axis (paired with `gemm_attn` on/off) in the holistic
  prefill BO** (`tune_prefill_attn.py --proxy`) so TPE can test the joint
  {E17, ILP2, tiles} space — single-axis A/B at default settings cannot see
  levers that only activate in combination or in the off-default path.
  Kept behind default-OFF FC31; not wired to production pending the BO
  result. Lesson: a single-axis A/B that doesn't exercise the modified path
  is not a disproof — verify the dispatch path before declaring a lever
  inert, and prefer the joint BO for levers that interact with path
  selection.

**Quality levers, disproven by measurement:**
- **Hard freeze of accepted positions** — WAS the flat-row wart driver
  (census 4/10 → 0/10 on removal); reference semantics have no freeze.
- **Expert quantization as wart driver** — q6 experts (2% err vs q4's 7.9%)
  changed nothing; quantization is exonerated. Close quality gaps
  memory-neutrally.
- **q8 embed** — flattened hard-tail logits, stalled convergence; embed
  stays bf16.
- **First-step eos-guard (token suppression)** — recovers a ceremony token,
  not the answer; token-suppression is a dead end for the empty-reply class
  (the shipped fix is the canvas re-roll).
- **Fast prefill for short prompts** — regresses the multi-seed gate
  (empty-reply class spreads); the ≤256 engine floor is a quality mitigation.
- **f32 hidden / f16 arena for short-context quality** — self-noise is bf16
  branch storage, not the trunk; both null.
- **fp16 prefill stream (E11) & f32 side-KV (E14) for the long-context
  collapse** — every precision variant still failed while a 1%-KV-noise
  engine run stayed correct; the cause was a computational defect (spurious
  encoder norm), since fixed. Precision-accumulation theories for this class
  are disproven; machinery kept opt-in.
- **Un-RoPE / rotated KV for q4** — pre-RoPE K quantizes identically
  (0.99-1.00× all formats); group-32 scaling and rotation are SUBSTITUTES,
  and nothing bridges the ~17× q4→q8 resolution gap. Feasible (QK-norms are
  exact scalars, the fold is sound) but valueless; revive only if q4-KV
  becomes necessary (18-24 GB Macs).
- **Global weight rotation** — residual-stream norms are genuinely
  per-channel (std/mean 0.4-2.1); no global orthogonal fold exists. Per-op
  rotation is modest+costly; the practical lever is plain q6.

**Validation lessons (encoded in the gates):**
- Needle probes are blind to comprehension loss — retrieval rides a few
  sharp attention edges. Long-context claims use the doc-QA ladder.
- Single-seed step comparisons are meaningless for trajectory-reshuffling
  changes — use matched-canvas multi-seed.
- A single-prompt delta can be trajectory chaos, not a bug — check that it
  is systematic before chasing it.
