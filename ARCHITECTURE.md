# diffgemma — architecture & implemented contract

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

**Consequence: joint constraints across positions are not representable.**
Each step samples every position from its OWN marginal. Where correctness
depends on a relationship BETWEEN adjacent positions, nothing in the update
enforces it — each position independently picks what is locally plausible.
The visible symptom is a **convention blend**: when two valid surface forms
compete, the canvas can settle on a mixture that is valid under neither.
Observed in generated shell code, in both directions:

    emitted            valid form A        valid form B
    if [ $#" -lt 1 ]   if [ $# -lt 1 ]     if [ "$#" -ne 1 ]
    == *$SEARCH"*      == *$SEARCH*        == *"$SEARCH"*

Both are single-token surface defects inside otherwise-correct programs, and
both commit at p_max ~1.0 — after the blend each position is individually
plausible; only the PAIR is wrong. A sequential decoder cannot produce this
class, because it conditions on its own previous emission. No confidence
threshold detects it either (see the trim tiers in §6 and Negative
Knowledge): the tokens are not uncertain, they are jointly inconsistent.

The blending mechanism is inferred from step traces rather than proven, and
its RATE is unquantified; what is measured is that the model does not
coordinate such a pair (37.5% of doubled-delimiter states resolve by both
positions correcting at once, n=16, rejecting a coordinated null at
p=0.003).

## The loop in one paragraph

One set of weights serves two attention phases: **causal prefill** builds the
KV cache from the prompt (and from each committed block), then
**bidirectional denoise** refines a canvas that attends causally to the cache
and symmetrically to itself. After each denoising forward, per-position
prediction entropy decides which positions are kept and which are re-noised —
low entropy means the model has made up its mind, and ~15–20 tokens commit
per forward. A converged canvas is committed, causally prefilled to extend the
cache, and a fresh canvas begins; blocks are immutable once committed. Part II
§2, §6 and §7 give the exact contract for each of those three pieces.

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

`should_fast_prefill(prompt_len)` (src/flags/): prompts ≤ 256 tokens use
the **f32 engine** prefill; longer prompts use the **fast quantized (bf16
activation) prefill** (~3 ms/token, ~20× the engine, doc-QA-grounded to 13k+
and needle-exact to 121k). `DGQ_FAST_PREFILL_MAX` (default 0 = uncapped) can
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

**Full-layer attention runs TOP-K SPARSE by default with kv-adaptive k on
BOTH the prefill (E20 + `DGQ_ATTN_TOPK_DYN`) and denoise
(`DGQ_ATTN_TOPK_DECODE`) paths:** per query row,
keep the k = clamp(t_total/128, 64, 512) highest-scoring keys (exact top-k
by f32 score via 4-level radix on a u16 key plane; deterministic
count/prefix-sum emission), softmax over that set, gathered-V PV. Prefill
dispatches causal=1; denoise dispatches the same 3-kernel pipeline causal=0
(bidirectional canvas), reading the main f16 cache only (never the E14 f32
side ring, which denoise-step canvas writes do not maintain). The k fraction
(~0.8% of context) exists because attention mass DIFFUSES with depth —
fixed k=64 measurably drops the deepest needle at 121k while dyn matches
dense 4/4. NOT bit-identical to dense (`DGQ_ATTN_TOPK=0` /
`DGQ_ATTN_TOPK_DECODE=0` restore E17 dense prefill / mma_full dense
denoise; golden blessed on the sparse defaults). Perf: prefill −16% @30k,
−28% @100k → within ~2.5% of MLX-4bit chunked; decode 4.42→1.84 s/step
@100k (isolated 3.1× — mma_full is issue-bound at the denoise shape,
705 ms/layer @100k vs 229 top-k / 352 dense-E17). Sliding layers are
untouched (E18 flash / mma2, window-bounded).

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

Production paths (`src/flags/`):
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
`expert_split` into two no-copy MTLBuffer regions. Raw tensors are written
byte-for-byte VERBATIM from the source safetensors (no transform) — the
property layered packs (below) depend on. Every other kind is a genuine
transform (quantized) and can only ever live in a pack's own blob.

`--profile` picks one of five fixed policies (q4/q5/q6/nvfp4/nvfp4x);
`quantize --set class=format` (§8.2) overrides `classify_tensor`'s output
per tensor CLASS on top of a chosen profile, for mixed-precision experiment
arms that don't fit a canned profile.

**Every tensor's canonical offset is 64-byte aligned, unconditionally, in
every pack** — a hard GPU-correctness requirement, not a packing nicety:
`gemm_rowk.metal` and structurally identical kernels reinterpret
`blob + w_off` as `device const ushort *` (feeding `half4`/`simdgroup_load`
tiles) and index it, which is only well-defined for a sufficiently aligned
`w_off`. A writer variant that relaxed this produced a pack with
byte-IDENTICAL tensor content that generated garbage from decoder layer 1
onward: a misaligned typed read is silently wrong while every byte-granular
check stays green, because those are untyped reads. Golden caught it; nothing
byte-level can. `assert_tensor_offset_alignment` (`src/metal/dgq_gpu.rs`)
enforces it at `DgqGpuBlob::from_store` for every pack and refuses to load
before any GPU buffer is created.

### 8.2 Custom quantization classes (`--set`)

`quantize --profile BASE --set class=format [--set class=format ...]`
layers per-class format overrides on top of a base profile
(`classify_tensor_custom`, src/dgq/layout.rs). Zero overrides is IDENTICAL
to the base profile — `classify_tensor_custom` only ever consults the
override map for a tensor `tensor_class` maps to a class, and every other
tensor (including every locked one) falls through to `classify_tensor`
unchanged.

**Classes and their legal formats** (`TensorClass`; enforced at CLI parse
time by `supported_formats`, before the source model is opened). A class only
accepts formats the step-kernel runtime has a compiled kernel for:

| Class | Tensors | Formats | Why not the others |
|---|---|---|---|
| `experts` (also `experts.gate_up` / `experts.down`) | `.experts.` 3D stacks | q4, q6, nvfp4 | raw/q8 would silently fall back to the scalar per-expert kernel — a probe/oracle surface, never a production dispatch path |
| `attn` | decoder `self_attn` q/k/v/o_proj | raw, q8, q4, nvfp4 | q6 dequant is wired into the block-SPARSE kernel body only; the dense-linear kernel has no q6 branch |
| `dense` | decoder dense-FFN gate/up/down | raw, q8, q4, nvfp4 | as `attn` |
| `sc` | the 3 self-conditioning MLP tensors | raw, q8, q4, nvfp4 | as `attn` |
| `vision` | vision-tower linears + `embed_vision.embedding_projection` | all five | inert: no forward path references the vision tower, so any codec is a pure disk-size lever |

`experts.gate_up` and `experts.down` must resolve to the SAME format if both
are set — the batched grouped-GEMM dispatch and blob-byte math key on one
shared `StepBlockProfile`.

**Locked classes** — not knobs, fatal if targeted, reason printed:
`embed`/`embed_tokens` (tied to the lm_head and SC soft-embed; bf16 keeps
tail-token logits sharp), `router` (routing is precision-sensitive, tensors
are tiny), `norms`/`layer_scalar` (same). `tensor_class` never maps a locked
tensor to a `TensorClass` at all, so no override can reach one by accident.

**Validation before any bytes are written**: dimension constraints per
resolved (tensor, format) — q4/q6 need K a multiple of 32, nvfp4 a multiple
of 16, q8 needs rank 2 (`validate_format_dims`). An invalid combo names the
offending tensor and fails before the output dir exists.

**Manifest**: `DgqManifest::custom_classes` records the resolved overrides
alongside the unchanged `profile` base, purely descriptively — the loader
never consults it, because every tensor's `kind` is self-sufficient for
dispatch. The manifest version bumps to `DGQ_VERSION_NVFP4` whenever an nvfp4
tensor is actually PRESENT (checked over resolved kinds, not `profile`), so a
pre-nvfp4 binary refuses rather than misreads.

**Runtime dispatch reads the manifest, never `profile`**: the step-kernel
runtime (`src/metal/step_kernel/build.rs`) derives THREE independent formats
from the tensors it actually finds — `attn_format` / `dense_format`
(`DenseWeightFormat`: bf16 / q8 / block(q4 or nvfp4), from the q_proj /
mlp.gate_proj kind respectively) and the expert `StepBlockProfile` (from the
gate_up kind, asserted equal to down's). Deriving them per tensor rather than
from one shared `QuantProfile` is what makes a mixed pack dispatchable at all:
a profile-derived format silently assumes attention, dense-FFN, and experts
all match. `QuantProfile::Nvfp4Experts` still deserializes so old `nvfp4x`
manifests keep loading, but `quantize --profile nvfp4x` now expands to
`--profile q4 --set experts=nvfp4`.

**Tooling**: `quantize --set ... --overlay` composes with layered overlays
(§8.1) unchanged — a tensor class switched from raw to a quantized format
simply moves from an `External` HF-safetensors ref to a `Local` blob entry,
same as any other Raw→quantized transition; `--set` only changes what
`classify_tensor_custom` returns per tensor, not how the writer decides
`External` vs `Local` (still keyed on `kind == Raw`).

### 8.1 Layered / overlay packs

A `.dgq` pack is either **self-contained** (every tensor's bytes in its own
`model.dgq.bin`, `DgqTensorMeta::source == None` — the default `quantize`
output and the only shape that ships to users) or **layered**: some entries
carry a `source` saying where their bytes actually live.
`DgqManifest::is_layered()` is true iff any entry has one.

- **`TensorSource::Local { local_offset }`** — in this pack's own compact
  blob, at a different position than the entry's canonical `offset` (a
  layered pack's blob has no gaps for tensors it doesn't store).
- **`TensorSource::External { file, offset }`** — in an external file keyed
  into `DgqManifest::external_files` (role `hf_safetensors`: a shard in the
  pinned `base_model` HF snapshot; role `pack_bin`: another pack's blob).

**The addressing contract is unchanged by layering.** An entry's own
`offset`/`byte_len` describe a canonical unified address space, identical in
shape to what a self-contained pack's blob would have been. `source` only
tells the loader where to copy FROM; it never changes what any consumer reads
offsets AGAINST (`build_offsets_from_store`'s `w_off` constants,
`DgqGpuBlob::buffer_for`, the split-blob region math).

**Overlay packs** are today's only layered shape: every Raw tensor becomes an
`External` ref into the HF base's safetensors shards (byte-identical bf16, so
no requantization), every quantized tensor stays `Local` in a small blob
(~13.3 GiB for the q4 profile vs ~18.84 GiB self-contained). `external_files`
pins each shard's byte size and a `header_sha256` (the header only, not the
multi-GiB payload) so a stale cache fails loud with the exact `hf download`
command to fix it (`src/dgq/hf_resolve.rs`; cache root `DGQ_HF_HOME` →
`HF_HOME` → `~/.cache/huggingface`).

**Loading** assembles the canonical space from two regions: the expert tail
`[expert_split, total)` is offset-mmap'd directly off the pack's own blob at
`local_expert_split` (never copied — file-backed, clean, evictable), and the
raw head `[0, expert_split)` is gather-copied into a private anonymous
mapping on every load (`materialize_layered_head_only`, ~5.5 GiB of dirty
pages). That per-load copy is an accepted cost for what layered packs are:
dev tooling for experiment arms. Distribution is monolithic and pays none of
it. Oldest layered packs (no `local_expert_split`) take the whole-blob
`materialize_layered_blob` fallback — same math, head and tail both gathered.

**Why the head is copied, not spliced**: serving it zero-copy by
MAP_FIXED-splicing HF shard ranges into one reserved VA range was built,
mechanically validated (Metal does accept a no-copy buffer spanning multiple
mappings), and removed — coverage on the real model is ZERO. Splicing needs
`canonical ≡ file (mod 16384)` page congruence while the 64-byte invariant
above forces `canonical ≡ 0 (mod 64)`, which together hold only when a shard's
own file offset is a multiple of 64: an ~1-in-64 draw per shard that came up
zero across all 11. Reviving it means first auditing every Raw-weight kernel's
true minimum alignment, since a floor below 64 reopens the odds.

**Tooling**: `repack --overlay -m PACK -o DIR` splits an existing
self-contained pack: it verifies each Raw tensor's bytes are byte-identical
to the HF base while streaming (doubling as the verbatim audit — a mismatch
falls back to storing that tensor locally rather than failing the whole
repack) and auto-detects `(repo, revision)` from the source path's
`models--org--name/snapshots/<rev>` HF-cache layout (`--hf-repo`/
`--hf-revision` override when it isn't one). `quantize --overlay` produces
an experts-only overlay directly from an HF snapshot (any profile,
including `nvfp4`) without a self-contained intermediate — the same
`classify_tensor`/quantization code path, only the emission target differs
for Raw tensors. `repack --monolithic -m PACK -o DIR` is the dual: it
flattens a layered pack back into a self-contained one, streaming every
tensor's resolved bytes (local or external) to a fresh blob at canonical
offsets — no requantization, and no splice-layout requirement, since a
monolithic pack has no external refs to splice.

**Product framing**: monolithic (self-contained) packs are how models are
distributed to users (`download`, HF-hosted packs). Layered/overlay packs
are local-only dev tooling for experiment arms (`quantize --overlay`,
`repack --overlay`/`--monolithic`) — never the distribution artifact.

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
| 9 | Entropy-only early stop (mean H < 0.05, §6) | ours only | tail steps only micro-flip near-ties; census cost = creative-tail only |
| 10 | Degenerate-first-block re-roll + shrink-on-retry | ours only | empty-reply attractor is intrinsic; deterministic per seed; no-op on good replies |
| 11 | Commit-time confidence trim, dup tier (`DGQ_COMMIT_CONF_TRIM=0.9`, §6) | ours only | a proposed block is cut at the first answer-region row that is BOTH below τ and argmax-duplicating a neighbour — an unresolved row copies its neighbour, so the tail re-denoises next block against committed context. Default ON; never fires on a golden case (golden 8/8 byte-identical). Enables the `max_blocks *= 2` token-budget headroom. The UNCONDITIONAL hard tier is the separate `DGQ_COMMIT_CONF_HARD`, still OFF |

## 10. Validation harnesses

- **Census campaigns** (`census`): flag ARMS × BATTERIES with explicit
  acceptance gates in ONE process, stats to a directory. Arms are `DGQ_*`
  overrides in env form parsed through the same validation as the process
  env (a typo'd arm dies before any model load); metrics come from the
  denoise p_max trace so they are battery-independent
  (`contested_per_1k`, `hard`, `dup`, `steps_committed`/`steps_run`/
  `steps_retry`, `retrieval_pct`). `--analyze DIR` re-reports an existing
  campaign with no GPU. This is how a quality lever is decided. Batteries:
  `smoke`, `longctx`, `programmatic`.
- **Executable correctness** (`census --battery programmatic`): the reply is
  compiled (rust) or syntax-checked (python, bash) and RUN against fixture
  cases; stdout and exit code must both match. The only harness that judges
  whether output is executably correct rather than textually plausible. Three
  outcome states (`compile_fail` / `wrong_output` / `pass`) stay distinct
  because well-formed-but-wrong is a different finding from unparseable —
  compare `compile_fail` within a language, never across (`bash -n` accepts
  prose). A probe fails on any of three independent axes: a wrong case, a
  blown step budget, or a markdown fence (every prompt forbids fences, so a
  fenced reply disobeyed an instruction — the fence is still stripped and the
  program still run, so `fenced%` keeps the rate visible beside full case
  credit). Each case runs in a fresh temp cwd under a hard 10 s timeout with
  captured pipes: a hanging generated program is a failed case, never a wedged
  machine.
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

- `crates/gpukit` — GPU mechanism with no model or policy knowledge:
  device/queue context, `#include` expansion, function-constant
  specialization with cache labels derived from the full input set (FC
  values + source hash), the pipeline binary-archive cache (keyed on the
  whole shader-tree hash), buffer pool, one-shot dispatch helpers.
- `src/shaders/<group>/<kernel>/` — every kernel's Rust wrapper, Metal
  source, CPU oracle, and manifest `SPEC` colocated (see AGENTS.md §4).
- `src/metal/` — the runtime: the gpukit policy layer (`device.rs`: flag
  wiring, `KernelVariant` → FC mapping), buffer arenas, KV cache, the
  production step dispatch (`step_kernel.rs`), and the f32 validation engine.
- One production step path (the "monolith"); the f32 engine survives for
  ≤256-token prefill and as the step-parity oracle.

## The token pipeline (the runtime's front door; P0–P4 shipped)

`src/pipeline/`. One thread owns the GPU + KV; every client (ask, chat,
serve) speaks serialized **ops** in and **events** out. Ops carry token ids
only, never strings — all text, parsing, and policy live in the message
layer (`src/server/worker.rs`). Motivation: serve's scattered state mutations
were un-auditable; the OpenCode collapse took a night to root-cause, and a
serialized replayable op-log turns any field failure into a golden-style
artifact.

- **Ops**: `Extend`, `Generate`, `Rewind`, `Splice {start,end,replacement}`
  (surgical mid-log replacement = truncate + re-extend), `SyntheticFill`,
  `KvFingerprint`, `Mark`, `Activate`, `Finalize`, `AlignTo`, `Ping`,
  `Cancel`, per-block `BeginTurn`/`ProposeBlock`/`CommitBlock {kept_len}`/
  `DiscardBlock`/`EndTurn`, `Shutdown`. `KvId = (epoch, pos)`; lineage-
  invalidating ops bump the epoch and a stale-epoch rewind fails loudly.
- **Per-block protocol**: `generate_with_session` is a thin driver over
  `begin_turn` / `propose_block` / `commit_block` / `finish_turn`
  (`default_commit_policy` = stop-scan/defer/ws-guard); op-driven turns are
  byte-identical to the monolithic path (golden-pinned).
- **Stage chain**: `PipelineStage` trait, Pipeline terminal, wrappers
  compose freely. `OpLogStage` journals every op+event as full-fidelity
  JSONL (`serve --log-dir`); `diffgemma replay <ops.jsonl>` re-executes
  a session and diffs every event — field sessions are executable repro
  artifacts. `ToolValidatorStage` (opt-in `--tool-validate`) retries
  malformed tool grammar with a bumped seed. `ToolRepairStage` (opt-in
  `--tool-repair`) is the **evaporating-draft** primitive: Mark → Generate →
  on invalid tool reply, feed one error tool-response per invalid call →
  regenerate to natural `<eos>` (response-opener removed from stops; forced
  early cuts are a degeneracy source) → trim any hallucinated response
  tail → Rewind to prompt end. The corrupt exchange and the feedback never
  enter canonical KV; the conversation pays only for the clean final reply.
- **Cancel**: a `CancelToken` rides in `StepGenerateConfig` like the
  observer Arcs; checked between denoise steps and blocks. Serve cancels
  when the SSE socket dies (observer send fails) and skips finalize.
- **Stop conditioning**: `continue_incomplete_tool_calls` is DEFAULT OFF
  (`DGQ_CONTINUE_PAST_STOP=1` restores). The defer was the load-bearing
  amplifier of the OpenCode collapse chain (premature stop → forced
  continuation → OOD → filler fixed point → commit amplifier → flood).
- **KV-reuse-first serving** (see the principle in AGENTS.md §6):
  tool-calling turns finalize with EMPTY content in the canonical render
  (prefix-stable: ends at the `<|tool_response>` opener; the client echoes
  the prose and the next render places it); `route()` picks the
  conversation with the longest common prefix and salvages tail divergence
  (lcp ≥ 256 && tail ≤ 512 → O(1) ring truncate on Activate); anything
  deeper is an explicit, logged decision to re-prefill.
- **Tool-turn continuation** (`DGQ_SERVE_TOOL_CONTINUATION`, default ON):
  a tool-mode generation prompt seeds the thought channel OPEN (reasoning
  lands in the thought block by construction; `DiffusionStreamMapper`
  starts in-thought so the wire split agrees), and a request recognized as
  the previous tool turn's next round (`match_tool_continuation`: same
  message prefix + our echoed calls + their responses) prompts as the raw
  token log — reasoning intact — plus the rendered responses and a
  reopened thought. Mid-turn rounds finalize the raw log (a no-op rebuild;
  pure KV extension per round); the round that ends the turn finalizes
  thought-free as always, so reasoning still never crosses a turn
  boundary. Fixes the intra-turn amnesia of the thought-stripping
  re-render (the model re-planned from scratch every round).
- **Standing gate — rewind byte-consistency**: seeded
  generate → rewind → generate loops must restore KV bytes exactly
  (position-ordered `live_kv_fingerprint`) and regenerate bit-identically.
  Covered below the wrap, across the wrap, through the ring-rebuild path,
  and after in-block re-rolls; golden's `ring_wrap_4p6k` crosses the
  DEFAULT `DGQ_KV_RING` (4096) — re-aim it if the default changes.

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
  long-context attention enabler. **q8 KV is wired but does not work**:
  `kv_format(max_seq)` auto-selects group-32 q8 once estimated f16 resident
  exceeds 85% of the GPU working-set cap (~178k tokens on 36 GB), and that
  path goes NaN after layer 0 on fast prefill. No gate reaches the crossover,
  so it has never run in production. `DGQ_KV_Q8=0` forces f16. Open v1 item.
- **Always-f32**: moeout plane, rowstats planes, MoE grouped scratch, router
  logits.
- The real hazard is producer/consumer dtype mismatch, not dtype choice
  (AGENTS.md §6).

## Performance regime (M3 Pro, measured)

- **Denoise GEMMs: compute-bound at the MPS matmul wall** (~3.7-3.9 TF/s
  tunable dense = MPS 3.65-3.69; MLX qmm 3.96-4.15 — the residual
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
  needle-exact to 121k (KV 2.4 GiB); denoise convergence parity-class
  (~1.15× steps vs MLX's best config, matched-canvas multi-seed).
- **Long-context prefill (dynamic top-k default): parity with MLX-4bit
  chunked across the range** — see README for the canonical head-to-head
  numbers. The gap does not widen with context; it is a dead heat at 100k.
  The dense-attention fallback (top-k off) trails MLX ~1.43× at 100k.
- **Decode is well ahead at long context**: dynamic top-k attention runs on
  the denoise path too (`DGQ_ATTN_TOPK_DECODE`, default on), cutting 100k
  decode from ~4.4 to ~1.8 s/step — ~3.9× vs MLX generation (see README).
  The dense `mma_full` fallback was the wrong decode kernel at this depth.
- Memory: single 36 GB machine is NOT swap-bound to 105k; the cliff is
  ~262k f16 KV or running beside another model. Nothing covers past the cliff
  today — q8 KV is the intended lever and does not work.

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
- **int8/int4 dot-product MoE expert GEMM (E18 redirect)** — `gemm_int_sparse`
  (int8-accumulated block-sparse expert GEMM), correctness-gated first
  (cos=1.000000 vs CPU oracle at every tile), measured 0.43 vs the production
  q4→half-MMA path's 3.78 TF/s at real MoE shapes. The 9.7× gap decomposes
  into three independently measured factors: **1.72×** the prototype's own
  un-register-blocked inner loop (384 vs 80 tgmem loads per lane per BK=32;
  register-blocking it measured 0.755 TF/s), **2.22×** int32-multiply vs
  f32-mad on M3, **2.54×** what simdgroup-MMA is actually worth vs scalar f32.
  So the honest bar for a future attempt is **5.6×, not 9×** — and it is still
  a dead end, because the two residual factors are hardware properties no
  restructuring addresses. There is also **no DP4A-style packed integer dot in
  MSL** (`dot(char4,char4)`, `simd_dot_acc_int8`, `dot_product_4x8_packed` all
  rejected by this M3 Pro's compiler), so the manual 4-scalar expansion was the
  only expressible form; int8 weights cost ~2× q4's bytes on top. Kept as a
  documented negative (`gemm_int_sparse` + `bench-gemm --shapes sparse`), not
  wired to production. Lesson: **a disproof whose stated mechanism was never
  independently measured is a curve fit** — the original write-up credited MMA
  width with the whole 9× and "matched exactly" only by absorbing a 1.7× own
  bug and a 2.2× effect nobody had identified.
- **E21: SLC-chunked online-softmax E17 (flash at dispatch granularity)** —
  parked at its pre-registered kill bar: E17's S/P DRAM round-trips
  are only ~8% of the attention stage and the share is FLAT in T (S/P traffic
  and QK/PV compute both scale linearly — no long-context regime where it
  grows). ~3% end-to-end, non-bit-identical: below the bar. Revive only for
  the memory co-benefit (chunking makes S/P scratch T-independent, 1.6 GiB →
  ~130 MB @100k) if scratch pressure ever bites.
- **E22: block-granular pre-QK attention selection (Quest/MInference-class)**
  — killed by measurement: attention mass is NOT block-concentrated
  at depth on this model. Real Q/K planes (`step-attn-qk-dump` +
  `python/scripts/e22_block_mass.py`), real 121k mixed corpus: even ORACLE
  top-32-of-944 blocks holds 17% of mass; the across-head union reaches 36%
  only by going ~dense. Selection fidelity was fine (centroid Spearman
  0.73-0.86) — the territory failed, not the mechanism. Parked toward E16
  token compaction: the measured structure (sharp retrieval peaks + near-
  uniform diffuse background) is EVIDENCE FOR fusing aged tokens.
- **Fixed-distance locality band** (attend only within ±N of the query) — a
  2k-wide band holds just 39% of attention mass at kv=10k (measured on a full
  layer, hd=512). The model has BOTH a prompt-start anchor (first 64 prompt
  tokens carry 19% of mass) and retrieval from the far end of the prompt
  (40%); a band structurally misses both. Top-k selection is the lever that
  works here, not distance.
- **A prompt-start anchor as a separate lever** (always include the first N
  prompt positions on top of top-k) — adds nothing: top-128 retains the same
  71.3% of mass with or without a 64-position anchor, because those positions
  are already inside the natural top-k for most heads. The anchor is a subset
  of top-k, not additive.
- **Attention-mass coverage as a quality oracle — IMPEACHED on this model**:
  exact row-top-512 holds only 30% of mass at 121k and row-top-64
  only 13%, yet behavioral retrieval at those k values is clean (doc-QA 4/4 to
  20.6k at k=64; needle-exact 4/4 at 121k with dynamic k). Mass and quality
  DECOUPLE: the diffuse background's average — not its composition — is what
  the output needs. **Never gate a sparsity lever on mass coverage here;
  behavioral retrieval probes only** (and a probe's markers must be
  corpus-unique: a "NEEDLE"-named marker inside a corpus of our own docs that
  discuss needle tests produced pure confabulation artifacts).
- **E18 flash BQ/BK tile geometry** — non-lever: interleaved
  same-batch A/B at kv=15k puts BK=128 at −1.9% and BQ=32 at +0.7%; BK=32/256
  don't compile. Default (16, 64) stands. The first single-pass sweep showed
  BK=128 at "−7.9%" — cross-process `bench-prefill-super` runs drift up to
  ~±4% same-config same-day (vs 0.08% within-process): **never read a
  cross-process proxy delta <5% without interleaved repeats.**
- **Flash-decode / sequential KV blocking** — attention is issue-bound at
  SLC service; the lockstep sweep already gets SLC locality. Restructuring
  the sweep bought nothing (`DGQ_ATTN_KV_BLOCK`, default off).
- **Low-bit KV for speed** (q8, and TurboQuant-class by the same physics) —
  +9% prefill / +54% denoise at 33k; only −6% even at 105k. Kernels are
  issue-bound, not byte-starved. q8 KV would only ever be a MEMORY lever, and
  is currently broken (see the precision policy).
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
- **Software-pipelined double-buffering GEMM port** — `gemm_tunable_db`
  (sibling entry in `gemm_tunable.metal`) doubles the tgmem tiles, prologue-
  loads tile 0, then overlaps the device→tgmem load of tile N+1 with the MMA of
  tile N. Bit-exact vs single-buffered `gemm_tunable`, and **7-9% SLOWER** at
  the prefill-relevant dense shapes (256×2816×2816 and 1024×2816×2816, tile
  64×64, q4): 3.377 / 3.566 vs 3.611 / 3.919 TF/s. PHYSICS: there is no
  async-copy engine on Apple GPU, so the "overlap" is just issuing the loads
  before the MMA — which the single-buffered kernel already does implicitly,
  because the load is fully hidden behind the ~6× compute margin. The extra
  barrier and the doubled tgmem footprint (occupancy, SLC pressure) cost more
  than the overlap returns. Kept as a documented negative (`gemm_tunable_db` +
  `bench-gemm --shapes db`), not wired to production. Lesson: when
  compute >> load, double-buffering is a regression, not a lever.

**Quality levers, disproven by measurement:**
- **Confidence trim as a fix for CODE-correctness errors** — the
  `programmatic` battery's failures looked like the low-p_max tool-arg
  stutter class. `DGQ_TRACE_PMAX_JSONL` over seeds {7,42,123} says
  otherwise: the wrong tokens commit at the TOP of the distribution (seed
  7's `"*` at **0.9993**, ` final` at **1.0000**; seed 123's failing rows
  0.83–0.98), inside blocks that are ~1.0 throughout with
  `conf_trim_row=None`; committed rows below 0.9 number just 9/4490,
  5/4343, 12/4352. Dup-stutter commits live at 0.40–0.86 — a different
  regime. **No threshold separates a confidently-WRONG token from a
  confidently-right one**, so no trim tier addresses this class. The
  failures are trajectory-dependent, not a fixed model limitation (seed 42
  scores 14/14; `bash_stdin_and_argv` is correct at seed 123 and wrong at
  seed 7), and the defect sits UPSTREAM of the visible error: seed 7 wrote
  `*` `$` `SEARCH` `"*`, confidently closing a quote it never opened, while
  seed 123 wrote `*"$` `SEARCH` `"*` — the same closing token, correct.
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
