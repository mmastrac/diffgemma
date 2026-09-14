# diffgemma — plan

Open work only, split into **v1** (what a 1.0 needs) and **v2** (everything
after). Design + contract: **ARCHITECTURE.md** (incl. Negative Knowledge —
check it before planning perf/quality work). Working discipline + commands:
**AGENTS.md**. User-facing scope and benchmarks: **README.md**. Everything
done lives in git history — commit messages are the changelog. When an item
lands, REMOVE it from this file; keep only any residual open sub-task.

## State (orientation, not history)

Engine is feature-complete for v1. Wall-clock beats MLX-4bit on short/medium
chat; long-context prefill is at MLX parity across the range and long-context
decode is well ahead (dynamic top-k attention default-on both phases). The
**token pipeline is the core** (P0–P4 shipped; ARCHITECTURE Part III):
ask/chat/serve are op-stream clients, sessions are op-logged and bit-exactly
replayable. Cross-turn KV reuse is KV-reuse-first with tail-salvage routing.
Gates are green (smoketest × {7,42,123}, golden 8/8, full suite). Remaining
decode headroom is STEP COUNT (E7 territory), not per-step attention.

---

# v1

## Blockers

- **q8 KV auto-enables into a broken path.** `kv_format()`
  (src/flags/accessors.rs) switches the KV cache to q8 with NO flag once
  estimated f16 resident exceeds 85% of the GPU working-set cap — ~178k
  tokens on a 36 GB machine. q8 KV goes NaN after layer 0 on the fast-prefill
  path and has never actually run in production, so `--ctx 200000` silently
  selects a format that does not work. Either fix the NaN or make the auto
  policy refuse (fail loud at open) instead of switching. ARCHITECTURE's
  precision policy currently describes q8-auto as a working memory lever and
  must be corrected either way. See [[kv-quant-status]].
- **Release tagging + `--version`** with a `.dgq` manifest-version gate.
  Cargo.toml is still 0.1.0 and no `--version` flag exists.
- **CI completion.** The nightly model-gated tier is a commented-out block in
  `.github/workflows/ci.yml`, not wired to a runner: smoketest + golden +
  longctx + perf floors (step ≤ 1.1 s chat lengths; 33k prefill ≤ 140 s;
  100k ≤ 16 min prefill, ≤ 5 s/step), plus a weekly multi-seed aggregate +
  census.
- **Install-to-first-token < 30 min is unverified.** README documents the
  path; the clean-36 GB-Mac timing run has not been done.
- **Refresh the MLX head-to-head before publishing.** The README table cites
  MLX's 4bit build as of one measurement day, and a later prefill A/B round
  read the same comparison differently (30k at 1.04×, 100k at 0.975×, vs
  README's 32k 1.09× / 100k 1.02×). Re-measure both engines in one session
  and make README the single source of record.
- **Long-context re-validation.** Re-run needle 33k/105k and the 100k
  field-incident repro on the uncapped fast path.

## Nice to have before shipping

- **Benchmark page** with the full harness invocations (README carries the
  headline numbers; decide whether that is enough for 1.0).
- **`--tool-repair` / `--tool-validate` default-ON decision** after the field
  trial (op-logs in `--log-dir` are the evidence base; first organic session
  fired one repair that recovered cleanly).
- **Thinking-flag flip silently loses KV reuse** — add a log line so the perf
  cliff is explicable in the field.
- **Inline flag validation gap**: ~29 `DGQ_*` flags still parsed via bespoke
  `var(...)` chains (enums, u32 ranges, paths) silently swallow bad values.
  The shared checked helpers exist (AGENTS.md §7); convert as touched.
- **Tier-1 attention fixture is below the worst tile**: `full_grp8_hd512_fixture`
  (canvas=16, t_total=44) vs E17/E20's BM=BN=64 — every parity test is
  single-tile, and `topk_k128_matches_cpu` claims k>64 coverage on 44 keys.
  Add a canvas ≥ 65 / kv ≥ 65 full-layer fixture.
- **Missing CPU twins**: `kv/unpack_encoder_kv` and `kv/kv_f32_side_hydrate`
  (~40 lines each).
- **Oracle sampler tests**: `sample_from_probs_rows` (the one worth a
  fixture), `scale_logits`, `logit_softcapping`.
- **`metal/oracle/` quarantine**: audit done, the safe-to-quarantine set has
  zero production callers; the MIXED files (`decoder.rs`, `decoder_layer.rs`,
  `memwatch.rs`) need a prod/validation split first. Blocked on user sign-off.
- **Broader eval**: the 17-prompt gate is sensitive but narrow; add a
  ~100-prompt adherence set, weekly, non-blocking.
- **Second machine**: every published number is one M3 Pro. Recruit at least
  one other M-series config before publishing claims — SLC-locality physics
  may differ on M1.

---

# v2

## Token pipeline follow-ups

- **P5 Refine / canvas-edit primitives** (`Refine {mask|forced_ids}` —
  continue denoising the same uncommitted canvas). Quality-gated: census,
  multi-seed; the freeze lesson applies to reject-masks. Later: multi-conv
  absorption, `Reground` (idle re-prefill = lineage reset), lineage-drift
  gate. Prerequisite for controlled canvas inspection / tree-sitter
  rerolling — "advance the canvas under control, then inspect" is not
  available today.
- **Replay across serve restarts** — a second `{"meta":…}` line stops replay.
- **serve ops.jsonl is no longer token-level replayable** — the registry op
  format (activate/generate/finalize summaries) is skipped by `replay`
  ("unknown op shape"). Either teach `replay` the registry format (activate
  carries the full prompt token array) or log a parallel token-level stream.
- **Grammar-aware `kept_len`** at the per-block layer (message layer owns it;
  today stop metadata is driver policy via block stats).
- **Non-streaming disconnect detection** (socket read-EOF polling on the
  connection thread; today only the final write notices).
- **Prefill chunks as cancellation points** (a 100k prefill runs to
  completion; `prefill_chunks*` needs a consistent partial-prefix story first).
- **Snapshot-restore-to-ring lever** — deep truncates past ring slack pay a
  full rebuild. A KV snapshot restore path could make deep rewinds cheap;
  measure demand from field op-logs first.

## Message-layer designs (user-directed, not started)

- **Interleaving-restoration blob**: any assistant-turn interleaving not
  representable in OpenAI format (call/prose/call/prose ordering) must
  round-trip via an opaque blob so re-prefill restores the exact KV ordering.
  Tool calls ONLY — thinking must never return to KV re-prefill. Needs
  per-client verification that unknown fields are echoed. (≈ OpenAI's
  encrypted reasoning items; also the cross-restart/cross-client hardening of
  the thinking-persistence option below.)
- **Tool-call triage via internal re-prompt**: on questionable calls, ask the
  model WHICH call(s) to keep, then splice so context retains only the chosen
  call. The evaporating-draft choreography is the substrate; the validator's
  retry-on-malformed is the degenerate case (keep zero).
- **Narrate-instead-of-act / act-without-narrating policy**: residual model
  warts — a turn ends after announcing a write with no call emitted, or a
  whole session is tool_calls-only (the amnesia driver). A triage-layer
  policy, not a serve defect: push toward "one line of visible narration per
  tool turn".
- **Confident-miscount loop breaker** (same policy family): repetition
  arithmetic can commit a wrong count at p_max ≈ 1.0, then spiral on
  self-contradiction retries. Mitigation is a message/triage-layer breaker
  that ties toward observed tool output after N failed reconciliations;
  confidence gating is blind here by construction (ARCHITECTURE Negative
  Knowledge, "Confidence trim as a fix for CODE-correctness errors").
- **Thinking persistence across tool turns (design option, NOT decided)**:
  all three frontier labs preserve full reasoning across the tool-calling loop
  and drop it at user-turn boundaries; we sit at the DeepSeek-R1 end (strip
  everything, every turn), and the cost is measured (a tool session spent
  32.7 s re-deriving state it had already reasoned out). Because canonical KV
  is SERVER-OWNED we could defer thought-stripping to the TASK boundary with
  zero client cooperation. Costs to weigh: ring pressure (thoughts × tool hops
  vs DGQ_KV_RING=4096 and the salvage window) and larger canonical-vs-client
  divergence mid-task. Standing user directive until revisited: thinking never
  reaches KV RE-PREFILL from the client side.

## Quality track

- **`DGQ_COMMIT_CONF_HARD` (unconditional p_max floor) — stays OFF, parked.**
  The dup tier (`DGQ_COMMIT_CONF_TRIM`, conjunctive at τ=0.9) is shipped
  default-ON; the hard tier is split out and off. It is the tier that kills
  the insertion/omission class but it fires ~4× as often, costs a few percent
  of steps, and tips `transformer` one step over its convergence budget.
  Blocked on a metric mismatch, not on the tier being bad: no battery both
  reaches the sub-0.5 trigger AND carries a quality signal — grounded
  retrieval never goes below ~0.73, and the wart proxy (`contested_per_1k`)
  does not predict executable correctness
  ([[wart-proxy-doesnt-predict-quality]]). To settle it, `soft` needs
  long-form (multi-sentence) answers past the 16-row floor; the `doc_tokens`
  field is already in the fixture schema.
- **`DGQ_PREFIX_EXIT` early block commit** (landed default OFF; quality-safe,
  not a reliable speed lever): NEXT is the quality-mode experiment —
  aggressive exits at matched TOKEN budgets, judged on census multi-seed +
  strain tool-arg typo rates (the "commit-when-stable ≈ more-causal
  factorization ≈ fewer independence violations" hypothesis; VSB
  arXiv:2604.23994 reports +4–10% from the trained analog). Trajectory-
  affecting → golden re-bless + live A/B, never a replay sim
  ([[trajectory-feedback-sim-bias]]).
- **E16 token fusion / KV merging** — the only unexplored long-context denoise
  SPEED lever (cuts token count, not bytes). Status and next steps live in
  [[token-fusion-e16]]. MUST gate on the doc-QA ladder, not needles.
- **E3 canvas shrink near max_tokens**: close divergence #5 (MLX shrinks to
  max(remaining, 64)); minor tail win; trajectory-affecting → multi-seed gate.
- **MLX matched-canvas dig on the preserved collapse trajectory** — can MLX's
  sampler survive the same conditioning? Artifacts in-repo:
  `debug/strain_battery/collapse_seed42/` (ops.jsonl + serve log),
  `debug/strain_battery/prompts/` (matched clean/collapse pair);
  `debug/strain_battery/battery.py` is the harness.
- **Structural repair (delimiter checker → repair).** The checker SHIPS
  observational: `src/delimiter.rs` behind `DGQ_DELIM_CHECK` /
  `DGQ_DELIM_CHECK_JSONL`, hooked at `commit_block`, read-only. It is the
  first thing to detect the convention-blend class, which no confidence
  threshold reaches. Repair is NOT built: the trigger is well-defined (~1 in
  11 judgeable code blocks, all `quote_region`/`bracket`, all terminated), and
  `ToolRepairStage` + `KvCheckpoint`/`rollback_to` (currently dead code) are
  the substrate. Regeneration must change something — reuse shrink-on-retry
  (256→128→64). The model is the WRONG adjudicator for its own defect (it was
  certain when it emitted the blend), so gate the CHECKER by block mode:
  prose → skip, code → regenerate without asking. Trajectory-affecting →
  golden re-bless + live A/B. Re-validate the probe's block labels first: a
  curated set showed 5/5 but a `smoke` run mislabelled 17/19 (the language
  SNIFFER, not the probe, carries the prose gate today).

## Census / evaluation

`census` (arms × batteries × gates, `src/commands/census.rs`) has `smoke`,
`longctx`, `programmatic`, and `soft`.

- **Quantify the convention-blend rate broadly.** The delimiter checker
  measures it per-battery (~9.3% of judgeable `programmatic` blocks); run it
  across convention-ambiguous constructs at scale to publish a rate rather
  than specimens. Everything needed is in `DGQ_TRACE_PMAX_JSONL` traces / the
  checker — GPU only for generating more samples.
- **`scan_trace` answer-region fix (low priority).** Mirroring the trim's
  `(MIN_CONF_KEEP..region_end)` rule would make `contested_per_1k` mean what
  its name implies. Arm ordering survives the correction, so this is
  legibility, not correctness — but it still moves a published number.

## Output-mode classification

Per-BLOCK classification SHIPS and works: `fit-token-probe` (refit locally per
checkpoint) + `DGQ_TOKEN_CLASS=<probe.json>` classify a block at its first
forward, while the canvas is still seeded noise. Per-TOKEN classification does
NOT work and is not one fix away ([[token-mode-probe]]).

Remaining (block-level only): colour the per-block labels on
`ChatEvent::BlockCommit` in `chat::render`. Perf obligations are HARD
requirements: exactly zero cost when OFF (capture not encoded at all — flag
parsed once into `RuntimeConfig`), and MEASURED cost when ON (~1.44 MiB device
copy per block, `bench-step-kernel --profile-steps` adjacent A/B). Do not
spend a capture kernel on per-token without new evidence that an earlier layer
clears 0.641 by a lot.

## Perf backlog

- **MoE weight-stationary `DGQ_MOE_PREFILL_BM` honest re-measure never
  concluded** (post cache-collision fix; early 3-trial signal: correct bm=64
  is SLOWER). Run the honest sweep or close the item.
- **MoE adaptive-M / partial-tile padding at M=1024** — predicted to activate
  when per-TG goes compute-bound (this regime); the live kernel comment still
  calls padding "immaterial", a stale denoise conclusion.
- **E5 QK-ILP2 chain-split — PENDING, inconclusive.** Splitting the 32-deep
  serial QK MMA chain in `attention_mma_full` into two independent 16-deep
  chains (FC31 `DGQ_ATTN_MMA_FULL_QK_ILP2`, default OFF) should halve QK
  dependency depth. Every A/B so far was INVALID: with E17/top-k default-on,
  full layers route to `attention_gemm`, so `attention_mma_full`+ILP2 is inert
  on the dominant attention cost. Test it as a categorical axis (paired with
  `gemm_attn` on/off) in the holistic prefill BO (`tune_prefill_attn.py
  --proxy`) — a single-axis A/B at default settings cannot see a lever that
  only activates on the off-default path.

## Code structure

- **cli.rs parser structure**: the usage string is true, but `parse_cli` still
  uses ~90 shared mutable locals (cross-wiring hazard); a per-command
  arg-struct redesign remains open.
- **Port the engine's kernels to CUDA.** `crates/dgops` proves the pattern on
  the shared subset (one CPU oracle, Metal + CUDA bodies, tier-1 parity) and
  `crates/nanogpt` exercises it end to end. The diffusion tranche is started:
  `dgops::ops::{embed_gather, rms_norm_rows, swiglu_gelu, apply_rope_heads,
  gqa_attention, moe_router_topk}` are the same kernels with a CUDA body
  beside the Metal one, and `crates/dgops/tests/golden_slice.rs` runs seven
  real-weight stages of the `engine_prefill` golden case on the GB10
  (`DGQ_MODEL_DIR=<pack>`, `--features cuda`). `src/shaders/**` is still
  Metal-only and the crate does not build on Linux at all (`main.rs`
  compile_errors off macOS; `src/metal/`, `chat/`, `server/`, `decoder/` are
  macOS-gated). Order: make `src/shaders` + `src/model` build on Linux behind
  the cuda feature (their GPU halves are already cfg-gated), move each ported
  kernel onto the portable body so the engine and the CUDA slice share one
  oracle, port the rest as `cuda.cu` beside each `.metal` (the quantized GEMM
  family next), then the step kernel. A CUDA box runs the tier-1 parity suite
  without the 19 GiB pack; the golden slice additionally needs the pack.
  `crates/dgqcuda` goes further: it runs the whole 30-layer forward and the
  denoise loop on CUDA against a real pack, with a CPU oracle per stage and a
  per-step parity mode. The MoE experts use a bucketed grouped GEMM
  (`moe_grouped.rs` + `moe_grouped.cu`): tokens are bucketed by expert, the
  expert-major rows feed one tiled GEMM per bucket with the q4 weight tile
  decoded into shared memory once per tile, and the routing weight is folded
  into the SwiGLU. That also removes the f32 expert copies, so the resident
  model drops from ~52 GiB to ~35 GiB. The grouped launch reads each bucket's
  expert from the plan (`experts[job]`), never from the job index: a bucket
  with no tokens holds no rows, so job j is not expert j. Every bucketed
  kernel is element-wise and must be dispatched over `rows * width` elements,
  never over rows: a row-count grid silently truncates the gather and the
  expert path degrades to zeros without failing. Text prompts work
  (`--prompt`, tokenizer + chat template), and the remaining port is the same
  quantized-GEMM treatment for the attention/dense weights.

  **Fixed: the step re-applied the layer stack to the prompt rows.** The port
  has no resident KV cache, so each step runs `[prompt][canvas]` through the
  layers and rebuilds the prompt's K/V that way. It seeded the prompt rows
  with the post-layer hidden state from `prompt_hidden`, which feeds a state
  that has already been through all 30 layers back into layer 0: the prompt
  residual stream leaves the stack 30 layers ahead, and the canvas attends the
  resulting wrong K/V. Seeding from the embeddings instead is exact, not an
  approximation -- a prompt row is causal (`causal_split` is the prompt
  length), so it attends only prompt keys up to itself and cannot see the
  canvas, which reproduces the standalone causal prefill's hidden and K/V bit
  for bit. `forward_sc` and the device `Session::step` now both embed the
  prompt; `prompt_hidden` became `warm` (a device warm-up that no longer feeds
  a probe), and the `--diag` line that compared it was the misleading
  probe-vs-production pair AGENTS.md warns about: it exercised a code path the
  step never used, so "prompt rows match the engine" and "the canvas drifts"
  were both true at once. `tests/step_prompt.rs` pins the prompt rows to the
  causal prefill and to canvas-independence; with the bug reinstated it reports
  cos 0.0061, and green it passes on the real pack.

  **Disproved: bf16-versus-f32 activations are not the residual drift.** Every
  attention and dense weight in the production pack is raw BF16 and bf16->f32
  widening is exact, so weights were never a divergence; the only delta is that
  the engine bf16-rounds activations at each arena store while the port keeps
  f32. Simulating that faithfully (norm-bounded 30-layer residual,
  round-to-nearest-even at every store) gives hidden cos 0.9997 and leaves
  argmax and entropy unchanged (8.4953 vs 8.4954) across 12 seeds. A relative
  cos of 0.4-0.7 is a structural mismatch, not a 2^-9 per-store accumulation.

  **Latent, not yet active.** `Session::step` builds its runner with `pos0: 0`
  although the canvas starts at absolute position `prompt_len`; the sliding
  window is still correct only because `window` (1024) exceeds the canvas and
  prompt lengths in use. Likewise the port windows the canvas where the engine
  windows only the prompt (`t_lo = kv_len - (window-1)`, canvas always
  visible), which the same sizes keep inert. Both bite at canvas > window.

  **Fixed: step 1 self-conditions on the canvas embedding.** The SC MLP runs
  on EVERY step; only its input changes. From step 2 on it is the soft
  embedding of the previous step's logits, and on step 1 -- no previous
  prediction -- the engine seeds it with the canvas embeddings themselves,
  reading the initial canvas as the step-0 prediction. The port skipped the
  MLP outright, which is the SC=0 case the engine's own comment on that branch
  records as degenerate ("cold-start empty reply"). `forward_sc` and
  `Session::self_condition` now both take the signal as a parameter, and
  `tests/step_first.rs` pins the row to the engine's PRODUCTION value.

  This moved the preamble onto the engine (cos 0.999988, from a row that was a
  different computation before) and moved step-1 entropy 11.10 -> 10.37. It did
  NOT fix generation: the logits still saturate and the canvas still does not
  sharpen, so the SC branch was a real divergence but not the whole one.

  The earlier reading came from `encode_step_preamble`, which production does
  not run on step 1 -- see the engine fix below. Every port conclusion drawn
  against a `step-*-dump` before that fix is unverified and worth re-deriving.

  **Open: the canvas still does not sharpen, and the failure is absolute.**
  A canvas row's logits come out around 144 raw, so the 30.0 softcap saturates
  every one of them (top sixteen post-cap all 29.99) and the row is flat. The
  engine's same row spans 18 to 28.4. That is not "a different trajectory" --
  it is a broken distribution, and it is what starves the accept rule.

  What is measured, all against the now-production-faithful engine probes on
  the same prompt, seed and canvas (`DGQCUDA_LAYER_DUMP` / `DGQCUDA_ATTN_DUMP`
  / `DGQCUDA_ROUTE_DUMP` vs `step-layer-probe` / `step-attn-dump` /
  `step-moe-route-dump`):

    preamble         cos 0.999988  (the step-1 SC fix; row 1 agrees, 0.999980)
    layer 0 attention cos 0.999984 -- exact, carries its input through
    layer 0 whole     cos 0.999602 -- the error enters in the FFN/MoE half
    layer 0 moe_out   cos 0.9999694, l2 ratio 1.0214 (row 0)
    layer 29          cos 0.183
    final norm        cos 0.175, l2 ratio 6.62

  Ruled out. The attention kernel is exact at the real full-attention geometry
  (head_dim 512, 2 KV heads, 276 keys) and across tile boundaries, so the
  amplification at layer 5 is a real input error, not a kernel bug -- that also
  closes the multi-tile fixture gap this file used to list. Routing is 99.1%
  identical (2029 of 2048, 68 experts both sides), so no wrong expert index.
  The MoE difference is a 2.1% SCALE with the direction intact, which is the
  same size as the engine's own GPU-vs-CPU-oracle MoE spread (rel_l2 0.0141,
  printed by the route dump) -- f32 against the engine's bf16 arithmetic, not a
  formula error: both routers compute `softmax(top_k) * per_expert_scale` with
  no renormalization, and both gate with `gelu_tanh`.

  The amplification is measured, not assumed. Two identical port runs (same
  seed, same binary) diverge by 6e-6 at layer 1 and 1.7e-3 by layer 29 -- a
  gain of about 1.2x per layer, ~300x over the stack. The port's 0.49% relative
  difference at the preamble, amplified 260x, is the observed 128% at layer 29.
  So per-layer divergence from the engine is EXPECTED at f32-vs-bf16 and bit
  parity is unattainable without matching the engine's arithmetic. Do not spend
  another session bisecting layers for a single wrong kernel; the trace is
  consistent with amplification everywhere.

  What that does NOT explain is saturation. A different-but-valid trajectory
  should still produce a sane logit row, and the port's prompt-path logits do
  (top 27.4, clear of the cap) while its canvas rows do not. The open question
  is therefore narrower than "why does the port differ": why does the canvas
  hidden land where `model.decoder.norm.weight` (l2 4746, mean 29.5, max 588)
  blows it up 6.6x, when the engine's lands on the small-weight coordinates.
  It is pathological in an absolute sense, and the shape is specific. The
  port's canvas row after the final norm carries 79% of its energy in ONE
  coordinate (216): l2 789.4, max 700.3. Neither control looks like that --
  the port's own last PROMPT row is l2 371.8 / max 105.4 (8%), and the engine's
  canvas row is l2 119.2 / max 36.0 (9%). Coordinate 216 holds -7.29 in the
  port's layer-29 hidden against the engine's -0.20, and the final norm's
  weight there (36.25, not even its largest) turns 19.3 sigma into 700. A
  normed row dominated by one coordinate makes every logit a multiple of one
  embedding column, which is why they are all large and all saturate.

  Where it happens is the last four layers. Tracing max_abs down the stack, the
  engine CONTRACTS hard at the end -- 10.9, 5.8, 2.8 at layers 26, 27, 28 --
  while the port grows: 10.7, 11.5, 18.8. Coordinate 216 does the same, engine
  1.04 -> 0.30 -> -0.20 against the port -7.9 -> -12.5 -> -7.3. The engine's
  late-layer row is flat (max/rms 4.8 at layer 28), the port's is spiky (17.5).
  Cos is already 0.72 by layer 26, so the failure to contract may be a
  consequence rather than the cause -- but it is where a merely-different
  trajectory turns into a broken one, and it is the first thing to look at that
  is NOT explained by the amplification above. Layers 26-29 are the place to
  put the next probe, not layer 3.

  **Open: the port is not deterministic.** Three identical runs give different
  bytes. `dgq_moe_scatter` accumulates each token's experts with `atomicAdd`,
  which is correct but order-dependent, so the sum varies run to run: layer 0
  is bit-identical, layer 1 differs by 6e-6, layer 29 by 1.7e-3. Harmless for
  the current bug (cos 0.9999997) but a blocker on its own terms -- this
  project gates on golden BYTE-identity, which a port with an order-dependent
  reduction can never pass. Fix by giving the scatter a deterministic order
  (accumulate per token over its k experts in index order) rather than racing
  slots into one address.

- **Model-gated tests treat a manifest-only pack as present.**
  `test_util::dgq_model_dir()` returns `Some` when `model.dgq.json` exists, so an
  interrupted pack download (manifest present, `model.dgq.bin` missing or a
  0-byte `.incomplete`) turns the ~25 graceful skips into ENOENT failures
  instead of skipping. Gate on the blob (or a complete-pack marker) too.
- **Move the engine GEMM family into `crates/dgemm`.** Stages 1-2 have
  landed: the f32 body (shared with nanogpt) and the decode-only subset of
  `src/dgq` (layout, bf16/fp4 codecs, q4/q6/q8/nvfp4 decode plus their CPU
  GEMM oracles) are in the crate, with the engine re-exporting them so no call
  site moved. Next: move
  `gemm_tunable` plus its bit-exact oracle twins (`gemm_block`,
  `gemm_block_stacked`, `gemm_block_grouped`) in verbatim -- gated by
  `bench-gemm --oracle` per-element bit-exactness across all production shapes,
  which needs the 19 GiB pack. Stacked/gather/arena fusion stay compile-time
  axis values on that body. Then the CUDA bodies, one per axis combination.

## Parked / speculative

- **Read the generation language off the model's own state.** Speculative,
  nobody is convinced; recorded because a cheap experiment can kill it.
  Cold-canvas probes establish a real linear "about to write code" axis (L6,
  code vs prose, survives four confounds) and a weaker rust-vs-python
  sub-axis; tool-emission is its own mode. The blocking unknown is whether the
  signal survives on a RESOLVED canvas at output time (one warm test showed it
  drop hard, but probed junk tail filler, not real generated code) — and it is
  blocked on the P5 `Refine` primitive / a controllable warm canvas. Cheapest
  kill-first order: `step-moe-route-dump` expert-histogram separation → linear
  probe with the two controls → only then any on-device work. Adjacent and
  more tractable: an on-device NUMERIC delimiter-balance prefix-scan detector
  (per-token delimiter-delta table + SC prefix scan) that produces the blend
  rate at zero trajectory risk.
- **Span handles / compact history.** Keep large code/tool bodies out of
  long-lived canonical KV as opaque handles that expand on demand and
  evaporate at finalize (tool-compact sibling; `Splice` in the message layer).
  Known constraints: attention only sees tokens in the active sequence (a
  handle grants nothing without expand); ring vs full-layer KV are parallel
  classes, not a promote-and-drop pipeline; invisible litter breaks prefix
  identity (handles must be canonical or deterministically rehydrated before
  `activate`; prefer content-hash ids over line numbers). Phases when revived:
  span store MVP → handle substitution + expand tool → client soft-expand →
  (only if evidence demands) sparse visible anchors.

## Headline v2 features

- **Vision tower** (SigLIP encoder + image splicing; ~2+ weeks; v2 headline).
- **E9 rotated experts** (near-bf16 fidelity within the 4-bit budget; prove
  with plain absmax q4 first).
- **E10 precision-decay KV** (value is 18–24 GB Macs / >262k, not 36 GB).
- **q6/q5 non-expert weights** — REVISIT: judged redundant with q8-KV-auto on
  36 GB, but q8 KV is broken, so that fallback does not currently exist
  ([[kv-quant-status]]).