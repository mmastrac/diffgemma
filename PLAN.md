# diffgemma-mps — plan

Open work only. Design + contract: **ARCHITECTURE.md** (incl. the token
pipeline and Negative Knowledge — check it before planning perf/quality
work). Working discipline + commands: **AGENTS.md**. User-facing scope,
quickstart, and benchmarks: **README.md**. Everything done lives in git
history — commit messages are the changelog. Do not re-track shipped work
here; when an item lands, REMOVE it from this file (keep only any residual
open sub-task).

## State (orientation, not history)

Engine is near-done. Wall-clock beats MLX-4bit on short/medium chat;
long-context prefill is at MLX parity across the range and long-context
decode is well ahead (dynamic top-k attention default-on both phases). See
README for the measured numbers. The **token pipeline is the core** (P0–P4
shipped; ARCHITECTURE Part III): ask/chat/serve are op-stream clients,
sessions are op-logged and bit-exactly replayable. Cross-turn KV reuse is
KV-reuse-first with tail-salvage routing. Gates are green (smoketest ×
{7,42,123}, golden 8/8, full suite). Remaining decode headroom is STEP COUNT
(E7 territory), not per-step attention.

## Token pipeline follow-ups

- **P5 Refine / canvas-edit primitives** (`Refine {mask|forced_ids}` —
  continue denoising the same uncommitted canvas). Quality-gated: census,
  multi-seed; the freeze lesson applies to reject-masks. Later: multi-conv
  absorption, `Reground` (idle re-prefill = lineage reset), lineage-drift
  gate. This is also the prerequisite for controlled canvas inspection /
  tree-sitter rerolling — "advance the canvas under control, then inspect"
  is not readily available today.
- **`--tool-repair` / `--tool-validate` default-ON decision** after the
  field trial (op-logs in the user's `--log-dir` are the evidence base;
  first organic session fired one repair that recovered cleanly).
- **Replay across serve restarts** — a second `{"meta":…}` line stops the
  replay today.
- **Grammar-aware `kept_len`** at the per-block layer (message layer owns
  it; today stop metadata is driver policy via block stats).
- **Non-streaming disconnect detection** (socket read-EOF polling on the
  connection thread; today only the final write notices).
- **Prefill chunks as cancellation points** (a 100k prefill runs to
  completion; `prefill_chunks*` needs a consistent partial-prefix story
  first).
- **Snapshot-restore-to-ring lever** — deep truncates past ring slack pay a
  full rebuild (~fresh prefill of the kept prefix; timing is printed). A KV
  snapshot restore path could make deep rewinds cheap; measure demand from
  field op-logs first.
- **serve ops.jsonl is no longer token-level replayable** — the registry op
  format (activate/generate/finalize summaries) is skipped by `replay`
  ("unknown op shape"). Either teach `replay` the registry format (activate
  carries the full prompt token array) or log a parallel token-level stream.

## Channel hygiene (remaining)

Special-token injection, quote-aware `strip_thinking`, and
the gated channel-unaware stop-scan all shipped (see
[[channel-hygiene-masked-scanner]], [[special-token-injection-guard]]).
Remaining:

- **Dual thought-splitters** — one walk implementation. Both splitters now
  share the marker vocabulary and the masked literal-region rule, and the
  CHANNEL MISMATCH tripwire + salvage covers the residual disagreement
  window; the open work is collapsing the mapper's string-scanner path and
  the engine walk into a single implementation.
- **`channel_id` None fallback hardening**.
- **Thinking-flag flip silently loses KV reuse** — add a log line.

## Message-layer designs (user-directed, not started)

- **Interleaving-restoration blob**: any assistant-turn interleaving not
  representable in OpenAI format (call/prose/call/prose ordering) must
  round-trip via an opaque blob so re-prefill restores the exact KV
  ordering. Tool calls ONLY — thinking must never return to KV re-prefill.
  Needs per-client verification that unknown fields are echoed. (≈ OpenAI's
  encrypted reasoning items; it is also the cross-restart/cross-client
  hardening of the thinking-persistence option below.)
- **Tool-call triage via internal re-prompt**: on questionable calls, ask
  the model WHICH call(s) to keep, then splice so context retains only the
  chosen call. The evaporating-draft choreography is the substrate; the
  validator's retry-on-malformed is the degenerate case (keep zero).
- **Narrate-instead-of-act / act-without-narrating policy**: residual model
  warts — a turn ends after announcing a write with no call emitted, or a
  whole session is tool_calls-only (the amnesia driver). A triage-layer
  policy, not a serve defect: push toward "one line of visible narration per
  tool turn".
- **Confident-miscount loop breaker** (same policy family): repetition
  arithmetic can commit a wrong count at p_max ≈ 1.0, then spiral on
  self-contradiction retries. Mitigation is a message/triage-layer breaker
  that ties toward observed tool output after N failed reconciliations;
  confidence gating is blind here by construction.
- **Thinking persistence across tool turns (design option, NOT decided)**:
  all three frontier labs preserve full reasoning across the tool-calling
  loop and drop it at user-turn boundaries; we sit at the DeepSeek-R1 end
  (strip everything, every turn), and the cost is measured (a tool session
  spent 32.7 s re-deriving state it had already reasoned out). Because the
  canonical KV is SERVER-OWNED we could defer thought-stripping to the TASK
  boundary with zero client cooperation — keep thinking in canonical KV
  across consecutive tool-call turns, strip only at a real user message.
  Costs to weigh: ring pressure (thoughts × tool hops vs DGQ_KV_RING=4096
  and the salvage window) and larger canonical-vs-client divergence mid-task.
  Standing user directive until revisited: thinking never reaches KV
  RE-PREFILL from the client side.

## Quality track

- **`DGQ_COMMIT_CONF_HARD` (unconditional p_max floor) — stays OFF, decision
  parked.** The dup tier (`DGQ_COMMIT_CONF_TRIM`, conjunctive at τ=0.9) is
  shipped default-ON; the hard tier is split out and off. It is the tier that
  kills the insertion/omission class but it fires ~4× as often, costs a few
  percent of steps, and tips `transformer` one step over its convergence
  budget (a budget ratcheted on a non-trimming baseline). Blocked on a metric
  mismatch, not on the tier being bad: no available battery both reaches the
  sub-0.5 trigger AND carries a quality signal — grounded retrieval never
  goes below ~0.73, and the wart proxy (`contested_per_1k`) does not predict
  executable correctness ([[wart-proxy-doesnt-predict-quality]],
  [[code-errors-are-confident]]). To settle it, `soft` needs long-form
  (multi-sentence) answers past the 16-row floor; the `doc_tokens` field is
  already in the fixture schema.
- **`DGQ_PREFIX_EXIT` early block commit (landed default OFF; quality-safe,
  not a reliable speed lever)**: NEXT is the quality-mode experiment —
  aggressive exits at matched TOKEN budgets, judged on census multi-seed +
  strain tool-arg typo rates (the "commit-when-stable ≈ more-causal
  factorization ≈ fewer independence violations" hypothesis; VSB
  arXiv:2604.23994 reports +4–10% from the trained analog). Trajectory-
  affecting → golden re-bless + live A/B (never a replay sim,
  [[trajectory-feedback-sim-bias]]).
- **E16 token fusion / KV merging** (IN PROGRESS): the only unexplored
  long-context denoise SPEED lever (cuts token count, not bytes). Oracle so
  far: gist-preserving/verbatim-lossy; residual-gated r=2 ≈ control quality
  but 1.4× (under bar). Next: min-pairwise/outlier gates, mass keep-lists,
  multi-seed, non-English. MUST gate on the doc-QA ladder, not needles.
  See [[token-fusion-e16]].
- **E3 canvas shrink near max_tokens**: close divergence #5 (MLX shrinks to
  max(remaining, 64)); minor tail win; trajectory-affecting → multi-seed gate.
- **MLX matched-canvas dig on the preserved collapse trajectory** — can
  MLX's sampler survive the same conditioning? Artifacts in-repo:
  `debug/strain_battery/collapse_seed42/` (ops.jsonl + serve log),
  `debug/strain_battery/prompts/` (matched clean/collapse pair);
  `debug/strain_battery/battery.py` is the harness.

## Census / evaluation

`census` (arms × batteries × gates, `src/commands/census.rs`) has `smoke`,
`longctx`, `programmatic`, and `soft`.

- **Quantify the convention-blend rate broadly.** The delimiter checker
  (below) measures it per-battery (~9.3% of judgeable `programmatic` blocks);
  run it across convention-ambiguous constructs at scale to publish a rate
  rather than specimens. Everything needed is in `DGQ_TRACE_PMAX_JSONL`
  traces / the checker — GPU only for generating more samples.
- **`scan_trace` answer-region fix (optional, low priority).** Mirroring the
  trim's `(MIN_CONF_KEEP..region_end)` rule would make `contested_per_1k`
  mean what its name implies. Arm ordering survives the correction, so this
  is legibility, not correctness — but it still moves a published number.
- **Method rule to keep enforcing** (earned, not open work): a battery can
  only evaluate a lever whose trigger its outputs actually produce — check
  `trims > 0` in the treatment arm before reading any arm comparison; judge
  by the multi-seed aggregate, never one seed; use PROBE-level counts for arm
  comparisons, case-level only for diagnosing what failed. See [[census-command]].

## Correctness debt

- **Exact-prefix repeat misses reuse**: re-POSTing an identical request paid
  a `truncate_kv_to` ring rebuild (~39 s) then re-prefilled everything with
  `reused 0`. Should be ~100% reuse (KV-reuse-first). Read
  `route()`/`reset_kv_unless_extends` for the delta==0 edge.
- **`rollback_to`'s "restores the conversation" contract is still false** for
  its only production caller (residue from the reuse-guard fix).
- **Inline flag validation gap**: the ~29 `DGQ_*` flags still parsed inline
  via bespoke `var(...)` chains (enums, u32 ranges, paths) silently swallow
  bad values. The shared checked helpers exist ([[flags-config]]); convert
  the inline ones as they are touched.
- **Long-ctx re-validation debt**: re-run needle 33k/105k and the 100k
  field-incident repro on the uncapped fast path.
- **Tier-1 attention fixture below the worst tile**: `full_grp8_hd512_fixture`
  (canvas=16, t_total=44) vs E17/E20's BM=BN=64 — every parity test is
  single-tile; `topk_k128_matches_cpu` claims k>64 coverage on 44 keys. Add a
  canvas≥65 / kv≥65 full-layer fixture.
- **Missing CPU twins**: `kv/unpack_encoder_kv` and `kv/kv_f32_side_hydrate`
  (~40 lines each).
- **tool-compact sizing closure thinking-flag**: `render_prompt` (the one
  render+encode seam) preserves the sizing closure's `thinking: false` where
  siblings pass the request flag — intent still undecided (NOTE at the call
  site).
- **Stale "prefill-only" code docs** on the shipped decode path:
  `step_kernel.rs` `attn_gemm`/`attn_topk` field docs and
  `attention_gemm/mod.rs` module doc still say "denoise keeps
  attention_mma_full".

## Perf re-test backlog

- **MoE weight-stationary `DGQ_MOE_PREFILL_BM` honest re-measure never
  concluded** (post cache-collision fix; early 3-trial signal: correct bm=64
  is SLOWER). Run the honest sweep or close the item.
- **MoE adaptive-M / partial-tile padding at M=1024** — memory predicted it
  activates when per-TG goes compute-bound (this regime); the live kernel
  comment still calls padding "immaterial" (a stale denoise conclusion).
- **E5 QK-ILP2 chain-split — PENDING, inconclusive.** Splitting the 32-deep
  serial QK MMA chain in `attention_mma_full` into two independent 16-deep
  chains (FC31 `DGQ_ATTN_MMA_FULL_QK_ILP2`, default OFF) should halve QK
  dependency depth. Every A/B so far was INVALID: with E17/top-k default-on,
  full layers route to `attention_gemm`, so `attention_mma_full`+ILP2 is
  inert on the dominant attention cost. Test it as a categorical axis (paired
  with `gemm_attn` on/off) in the holistic prefill BO
  (`tune_prefill_attn.py --proxy`) — a single-axis A/B at default settings
  cannot see a lever that only activates on the off-default path.

## Code structure / cleanup

- **Clippy residue**: ~60 warnings (arg-count/type-width allowed crate-level);
  opportunistic, not a campaign.
- **cli.rs parser structure**: usage string is true, but `parse_cli` still
  uses ~80 shared mutable locals (cross-wiring hazard); a per-command
  arg-struct redesign remains open.
- **Oracle sampler tests missing**: `sample_from_probs_rows` (the one worth a
  fixture), `scale_logits`, `logit_softcapping`.
- **`metal/oracle/` quarantine**: audit done; the safe-to-quarantine set has
  zero production callers, and the MIXED files (`decoder.rs`,
  `decoder_layer.rs`, `memwatch.rs`) need a prod/validation split before
  moving. Move pending user sign-off.

## Output-mode classification

Per-BLOCK output-mode classification SHIPS and works: `fit-token-probe`
(refit locally per checkpoint) + `DGQ_TOKEN_CLASS=<probe.json>` classify a
block at its first forward, while the canvas is still seeded noise (the
regime the probe is fitted in). Per-TOKEN classification of committed tokens
does NOT work and is not one fix away — a committed position's hidden state
is dominated by its own token identity, so mode is a weak residual (in-regime
LOO 0.641 vs cold 1.000). See [[token-mode-probe]].

Remaining (block-level only): colour the per-block labels on
`ChatEvent::BlockCommit` in `chat_ui`. Perf obligations are HARD requirements
([[perf-timing-methodology]]): exactly zero cost when OFF (capture not
encoded at all — flag parsed once into `RuntimeConfig`), and MEASURED cost
when ON (~1.44 MiB device copy per block, `bench-step-kernel --profile-steps`
adjacent A/B). Do not spend a capture kernel on per-token without new
evidence that an earlier layer clears 0.641 by a lot.

## Structural repair (delimiter checker → repair)

The delimiter checker SHIPS observational: `src/delimiter.rs` behind
`DGQ_DELIM_CHECK=1` / `DGQ_DELIM_CHECK_JSONL`, hooked at `commit_block`,
read-only (golden needs no re-bless). Measured: `quote_region` + `bracket`
earn their place (2/2 precision), `quote_line` does not (2/8 — cross-quoting
false positives, pinned as a test); prose blocks skip content checks. It is
the first thing to detect the convention-blend class, which no confidence
threshold reaches ([[delimiter-parity-checker]], [[code-errors-are-confident]]).

Open:
- **Repair is not built.** Trigger is well-defined (~1 in 11 judgeable code
  blocks, all `quote_region`/`bracket`, all terminated). `ToolRepairStage` +
  `KvCheckpoint`/`rollback_to` (currently dead-code) are the substrate.
  Regeneration must change something — reuse shrink-on-retry (256→128→64).
  The model is the WRONG adjudicator for its own defect (it was certain when
  it emitted the blend), so gate the CHECKER by block mode rather than asking:
  prose → skip, code → regenerate without asking. Trajectory-affecting →
  golden re-bless + live A/B.
- **Re-validate the probe's block labels** before building on them: PLAN's
  curated set showed 5/5, but a `smoke` run mislabelled 17/19 (the language
  SNIFFER, not the probe, carries the prose gate today).

## Parked / speculative

- **Read the generation language off the model's own state.** Speculative,
  nobody is convinced; recorded because a cheap experiment can kill it. Cold-
  canvas probes establish a real linear "about to write code" axis (L6, code
  vs prose, survives four confounds) and a weaker rust-vs-python sub-axis;
  tool-emission is its own mode. The blocking unknown is whether the signal
  survives on a RESOLVED canvas at output time (one warm test showed it drop
  hard, but probed junk tail filler, not real generated code) — and it is
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

## v1 productization

- **Benchmark page** with the MLX methodology (README carries the headline
  numbers; a fuller page with the harness invocations is still to write).
- **Release tagging + `--version`** with a `.dgq` manifest-version gate.
- **CI completion**: nightly model-gated tiers are scaffolded; wire fully
  (smoketest + golden + longctx + perf floors), plus a weekly multi-seed
  aggregate + census.
- **Broader eval**: the 17-prompt gate is sensitive but narrow; add a
  ~100-prompt adherence set, weekly, non-blocking.

## v1 acceptance (unmet items only)

- Install-to-first-token < 30 min VERIFIED on a clean 36 GB Mac (README
  documents the path; the clean-machine timing run is not yet done).
- Perf floors regression-gated in CI (step ≤ 1.1 s chat lengths; 33k prefill
  ≤ 140 s; 100k ≤ 16 min prefill, ≤ 5 s/step).

## v2 parking lot

- **Vision tower** (SigLIP encoder + image splicing; ~2+ weeks; v2 headline).
- **E9 rotated experts** (near-bf16 fidelity within the 4-bit budget; prove
  with plain absmax q4 first).
- **E10 precision-decay KV** (value is 18–24 GB Macs / >262k, not 36 GB).
- **q6/q5 non-expert weights** — REVISIT: this memory lever was judged
  redundant with q8-KV-auto on 36 GB, but q8 KV is broken and ungated
  ([[q8-kv-broken]]), so that fallback does not currently exist.
- **E8 rotated/un-RoPE'd KV** — parked on value; revive only if q4-KV becomes
  necessary (see Negative Knowledge).
- **E22 block-granular pre-QK top-k** — KILLED on the mass oracle; revival
  path = E16 token compaction, not kernels. Dumps + analysis kept:
  `step-attn-qk-dump` + `python/scripts/e22_block_mass.py`.

## Risks

- **Single-machine evidence** — every number is one M3 Pro; recruit at least
  one other M-series config before publishing claims (SLC-locality physics
  may differ on M1).
- **Upstream drift** — the MLX head-to-head cites their current 4bit; refresh
  before publishing benchmarks.
- **Gate breadth** — 17 prompts; see the broader-eval item.
