# diffgemma-mps — plan

Open work only. Design + contract: **ARCHITECTURE.md** (incl. the token
pipeline and Negative Knowledge — check it before planning perf/quality
work). Working discipline + commands: **AGENTS.md**. Everything done lives
in git history — commit messages are the changelog; do not re-track
shipped work here.

## State (for orientation, not history)

Engine is near-done: wall-clock beats MLX-4bit on short/medium chat;
long-context PREFILL is at MLX parity across the range (30k 1.04×, 100k
0.975×) and long-context DECODE is ~3.9× ahead (dynamic top-k attention
default-on both phases). Needle-exact to 121k, doc-QA-grounded to 20.6k.
The **token pipeline is the core** (P0–P4 shipped 2026-07-17; see
ARCHITECTURE Part III): ask/chat/serve are op-stream clients, sessions are
op-logged and bit-exactly replayable. The OpenCode collapse is SOLVED
(defer default OFF) and `ToolRepairStage` turns invalid tool calls into
evaporating drafts. Cross-turn KV reuse for tool sessions ~99.8%
(KV-reuse-first canonical + tail-salvage routing). Gates green: smoketest
17/17 ×{7,42,123}, golden 8/8, suite 621/0. Remaining decode headroom is
STEP COUNT (E7), not per-step attention.

## Open items

### Token pipeline follow-ups

- **P5 Refine / canvas-edit primitives** (`Refine {mask|forced_ids}` —
  continue denoising the same uncommitted canvas). Quality-gated: census,
  multi-seed; the freeze lesson applies to reject-masks. Later: multi-conv
  absorption, `Reground` (idle re-prefill = lineage reset), lineage-drift
  gate.
- **`--tool-repair` default-ON decision** after the field trial (first
  organic session: one repair fired, recovered cleanly; op-logs in the
  user's `--log-dir` are the evidence base). Same decision pending for
  `--tool-validate`.
- **Replay across serve restarts** — a second `{"meta":…}` line stops the
  replay today.
- **Grammar-aware `kept_len`** at the per-block layer (message layer owns
  it; today stop metadata is driver policy via block stats).
- **Non-streaming disconnect detection** (socket read-EOF polling on the
  connection thread; today only the final write notices).
- **Prefill chunks as cancellation points** (a 100k prefill runs to
  completion; `prefill_chunks*` needs a consistent partial-prefix story
  first).
- **Snapshot-restore-to-ring lever** — deep truncates past ring slack pay
  a full rebuild (~fresh prefill of the kept prefix; timing now printed).
  A KV snapshot restore path could make deep rewinds cheap; measure demand
  from field op-logs first.

### Channel-hygiene fixes (survey 2026-07-17, unimplemented)

**Field incident 2026-07-17 late (regex_lite session, turn 14) — the dual-
splitter disagreement has a live casualty.** The model skipped the thought
ceremony and emitted a bare, WELL-FORMED edit call (no channel markers in
the reply at all; finish=stop, 170 tok). The wire mapper's rule is
"thinking mode ⇒ everything is reasoning until a `<channel|>` CLOSE id
appears" (`split()`, no open required — lenient because the model
sometimes emits the opener as text BPE, not the special id), so the whole
call was streamed as `reasoning_content` and the client got an EMPTY
message; the validator/repair side (string-based, explicit-span-driven)
correctly judged the call visible+valid → verdict Ok → repair rightly
silent. Verified by probe: every reply shape containing a `<|channel>`
opener flags for repair; only the bare-call shape passes Ok — and the turn
prompt was byte-clean (no leaked thought, correct ordering). So the loss
is OURS (misclassification), not the model's grammar. Candidate tactical
fix ahead of the full unification: in the mapper's
reasoning-until-close branch, if the committed region contains NO channel
open (id or text form) and parses as a complete tool call, classify it as
content — plus a worker-level loud log when mapper and validator disagree
about whether a reply contains a call.

Findings, in proposed fix order:

1. ~~**Special-token injection**~~ **FIXED (2026-07-17, session freebie).**
   Client text (message bodies, tool outputs, tool names/descriptions/keys/
   values) is wrapped in `CLIENT_GUARD` (U+E000) sentinels by
   `render_conversation_guarded`, and `Tokenizer::encode_prompt` refuses to
   special-match inside guarded ranges — a `<|tool_response>`/`<|turn>` in
   a file body or web page now encodes as PLAIN TEXT, not protocol tokens.
   Benign input is bit-identical to the old
   `encode_with_specials(render_conversation(…))` (compat-pinned; golden
   8/8 unchanged, same kv hash). Serve logs `neutralized N special-token
   literal(s)` when it fires. Live-verified: an injected fake `<|turn>user`
   boundary in a tool response was neutralized and the model treated it as
   content instead of structure. Tests: `encode_prompt_*` (tokenizer +
   tools). Follow-up: the same guard vocabulary is the substrate for the
   unified channel scanner (#4).
2. **`strip_thinking` is not quote-aware** — thought markers inside string
   literals/code fences get treated as channel boundaries.
3. **Channel-unaware stop-scan** — the stop scanner doesn't know which
   channel it's in; directly protects the repair loop.
4. **Dual thought-splitters** — mapper (id-based) vs strip (string-based)
   can disagree; unify on one scanner.
5. `channel_id` None fallback hardening; 6. thinking-flag flip silently
   loses KV reuse (add a log line).

Plan: #1+#2 together via one quote/special-aware sanitization/scanner
layer → #3 → #4 falls out of the same scanner.

### Message-layer designs (user-directed, not started)

- **Interleaving-restoration blob**: any assistant-turn interleaving not
  representable in OpenAI format (call/prose/call/prose ordering) must
  round-trip via an opaque blob so re-prefill restores the exact KV
  ordering. Tool calls ONLY — thinking must never return to KV re-prefill.
  Needs per-client verification that unknown fields are echoed.
- **Tool-call triage via internal re-prompt**: on questionable calls, ask
  the model WHICH call(s) to keep, then splice so context retains only the
  chosen call. The evaporating-draft choreography is the substrate; the
  validator's retry-on-malformed is the degenerate case (keep zero).
- **Narrate-instead-of-act policy**: residual model wart — turn ends after
  announcing a write with no call emitted; a triage-layer policy, not a
  serve defect. The inverse (act-without-narrating: every turn of the
  2026-07-17 regex_lite session was tool_calls-only) is the amnesia
  driver below — the policy should push toward "one line of visible
  narration per tool turn", not just "don't narrate without acting".
- **Thinking persistence across tool turns (design option, NOT decided —
  survey 2026-07-17):** all three frontier labs preserve FULL reasoning
  across the tool-calling loop and drop it at user-turn boundaries —
  OpenAI encrypted reasoning items round-tripped with tool outputs
  (docs: measurably better function-calling), Anthropic signed thinking
  blocks required back with `tool_result`, Gemini thought signatures.
  Nobody feeds back the display summaries; harness-level compaction
  (summarize-history-into-a-note) is the only summary-as-context pattern.
  We currently sit at the DeepSeek-R1 end (strip everything, every turn),
  and the cost is measured: turn 7 of the regex_lite session spent 32.7 s
  re-deriving state it had already reasoned out (prompt verified
  byte-clean — pure thought-evaporation amnesia, model recovered by
  reading its own edit call out of context). Because the canonical KV is
  SERVER-OWNED, we could do what the labs do with zero client
  cooperation: defer thought-stripping to the TASK boundary — keep
  thinking in canonical KV across consecutive tool-call turns (each
  `<|tool_response>` extend continues the same reasoning context), strip
  only when a real user message arrives or the turn truly ends. Within a
  task the client only appends tool responses, which the LCP routing
  already handles. Costs to weigh before committing: ring pressure
  (thoughts × tool hops accumulate against DGQ_KV_RING=4096 and the
  salvage window) and a larger canonical-vs-client divergence mid-task.
  The interleaving-restoration blob (above) is the cross-restart/
  cross-client hardening of the same idea (≈ OpenAI's encrypted items).
  User directive that still stands until revisited: thinking never
  reaches KV RE-PREFILL from the client side.

### Quality track (the current frontier)

- **Duplication micro-stutter — mechanism FOUND (E7 M0, 2026-07-18,
  commit 07ace48)**: `DGQ_TRACE_PMAX_JSONL` traced smoketest ×{7,42,123}
  + strain battery ×3 seeds. Every strain adjacent-dup commit (`",",
  `<|"|><|"|>`, doubled 8-space indent, `the the`) committed at final-step
  p_max 0.40–0.86 while 98% of non-dup commits sit ≥0.9. Mechanism: the
  MEAN-entropy stops fire while individual rows are unresolved (one
  1.9-nat row hides inside the 0.05×256 budget), and an unresolved row's
  argmax copies its neighbor (uncertain diffusion marginals at i and i+1
  look alike). Three observed sub-cases: late destabilization (answer
  region grows past a settled eos, stop fires mid-re-resolution),
  canvas-tail ambiguity (last rows unknowable at the block edge), and
  contested flat rows (never resolve in-block). Commit-level dups mostly
  get absorbed by downstream defenses before user-visible text; the tool-
  arg stutter typos are the survivors.
- **E7 confidence-threshold sampler: speed lever KILLED per pre-registered
  criterion** — M0 would-accept proxy measured 6.5% (smoke) / 1.2%
  (strain) marginal steps at τ=0.9 vs the 8–10% KILL line. The wart
  hypothesis half was CONFIRMED (see above) and is now pursued as
  commit-time confidence trim instead of a sampler swap.
- **Confidence trim `DGQ_COMMIT_CONF_TRIM` — A/B CLEAN, default-on
  decision pending**: final rule (2026-07-18) is CONJUNCTIVE and
  answer-region-scoped — trim at the first row that is both low-confidence
  (p_max < τ) AND argmax-duplicating a neighbor, scanning only before the
  first stop/pad/filler row (floor 16, 2× max_blocks headroom). p_max
  alone does NOT separate wart from art (benign creative soft rows sit at
  0.58–0.87 and judged fine; fixed-τ trim regressed smoketest 15-16/17 via
  step-budget churn; eos-padding runs are structural dups). τ-insensitive
  across 0.9–0.95. A/B at τ=0.9: smoketest 17/17 ×3 seeds with dup commits
  6→0 (one trim in 51 turns); strain battery dup commits 6→0 with trims at
  exactly the 3 known stutter sites, tool health/flood identical, +10.6%
  denoise on stutter-heavy turns; real OpenCode bug-fix run clean (7
  turns, correct edit, verified, zero trims). Remaining before default-on:
  census multi-seed + longctx.
- **Prefix-exit early block commit (user-proposed, simulated PROMISING —
  next speed lever)**: per step, if a head prefix (largest of active/2,
  /4, /8) has mean entropy < 0.05 while the tail churns (≥2×), and the
  head argmax is 2-step stable, commit the head and let the next block
  re-denoise the rest against real context. Offline simulation on the 70
  traced M0 blocks: fires on 29%, saves ~28% of denoise steps gross
  (extend cost ≈0.6 step vs ~7 saved per firing), head bit-identical to
  the eventual commit in 17/20 firings (other 3 differ by 1 token). Must
  run the conjunctive dup-scan on the exited head (prefix MEAN has the
  same tail-hiding flaw). Net savings need a live A/B (`DGQ_PREFIX_EXIT`,
  default off); gates: smoketest ×3, strain, census multi-seed.
- **MLX matched-canvas dig on the preserved collapse trajectory** — can
  MLX's sampler survive the same conditioning? Artifacts preserved in-repo:
  `debug/strain_battery/collapse_seed42/` (ops.jsonl + serve log),
  `debug/strain_battery/prompts/` (matched clean/collapse pair);
  `debug/strain_battery/battery.py` is the harness.
- **E16 token fusion / KV merging** (IN PROGRESS): the only unexplored
  long-context denoise SPEED lever (cuts token count, not bytes). Oracle so
  far: gist-preserving/verbatim-lossy; residual-gated r=2 ≈ control quality
  but 1.4× (under bar). Next: min-pairwise/outlier gates, mass keep-lists,
  multi-seed, non-English. MUST gate on the doc-QA ladder, not needles.
  E22's diffuse-background finding is the supporting evidence; E22's block
  summaries (centroids) are fusion candidates.
- **E3 canvas shrink near max_tokens**: close divergence #5 (MLX shrinks to
  max(remaining, 64)); minor tail win; trajectory-affecting → multi-seed
  gate.

### Correctness debt

- **`generate_with_session` ring-truncate duplicate (task #93, still
  open)**: `step_generate.rs` reuse path calls `prefill_chunks_from(reuse,
  delta)` with no `kv_truncate_needs_ring_rebuild` check (the fixed
  predicate guards `truncate_kv_to` only). serve is shielded by `route()`'s
  prefix guarantee; `chat` (raw-vs-sanitized divergence) and
  `run_summarize_pass` are NOT. Also `rollback_to`'s "restores the
  conversation" contract is still false for its only production caller.
- **Long-ctx re-validation debt**: re-run needle 33k/105k and the 100k
  field-incident repro on the uncapped fast path (post root-cause fix).
- **Tier-1 attention fixture below the worst tile**: `full_grp8_hd512_fixture`
  (canvas=16, t_total=44) vs E17/E20's BM=BN=64 — every parity test is
  single-tile; `topk_k128_matches_cpu` claims k>64 coverage on 44 keys.
  Add a canvas≥65 / kv≥65 full-layer fixture.
- **Missing CPU twins**: `kv/unpack_encoder_kv` and `kv/kv_f32_side_hydrate`
  (~40 lines each).
- **tool-compact sizing closure thinking-flag**: `render_prompt` (the one
  render+encode seam, landed 2026-07-18) preserves the sizing closure's
  `thinking: false` where siblings pass the request flag — intent still
  undecided (NOTE at the call site).
- **Stale "prefill-only" docs** on the shipped decode path:
  `step_kernel.rs` `attn_gemm`/`attn_topk` field docs and
  `attention_gemm/mod.rs` module doc still say "denoise keeps
  attention_mma_full".

### Perf re-test backlog

- **MoE weight-stationary `DGQ_MOE_PREFILL_BM` honest re-measure never
  concluded** (post cache-collision fix; early 3-trial signal: correct
  bm=64 is SLOWER). Run the honest sweep or close the item; the
  kv-adaptive ship plan is moot unless it wins.
- **MoE adaptive-M / partial-tile padding at M=1024** — memory predicted it
  activates when per-TG goes compute-bound (this regime); the live kernel
  comment still calls padding "immaterial" (a stale denoise conclusion).

### Code duplication / structure (2026-07-17 audit; burn-down 2026-07-18)

DONE 2026-07-18 (commit messages carry details): big-file splits
(pipeline/, step_generate/, tools/, server wire+log, diagnostics
probe/moe/bench, flags/, cli/, step_debug dumps, bench_gemm sparse,
step_kv fusion tests), server/step_kv/step_kernel directory folds,
attention-harness AttnRig dedup (~1,050 dup lines out; bench `causal`
now a real arg), encode_attn_gemm/topk merged into one decomp driver
(shared QK stage), step_kernel exec.rs carve (StepEnc+StepRuntime+build
as one child; golden 8/8 + suite gated), cli usage truth pass, flags
EnvReader Default-dedup + dead accessor, worker render_prompt seam,
membudget permits. Still open below.

- **Clippy residue**: ~60 warnings (arg-count/type-width allowed
  crate-level); opportunistic, not a campaign.
- **cli.rs parser structure**: usage string is now true, but parse_cli
  still uses ~80 shared mutable locals (cross-wiring hazard); a
  per-command arg-struct redesign remains open.
- **Oracle sampler tests missing**: `sample_from_probs_rows` (the one worth
  a fixture), `scale_logits`, `logit_softcapping`.
- **`metal/oracle/` quarantine**: audit DONE (2026-07-17). Confirmed:
  `engine`/`decoder`/`kv_cache`/`decoder_layer`/`decoder_attention`/
  `weights`/`moe`/`router` are production via the prefill path
  (`MonolithicEncoderCache` owns `GpuDecoderEngine` and drives
  `forward_encoder_prefill_resident`). Safe-to-quarantine set (zero
  production callers): `step_m0`, `step_kernel_diagnostics`,
  `step_kv_audits`, `bench_gemm`, `probe`, `step_attn_dump`,
  `step_logits_dump`, `step_moe_{,route_,batched_pin_,single_}dump`,
  `step_preamble_dump`. MIXED, split before moving: `decoder.rs`
  (`load_weight_cache*`/`GpuDecoderScratch` prod vs `forward`/
  `forward_inner` validation-only — the validation forward is what pulls
  in `lm_head.rs` and the `sampler.rs` GPU path), `decoder_layer.rs`
  (prefill fns prod vs `forward_decoder*` validation),
  `memwatch.rs` (`physical_ram_bytes` prod). Move pending user sign-off.

### Concept — span handles / compact history (parked)

Keep large code/tool bodies out of long-lived canonical KV as opaque
handles that expand on demand and evaporate at finalize (tool-compact
sibling; `Splice` choreography in the message layer). The collapse
diagnosis that blocked it is DONE. Constraints already known: attention
only sees tokens in the active sequence (a handle grants nothing without
expand); ring vs full-layer KV are parallel classes, not a
promote-and-drop pipeline; invisible litter breaks prefix identity —
handles must be canonical or deterministically rehydrated before
`activate`; prefer content-hash ids over line numbers. Phases when
revived: span store MVP → handle substitution + expand tool → client
soft-expand → (only if evidence demands) sparse visible anchors.

## v1 productization

- README quickstart (none exists), fetch/quantize UX one-liner, benchmark
  page with the MLX methodology, release tagging + `--version` with `.dgq`
  manifest-version gate.
- CI completion: nightly model-gated tiers are scaffolded; wire fully
  (smoketest + golden + longctx + perf floors), weekly multi-seed
  aggregate + census.
- Broader eval: the 17-prompt gate is sensitive but narrow; add a
  ~100-prompt adherence set, weekly, non-blocking.

## v1 acceptance (unmet items only)

- Install-to-first-token < 30 min documented on a clean 36 GB Mac (README).
- Perf floors regression-gated in CI (step ≤ 1.1 s chat lengths; 33k prefill
  ≤ 140 s; 100k ≤ 16 min prefill, ≤ 5 s/step).
- Scope statement in README: text-only v1, 36 GB minimum RAM.

## v2 parking lot

- **Vision tower** (SigLIP encoder + image splicing; ~2+ weeks; v2 headline)
- **E9 rotated experts** (near-bf16 fidelity within the 4-bit budget;
  prove with plain absmax q4 first)
- **E10 precision-decay KV** (value is 18-24 GB Macs / >262k, not 36 GB)
- **q6/q5 non-expert weights** (memory lever redundant with q8-KV-auto on
  36 GB)
- **E8 rotated/un-RoPE'd KV** — parked on value; revive only if q4-KV
  becomes necessary (see Negative Knowledge)
- **E22 block-granular pre-QK top-k** — KILLED on the mass oracle
  (2026-07-16); revival path = E16 token compaction, not kernels. Dumps +
  analysis kept: `step-attn-qk-dump` + `python/scripts/e22_block_mass.py`.

## Risks

- **Single-machine evidence** — every number is one M3 Pro; recruit at least
  one other M-series config before publishing claims (SLC-locality physics
  may differ on M1).
- **Upstream drift** — MLX head-to-head cites their current 4bit; refresh
  before publishing benchmarks.
- **Gate breadth** — 17 prompts; see broader-eval item.
