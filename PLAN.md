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
17/17 ×{7,42,123}, golden 8/8, suite 603/0. Remaining decode headroom is
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

Findings, in proposed fix order:

1. **Special-token injection**: client-supplied text is encoded with
   `encode_with_specials` at ~6 render sites — a user message containing
   `<|tool_response>` etc. becomes real protocol tokens.
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
  serve defect.

### Quality track (the current frontier)

- **Duplication micro-stutter is the core quality signature**: `the the
  the`, `},{{`, `<|"|><|"|>`, `(".` → `....` — one family at all scales;
  the collapse was its amplified endpoint, and the surviving field bugs
  (stutter typos inside tool args) are its small end. E7 is the open
  lever aimed at it.
- **E7 confidence-threshold sampler**: accept canvas positions at top-token
  confidence ≥ τ instead of the entropy budget (MLX
  `diffusion_threshold=0.9`). Semantics pinned: p_max from the
  distributions the entropy reduction already reads; floor = schedule's
  per-step count (hybrid), literal-MLX as parity mode; measure WITH
  early-stop on, report MARGINAL steps. Wart hypothesis: threshold refuses
  the flat-distribution creative-tail rows that budget-accept commits. M0
  (zero behavior change): p_max histograms + would-accept counts at
  τ∈{0.8,0.9,0.95} across the 17-prompt gate; KILL if predicted marginal
  savings <8-10%. M1 `DGQ_ACCEPT_THRESHOLD` + matched-canvas multi-seed
  A/B. M2 gates: smoketest ×{7,42,123}, census multi-seed (decision gate),
  longctx; golden negative expected.
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
- **server_worker render inconsistency**: `encode_with_specials(render_conversation(…))`
  repeats at ~6 sites; the tool-compact sizing closure passes
  `thinking: false` where siblings pass the request flag. Collapse into one
  `render_prompt` helper (also the natural seam for channel-hygiene #1).
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

### Code duplication / structure (2026-07-17 audit)

- **Attention GPU/bench harnesses**: ~1,300 duplicated lines across
  `attention_topk`/`attention_gemm`/`attention_flash` `gpu()`/`bench_*()`;
  ~600-800 removable via a shared Fixture→buffers builder + dispatch
  closure + min-of-rounds timer. Drift hazard: benches hard-code
  `causal: 1` while the oracle takes it as an arg.
- **`encode_attn_gemm` vs `encode_attn_topk`** are ~80% identical (~120
  lines): shared QK stage + head-loop scaffold, stage-2/3 closure.
- **`step_kernel.rs` core carve (deferred, own gated task)**: StepEnc (152
  fns) is struct-literal-constructed by StepRuntime and StepPipelines is a
  50-field struct read throughout — a peer-module split forces ~50-100
  promotions. If done: move StepEnc+StepRuntime+build as ONE execution
  child. Full golden+suite gate mandatory.
- **`metal/oracle/` quarantine**: needs an accurate oracle-vs-production
  audit first — `engine`/`decoder`/`kv_cache` are production
  (encoder-prefill path), only a subset is genuinely oracle-only.
- **Clippy residue**: 60 warnings remain post-cleanup (arg-count/
  type-width allowed crate-level); opportunistic, not a campaign.
- **cli.rs usage string**: six advertised commands don't exist,
  `--compare-cpu` never parsed, ~30 real commands undocumented; shared
  mutable parse vars invite cross-wiring.
- **flags.rs**: test-only `impl Default` duplicates env defaults (can go
  stale silently); dead `attn_topk()` accessor.
- **Oracle sampler tests missing**: `sample_from_probs_rows` (the one worth
  a fixture), `scale_logits`, `logit_softcapping`.

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
