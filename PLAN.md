# diffgemma-mps — plan

Open work only. Design + contract: **ARCHITECTURE.md** (incl. Negative
Knowledge — check it before planning perf/quality work). Working
discipline + commands: **AGENTS.md**. Everything done lives in git history.

## State (for orientation, not history)

Engine is near-done: wall-clock beats MLX-4bit on short/medium chat, and —
since 2026-07-16, with dynamic top-k attention default-on for BOTH
prefill (`DGQ_ATTN_TOPK`) and decode (`DGQ_ATTN_TOPK_DECODE`) — **long-
context PREFILL is at MLX parity across the range** (30k: 343 vs 355 tok/s;
100k: 242 vs 248) and **long-context DECODE is ~3.9× ahead of MLX**
(100k: 4.57 vs 1.16 tok/s; step 4.42→1.84 s — the full-layer denoise
attention that was mma_full-issue-bound now runs the E20 GEMM-decomp
top-k, 3.1× isolated). Denoise convergence parity-class; long context
needle-exact to 121k, doc-QA-grounded to 20.6k; OpenAI-compatible `serve`
with multi-conversation KV reuse + tool calling; gates green (smoketest
17/17 ×{7,42,123}, golden 8/8, suite green, census signed off). Remaining
decode headroom is STEP COUNT (E7), not per-step attention.

## Open items

### Token pipeline — the new core (designed 2026-07-17, implementation starting)

Serialized op-stream architecture around the model thread; everything else
(serve, chat, tool handling, span handles) becomes a client. Motivated
directly by the OpenCode-collapse investigation
(`debug/opencode_collapse/`): serve's state mutations are scattered
side-effects today — un-auditable, un-replayable, and append-only context
turned out to be the collapse trigger.

**Core (pipeline thread, ids only — never strings):**

- One thread owns the GPU + KV; input = serialized ops, output = events.
- Ops: `Extend(ids)`, `GenerateBlock(params) -> Proposal`,
  `Refine {mask|forced_ids, budget}` (continue denoising the SAME
  uncommitted canvas), `Commit(kept_len)`, `Discard {reroll}`,
  `Rewind(kv_id)`, `Snapshot/Restore`, `Cancel`, later `Reground`
  (idle re-prefill of the canonical stream = lineage reset).
- `kv_id = (epoch, position)`; lineage-invalidating ops bump the epoch;
  rewind to a stale-epoch id fails loudly (the turn-6 drift class becomes
  type-checked). Rewind is token-granular; cheap within ring slack, else
  the ring-rebuild fallback (`rollback_to` already implements both) — the
  pipeline reports which one you paid.
- Events: `Proposal {ids, stats}`, `Committed {ids, kv_id, stats}`,
  `Strained`, `Aborted`, `Rewound`; drafts on a lossy side channel.
  Cancellation: epoch bump checked between denoise steps / prefill
  chunks; purges stale queued ops; ack reports last committed kv_id.
- Per-block protocol (decided): the upper layer inspects every proposal
  before commit — partial commit (`kept_len`) covers "good up to X, bad
  tail"; commit-then-regenerate covers tail replacement. Commit re-encodes
  the kept prefix causally, so refine/reject cycles leave no lineage
  residue.

**Message layer (owns all text/parsing/policy):**

- Channel/tool-grammar parsing, canonicalization (thought-strip becomes
  explicit Rewind+Extend, not a buried finalize), validator FSM:
  propose -> validate -> refine/splice -> commit; on invalid content the
  policy is immediate re-`GenerateBlock` after a surgical
  `Splice(range -> replacement)` (= Rewind + Extend(replacement +
  re-encoded tail)) that REMOVES failed attempts from context — the
  anti-collapse move; transient hints extend in and evaporate the same
  way. The commit guard (`DGQ_BLOCK_COMMIT_MAX_ENT`) folds in as one
  validator policy. Tool-compact and span handles become Splice clients.

**Op-log:** every op + seed durably logged; any session replays
bit-exactly (the oc_sim/seed-42 reproducibility, but by construction);
field failures become golden-style artifacts; KV lineage becomes
measurable (replay with fresh-prefill-per-op and diff).

**Standing gate — rewind byte-consistency:** seeded
generate -> rewind -> generate -> rewind loops must (a) restore the KV
bytes exactly after every rewind (FNV of the valid snapshot) and (b)
regenerate bit-identical proposals at the same seed. Any lineage residue
fails one of the two. Runs below the sliding-ring wrap in Tier 2; a
wrap-crossing variant exercises the rebuild path.

**Phases:** P0 op/event types + pipeline thread wrapping the existing
session (Extend/Generate/Rewind), rewind gate green, nothing rerouted.
P1 epochs + Cancel; reroute ask/chat. P2 per-block protocol (decomposes
`generate_with_session` along op boundaries). P3 message layer v1 + serve
reroute (client disconnect cancels — today serve generates 4k tokens into
a dead socket). P4 Splice + tool-grammar validator + op-log persistence
and replay tool. P5 Refine/canvas-edit primitives (quality-gated: census;
the freeze lesson applies to reject-masks). Later: multi-conv absorption,
Reground, lineage-drift gate. Every phase: golden 8/8 + suite; behavior
changes additionally smoketest ×{7,42,123}.

**P4 Splice + op-log replay + tool validator (SHIPPED):** (1) `Splice
{start, end, replacement}` op — surgical mid-log replacement (truncate to
start + re-extend replacement + tail), epoch-bumping;
`splice_matches_fresh_build` pins spliced state byte-identical to a
fresh build of the target log. (2) Full-fidelity op-log:
`PipelineOp::log_json` records every id + the replayable cfg core
(seed/budget/steps/stops/E6-flag); events log digests (ids FNV,
fingerprints, lineage ids); serve writes a `{"meta":…}` header
(model/ctx/steps). `diffgemma-mps replay <ops.jsonl>` re-executes the
log on a fresh pipeline and diffs every event
(`oplog_roundtrip_replays_bit_identically` + live serve-session replay
7/7 ops, 0 diverged — a field ops.jsonl IS a repro artifact). Replay
caveat: DGQ_* flags come from the current env; a second meta line
(serve restart) stops the replay. (3) `ToolValidatorStage` — chain
stage wrapping Generate with Mark→validate→Rewind→re-Generate (bumped
seed): malformed tool grammar (`tools::validate_tool_reply`) never
enters causal context; retries flow through the op-log (validator sits
OUTSIDE OpLogStage). Opt-in `serve --tool-validate` /
`DGQ_TOOL_VALIDATE=1`, default OFF (quality-affecting: a retry replaces
the reply; needs field sign-off). Remaining P4-adjacent: replay across
serve restarts; grammar-aware kept_len at the per-block layer.

**Tool-turn canonicalization vs cross-turn reuse (OPEN, root-caused
2026-07-17):** the grammar renders an assistant tool turn as
`calls → responses → content`, so a canonical finalized BEFORE the tool
response exists (calls → content → dangling `<|tool_response>` opener)
is structurally never a prefix of the next request — every OpenCode tool
turn routes to a fresh conversation and re-prefills the full prompt
(observed: 12,794/12,868-token shared prefix discarded, 33 s re-prefill;
surfaced by the prefill-status heartbeat). Candidate fixes, decision
pending (user: don't be hasty — the trailing prose is the model's own
explanation of its next call):
1. *Empty-content canonical*: finalize tool-calling turns through the
   calls + response opener only (content omitted from the CANONICAL KV
   log, not from the conversation — the client echoes it and the next
   render places it after the responses; next-turn prompt provably
   byte-identical). Restores full-prefix reuse with no client
   cooperation. Risk surface: none found for content; clients that DROP
   assistant content alongside tool_calls lose the prose in both designs
   (client-owned conversation).
2. *Private re-canonicalization blob*: return an opaque field the client
   must round-trip (OpenAI encrypted-reasoning-item style) carrying the
   exact finalized token log / digest + splice points, letting serve
   re-canonicalize under ANY client re-rendering; same vehicle could
   restore thinking and carry compaction state. Needs per-client
   verification that unknown fields are echoed.
3. *Tail-slack routing* (fallback, orthogonal): route by longest common
   prefix when divergence is confined to the conversation tail, truncate
   resident KV to the LCP on activate (74-token truncate is O(1) in ring
   slack) — recovers most reuse under any tail re-render.

**Tool-call triage via internal re-prompt (FUTURE, user design
2026-07-17):** on questionable tool calls, run an internal model
re-prompt asking WHICH call(s) to keep, then rewind/splice the KV so the
context retains only the chosen call — the propose/inspect/commit and
Splice primitives are the substrate (P2/P4 shipped); this is a message-
layer policy on top. Related: the tool-grammar validator's
retry-on-malformed is the degenerate case (keep zero, re-roll).

**P2 per-block propose/commit (SHIPPED):** `generate_with_session`
decomposed into `begin_turn` (prefill selection) / `propose_block` (one
block's attempt loop, E6 + commit-guard re-rolls intact, commits
nothing) / `commit_block(kept_len, extend)` / `finish_turn`
(`kv_valid_tokens` + output) in step_generate.rs, with the monolithic
path now a thin driver over exactly those primitives
(`default_commit_policy` = stop-scan/defer/ws-guard). Pipeline ops:
`BeginTurn`/`ProposeBlock` (event carries the argmax + advisory
stop-scan)/`CommitBlock {kept_len, extend}`/`DiscardBlock`/`EndTurn`;
lineage-mutating ops are rejected while a turn is open. Gates:
`per_block_ops_match_monolithic_generate` (byte-identical to the
whole-turn op), `per_block_partial_commit_and_discard_consistency`
(partial commit + discard leave zero residue past a rewind), golden 8/8
byte-identical post-decomposition. Notes: op-driven turns report stop
metadata only via block stats (`stopped_on_eot` is driver policy); the
message layer (P4) owns grammar-aware kept_len.

**Cancel (SHIPPED):** a `CancelToken` rides in `StepGenerateConfig` the
same way the observer Arcs do (works through any stage chain, per-op
scoped so no reset races). Checked between denoise steps and between
blocks; the in-flight canvas is abandoned uncommitted and
`GenerateOutput.cancelled` reports it; KV stays consistent with the
returned token log (`pipeline_cancel_stops_generate_kv_clean` pins
residue-free rewind past a cancelled generate). Serve wires it in
`attach_stream_observer`: the connection thread drops its receiver when
the SSE socket dies, the next delta send fails, the observer cancels —
and the worker skips the finalize rebuild. Remaining gaps: (1)
non-streaming requests only detect disconnect at the final write (needs
socket read-EOF polling on the connection thread); (2) prefill chunks
are not yet cancellation points (a 100k prefill runs to completion; the
chunk loops live in `prefill_chunks*`/`extend_monolithic_kv_chunked` and
need a consistent partial-prefix story before a mid-prefill stop).

### Code-quality survey backlog (audited 2026-07-17, unfixed by design)

Correctness-adjacent (do these first; each is small):

- **Tier-1 attention fixture is below the worst tile**: `full_grp8_hd512_fixture`
  (canvas=16, t_total=44) vs E17/E20's BM=BN=64, K_PAD=64 — every parity test
  is single-tile; multi-M-tile stepping, ragged N-tile masking, and K_PAD
  saturation are unexercised on the default-ON path, and
  `topk_k128_matches_cpu` claims k>64 coverage while running on 44 keys.
  Add a canvas≥65 / kv≥65 full-layer fixture.
- **Missing CPU twins + Tier-1 tests**: `kv/unpack_encoder_kv` (ring-aware
  hydrate; its inverse IS tested) and `kv/kv_f32_side_hydrate`
  (producer/consumer dtype surface). ~40 lines each.
- **server_worker render inconsistency**: the `encode_with_specials(render_conversation(…))`
  pattern repeats at ~6 call sites and the tool-compact `too_big` sizing
  closure passes `thinking: false` where its siblings pass the request flag —
  sizing and rendering can disagree. Collapse into one `render_prompt` helper.
- **Stale "prefill-only" docs** on the shipped decode path:
  `step_kernel.rs` `attn_gemm`/`attn_topk` field docs and
  `attention_gemm/mod.rs` module doc still say "denoise keeps
  attention_mma_full".

Duplication (largest mechanical wins):

- **Attention GPU/bench harnesses**: ~1,300 duplicated lines across
  `attention_topk`/`attention_gemm`/`attention_flash` `gpu()`/`bench_*()`
  (same buffer-setup + head-chunk dispatch body, 3-4× per file); ~600-800
  removable via a shared Fixture→buffers builder + dispatch closure +
  min-of-rounds timer. Also a drift hazard: benches hard-code `causal: 1`
  while the oracle takes it as an arg.
- **`encode_attn_gemm` vs `encode_attn_topk`** are ~80% identical (~120
  lines): shared QK stage + head-loop scaffold, stage-2/3 closure.

Structural (absorbed by the token-pipeline phases where noted):

- **`step_kernel.rs` (6.2k lines)**: ~90-method encoder object + 60-field
  `StepPipelines` with a 381-line compile fn. Extractable seams: GEMM-encoder
  family, SC-softembed encoders, ~1,300 lines of diagnostics/logging.
- **`generate_with_session` (~830 lines, three nested retry axes)** —
  decomposes along op boundaries in pipeline P2 (per-block protocol);
  `GenerateOutput.block_stats` (currently write-only) becomes the
  Proposal/Committed event payload there — do not delete it before P2.
- **`handle_tool_compact` (346 lines)** — becomes message-layer choreography
  over pipeline ops (checkpoint/rollback ≡ KvId mark + Rewind) in P3/P4.

Trivia sweep (one batched cleanup commit):

- cli.rs usage string: six advertised commands don't exist (`bench-step`,
  `bench-prefill`, `decoder`, `decoder-gpu`, `prefill`, `generate-parity`),
  `--compare-cpu` is never parsed, ~30 real commands undocumented; shared
  mutable parse vars (`bench_layers` quadruple-duty) invite cross-wiring.
- flags.rs: test-only `impl Default` duplicates the env defaults (can go
  stale silently — derive or generate one from the other); dead
  `attn_topk()` accessor; `DGQ_MOE_PREFILL_BM` carries disproven-experiment
  surface.
- Oracle sampler tests missing: `sample_from_probs_rows` (RNG/CDF walk — the
  one worth a fixture), `scale_logits`, `logit_softcapping`.
- Clippy: ~230 style warnings — 95 too-many-arguments (encoder plumbing;
  bundle into dims/param structs as files get touched), 19 hand-rolled
  clamps, collapsible-ifs. Opportunistic, not a campaign.

### Concept — span handles / compact history (engine revisit)

*(Substrate note 2026-07-17: implement on the token pipeline above — span
handles are `Splice` choreography in the message layer; the collapse
diagnosis it called for is DONE, see `debug/opencode_collapse/`.)*

**Symptom to diagnose first:** long OpenCode/agent turns look stuck in a
**newline / structure collapse** — indent and line grid degrade across
repair loops; the model rewrites the same file, emits mid-tool self-talk,
and burns context. Localize before building (decode `▁→space` is already
fixed; remaining collapse may be sampler/early-stop, attention on dense
code, or history shape). Do not treat "more context" as the fix until the
collapse axis is named.

**Idea (tool-compact sibling):** keep large code/tool bodies out of the
long-lived canonical KV as full token streams; store them as **handles**
(opaque id + optional line/span metadata) that expand on demand for the
client and for turn-local generation, then evaporate again at finalize.

Rough grammar (illustrative, not committed):

```text
<|copy id=sp_… start="pub fn foo()" lines=10 replace=… |>
```

Server holds the span blob (same lifecycle class as `ToolOutputStore`).
Serve soft-expands when packaging `write`/`edit` / when the model calls
`expand_*`; finalize keeps only the handle in the conversation prefix so
OpenAI re-send + KV reuse still match.

**Constraints already known (do not re-explore as free wins):**

- Attention only sees tokens in the active sequence. A handle in KV does
  **not** grant access to the full span without expand / turn-local inject.
- Sliding-ring KV vs full-layer KV are **parallel layer classes**, not a
  promote-and-drop pipeline. Ring-only litter handles that "throw away on
  promote" are not available; any real token position is written through
  the residual and full layers keep it for the whole context.
- Invisible litter that only the GPU sees breaks conversation prefix
  identity (clients re-send history without the litter). Handles must be
  in the canonical log or deterministically rehydrated before `activate`.
- Prefer content-hash / span ids over positional newline markers; edits
  invalidate line-number maps.
- Complementary to (not replacing) current serve tooling: incomplete-tool
  continue, trailing-after-tool continue, thought strip on finalize,
  tool-compact summarize/expand.

**Proposed phases (concept only):**

1. **Diagnose newline/structure collapse** on a fixed OpenCode repro
   (token dumps per block: newline/`▁` indent stats, entropy, stop deferral
   counts). Name the axis before shipping handles.
2. **Span store MVP** — on `write`/large tool text, hash→blob; messages
   may keep full text until a compact substitution path exists.
3. **Handle substitution + expand tool** — mirror tool-compact
   (`full_output_id` / `expand_summary`) for code spans; gate on
   prefix-stable KV reuse + smoketest/longctx.
4. **Client expand** — soft-expand handles in emitted tool args so
   OpenCode never race-compiles a handle string; keep serial tool policy
   (or one side-effect tool per turn) as a separate harness concern.
5. **Optional sparse visible anchors** only if (1) shows attention needs
   a line spine — explicit, id'd, in-canonical; not ring-side-band.

Parked until (1) is clear: regex-patch-in-handle grammar, per-newline
special-token litter, ring-only markers.

**Standing lever-hygiene principle (2026-07-13):** a lever disproven on one
architecture/shape is not disproven forever. When the shape changes (new path,
new tiler, new M regime), *cheaply* re-test the disproven levers it could have
unblocked. **CAUTION 2026-07-13 — the lever-hygiene re-test that "vindicated"
`DGQ_MOE_PREFILL_BM` (E1 revival) measured a FAKE win:** the −11/−20% came from a
pipeline-cache-label collision that silently ran a `bm=32` kernel (half the rows
zeroed), not real weight-reuse. Root-caused + FIXED (source-hash in the GEMM cache
label); a *correct* wide block does full work and the honest re-measure is in
progress (early signal: it's SLOWER). See the MoE-weight-stationary row below.
LESSON REINFORCED: a re-test that reports a speedup with NO correctness oracle at
the new config is not a vindication — gate perf levers on an output oracle FIRST
(the missing `sparse_block_m_invariant` gate is what let the fake win stand).

**The DUAL (2026-07-16, from the int8 audit — same campaign, opposite sign): a
DISPROOF whose stated mechanism was never independently measured is a curve fit,
not a mechanism.** `86acced` explained its 9× int8 deficit as "the half-MMA's
per-instruction width dominates the int8 dot's per-lane scalar chain by ~9x,
**exactly** the measured gap". It matched — because it silently multiplied 1.72×
(their own un-register-blocked loop nest, a real bug) × 2.22× (int32-mul vs
f32-mad on M3, the actual int8 penalty, never identified) × 2.54× (what MMA is
really worth). The conclusion survived (honest bar 5.6×, still un-clearable), but
the *bar* and the *mechanism* on record are both wrong, and the next person to
register-block that kernel would have measured a 1.7× win against a strawman.
**A physics story that "matches exactly" is the failure mode to distrust, not the
evidence** — decompose the claimed mechanism one variable at a time before
recording it. Corollary from the same review: **settle bandwidth-vs-compute by
effective GB/s before optimizing** — E20's findings doc proposed a "~2×" softmax
win from killing atomic contention while the kernel was already running at 78-83%
of the DRAM wall (ceiling: 1.2×). Divide bytes by time first.

**A fix scoped to one call site does not close a CLASS (2026-07-16, task #99).**
`0281c0a` hashed the source into the **GEMM** cache label and was written up — here
and in memory — as closing "the whole class of source-define collisions". It did not:
`compile_subkernel_ex` has its own labeling, and E18 walked into the **identical** bug
(`pipeline_flash` bakes `FL_HD/FL_BQ/FL_BK` into the source but labels every variant
`"flash"`, so hd=256 and hd=512 collided and E18's own hd=512 oracle was validated
against the hd=256 pipeline). Now fixed generically in `compile_subkernel_on_device`.
When a root cause is a **pattern**, the fix and the write-up must each name every site
the pattern reaches — or say plainly which ones were left. And prefer fixes that make
the bug unexpressible: **derive the key from the actual inputs** rather than asking
each caller to remember a distinguishing string.

Open re-test backlog (disproven-scan 2026-07-13):
(1) **MoE adaptive-M / partial-tile padding at M=1024** — memory *predicted* it
activates when per-TG goes compute-bound (= this regime); the live kernel comment
still calls padding "immaterial" (a stale denoise conclusion). (2) **E5 QK-ILP2
chain-split — PENDING BO 2026-07-14**: a single-axis A/B was inconclusive (and
didn't exercise the path — E17-on routes full layers to `attention_gemm`, making
`attention_mma_full`/ILP2 inert at default settings). Added as a categorical
axis (paired with `gemm_attn` on/off) in `tune_prefill_attn.py --proxy` so TPE
can test the joint {E17, ILP2, tiles} space. Kept behind default-OFF FC31.

| Item | Notes |
|---|---|
| **E22 — block-granular PRE-QK top-k selection** — **M0 RAN 2026-07-16 (same day): KILLED by the pre-registered bar — PARKED, revival path = TOKEN COMPACTION (E16), not kernels** | **M0 verdict (real Q/K planes, L5+L23, kv 12.3k/45.5k/120.8k, real mixed prose+code corpus):** attention mass is NOT block-concentrated at depth — even the ORACLE (perfect top-B block selection) captures only 34%@45.5k / **17%@120.8k at B=32** (bar was ≥90%); the across-head union reaches 36%@120.8k only by spending 196/944 blocks ≈ dense. Selection fidelity was fine (centroid Spearman 0.73-0.86) — **the territory failed, not the mechanism**; no kernel investment can fix "the mass is smeared across hundreds of blocks". **THE BIGGER FINDING — the mass-coverage ORACLE ITSELF IS IMPEACHED as a quality predictor:** exact row-top-512 = only 30%@120.8k and 64%@12.3k (the findings doc's "82-99%" was a 3-position, head-aggregated artifact), yet E20 k=64 (13% mass @120.8k, 41% @12.3k) **passes every behavioral retrieval gate at ≤20.6k**. Mass and quality have DECOUPLED: long-range attention ≈ sharp retrieval peaks + near-uniform diffuse background whose *average* (not composition) matters. Ledger entry: **never gate a sparsity lever on mass coverage for this model — behavioral retrieval only.** **REVIVAL PATH (2026-07-16, user-directed): the diffuse-background structure is EVIDENCE FOR E16-class token compaction** — if the background is near-uniform, a fused/averaged representation of aged tokens is exactly the right approximation; E22's block summaries (centroids) are E16 fusion candidates, and the M3-v0 zero-kernel harness (mma_full range-dispatch, scratchpad `e22_m3_decode_blocktopk.md`) is the behavioral test rig for any such experiment. Dumps + analysis kept: `step-attn-qk-dump` subcommand + `python/scripts/e22_block_mass.py`. Original playbook (historical): | **The structural flaw of E20 it fixes:** E20 computes **dense QK first**, then discards — `debug/sparse_attn_findings.md`'s own numbers show every k converging to ~50% of dense FLOPs at long T because QK dominates and QK is untouched; and QK is at the half-MMA wall (3.82-3.87 TF/s, independently reconfirmed in the 2026-07-16 review) so it cannot be made *faster*, only *smaller*. Selecting at 128-token-**block** granularity **before** QK skips QK+softmax+PV for pruned blocks → ceiling goes from E20's 2× to ~`T/(B·128)`× on full-layer attention. **Stakes per the 2026-07-16 head-to-head (30k, fresh, same day):** MLX-4bit chunked 355 tok/s prefill; ours-default 294 (1.21×); ours+E20-row-topk **343 (1.04× — row-topk nearly closes 30k already)**; 100k round pending, and 100k is where attention share grows — at B=32 (4,096 of 100k keys = 4%) full-layer attention approaches the sliding-layer floor, i.e. we'd **pass** dense MLX at 100k rather than chase it. **Design sketch:** (a) *summaries* — per 128-token K block, per KV head (full layers only: nkv=2, hd=512): f16 centroid (option +per-dim min/max bounds); @100k that's 782 blocks × 2 × 512 × 2B ≈ **1.6 MB, trivial**, maintained **incrementally at super-chunk commit** (the chunked-prefill structure gives the hook for free); full layers are linear KV — no ring-wrap complexity; sliding layers excluded (window 1024 is already the floor). (b) *selection* — per 64-row query tile (16 tiles per M=1024 super-chunk), tile-Q centroid × summaries GEMM = 16×782×512 @100k ≈ 6.4 MFLOP = noise; take the top-B block **union per tile** + forced blocks: causal frontier/local band + block 0 (findings: anchor ⊂ top-k, forcing is cheap insurance). **Deterministic BY CONSTRUCTION** — exactly-B with block-index tie-break, emitted sorted ascending → fixed summation order; no atomic compaction anywhere (E20's #94 race class is unexpressible, not merely fixed). (c) *execute* — the existing E17 decomp over the selected blocks as **dense 64×128 MMA tiles**. That last point is the second structural win: row-topk scatters columns per row → `attn_topk_pv` is a scalar gather **linear in K_PAD** (why E20's k is stuck at 64 — see the E20 row's coupled-caveat); block-topk keeps QK *and* PV at full MMA rate at any effective k = B·128, so the quality knob finally reaches the k=256-512 region where the mass table says quality lives. **Why block granularity should hold quality (to be PROVEN, not assumed):** the findings' mass structure is spatially clustered (prompt-start anchor 19%, far-half 40%, ±2k local band 39%) — coarse selection matches that shape; per-row k=64 spends selection precision the structure doesn't need and ships below the measured mass floor (the table starts at k=128). **M0 — offline mass oracle (~afternoon, no kernel, extends the findings methodology):** on `fixtures/longdoc.md` REAL text (E16 lesson: random-text similarity is common-mode, never oracle on it) at kv 10k/33k/100k, measure (a) attention-mass coverage of the top-B-blocks-per-tile union, B ∈ {8,16,32,64}, vs row-top-{128,512} reference; (b) **selection fidelity** — rank-correlation of centroid-score vs true block max-score (the mass can exist and the centroid still miss it: a block with one hot key + 127 cold ones dilutes to a cold centroid; the min/max-bounds variant is the fallback). **KILL if** union-top-32 < ~90% mass @100k or materially under row-top-512 (82-99%). **M1** — selection+gather prototype on the tunable-GEMM sparse machinery; isolated A/B vs E17/E20 at kv 15k/30k/100k; **×8 determinism probe (cd66294 protocol) before reading any gate**. **M2** — e2e opt-in `DGQ_ATTN_BLOCK_TOPK`; gates: multi-needle **aimed at pruned depths** (E16 lesson), longctx 4/4, smoketest 17/17×{7,42,123}, golden negative expected (non-bit-identical); **the quality-vs-B curve is the ship blocker** — do not repeat E20's k=64-unmeasured mistake. **M3 — decode arm (separate sign-off; design deepened 2026-07-16, full doc in session scratchpad `e22_m3_decode_blocktopk.md`):** the same summaries serve **denoise** attention at long kv. Physics check against the disproof ledger: decode attention is **issue-bound** (settled) → time ∝ keys touched → block selection cuts instructions linearly; compatible with the flash-decode/q8-for-speed disproofs (those changed the *rate*, this changes the *amount*). Stakes measured today: 1.97s/step @30k vs ~0.97s short-ctx → ~1s/step of linear-in-T attention growth; napkin at B=32: step 1.97→~1.15s @30k (+70% decode tok/s), ~4.6→~1.5s @100k (~3×). **LOAD-BEARING DISCOVERY: `attention_mma_full` already takes a KV RANGE per dispatch (`blk{t_begin,t_end,is_first}` + `st_o`/`st_ml` resume state — the d-split machinery), so M3-v0 needs ZERO kernel changes**: host-side selection → coalesce selected blocks into runs → one dispatch per run chaining the resume state (≤~160 extra dispatches/step ≈ 1-2ms). v0 = the end-to-end QUALITY harness — census/ladder/smoketest at a generous budget before any kernel is written; if quality fails at v0 the arm dies for the price of host code. **The design fork M0 now measures (`union_top{B}` in `e22_block_mass.py`): one list shared across all 16 heads (v0-shippable, mma_full grid covers all heads) vs per-head lists (needs the FC33 gather indirection — one-line remap of the K-stream tile base `t0 = blk_list[s0>>7]*128 + (s0&127)`, same remap for q8 dequant row/V tile/causal position).** Summaries lifecycle: compute at commit/extend only (prompt KV frozen during denoise; canvas keys ALWAYS forced), recompute-on-restore rather than snapshotting (no staleness state — the ring-rewind bug class gets no new member). Risk higher than prefill (every step compounds; selection error feeds back through the canvas) → census + multi-seed gate, doc-QA ladder not needles; ×8 determinism probe (cheap, and the 6/6-distinct precedent is fresh). **Relation to E20 (#94-#97):** row-topk stays the *interim* ship candidate (its 30k win is real, measured today); if E22-M0 passes, E20 becomes the donor (QK kernel + gather infra reused; radix select dies) and #95's quality-vs-k protocol transfers as quality-vs-B. |
| **E21 — SLC-chunked online-softmax E17** — **M0 RAN 2026-07-16 (same day): PARKED at the kill bar** | **Verdict:** measured attention stage (`bench-prefill-super --stages`, hc=16, E18 on) = 525/1433/2783 ms per super-chunk at kv 2k/15k/29k; analytic S/P bytes (S f32 + P f16 = 12 B per row·key·head ×4 subs ×6 full layers ×16 heads ×256 rows) ≈ 17.7 GB @15k ≈ 118 ms at the 150 GB/s wall = **~8% of the attention stage, and the share is FLAT in T** (S/P traffic and QK/PV compute both scale linearly — there is no long-context regime where it grows). ~3% end-to-end, non-bit-identical, at the pre-registered <8% kill bar → **not worth the gate. PARKED.** Approx diff pre-written (scratchpad `e21_chunked_online_softmax.md`) — revive ONLY for the memory co-benefit if 100k+ scratch pressure ever bites: chunking shrinks S/P scratch from `hc×m×n_pad(T)` (1.6 GiB @100k) to `hc×m×n_pad(Tc)` (~130 MB, T-independent) and skips fully-masked causal chunks. Design notes (kc0/kc_len window, `attn_gemm_softmax_chunk` running-(m,ℓ), PV fragment-init from rescaled O accumulator) are in the sketch. Original playbook: | **The middle point E18 never tried:** E17 materializes S and P to device — per head-chunk, ~4 round-trips of the `canvas×T` half-plane (QK writes S, softmax reads S + writes P, PV reads P). E18 proved **monolithic** fusion is dead at hd=512 (BQ=16 register cap → tiny MMA tiles, 3× slower even after the device-load rewrite — do NOT re-litigate). But between "materialize everything" and "fuse everything" sits: keep E17's **three dispatches and 64×64 tiles**, iterate K in Tc-sized column chunks (Tc ≈ 2-4k → S-chunk = M×Tc×2B ≈ 4-8 MB, SLC-class), carry running (m,ℓ) per row and rescale the f32 O accumulator between chunks — S/P stay cache-hot across three back-to-back dispatches instead of round-tripping DRAM at full T width. **M0 — KILL PROBE FIRST (per the standing lesson: divide bytes by time before optimizing):** decompose E17's measured attention stage at kv 8k/30k/100k into moved bytes (S/P passes ≈ 4×M×T×2B×heads; K/V reads irreducible; O writes constant) ÷ stage ms → the S/P DRAM share. Envelope: 4×1024×100k×2B×16 heads ≈ **13 GB/super-chunk/full-layer @100k ≈ 87 ms at the 150 GB/s wall** — expected share ~10-15% of the attention stage. **KILL if <8%** (below the complexity bar). **M1** — 4th mode of `attention_gemm` (K-column-range params + a small rescale kernel; reuse `frag_mma_ktile`); Tc tunable, joins the holistic BO surface. **RISKS, named up front:** (1) SLC residency across dispatch boundaries is a *bet*, not a contract — if the GPU flushes between encoders the traffic doesn't drop; M0's GB/s math vs one honest A/B settles it, kill on no-effect; (2) non-bit-identical (online softmax reorders the reduction) → full gate + golden re-bless sign-off — with the silver lining E17b already recorded: E17's golden flips ARE batch-vs-online softmax, so E21 may *re-align* golden with the `attention_mma_full` baseline; (3) **sequencing:** E22 landing first shrinks effective T and with it E21's value — run E21-M0 (cheap) now, but spend build effort on E22-M0's verdict first. |
| **E18 sliding flash — RACE + CACHE-LABEL COLLISION FOUND + FIXED 2026-07-16** (task #99; `debug/review_2026-07-16.md` Part 4) | Found while building the #92 ring-truncate oracle: prefill was **not reproducible run-to-run** (~80% of KV bytes differed across identical `reset_kv`+`extend_kv(1200)` cycles). Bisected to E18 in ONE flag (`DGQ_FLASH_PREFILL=0` → 0 bytes differ). **BUG 1 — missing barrier:** `attention_flash.metal` stages `Qs` STRIDED BY `tid` but every simdgroup then reads the WHOLE tile via `simdgroup_load` in the QK step, with **no `threadgroup_barrier` between the staging loop and the first read** (the first barrier sat at the END of the QK step — too late). Only the FIRST K-block iteration was exposed; later ones are covered by the barrier closing the loop. Same class as the SC GEMM "missing gemm_b barrier". FIX = one barrier after staging. Probe (`prefill_nondeterminism_probe`, `#[ignore]`, `DGQ_PROBE_N`): flash-ON pre-fix A-vs-B=216182753 / B-vs-C=212327667 of 270336000 → **post-fix 0 / 0**; end-to-end, `golden` longctx_13k emitted **13065 then 13067** tokens on consecutive runs of the same binary+seed pre-fix, and two full golden runs are **byte-identical** post-fix. **PERF FREE** (A/B `flash_bench_hd256` bq16 kv 8192/30000/60000: with-barrier 33.07/121.81/255.61 ms vs without 35.14/133.08/260.51 — once per threadgroup, not per K-block; delta inside the bench's own ~5% noise, cf. the e17 baseline column moving 15.7→16.6 across the same two runs). LOCALIZING CLUE worth reusing: **layer 0's KV stayed bit-identical while layers 1-29 diverged** — layer 0's K/V is a pure function of `embed(tokens)`, written BEFORE attention, so a clean layer-0 KV + dirty layer-1 KV points at layer 0's attention OUTPUT. **BUG 2 — pipeline-cache label collision (`0281c0a`'s class, second instance):** `flash_gpu_full_grp8_causal_vs_cpu` failed IN-SUITE but passed in isolation on clean HEAD — `pipeline_flash` bakes `FL_HD/FL_BQ/FL_BK` into the SOURCE via `tuned_source` but labels every variant with the constant literal `"flash"`, so hd=256 (sliding, production) and hd=512 (full, the test) shared a cache label and **E18's own hd=512 oracle was validated against the hd=256 pipeline**; `attention_gemm`'s `"tune"` is identical, and `extra_bools`/`extra_uints` never reached the label at all (encoded by CONVENTION: `"side"`/`"default"`, `fmt.label()`). **FIX = fold `source_hash(source)` + every extra FC index/value into the label inside `compile_subkernel_on_device`** — closes the class for EVERY `compile_subkernel_ex` caller, not one kernel. **Production impact of bug 2: NONE** (production compiles only the hd=256 variant) — it corrupted TESTS, i.e. it silently weakened the gate meant to validate E18. **LESSON (recorded in the lever-hygiene note): a fix scoped to one call site does not close a class** — `0281c0a` hashed the source into the *GEMM* label only and was written up as closing "the whole class of source-define collisions"; it didn't, and E18 walked into the identical bug. **CORRECTION to my own reasoning, kept because it nearly misled the diagnosis:** "B≠C rules out a stale-residue function" is WRONG (each round's residue IS the previous round's output, so a deterministic stale-reader also gives B≠C) — the flag bisect discriminated, not the 3-way compare. VERIFIED: suite **582/0** (was 581/1 — that failure WAS bug 2), golden 3/8 unchanged and now stable (still red for an unrelated reason = the golden-red row). |
| **Golden Tier-1 gate** — **RED (3/8) since E18; ROOT-CAUSED + RE-BLESSED 2026-07-16 → 8/8** (task #100; `debug/review_2026-07-16.md` Part 5) | Two causes, both intentional, **neither an engine regression**. **(1) `cross_turn_reuse` + `multi_block_reply`** ("KV hash differs, token_ids identical") = **E18 sliding flash** (`42de5cd`), non-bit-identical by design (online softmax), signed off on smoketest 17/17×{7,42,123} + longctx 4/4 but **never re-blessed**. Confirmed: `DGQ_FLASH_PREFILL=0` → both pass (3/8→5/8). **(2) `fast_prefill_3k` + `ring_wrap_2p5k` + `longctx_13k`** ("token_ids diverge at index 237: golden=17723 got=138") = **the DECODE FIX in `83b184e`** — bisected (parent `466cee4` = 8/8). **MECHANISM, proven by code not inference: golden's doc cases build the prompt ENCODE → truncate → DECODE → re-encode** (`src/commands/golden_cmd.rs:269-271`, `let excerpt = tokenizer.decode(&ids[..n]);`), so `tokenizer.decode` is **on the golden prompt-construction path**. `83b184e` correctly changed decode from "collapse a leading U+2581 run to ONE space" to "replace EVERY U+2581 with a space" (matches `tokenizer.json`'s Replace decoder; regression test `"    return"` → `[140, 2060]` → `"    return"`). The old decode destroyed indentation (`▁▁▁▁` id 140 → 1 space) — the bug that broke OpenCode tool writes. **So the blessed prompts ENCODED THE DECODER BUG**; index 237 is *inside* the prompt (prompt_len=3005). **Bless diff confirms it independently of the bisect:** prompt_len 3005→3038 (+33), 2505→2538 (+33), 12834→13040 (+206) — the 3k and 2.5k docs share their leading ~3000 tokens and move by *exactly* the same +33; the 13k doc moves +206, proportional to restored indentation. **RE-BLESSED per policy** — gates run FIRST: smoketest **17/17**, longctx **4/4** (8/8 kw, drift 0.0%), suite 582/0; determinism precondition met by task #99 (pre-#99 `longctx_13k` emitted 13065 then 13067 on the same binary+seed — blessing then would have frozen a sample of a random variable). Now **8/8, stable across runs**. **COUPLING: the bless MUST land with #99's barrier fix** — blessing without it freezes racy flash output; if that fix is ever reverted, golden going red is CORRECT. **LESSONS: (a) a gate whose harness ROUND-TRIPS THROUGH THE CODE IT GATES restales itself on any correct fix to that code** (golden's doc prompts are decode-dependent — worth a line in the golden README); **(b) self-test a bisect predicate against a known-bad commit before trusting it** — my first script grepped the case NAME, which golden prints on PASS lines too, so every commit tested "bad" and bisect returned the oldest candidate (`11bc846`, superficially plausible as "the commit that made tiles tunable"); caught by reading the diff (semantically identical at defaults) and verifying that commit + its parent directly (both 8/8). |
| **KV sliding-ring rewind (`632aa69`)** — **PREDICATE FIXED + ORACLE LANDED 2026-07-16 (tasks #91/#92); `generate_with_session` path STILL UNFIXED (task #93)** (review: `debug/review_2026-07-16.md` Part 3) | **FIXED:** the predicate now takes `window` and returns `deepest_needed < oldest_live` (saturating), replacing `old_len > ring`. Verified three ways: (1) `kv_truncate_needs_ring_rebuild_matches_corruption_model` checks it against an INDEPENDENTLY-DERIVED corruption model (written from the storage rule, shares no code with the predicate) over `old ∈ 0..6000` × 10 `new` — it caught my own hand-written boundary off-by-one (true edge is `old-769`); (2) `truncate_after_uncommitted_canvas_write_matches_fresh_prefill` drives the ORDINARY production flow (prefill 2000 → short reply whose final block writes canvas at `[2000,2256)` and never commits → `truncate_kv_to(1200)`) and byte-compares `snapshot_kv` vs a fresh 1200-token prefill — **on the pre-fix predicate it fails with exactly the predicted signature: `208 of 1200 layer-0 ring slots differing, slots 0..=207`** = the canvas write of positions 2048..=2255 landing in wrapped slots; (3) suite 581/1 + golden 3/8, **both identical on clean HEAD** ⇒ suite- and golden-neutral. **DISCOVERED WHILE BUILDING THE ORACLE (the valuable part):** the premise control — two identical `reset_kv`+`extend_kv` cycles → identical KV — **FAILS** (~80% of bytes differ, and differ AGAIN on a third run, ruling out a stale-residue function); layer 0 always bit-identical, layers 1..29 diverge from slot ~14 ⇒ layer 0's KV is reproducible (pure function of `embed(tokens)`) but its ATTENTION output is not, amplified through depth. Pre-existing (the control touches neither `truncate_kv_to` nor `rollback_to`) → **task #99**; corroborated end-to-end by golden's longctx_13k emitting 13065 then 13067 tokens on consecutive runs of the same binary+seed. The oracle therefore asserts on LAYER 0 only (still a ring layer ⇒ still poisoned by the identical mechanism), with `layer0_prefill_kv_is_bit_reproducible` as the documented premise control; widen when #99 lands. Also found: **golden is 3/8 RED on clean HEAD** (task #100), prime suspect E18 sliding-flash default-on (`42de5cd`, explicitly non-bit-identical) never re-blessed. **STILL OPEN (task #93):** `generate_with_session:650-659` is an unfixed duplicate of the same truncate-then-extend (`prefill_chunks_from(reuse, delta)`, no ring check) — serve is shielded by `route()`'s prefix guarantee, `chat` (CHAT_MAX_SEQ=8192, raw-vs-sanitized divergence) and `run_summarize_pass` (`server.rs:801`) are NOT; plus `rollback_to`'s "restores the conversation" contract is still false for its only production caller. Detail in §3.3/§3.4 of the review. |
| **E20 top-k sparse attention** — **#94-#97 RESOLVED + DEEP-RETRIEVAL PROBED (task #103) 2026-07-16: ship shape = DYNAMIC K (`DGQ_ATTN_TOPK_DYN`), pending sign-off + its gate run** — **Deep needle probe (clean corpus-unique markers at token depths ~5k/40k/80k/115k, real 121k mixed corpus): dense 4/4; fixed k=64 = 3/4 — DROPS THE DEEPEST NEEDLE at 121k (the E22-M0 mass-diffusion prediction has behavioral teeth: 13% mass coverage finally bites); k=256 4/4; DYNAMIC k (clamp(t_total/128, 64, 512) = ~0.8% of context, K_PAD=512) = 4/4 = dense.** At 45k all four configs 3/3. PROBE-HYGIENE LESSON: probe v1 planted "NEEDLE"-named markers into a corpus built from OUR OWN DOCS (which discuss needle tests) → the model confabulated documentation content as answers; a retrieval probe's markers must be corpus-unique strings at verified-clean insertion points. v1's "1/4 @121k for topk" was mostly that artifact — but the single real k=64 miss reproduced in v2. Fixed-k=64 keeps its measured cliff OUT of production; dyn's extra k is ~free at long T (QK dominates; PV overscan ~8ms/super-chunk at short kv). **DYN GATES ALL GREEN (same day): longctx 4/4, smoketest 17/17 ×{7,42,123}, proxy = k64 within noise. 100k H2H: ours-dyn 413.5s/242 tok/s vs MLX 402.7s/248 = WITHIN 2.5% (dense 574s/174 = 1.43× behind); 30k: 343 vs 355 = 1.04×. Prefill gap vs MLX ≈ CLOSED across the range with dyn on. Decode @100k was TIED (ours 1.21 vs MLX 1.16 tok/s — shared ceiling while dyn was prefill-only) → RESOLVED 2026-07-16: `DGQ_ATTN_TOPK_DECODE` default-on routes full-layer DENOISE dispatches through the same 3-kernel pipeline (causal=0); 100k decode 4.57 vs MLX 1.16 tok/s (~3.9×).** Earlier same-day state: — **#95 result: k=64 PASSES everything** — longctx doc-QA 4/4 (8/8 kw, 0.0% drift) at every k ∈ {64,128,256,512} + dense control, **smoketest 17/17 × {7,42,123}** at k=64. The k knob is LIVE (was silently clamped): `DGQ_ATTN_TOPK_K` bakes `AG_K_PAD = next_pow2(k)` via `tuned_source` (safe post-#99 label fix), P/Idx planes sized from the same `flags::attn_topk_k_pad()`, k=128 oracle test pins it. CAVEAT on record: behavioral evidence tops out at 20.6k ctx (the ladder's ceiling); the review's mass-table concern (k=64 < measured floor) did NOT materialize as a gate failure at any tested depth. Perf case for default-on: 30k prefill 343 vs MLX 355 tok/s (1.04×), +16% over dense E17. Earlier same-day fixes: — probe: old kernel gave **6 DIFFERENT outputs in 6 runs** (same binary/seed/prompt — the race perturbed every run, worse than the review predicted); new = 12/12 bit-identical. Fix = u16 pattern plane from QK (FC32) + 4-level radix (levels 3-4 refine u16-ties from f32 S, so selection is EXACT top-k by score — the k=1 oracle caught that u16-level exact-k selection can MISS THE ARGMAX inside a 2^16-wide tie bucket, a quality loss the old kernel's over-emit accidentally avoided) + count/prefix-sum/thread-major emit (cd66294 shape; atomic compaction gone) + all four hc sites unified on `DGQ_GEMM_ATTN_HC`. **PERF-NEUTRAL by A/B** (interleaved ×2: means within 0.5%): 6 u16 scans = the same bytes as the old 3 f32 scans — the u16 savings were SPENT on exactness+determinism, not speed (possible follow-up: merge to 4 passes, ~1% e2e, below bar unless topk defaults on). Gates: topk oracles 7/7 (k=1 at original tolerance), suite 582/0, golden 8/8 UNCHANGED (default path compiles identically, FC32 unset). 30k standing: topk 343 tok/s vs MLX 355 = 1.04×. Original review (still the map for #95): (review 2026-07-16, `debug/review_2026-07-16.md` Part 1; tasks #94-#97) | 3-kernel pipeline (reuse E17 `attn_gemm_qk` → `attn_topk_softmax` 2-level radix select+softmax → `attn_topk_pv` gathered-V), full layers only, takes precedence over E17. Measured 1.86× attention / 1.19× prefill @kv=15k; gate 17/17 ×{7,42,123} + longctx 4/4; non-bit-identical by design (sparse PV reorders FP-associativity). **The QK-at-the-wall finding in `7fcf3ff` is CORRECT and independently reconfirmed** (3.82-3.87 TF/s at the half-MMA ceiling) → chunking is dead as a top-k lever. Four things to fix, in order: **(1) DETERMINISM — this is `cd66294` character for character** (task #94): `attention_topk.metal:257-261` compacts via `slot = atomic_fetch_add(&emit_cnt,1u); if (slot < K_PAD) {...}` and `attn_topk_pv:344-354` then sums `acc0 += pj*vb[idx[j]...]` **in list order** — same atomic compaction → race-ordered list → non-associative f32 reduction that `cd66294` already fixed once in `sc_sparse_select` (fix was threadgroup count + exclusive prefix-sum → fixed thread-major order). **E20's version is STRICTLY WORSE**: `if (slot < K_PAD)` means when `emit_cnt > K_PAD` the surviving **SET** is race-determined, not just the order — and overflow is the *normal* path (threshold emits all ties; scores share one exponent so top16 varies in ~7 mantissa bits ≈ 128 buckets; at `n_valid≈16k` that's ~125 keys/bucket vs k=64). The 17/17 gate is blind **by construction** — `cd66294`'s own message: *"Token output stayed stable for clean prompts (the diffs don't move argmax), but borderline-convergence prompts flipped their early-stop step count."* **Probe first (~10 min, `cd66294`'s protocol): one prompt ×8, same seed+binary, `DGQ_ATTN_TOPK=1`, count distinct step counts** (old SC gave 5,6,7; fixed gave 6,6,6). Fixes cheapest-first: bitonic-sort the ≤64 `(score,idx)` pairs post-emit (one simdgroup, ~36 CE steps, fixes ORDER) → count+prefix-sum (fixes SET) → 3rd radix level (bits 8-15, narrows ties 256×, makes it actually compute top-k). **(2) k=64 SHIPS BELOW THE MEASURED FLOOR = the default-on blocker** (task #95): the mass table in `debug/sparse_attn_findings.md` **starts at k=128** (71% @pos129); k=64's retained mass appears nowhere, and mass is monotonic in k. The doc's own call was the opposite ("prototype at k=512… push higher"). The knob is dead: `DGQ_ATTN_TOPK_K` is **silently clamped** to compile-time `K_PAD=64` (`flags.rs:442` → `step_kernel.rs:3331` `.min(K_PAD)` → `mod.rs:28`), and `tuned_source(bm,bn,sm_tpg,k_pad)` (`mod.rs:44`) is **dead code** — `pipelines()` compiles `SHADER_TOPK` directly (`mod.rs:69-70`), so BM/BN/SM_TPG/K_PAD are frozen at `#ifndef` defaults and E20 is absent from the holistic BO surface. **COUPLED CAVEAT the findings doc misses**: raising k is near-free in FLOPs but NOT in *this* PV kernel — `attn_topk_pv` is a scalar gather looping `j < K_PAD` on a single 32-thread simdgroup, so cost is **linear in K_PAD** (est. @kv=15k: pv 4.4→~35 ms at K_PAD=512 → attention 102.6→~131 ms = 1.67× vs E17, down from 2.13×; @kv=100k it barely moves, ~2.15×). So "PV is <1%, no need for MMA sparse PV" holds **only at K_PAD=64** — the two conclusions are coupled and the doc treats them as independent. Points at **kv-adaptive k** (short kv → k=64 fine; long kv → PV relatively free *and* quality risk highest → k=256-512), now **safe** to do via source-define specialization precisely because `0281c0a` folded `source_hash` into the pipeline label (pre-fix, two K_PAD variants in one process would have collided exactly like `bm=32/64`). **(3) THE REMAINING PERF JUICE — and the findings doc's stated next step targets the WRONG bottleneck** (task #96): it proposes per-thread local histograms to kill atomic contention, est. "~2× on softmax" — **arithmetically impossible; `attn_topk_softmax` is BANDWIDTH-bound.** Recomputed from `7fcf3ff`'s own prod-shape bench (canvas=1024, hc=4, 16 q-heads, BN=64), sm reads the S plane 3×: kv 15k/30k/60k → 3.16/6.10/12.0 GB in 25.37/51.88/100.43 ms = **124.5/117.6/119.5 GB/s = 78-83% of M3 Pro's ~150 GB/s wall, FLAT across kv**. Zero atomic cost caps at 1.2× on sm (≈4% of attention, ≈1% e2e). **The real lever: all three passes compare only the TOP 16 BITS** of the monotonic key (`:164` `pat>>24`; `:217` `(pat>>16)&0xFF`; `:255-256` `top16 = pat>>16`) — the f32 score is needed **only** for the ~64-96 winners at `:259`. So emit a **u16 pattern plane** from `attn_gemm_qk` alongside S: sm's passes read half the bytes → 25.4→~12.7 ms (the true 2× the doc wanted, via the binding mechanism), attention 102.6→89.9 = **1.14×**, E20-vs-E17 2.13→2.43×, QK pays +526 MB of writes but is compute-bound at 10% bandwidth = free, and it is **BIT-IDENTICAL to today's E20** (the kernel already discards bits 0-15 in every comparison). It also *funds* the prefix-sum determinism fix (4 u16 passes = 16.9 ms, still under today's 25.4 ms AND deterministic). **DO NOT attempt full QK+top-k fusion at hd=512 — E18 already disproved it** (BQ=16 register cap → tiny MMA tiles → 3× slower even after the device-load rewrite); QK's grid splits columns across `t_total/BN`=251 threadgroups, so per-row partial top-64s need a 251-way merge. The E18 disproof transfers cleanly; the u16 plane is the version that fits the machine. **(4) `hc` HARDCODED to 4, ignores the shipped BO optimum 16** (task #97): `step_kernel.rs:3335` reads `TuneCfg::default().hc` (=4, `attention_gemm/mod.rs:45`) not `DGQ_GEMM_ATTN_HC` (=16, `flags.rs:181`, shipped `9957a81`, +3.6% @100k) → E20 leaves the win on the table, **and the A/B was E17@hc16 vs E20@hc4** (biases *against* E20, so 1.19× is a floor — but it is not the number it appears to be). With both flags on, `:5948-5955` sizes the S plane from the *flag* (16) while the encoder uses 4 → 4× over-allocation (1.6 vs 0.4 GiB @100k, where memory pressure bites). **LANDMINE: `hc` is coupled across FOUR sites** — `3335` (encoder) + `5984`/`5993`/`6002` (P/Idx/lrow scratch); change only the encoder and the scratch **overflows**. Same shape as `tunable_wide_n_tile` and the `bm=32/64` label bug → add a debug-assert at the dispatch seam. Smaller: `mod.rs:249-250` claims *"topk is exact (same selection + same renormalization)"* — **false** per (1); `topk_k1_matches_argmax_value:264-268` *does* document the tie behaviour, so the knowledge exists, it just didn't propagate; **no test exercises the overflow/tie regime, which IS the production regime**. `SM_TPG` power-of-2 assumed by the reductions at `:281`/`:297` — inert today (`tuned_source` dead), live the moment E20 joins the BO → `static_assert`. `a2a9e95` cites *"vs MLX single-shot: 230 tok/s → we are 1.56× faster"* — **drop it**: driving MLX single-shot is the known mis-drive that already produced one retracted "MLX can't do long ctx" conclusion (memory `mlx-prefill-ab`); the honest number is the chunked 433 tok/s the commit also cites. |
| **int8/int4 MoE disproof (`86acced`) — record correction** — **the DEAD END IS SOUND (do NOT re-test); the BAR and the MECHANISM in ARCHITECTURE.md are WRONG** (review 2026-07-16, `debug/review_2026-07-16.md` Part 2; task #98) | Audit reproduced the kernel standalone (**0.438 TF/s vs their reported 0.43; 9.70× vs their 9×**) so the numbers are directly comparable. Harness is honest (same `RouteScratch`/`BlockGroupedJob`, BM/BN/BK, 4-simdgroup 2×2, `gflops()`, shapes/slots/warmup as the baseline). **Decomposition, one variable at a time (identical tile geometry, bit-identical outputs): loop nest / no register blocking = 1.72× (a REAL implementation flaw) × int32-mul vs f32-mad = 2.22× (hardware) × simdgroup-MMA vs scalar-f32 = 2.54× (hardware) = 9.7 = the measured gap, fully accounted.** The flaw: `gemm_int_sparse.metal:171-189` loads `a4`/`w0`/`w1` **inside** the innermost k-loop (inner-product nest, operands re-fetched from tgmem per `(i,j)`) while the production baseline it's benched against does the opposite — `gemm_frag_tile.metal:46-68` hoists `A[TM]`/`B[TN]` into registers per k-chunk then runs a TM×TN outer product (**384 vs 80 tgmem loads per lane per BK=32**); register-blocking measured **0.755 vs 0.438 = 1.72×**. This contradicts the commit's own claim (*"Tile sweep did not close the gap — the loss is in the inner accumulation, not tiling"*): they swept tile **geometry** (2 points; BK never swept) but never the loop **nest** — the one axis on which the prototype differs from its own baseline. **CONCLUSION STANDS ANYWAY**: a *perfectly* register-blocked int8 kernel is still **5.6× slower** than production, nowhere near the 1.0× kill criterion; the two residual factors are hardware properties no restructuring addresses; int4 is deader still. **EDITS: (a) the bar is 5.6×, not 9×** — ARCHITECTURE.md tells a future attempt it *"has to clear the 9x bar"*; whoever register-blocks the kernel will measure a 1.7× win **against a strawman** and mistake it for progress. **(b) replace the mechanism** — `ARCHITECTURE.md:424-426` asserts *"the half-MMA's per-instruction width dominates the int8 dot's per-lane scalar chain by ~9x, exactly the measured gap"*; measured, **MMA is worth only 2.54×**, and the 512-vs-4 MAC arithmetic implies ~4× and *cannot* produce 9× — it matched only by silently absorbing their own 1.72× bug plus a 2.22× effect nobody identified. **The real specifically-int8 penalty is that int32 multiply is 2.2× slower than f32 mad on M3** — that is the fact worth keeping. **(c) record the hardware fact**: probed the runtime compiler directly on this M3 Pro — `dot(char4,char4)`, `simd_dot_acc_int8`, `dot_product_4x8_packed` are **ALL REJECTED**; only float `dot()` compiles. **There is no DP4A-style packed integer dot in MSL**, so the kernel's manual 4-scalar expansion at `:181-184` was the only expressible formulation and no fast primitive was left on the table. NB `0d41267` upgraded this redirect to PROMISING on the premise that *"the integer DOT-PRODUCT instruction (DP4A-style, family 7+) DO exist"* — factually wrong, and that false premise is what made the whole redirect look promising; **a 20-line compile probe up front would have killed it.** Minor (cuts against int8 anyway): int8 W is K bytes/row vs q4's ~K/2 → ~2× the weight bytes of the baseline, a bandwidth penalty independent of the math, material at rpe=16; the bench emits per-regime rows but both docs collapse to one headline. |
| **Span handles / compact history (concept)** — OPEN; blocked on collapse diagnosis | See section above. Engine revisit: newline/structure collapse under agent loops first; then tool-compact-like span store + expand. Not a sliding-KV free lunch. |
| **MoE weight-stationary prefill — `DGQ_MOE_PREFILL_BM`** — **ROOT-CAUSE FOUND + FIXED; the "win" was FAKE; honest re-measure IN PROGRESS 2026-07-13** | **The `block_m>32` "correctness bug" was NEVER in the kernel — it was a pipeline-cache-label collision (`src/metal/device.rs`).** The tunable tile geometry (`TUNE_BM`/`TUNE_BN`) is baked into the shader SOURCE `#define`, not a function constant, so the GEMM cache label omitted it. A process that compiled the sparse GEMM at *both* `bm=32` (denoise) and `bm=64` (prefill) → **same label → `PipelineArchiveCache` silently returned the first-compiled pipeline.** A `bm=32` kernel fed 64-row blocks processes rows 0-31 and leaves **32-63 exactly zero** — confirmed by per-row dump (`cos=0.707=√½` = "half the rows"; `[40]`-block → `cos=0.90=√(32/40)`). **FIX: fold a `source_hash` into the cache label** (closes the whole class of source-define collisions, not just tiles). Verified: `sparse_block_m_invariant` oracle **cos=1.000000 at every tile** {32,64,128}×{64,128} (was 0.59–0.74); **golden 8/8** (fix invisible to production — same source hashes identically); **suite 551/0**. The oracle now loops tiles *in one process* to force the collision if it regresses. **CONSEQUENCE: the earlier −11/−20% "win" was FAKE** — a `bm=32` kernel doing half the work is faster *because it skips rows*. Likewise the "golden 5/8 + longctx stochastic fact-dropping" quality story was the zeroed KV rows, not a real perturbation. A *correct* wide block does the FULL work, so it needs an honest re-measure (early 3-trial smoke: correct `bm=64` is **slower**, 3264/3377 vs 2838ms default — consistent with the fake-win theory). **`DGQ_MOE_PREFILL_BM` is now a correct, sweepable axis** (added to the 10-dim holistic BO; needs no Rust change — the flag drives the fixed machinery). Wide-path N-tile still pinned at 64 in two coupled sites (grid `step_kernel:1284` + compile `pipeline_for_sparse_bm`); unpinning it into a joint `(BM,BN)` sweep is a follow-up *if* corrected `bm>32` proves worth it. `DGQ_MOE_PREFILL_BM_MAXKV` / the kv-adaptive ship plan are MOOT unless the honest sweep shows a real win. |
| **Prefill floor decomposition** — **DONE 2026-07-13 (`bench-prefill-super --stages`)** | Stage-ablation of the M=1024 super-chunk (skip a stage group, delta = its cost; timing data-independent). @kv2k ms/super-chunk: **moe_experts 881 (35%)**, attention 657 (26%, grows w/ kv), qkv_proj+inorm 298 (12%), dense_ffn 296 (12%), o_proj 119 (5%), moe_postnorm+combine **only 102 (4%)**, router/embed ~0. FINDINGS: (1) floor ~1685ms is MoE-dominated + dead-constant across kv; (2) attention is 26% even @kv2k (large FIXED cost, not negligible); (3) the "fuse the norm/combine" shot is DEFLATED — the memory-bound part is only 4% (the 383ms "elementwise" was 296ms compute-bound dense-FFN GEMM + 102ms elementwise); a fusion recovers ~1.4% → below bar, NOT built. ~90% of prefill is compute-bound GEMM/MoE at the wall → residual MLX gap = per-flop constant factor; the one non-wall lever was the MoE weight-load regime above. |
| **E18 fused flash prefill** — **CLOSED (full hd=512) 2026-07-13; SLIDING SHIPPED default-on 2026-07-15 (`DGQ_FLASH_PREFILL`)** | Full hd=512 path CLOSED (3× slower than E17). **Sliding-layer revival SHIPPED default-on:** window + ring support on `attn_flash`; `encode_attn_flash_sliding` routes the 25 hd=256 sliding layers. Isolated: mma2 34.4 ms / 0.50 TF/s vs flash+window 14.1 ms / 1.22 TF/s = **2.44×**. E2E kv=15k M=1024 (with topk): attn 973→807 (−17%), total 2837→2666 (−6%). Gate PASS: smoketest 17/17 ×{7,42,123}, longctx 4/4. Non-bit-identical (online softmax). `DGQ_FLASH_PREFILL=0` restores mma2. **TILE SWEEP CLOSED 2026-07-16 (non-lever):** interleaved same-batch A/B at kv=15k (`bench-prefill-super`, ×2 each): BK=128 −1.9%, BQ=32 +0.7%, both under the tiny-gains bar; BK=32 (zero-length-array compile error), BK=256 and BQ32+BK128 (pipeline compile fail) don't build. Default (16,64) stands. **CAUTION:** a first single-pass sweep had shown BK=128 at "−7.9%" — it did not reproduce (1 of 4 runs, unexplained outlier); cross-process proxy runs drift up to ~±4% same-config same-day vs 0.08% within-process, so **never read a cross-process `bench-prefill-super` delta <5% without an interleaved repeat** (the topk −16% reproduced ×3; that's what a real effect looks like). | 
| **E16 token fusion / KV merging** (IN PROGRESS) | The only unexplored long-context denoise SPEED lever (cuts token count, not bytes — fewer score rows; ~4.5 s/step at 105k is the target). Oracle so far: fusion is gist-preserving/verbatim-lossy; residual-gated r=2 ≈ control quality but 1.4× (under bar). Next: min-pairwise/outlier gates, mass keep-lists, multi-seed, non-English. MUST gate on the doc-QA ladder, not needles |
| **Super-chunk size sweep (`--n-subs`)** — **DONE 2026-07-14 (single-axis); RE-VALIDATED in joint BO + 100k run** | Added `--n-subs` runtime cap to `bench-prefill-super` (sweeps n_subs 1/2/4 = M=256/512/1024 without recompile — arrays sized for 4, runtime cap is the only change). Per-token cost at kv=8192: n_subs=1 **3.16ms/tok**, n_subs=2 **2.93ms/tok** (0.93×), n_subs=4 **2.84ms/tok** (0.90×). At kv=30000: n_subs=1 **4.62ms/tok**, n_subs=2 **4.38** (0.95×), n_subs=4 **4.26** (0.92×). **n_subs=4 (M=1024) stays optimal at both short and long context** — the MoE/dense GEMM batching win (weight reuse at M=1024) outweighs the attention growth from more queries. The gap narrows with kv (0.90×→0.92×, consistent with attention growing) but 4 still wins; the crossover would need attention >50% of the floor (it's 36% at kv8k, ~50%+ only at very long ctx). MLX's 512 (n_subs=2) is actually **slower per-token** than our 1024. **RE-VALIDATED in the joint holistic BO (32 trials, n_subs as a categorical axis alongside {E17, ILP2, tiles}, ms/token objective — the raw ms/super-chunk objective was WRONG when n_subs is swept, it rewarded smaller super-chunks for doing less work; fixed to ms/tok = ms/(n_subs×256)):** top-8 configs at kv=15k are ALL n_subs=4, E17-on, ILP2-off; best = 3.242 ms/tok = 1.00× the shipped default. No interaction between n_subs and the other axes changed the optimum. **100k run (3-trial, the dynamic-sizing question):** n_subs=4 = 9.106 ms/tok, n_subs=2 = 9.196 (1.010×), n_subs=1 = 9.491 (1.042×). n_subs=4 still wins at 100k but the gap vs n_subs=2 collapsed to ~1% (real, ~4× per-sample std, but tiny). Trend of the n_subs=4-vs-2 gap: 8k n/a → 30k 5% → 100k 1%. The crossover (where smaller n_subs wins) lands around kv≈120-150k — beyond the 105k supported range. **Dynamic sizing NOT worth the complexity** — n_subs=4 wins at every measured kv, the 1% boundary win at 100k is below the "reject tiny speed gains that add complexity" bar. Fixed default `PREFILL_SUBS=4` confirmed optimal across the whole supported context range. (100k prefill = ~14.8 min total = within the v1 acceptance "100k ≤ 16 min" bar.) |
| **Steel-loader GEMM port (software-pipelined double-buffering)** — **CLOSED, DISPROVEN 2026-07-14** | The parked "≤2-3% end-to-end at ~10-15% kernel gap" estimate was optimistic; the real prototype REGRESSES. Built `gemm_tunable_db` (sibling entry in `gemm_tunable.metal`): doubles the tgmem tiles `Xs[2][BM][PAD]` + `Ws[2][BN][PAD]`, runs a prologue load of tile 0, then in the K-loop overlaps the device→tgmem load of tile N+1 with the MMA of tile N (one barrier per K-tile vs two in the single-buffered kernel). Bit-exact vs the single-buffered `gemm_tunable` (Tier-1 `gemm_tunable_db_bitexact_vs_single_buffer` test green — the K-accumulation chain, dequant, and store rounding are unchanged, only the tgmem buffering schedule differs). Benched head-to-head at the prefill-relevant dense shapes (256×2816×2816 + 1024×2816×2816, production tile 64×64, q4 weights): double-buf = **3.377 / 3.566 TF/s** vs single-buf = **3.611 / 3.919 TF/s** → **0.93× / 0.91× (7-9% SLOWER)**. PHYSICS: the device→tgmem load is already fully hidden behind the ~6× compute margin (compute >> load, so the single-buffered kernel's load already overlaps with the next K-tile's compute via the GPU's natural instruction issue). The extra barrier sync + doubled tgmem footprint (hurts tile occupancy / SLC pressure) costs more than the explicit load/MMA overlap gains. There is no async-copy engine on Apple GPU — the "overlap" is just re-issuing the load instructions before the MMA, which the single-buffered version already does implicitly because the GPU issues load/store and matrix-unit instructions concurrently. Same shape as the int8 dot disproof: the physics (compute-bound regime, load already hidden) predicted the result, and a real prototype on real shapes confirmed it. `gemm_tunable_db` + `bench-gemm --shapes db` kept as documented negatives; not wired to production. Lesson: when compute >> load (the 6× margin regime), software-pipelined double-buffering is a regression, not a lever — the doubled tgmem footprint + extra sync costs more than the already-hidden load overlap returns. The "no async copy on Apple GPU" framing was a red herring — the missing primitive is not the issue; the load is already overlapped by the GPU's natural instruction issue. |
| **Long-ctx re-validation debt** | Post-root-cause-fix: re-run needle 33k/105k and the 100k field-incident repro on the uncapped fast path |
| **E5 fragment-tile attention** — **CLOSED (2026-07-13); QK-ILP2 re-test CLOSED 2026-07-14 (BO-confirmed inert)** | Full axis sweep on `attention_mma_full` @kv 60k, all built+measured: MT 8→16 wash; NKT key-blocking (barriers ÷8) slower; fragment-array O 1.7× slower (Metal spills fragment arrays); grouped-PV unload (barriers 64→16) neutral; QG=4 (4 warps) neutral; occupancy disproven as binder (side-off = 2%); f32-side ring = 2%. The one survivor — QK ILP2 chain-split (~5% kernel / ~3% prefill on the *pre-E17* baseline, non-bit-identical) — was re-grafted 2026-07-14 as FC31 `DGQ_ATTN_MMA_FULL_QK_ILP2` (two interleaved 16-deep accumulator chains, even/odd chunks; `st_ilp` second tgmem slot; softmax sums 4 partials). Parity green (step-probe identical max_abs at every stage). **A single-axis A/B at default settings was INVALID — it didn't exercise the path** (E17-on routes full layers to `attention_gemm`, so `attention_mma_full`/ILP2 is inert at default). The correct-regime manual A/B (E17-off) showed a real but small signal: ILP2 = **−1.5%** (4611→4542 ms/super-chunk at kv=15k, 3-trial mean). **The holistic BO (24 trials, joint {E17, ILP2, tiles} space, `tune_prefill_attn.py --proxy`) then confirmed ILP2 is materially inert**: all top-8 configs have `E17=on, ilp2=off`; the best = the shipped default (3538 ms, 1.00× vs default). E17-on (1.37× over mma_full) dominates ILP2's 1.5%, and within E17-on the ILP2 flag is a no-op (mma_full doesn't run for full layers). Kept behind default-OFF FC31 as a documented conditional-negative; not wired to production. Kernel is pinned at ~1.25 TF/s: the hd=512 / 8×8-fragment shape ceiling. Prefill A/B vs MLX-4bit (chunked, `prefill_step_size=512`): MLX 1.4×(10k)→2.4×(100k) faster; the gap ≈ our full-attn kernel's TF/s deficit exactly |
| **E17 GEMM-attention for full-layer prefill** — **SHIPPED default-on 2026-07-13 (`DGQ_GEMM_ATTN`, 79d5829 M1 + b6be181 M2); golden 8/8 confirmed 2026-07-14** | `src/shaders/attn/attention_gemm/`: full-layer PREFILL attention decomposed to S=Q·Kᵀ → rowwise masked softmax → O=P·V, both matmuls reusing gemm_tunable's 64×64 fragment tiler (QK stages K native; PV stages V transposed). Head-batched (grid.z=head, per-head S/P scratch) — 3 dispatches + 2 barriers/layer; per-head serialization was a 1.1× trap. MEASURED: attention kernel 1.75-1.79× vs attention_mma_full; **real prefill 30k 120.8→98.2s (−19%)**, 8k −7%; closes MLX gap at 30k 1.64→1.25×; win grows with context. Gate PASS: smoketest 17/17 ×{7,42,123}, longctx 4/4. **Golden 8/8 byte-identical confirmed 2026-07-14** (default-on, with E17b f32-side-KV variant default-on — the earlier "3/8 diverge" was the f16-KV path; the f32-side path matches the `attention_mma_full_side` baseline exactly). Default ON (env_on_unless_zero). Follow-ups done: E17a head-chunk (HC=16 shipped, memory precondition met), E17b f32-side-KV (FC30, closed the divergence). |
| **E17a: head-chunk the S plane** — **DONE 2026-07-13 (fba19e3, `DGQ_GEMM_ATTN_HC`=4)** | S/P scratch now `[HC][CANVAS][n_pad(max_seq)]` — kernel `head_base` splits global data head (`head_base + tgid.z`) from batch-local scratch (`tgid.z`). Numerically invariant (golden HC=4 == HC=16 byte-for-byte). Sweep: HC16 1.78×/1.6 GiB, HC4 1.66×/0.4 GiB @100k (~5% kernel, 4× memory cut); real 30k −16% (99.9s) vs −19% (98.2s). Default HC=4 at the time; **superseded 2026-07-13 — `DGQ_GEMM_ATTN_HC` default is now 16** (holistic-BO optimum, bit-identical; see the Holistic prefill BO row). NB E20 top-k does NOT read the flag — it hardcodes `TuneCfg::default().hc`=4; see the E20 row. **Memory precondition for default-on MET.** |
| **E17b: f32-side-KV GEMM variant** — **DONE 2026-07-13 (913cf52, FC30)** | Generic: `frag_mma_ktile_f32` added to the shared `gemm_frag_tile.metal` (float staging + simdgroup_float8x8) + FC30-guarded loaders in attn_gemm_qk/softmax/pv — one seam, no 4th kernel. Reads the f32 side ring (buffer 9), all-float MMA, f32 probs; selected when `DGQ_GEMM_ATTN && DGQ_PREFILL_KV_F32` (default), f16 GEMM fallback otherwise. Oracle: matches raw-f32-KV CPU ref (tighter than the f16 path). Gate PASS: 17/17 ×{7,42,123}, longctx 4/4. **FINDING**: f32-side gives the SAME golden as f16 (3/8, identical diverge indices) — the flag-on flips are NOT KV precision (baseline `attention_mma_full_side` already uses f32-side KV) but the decomposition's batch-softmax vs the flash kernel's online softmax. So E17b is precision-matched (no KV regression) but doesn't reduce the divergence. Default-on remains gated on sign-off + golden re-bless. |
| **Holistic prefill BO (task #88)** — **DONE + SHIPPED 2026-07-13** | 9-dim TPE sweep (attn QK/PV tiles, HC, softmax TPG, dense-GEMM tile, MoE-sparse N-tile) against the faithful `bench-prefill-super` proxy, default pinned as trial 0 (proxy variance 0.08%, 5-run). Result: **`hc=16` the optimum at EVERY kv, monotonic** (−0.28/1.50/3.13/3.61% at 2k/8k/30k/100k vs hc=4); all other levers stay at default (GEMM/MoE floor already tuned, attn tiles a non-lever per the kv_block lesson). Shipped: default `DGQ_GEMM_ATTN_HC` 4→16 — bit-identical (HC numerically invariant, golden 8/8). Scratch ~1.6 GiB @100k. |
| **E18 fused flash prefill** (NEW 2026-07-13, **BUILT + DISPROVEN-AS-DESIGNED; rewrite pending**) | Kernel built + CORRECT (oracle 3/3 vs cpu_causal) but **5.8× SLOWER than E17** at hd=512 (isolated: 8k/30k/60k/100k = 0.17/0.17/0.18/0.18× E17). Root cause = the hd=512 constraints predicted up front: `BQ=16` (register-forced) → tiny under-utilized MMA tiles, and the 32 KiB tgmem limit forces V staging in 8-key sub-blocks → ~80k threadgroup barriers/head-tile at 100k → barrier-bound. The killed S/P device traffic is real but swamped. **Confirms E17's materialize-S/P was the right call for hd=512**, and answers the steel-lever question: we're not missing a free lever; flash doesn't fit hd=512 cheaply on M3 (32 KiB tgmem, no async copy). Kept behind default-OFF `DGQ_FLASH_PREFILL`. **Device-load rewrite DONE — still loses**: v2 simdgroup_loads K/V directly from device (K transposed, V untransposed, no tgmem staging, +BK KV pad, f16-KV only) → killed the ~75k staging barriers → 2× faster than v1 but **still 3× slower than E17** (0.33× flat, 8k–100k; oracle 2/2). FINAL: the floor is structural — hd=512 caps BQ at 16 (O=BQ×hd resident) → tiny under-utilized MMA tiles, + transposed device K-loads uncoalesced; E17 avoids both via materialize + 64×64 GEMMs. **Flash doesn't fit hd=512 on M3; E18 CLOSED as a documented negative** (default-OFF `DGQ_FLASH_PREFILL`). Revive only if hd shrinks / async copy / bigger tgmem. Original design notes: | Audit of MLX steel source (`steel/gemm/{gemm,mma,loader}.h`, `steel/attn/attn.h` vs ours): GEMM side has **nothing to steal** — our `frag_lane_coords` is a byte-identical port of steel's `get_coord`, our K-loop is the same barrier-load-barrier-mma, and **steel is single-buffered too** (no async/prefetch trick on Apple GPU). The gap is attention *dataflow*: E17 `attention_gemm` writes **S=Q·Kᵀ to device**, reads it back for softmax, writes **P to device**, reads it back for PV — ~4 full `canvas×T`-plane device round-trips/head (this is exactly the `hc` scratch the holistic BO is throwing head-parallelism at to *hide*; the `hc=16` win grows with T because the S/P traffic grows with T). Steel's SDPA is **flash**: online softmax, O in registers, K/V streamed in blocks — S/P never touch device. **Design**: new kernel that drives QK and PV through the shared `frag_mma_ktile` (E17's steel-efficient 64×64 MMA) but keeps O resident + running max/sum — i.e. E17's GEMM throughput *without* E17's S/P traffic. We own both halves separately today (`attention_mma_full` = flash-but-less-GEMM-efficient, lost to E17 for prefill; `attention_gemm` = GEMM-efficient-but-materializes); E18 = the union steel already ships. **Ceiling**: attention is SLC-bandwidth-bound; flash removes ~4×(`canvas·T`) half-plane traffic but the K/V read (`T·hd`) is irreducible → at hd=256/canvas=256 roughly *halves* attention device traffic at long T ≈ the residual MLX gap (1.2–1.5×). **Bonus** (per E17b finding): online softmax may re-align golden with the `attention_mma_full` baseline (E17's flag-on flips were batch-vs-online softmax, not KV precision). **Risks**: must hit steel's occupancy (plain `mma_full` didn't for prefill); online-softmax rescale of the register O tile adds per-K-block work; correctness of the causal + sliding-window mask under blocked streaming. **Resurrect-along** (see principle above): fold E5's QK-ILP2 chain-split; pick online softmax deliberately. **Do NOT transfer the flash-*decode* disproof** — that was M=1 (online softmax over one query = pure overhead); prefill is M=256×large-T where not materializing the plane is the whole point. Gate: golden + smoketest 17/17×{7,42,123} + longctx 4/4 + real `ask --prompt-len` A/B + `bench-prefill-super` proxy vs E17. Opt-in flag first (`DGQ_FLASH_PREFILL`), default OFF pending sign-off |
| **E7 confidence-threshold sampler** — playbook deepened 2026-07-16 (full doc in session scratchpad `e7_confidence_sampler.md`) | MLX's alternate accept rule (`diffusion_threshold=0.9`): accept every canvas position with top-token confidence ≥ τ, vs our shipped `EntropyBoundSampler` (`sample.rs:389` — entropy-sorted prefix-sum BUDGET per step; easy spans still take ~scheduled steps because the budget, not the difficulty, sets the count). Threshold-accept lets easy spans commit in 1-2 steps. **Semantics pinned before coding:** p_max from the same distributions the entropy reduction already reads (~free); floor = at least the schedule's per-step count (hybrid) with literal-MLX as the A/B parity mode; MUST be measured WITH early-stop on — their savings overlap, report MARGINAL steps_eff. **Wart hypothesis (the actual reason to build it):** warts = flat-distribution creative-tail rows accepted by budget despite low confidence; threshold refuses exactly those until they sharpen — prediction: census warts at τ=0.9 ≤ the 6/10 early-stop baseline. **M0 (one evening, zero behavior change): instrument p_max histograms + would-accept counts at τ∈{0.8,0.9,0.95} across the 17-prompt gate; KILL if predicted marginal savings <8-10%** (early-stop already banks −15-35%; convergence is parity-class, so honest headroom = 5-15% steps + the census delta). M1 `DGQ_ACCEPT_THRESHOLD` (0=off) + matched-canvas multi-seed A/B (mlx-generation-equivalence harness — single-seed step counts are meaningless, lesson already paid). M2 gates: smoketest ×{7,42,123}, census multi-seed (the decision gate), longctx; golden negative expected; ship only gate-neutral-or-better. The M0 instrument doubles as a diagnostic for E22-M3's quality runs. |
| **E3 canvas shrink near max_tokens** | Close divergence #5 (MLX shrinks to max(remaining, 64)); minor tail win; trajectory-affecting → multi-seed gate |
| **v1 productization** | README quickstart (none exists), fetch/quantize UX one-liner, benchmark page with the MLX methodology, release tagging + `--version` with `.dgq` manifest-version gate |
| **CI completion** | Nightly model-gated tiers are scaffolded; wire fully (smoketest + golden + longctx + perf floors), weekly multi-seed aggregate + census |
| **Broader eval** | The 17-prompt gate is sensitive but narrow; add a ~100-prompt adherence set, weekly, non-blocking |

## Code-health refactor backlog (2026-07-12 audit; execute in order)

1. Warning zero + CI deny-warnings gate: `cargo fix` the mechanical ~48, triage the 57 dead_code (delete stale bench scaffolding / `#[allow]` kept diagnostics).
2. Dead-residue basket: dead encoder imports in generate.rs, `block2` dep (0 refs), `div_up` ×5 → `gpu_common`, 7 stale pre-refactor path comments, main.rs duplicated FAILURE dispatch arms, `flags::Config` → `RuntimeConfig`. Flag removal RESOLVED: no genuinely-dead `DGQ_*` flags exist (the disproven ones are still wired A/B toggles, kept per the ledger). NOTE: items 1-2's non-test dead-code pruning over-cut test-only re-exports (`metal::DgqGpuBlob`, `metal::build_offsets_from_store`, `sample::ARGMAX_HIST_MAX`, `LivenessCtx.first_step`) — restored test-gated so `cargo check --tests` is clean again while the non-test build stays warning-zero.
3. ~~Extract inline tests to sibling files~~ **DONE (0390762)**: test mods moved to flat `#[cfg(test)] #[path] mod NAME;` siblings (step_kernel_tests / step_kv_bench_tests + step_kv_encoder_moe_tests / server_tests + server_tool_smoke_tests + server_tool_compact_tests / step_generate_tests). step_kernel.rs 6193→5529, step_kv.rs 2336→912, server.rs 2098→1568, step_generate.rs 1177→1145. Verbatim move (child modules, `use super::*` unchanged), suite 541/0, golden 8/8.
4. Split step_kernel.rs along impl seams. **Part 1 DONE (0f072a2)**: diagnostics extracted to `#[path]` child modules — `step_kernel_diagnostics.rs` (1629; probe/capture/bench) + `step_kv_audits.rs` (691; run_step_kv_* parity/probe), so step_kernel.rs 7798→6193 and step_kv.rs 2989→2336; verbatim move, golden 8/8. **Core carve DEFERRED (own gated task, needs care)**: step_types / step_pipelines / step_dispatch (StepEnc) / step_runtime is NOT mechanical — StepEnc (152 fns) is struct-literal-constructed by StepRuntime and StepPipelines is a 50-field struct read throughout StepEnc, so a peer-module split forces ~50-100 `pub(crate)` field promotions or introduced constructors/accessors on the hottest production file. If done: move StepEnc+StepRuntime+build as ONE execution child (they keep seeing each other → far fewer promotions), leaving the parent = types + StepPipelines. Full golden+suite gate mandatory.
5. ~~Split main.rs (5.4k)~~ **SPLIT DONE (split-only)**: `cli.rs` (Cli/Command + parse_cli) + `commands/{mod,common,step_debug,step_gate,bench,gen_cmd,chat,smoketest,golden_cmd,model_ops}.rs`; main.rs now 50 lines. Every top-level item moved verbatim (no deletions, no MoE collapse). **Prune deferred**: the "cold" dump subcommands each pair with a `python/scripts/{dump,compare}_*.py` oracle (manual layer/attn/MoE/embed parity-debug harness) — not dead, so retiring them + collapsing the 4 `step_moe_*_dump` is its own sign-off decision, not a mechanical refactor.
6. ~~`GemmCompileConfig` struct in device.rs~~ **DONE (3bdd03f)**: named-field config replaces the trailing-4-bool positional footgun on `compile_gemm_subkernel_on_device`; 4 preset constructors (raw/out_bf16/gather/rowk_arena). The 4 public wrappers kept their signatures (17 call sites unchanged) → bit-identical, golden 8/8.
7. `crate::Error` — **DONE (signed off)**. Moved the enum `safetensors::Error` → `src/error.rs`, surfaced as `crate::Error`, migrated all 126 import sites (safetensors keeps a back-compat re-export). Added `Gpu` / `Runtime` variants and reclassified the 563 `Error::Format` catch-all sites (388 Gpu / 100 Runtime / 75 Format) — purely descriptive (nothing matches the variant), so cosmetic-but-clarifying. **The classification was done by DiffusionGemma itself** (`serve` driven over the 216 unique messages, 0 parse failures, labels used verbatim; where it disagreed with a keyword heuristic it was often the better judge). Parity-gated: build -D warnings clean, suite 540/0, golden 8/8.
8. `metal/oracle/` quarantine — **PREMISE PARTLY WRONG (audit erratum)**. `engine` / `decoder` / `kv_cache` are NOT oracle: they run the **production** encoder-prefill (`MonolithicEncoderCache` / `prefill_monolithic_kv`, called by `StepGenerateSession` at step_generate.rs:562-1070 for every real generation). Quarantining them would mislabel load-bearing code. Only a subset is genuinely oracle-only (attention_scalar has 0 non-test refs; verify sampler/self_conditioning/lm_head/decoder_attention/kernels/sampler_kernels per-module before moving). Needs an accurate oracle-vs-production audit + a decision on the encoder-prefill cluster; not the clean mechanical move the estimate assumed.
9. Judgment-call items (need sign-off). **`pack/` retired (signed off)**: `iris.pack` was confirmed dead (no `iris.pack.json` anywhere, gate model is `.dgq`, zero tests) — deleted src/pack/ (473 lines) + the `WeightStore::Packed` variant + `is_packed()` + the `convert-model` CLI command (existed only to produce iris.pack) + 3 now-dead fast_slice.rs helpers; golden 8/8, suite 540/0. **Oracle GEMM fixtures parametrized (signed off)**: the 4 near-copy dense fixtures (bf16/q8/q4/nvfp4, 824 lines) now share `oracle/gemm/fixture.rs` (Fixture + coefficient builder + `bind_gpu_buffers` + the `run_gpu` runner); each format file keeps only its CPU reference, weight-quant, and `QuantFormat` → 824→542 (the 4× duplicated GPU dance collapsed to 1). Parity-gated: suite 540/0, golden 8/8. **Renaming aliases finished (signed off)**: of the 8 `pub use X as Y` re-exports, only 2 were genuine legacy renames — `gemm_q8_rowk` (canonical `gemm_rowk` already exported → pure duplicate) and `gemm_q8_linear_kxn_f32` (pre-merge name for `gemm_q8_linear_f32::kxn`). Migrated their module-path refs to canonical and dropped the 2 aliases; the other 6 (`gemm_common`, `rms_norm_rows_tiled`, `moe_batched_pin`, `moe_grouped_nvfp4`, `sampler_ranged`, `sc_probs`) are descriptive **pass-through flat aliases** (KEEP — they flatten a nested/generic path into the intended `shaders::<name>`). The "148 sites" over-counted kernel-named struct fields; real module-refs migrated ≈18. Bit-identical, suite 540/0, golden 8/8. **CPU MoE routers + rms twin deduped (signed off)**: `route` and `route_with_cached_weights` (identical but for weight source) now share a `route_core` fed disjoint scratch fields (bit-identical, no clones; exact tie-break stays in `top_k_route_from_raw_logits`); `rms_norm` now delegates to `rms_norm_no_scale` + weight. No genuine RoPE twin existed — `apply_rope_tensor` was already the single CPU impl. Parity-gated: suite 540/0, golden 8/8. **model/decoder.rs — RESOLVED: KEEP (signed off)**. Not dead: the full-stack CPU `forward` is the reference half of `step-logits-dump` / `step-bf16-logits-dump` (pairs with `python/scripts/compare_step1_logits.py`), and its `DecoderForwardInput` type is shared with `step_m0`. Same category as the item-5 python-oracle-paired dump commands we kept; deleting it would break a live diagnostic and remove the only full-stack pure-CPU forward. The "(test-only)" label was imprecise — it's diagnostic-CLI-only, not suite-dead. **Item 9 complete.**
10. `serve/` cluster split — **DONE (worker + http/sse)**: `server.rs` 1568→687 via two `#[path]` children — `server_worker.rs` (the GPU-owning `Worker` impl) + `server_http.rs` (request parse + SSE/JSON layer). Wire types + `DiffusionStreamMapper` stay in the parent as shared vocabulary (heavy cross-use; not worth the pub(crate) churn). suite 541/0, golden 8/8. `lcp()` triplication CHECKED (server `lcp` / conversation `common_prefix_len` / step_generate `longest_common_prefix`, all `&[u32]`→usize) — left as-is: deduping a 6-line helper across three module clusters (incl. the step_generate hot path) isn't worth the coupling.

## v2 parking lot

- **Vision tower** (SigLIP encoder + image splicing; ~2+ weeks; v2 headline)
- **E9 rotated experts** (near-bf16 fidelity within the 4-bit budget;
  PolarQuant-class port, rotation ≈ 98% of the gain — prove with plain
  absmax q4 first; only after any KV-rotation infra exists)
- **E10 precision-decay KV** (f16 recent window + low-bit aged bulk; value
  is 18-24 GB Macs / >262k, not 36 GB)
- **q6/q5 non-expert weights** (memory lever redundant with q8-KV-auto on
  36 GB; scope = attention+FFN only, exclude embed/lm_head)
- **E8 rotated/un-RoPE'd KV** — parked on value; revive only if q4-KV
  becomes necessary (see Negative Knowledge)

## v1 acceptance (unmet items only)

- Install-to-first-token < 30 min documented on a clean 36 GB Mac (README).
- Perf floors regression-gated in CI (step ≤ 1.1 s chat lengths; 33k prefill
  ≤ 140 s; 100k ≤ 16 min prefill, ≤ 5 s/step).
- Scope statement in README: text-only v1, 36 GB minimum RAM.

## Risks

- **Single-machine evidence** — every number is one M3 Pro; recruit at least
  one other M-series config before publishing claims (SLC-locality physics
  may differ on M1).
- **Upstream drift** — MLX head-to-head cites their current 4bit; refresh
  before publishing benchmarks.
- **Gate breadth** — 17 prompts; see broader-eval item.
