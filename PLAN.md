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

### Channel-hygiene fixes (survey 2026-07-17; #1-#3 done, #4 partial)

~~**Field incident 2026-07-17 late (regex_lite session, turn 14) — the
dual-splitter disagreement has a live casualty.**~~ **FIXED (782345c)**:
the model skipped the thought ceremony and emitted a bare, well-formed
edit call; the mapper's old "everything is reasoning until a close id"
rule streamed it all as `reasoning_content` and the client got an EMPTY
message while the validator (rightly) judged the reply clean. `split()`
now classifies by EMISSION (reasoning is only what sits inside an explicit
thought span), the CHANNEL MISMATCH tripwire logs any mapper/validator
disagreement, and both assembly paths ADOPT calls salvaged from the raw
decode / round reasoning instead of only logging.

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
2. ~~**`strip_thinking` is not quote-aware**~~ **FIXED (2026-07-18,
   masked-scanner commit).** `tools::masked_ranges` computes the literal
   regions of a reply — complete `call:NAME{…}` arg bodies (anchored at the
   grammar via the same consumption scan as `parse_tool_calls`; global
   `<|"|>` parity deliberately NOT used — one displayed quote literal in
   prose would flip it and hide every later marker) plus closed ``` fences
   that don't straddle a real call's args. EVERY string-side marker scan
   goes through it: `strip_thinking`, `sanitize_model_reply` turn/channel
   cuts, `thinking_spans`, `has_incomplete_tool_call` (a quoted
   `<|tool_call>` literal no longer holds the turn open),
   `has_trailing_after_tool_calls`, `scan_call_attempts`,
   `parse_tool_calls`, `content_before_tool_calls`, `validate_tool_reply`.
   An edit call writing this repo's own `chat_template.jinja` now
   round-trips byte-exact. Unmasked semantics are pinned unchanged by the
   pre-existing suite; new tests: `quoted_*`/`fenced_*`/`masked_ranges_*`
   (tools), `sanitize_keeps_quoted_markers_cuts_unquoted`.
3. ~~**Channel-unaware stop-scan**~~ **DONE (same commit), gated**: engine
   stop-scan (`first_unquoted_stop`) and the mapper stop-cut skip stop ids
   inside an open `<|"|>` run (parity carried across blocks), but ONLY
   under `continue_incomplete_tool_calls` (`DGQ_CONTINUE_PAST_STOP`,
   default OFF) so engine and stream stay in lockstep and the 2026-07-17
   honor-the-stop policy is preserved; default path still ends the turn
   and lets tool-repair regenerate. `quote_token_id` rides the op-log cfg
   (absent in old logs → plain scan, replay-faithful). Revisit the gate
   only with field evidence of quoted-stop truncations.
4. **Dual thought-splitters — PARTIAL**: mapper `split()` is now
   quote-aware at the id level (quoted channel ids stay in content
   verbatim; unconditional channel-id filters removed — masked
   `sanitize_model_reply` handles text-form leftovers), so both splitters
   share the marker vocabulary and the literal-region rule. Remaining:
   one walk implementation (string-scanner-driven mapper) — the CHANNEL
   MISMATCH tripwire + salvage covers the residual disagreement window.
5. `channel_id` None fallback hardening; 6. thinking-flag flip silently
   loses KV reuse (add a log line).

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
- **Prefix-exit early block commit `DGQ_PREFIX_EXIT` (landed default OFF;
  reframed speed→quality)**: per step, commit the settled active/2 head
  (mean ent < τ, 2-step stable) when the tail is HARD-stuck (mean ≥
  max(2τ, 0.3) nats); tail re-denoises next block; conjunctive dup-scan
  always runs on the exited head. LESSONS: (a) the 28%-savings offline sim
  was wrong — replay never charges the successor block, and live small-head
  (/4, /8) exits CASCADED into mini-block chains (blocks 19→23, steps net
  worse) — trajectory-feedback sim bias, now a memory; (b) with the
  tightened rule: smoketest 17/17 ×3 (fires ≈0 on short prompts — correct),
  strain 3 faster / 2 parity / 1 healthy long-fork (trajectory variance
  dominates timing; zero stutters anywhere; stuck blocks salvaged at 9–19
  steps instead of schedule-burn at mean-ent 0.7–1.5). VERDICT: not a
  reliable speed lever; quality-safe. KEY EXHIBIT (clean/seed-123
  regex_lite turn): base arm thought 141 tokens (plan restatement, 2
  blocks); prefix-exit arm thought 2,040 tokens (glob-vs-regex semantics
  deliberation, UTF-8 hazard caught, full matcher drafted in-thought, 14
  test cases; 12 blocks) — stuck tails ARE mid-thought states, and exits
  honor them instead of steamrolling; both arms emitted clean tool calls
  (single sample; needs the quality gate). NEXT: quality-mode experiment —
  aggressive exits at matched TOKEN budgets, judged on census multi-seed +
  strain tool-arg typo rates (the "commit-when-stable ≈ more-causal
  factorization ≈ fewer independence violations" hypothesis; VSB
  arXiv:2604.23994 reports +4–10% from the trained analog).
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

- **Hard-kill flag validation (user-directed 2026-07-18)**: any `DGQ_*`
  flag present in the env with an unparseable or out-of-range value must
  exit the process with a clear message instead of silently falling back
  to the default (today `DGQ_PREFIX_EXIT=1` silently disables — the exact
  footgun). Implement centrally in `RuntimeConfig::from_reader` so every
  flag gets it; keep unset = default. Mind `EMPTY_ENV`/test parsing and
  the drift-tripwire pattern.
- **KV reuse+delta ≠ fresh prefill (PROVEN divergence, 2026-07-18; the
  likely "KV-lineage drift" root)**: chasing a live `existingIf`
  insertion-typo seam (which is NOT an off-by-one — it committed
  mid-block in a `confident`-stopped block, the non-dup cousin of the
  stutter; the analog reproduced in replay at p_max 0.405, `ownBut`)
  established: byte-identical prompt (58,017 chars), same seed/flags/
  binary, engine fully deterministic (two identical fresh-prefill
  requests → 47/47 blocks bit-identical argmax) — yet the live turn
  (reused 14,509 + 266 delta-extend, epoch 22, ring wrapped ~3.5×)
  forked from fresh prefill at block 1. Therefore reuse+delta KV is
  numerically ≠ fresh-prefill KV. The f16 lineage matrix (fresh vs
  prefix+delta vs overshoot+truncate(+re-extend)) is IDENTICAL at every
  config including wrapped and is a live Tier-1 gate. **The q8 arm's
  "ring-wrap path-dependence" was REFRAMED by forensics (2026-07-18,
  `q8_ring_wrap_divergence_probe`) — see the q8 item below; crucially,
  the live ctx-100000 serve ran F16 (measured: `q8_auto_threshold_crossover`
  — 27.00 GiB working-set cap, q8 engages only at max_seq >= 178,176), so
  the LIVE fork is NOT explained by q8.** Environmental caveat RESOLVED
  — the divergence is real and internal. Artifacts: scratchpad pmax/
  turn11 traces, /tmp/logs (live), ops line 232.
  **REPRODUCED ON f16, 2026-07-19 (`kv_lineage_unaligned_delta_offsets_*`,
  KNOWN RED)**: the aligned matrix could not reach the live shape. The
  sliding ring (4096) is a multiple of the 256-token prefill chunk, so a
  256-ALIGNED resume can never put a chunk boundary mid-ring; every
  matrix case was aligned (6144/5376) while the live turn resumed at
  14,509 (residue 173). Feeding the live shape in forks — on the
  production f16 path. Verdicts: misaligned + ONE-chunk delta = OK;
  misaligned + MULTI-chunk delta = FORK; the live 14,509+266 = FORK.
  The fork needs ALL THREE of wrapped ring + misaligned resume +
  multi-chunk delta: below the ring every alignment x chunk-count combo
  is clean (`kv_lineage_alignment_vs_chunkcount_sweep`), and the aligned
  matrix passes at every size. Batched super-chunks EXONERATED
  (`kv_lineage_unaligned_batch_bisect` — identical with prefill_batch
  on/off). Position map: layers 0-5 clean across the whole ring, every
  layer from 6 on diverges starting at EXACTLY the resume position,
  max|delta| growing 0.5 -> 9.3 with depth — so the seed is earlier and
  sub-ULP (f16 KV hides it until it grows), and layer 6 is a VISIBILITY
  threshold, not the origin. NEXT: pin the phase-dependent reduction.
  Leading hypothesis: the same absolute position occupies a different row
  index / a different `kv_len`+`t_total` tiling in the two arms, so the
  key-axis reduction is grouped differently (E17/E20 full-layer GEMM
  attention tiles by `t_total`; mma2's tile start is 8-aligned in
  ABSOLUTE position and so should be phase-free — start by diffing a
  single layer's attention output between arms at one position, and
  bisect flags: DGQ_GEMM_ATTN / DGQ_ATTN_TOPK / DGQ_ATTN_MMA). This is
  a bit-identity break that AMPLIFIES through depth into a different
  trajectory, which is exactly the field symptom.
- **q8 fast-prefill is broken WHOLESALE (forensics 2026-07-18,
  `q8_ring_wrap_divergence_probe` — supersedes the "ring-wrap requant
  seam" framing)**: in any q8 session the fast-prefill forward goes
  **NaN** after layer 0 — layer-0 Q/K/V land real (verified byte-level),
  yet the attention output plane comes back non-finite on BOTH the
  scalar and mma2 kernels, and the plane trace shows
  hidden/attn_out/ffg/dense/moein all `non_finite=true, max_abs=0.0`
  from row 0 on. Every later layer then quantizes a NaN hidden stream:
  the tell is a KV row of ALL `-127` codes, which is the q8 signature of
  a NaN source (`fmax(NaN,lo)=lo`, so Metal's `clamp(NaN,-127,127)`
  yields -127; it is NOT a quantized zero — proven by the scale-floor
  fix below, after which the scales became normal but the codes stayed
  -127). Established by elimination: scrubbed builds are
  deterministic AND path-independent (the RED "divergence" needed
  cross-build residue: unwritten rows keep whatever the previous build
  left); chunking only changes which DEAD bands survive ring
  overwrites; the f16 control writes every row in every layer.
  Exposure: auto-q8 needs f16-resident > 85% of the GPU working set
  (MEASURED on this 36 GB M3 Pro: cap 27.00 GiB, crossover max_seq
  178,176) or a forced `DGQ_KV_Q8=1` — so production has effectively
  never run q8; nothing gates it (the
  attention harness has ZERO q8 coverage; smoketest is short-prompt;
  golden deliberately sizes `GOLDEN_MAX_SEQ` to stay UNDER the q8-auto
  threshold, so it only ever gates f16). NEXT: (1) isolated q8
  attention harness case (scalar + mma2 vs the CPU oracle over a
  crafted q8 cache) to pin where the NaN is born — the session-level
  probes can see it happen but not inside one dispatch; (2) fix; (3)
  un-ignore `kv_lineage_paths_are_fingerprint_identical_q8`; (4) only
  then is q8 a usable >178k memory lever. The probe test documents
  every forensic arm and stays runnable.
  TWO QUANTIZER DEFECTS FOUND + FIXED ALONG THE WAY (upstream of the
  NaN, not its cause): the q8 group-scale floor was `1e-8`, which is
  unrepresentable in f16, so (a) it rounded to +0.0 and `x/scale`
  divided by zero — the NaN clamping to -127 on GPU but 0 in the Rust
  mirror, and (b) Metal emits subnormal halves while this crate's
  `f32_to_f16_bits` FLUSHES subnormals to zero, so every group with
  `max|x| < 127*2^-14 = 7.75e-3` stored a different scale on the CPU
  packer than the GPU kernel — two silent CPU/GPU divergences in a pair
  documented as bit-for-bit. Both closed by flooring the scale at the
  min NORMAL half (`Q8_MIN_SCALE` / `KV_Q8_MIN_SCALE`, 2^-14), which is
  a fixed point of both conversions; real KV rows sit orders of
  magnitude above it, so no realistic row's bytes change (golden 8/8
  unchanged). Test: `q8_small_groups_quantize_finitely_at_a_representable_scale`.
- **Exact-prefix repeat misses reuse**: re-POSTing an identical request
  paid `truncate_kv_to` ring rebuild 22263→14775 (39.45s!) then
  re-prefilled all 14,775 with `reused 0`. Should be ~100% reuse
  (KV-reuse-first). Read `route()`/`reset_kv_unless_extends` for the
  delta==0 edge.
- **Two-tier conf-trim LANDED; microscope A/B run (2026-07-18)**: the
  contested-row commit class has THREE surfaces — duplication, insertion
  (`냥`, `("."`, `ownBut` 0.405), and OMISSION ("Then I'll ~~start~~ with
  line 61" — the whole clause committed at 0.296–0.49). Hard tier =
  unconditional `p_max < 0.5` trim alongside the dup 0.9 tier (same
  answer-region scan/floor). Live OpenCode regex-lite microscope
  (tool-repair + PREFIX_EXIT=0.05, traced): baseline 2 runs = 29 sub-0.5
  commits in 152 blocks incl. a missing-word clause, a `,,` inside Edit
  args adjacent to the schema-error retry, a garbled phrase; treated
  (adds trim) 2 runs + full strain battery = **0 sub-0.5 commits / 0
  stutters in 137 blocks**, 36 of which PREFIX-EXITED — the layers
  compose: prefix-exit defuses stuck tails upstream, dup tier catches
  commit-time stutters, hard tier is the insertion backstop (validated
  on baseline data; not yet observed firing live because upstream layers
  starve it). CAVEAT recorded: the OpenCode-arm comparison is partly
  trajectory luck (baseline runs stumbled into repair cycles, treated
  didn't); the strain-battery result is the controlled evidence. Both
  trim flags remain default OFF pending census+longctx.
  FOURTH SURFACE, OUT OF SCOPE for confidence gates (field 2026-07-18,
  alpha/beta run): CONFIDENT MISCOUNT of repetitive content —
  `betababeta` committed at p_max ≈ 1.0 (`bet·ab·ab·eta`, one BPE piece
  dropped from "betabetabeta") inside thinking, contradicting the
  correct observed tool output and triggering a 68s "Wait, let me
  re-count" ×5 verification spiral (final answer still correct; cost is
  latency). Confidence-gating is blind here by construction — the model
  is wrong AND sure. Full anatomy (all confidence-verified): miscount
  (≈1.0) → faithful propagation ("alphaalphaalphabetababeta" at
  0.92–1.0) → INSTANT confident contradiction-detection ("Wait" at 1.0)
  → retry-by-re-rendering hits the same weakness ×5. Detection is not
  the deficit; the retry strategy is. Mitigation = message/triage-layer
  loop breaker that breaks the tie TOWARD observed tool output ("ground
  truth beats re-derivation after N failed reconciliations"), same
  policy family as narrate-vs-act. Or accept as intrinsic model
  weakness (repetition arithmetic) serving can only bound. Trace
  evidence: /tmp/logs pmax_trace + serve-00004.
- **Environmental confound audit (user-prompted 2026-07-18)**: swap/
  compression lossless; no purgeable Metal buffers anywhere; GPU RESETS
  were the real vector — the engine-prefill batch path (`batch.rs end()`)
  did NOT check `cmd.error()` after waitUntilCompleted, so a reset
  (sleep/wake edge, driver recovery — a yellow lock screen is a
  compositor-recovery symptom) could silently commit corrupt KV that
  poisons every later turn. Fixed: typed error there + loud
  `assert_cmd_ok` at all 6 diagnostic sites (hot path already checked).
  12h system-log scan: no GPU restart events found. The live-vs-replay
  KV divergence finding now carries an environmental-confound caveat —
  the controlled kv_hash two-path test is the decisive experiment.
- **serve ops.jsonl is no longer token-level replayable**: the registry
  op format (activate/generate/finalize summaries) is skipped by
  `replay` ("unknown op shape") — the collapse-repro workflow's replay
  path is broken for current serve logs. Either teach `replay` the
  registry format (activate carries the full prompt token array) or log
  a parallel token-level stream.
- **Task #93 FIXED (2026-07-18, reuse-bugs commit)**: `begin_turn` now has
  the rewind/divergence guard — resident causal log extending past or
  diverging from the prompt's common prefix is truncated before reuse
  (O(1) inside the ring slack) or, on a deep rewind, reset for a fresh
  prefill (no-rebuild-to-salvage: rebuild ≈ fresh-prefill cost + lineage
  KV). Same policy at conversation `finalize` (long-thinking thought-strip
  no longer pays the ~39s ring rebuild) and `route()` now accepts REWIND
  prompts (prompt is a prefix of the canonical log — repeat/retry/edit),
  verified live: exact-repeat request went `reused 0` + full prefill →
  `+0tok (reused 6900)`. Remaining from old #93 note: `rollback_to`'s
  "restores the conversation" contract is still false for its only
  production caller.
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
- **q6/q5 non-expert weights** (memory lever was judged redundant with
  q8-KV-auto on 36 GB — REVISIT: q8 KV is broken and ungated, so that
  fallback does not currently exist; see the q8 correctness-debt item)
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
