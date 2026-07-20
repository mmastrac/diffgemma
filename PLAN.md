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

### Census batteries — open items

- **Quantify the convention-blend rate.** The mechanism is now in
  ARCHITECTURE Part I ("Discrete diffusion, not continuous") as a property of
  parallel per-position decoding: where two valid surface forms compete, the
  canvas can settle on a mixture valid under neither (`if [ $#" -lt 1 ]`,
  `== *$SEARCH"*`). Characterised and directionally confirmed, NOT
  quantified. Next step: a detector for unbalanced delimiter counts in the
  committed answer region, run against convention-ambiguous constructs, to
  get a rate rather than six specimens. Everything needed is already in the
  `DGQ_TRACE_PMAX_JSONL` traces — no GPU required to build it, only to
  generate more samples.
- **RETIRE `soft` as an instrument for `DGQ_COMMIT_CONF_HARD`** (do not try
  to repair it). Grounded retrieval never produces the sub-0.5 rows the tier
  triggers on — min p_max across 881 trim-scannable soft rows is 0.733 — so
  the lever is unreachable by construction, and lengthening the answers did
  not change that. `soft` remains valuable as a retrieval/hallucination
  rate; it is simply the wrong instrument for that decision. `programmatic`
  and `smoke` DO exercise the tier.
- **`scan_trace` answer-region fix — optional, low priority.** Mirroring the
  trim's `(MIN_CONF_KEEP..region_end)` rule would make `contested_per_1k`
  mean what its name implies and sharpen effect sizes. The arm ORDERING was
  verified to survive the correction, so this is legibility, not
  correctness — and it still moves a published number.

### Census batteries

`census` (arms x batteries x gates, `src/commands/census.rs`) has `smoke`,
`longctx` and `programmatic`. One battery still to build:

**`soft` retrieval — indirect facts, NON-blocking.** The softness is in
the QUERY, not the judging: the planted fact shares NO tokens with the
question, so it needs semantic matching, but scoring stays keyword-based
(cheap, deterministic, no judge model). Doc plants "the cat had a mauve
collar"; query asks "there was an animal with a collar, what colour was
it?"; answer must still contain "mauve".
- `expect_any` (any-of), NOT require-all: soft answers vary in phrasing and
  all-of re-imports the brittleness this exists to escape.
- Reported as a rate, EXCLUDED from pass/fail. Gateable only if someone opts
  in (`--gate 'soft_pct>=baseline'`).
- Authoring rule: if the fact shares tokens with the query the probe
  silently degenerates into an ordinary longctx probe and measures nothing.
- Classes: attribute indirection, referential callback ("using something we
  wrote here before"), numeric restatement/unit conversion, negation
  /exception, aggregation, synonym paraphrase, multi-hop, ordinal,
  superlative, temporal ordering, and ABSENCE (ask about something not in
  the doc, expect "not mentioned"). Absence measures hallucination — nothing
  currently covers it — and INVERTS the metric, so it needs its own rate
  rather than being averaged in.

**`soft` — built and run (2026-07-19). It SATURATES, and more importantly it
CANNOT exercise the lever it was built for.**

Predictions were: soft_pct >= 70 (right); `soft_unit_convert` the likeliest
miss (wrong — it answered "1,200 centimetres"); absence_pct <= 50 (wrong —
2/2 declined). Result: **10/10 retrieval and 2/2 absence at every one of
seeds {7,42,123}** — 30/30 and 6/6, all eleven indirection classes, no
variation. Contrast `programmatic`, which swings 12/14–14/14 across the same
seeds: indirect retrieval at this document length is a stable solved
capability for this model, while code generation is trajectory-fragile.

Saturation alone would not be fatal — a ceiling battery cannot show
IMPROVEMENT but is a clean REGRESSION detector, which is the actual
`DGQ_COMMIT_CONF_HARD` risk. The fatal part is structural:

**`soft` cannot reach the hard trim — and the reason is CONFIDENCE, not
answer length.** An `off` vs `hard:DGQ_COMMIT_CONF_HARD=0.5` comparison
returned BIT-IDENTICAL metrics with `trims 0` in both arms, twice: once with
the original short probes, and again (campaign B) after six long-form probes
lifted the median answer region from 13 to 20 rows (max 94) and took
trim-scannable rows from ~0 to 881.

My first diagnosis — "soft answers are shorter than the 16-row
`MIN_CONF_KEEP` floor" — was only part of it, and lengthening them did NOT
make the lever reachable. The decisive number:

    battery          scannable rows   rows<0.5 in scan   MIN p_max in scan
    soft (long-form)      881               0                 0.733
    smoke                2856               5                 0.299

The hard tier fires at p_max < 0.5. Across every scannable soft row the most
uncertain is 0.733, so no threshold in the tier's range can engage. Grounded
retrieval is a HIGH-CONFIDENCE task; the model is never unsure. Open-ended
generation (smoke's convergence prompts) is what produces genuine
sub-0.5 rows, which is why the original sign-off could measure the lever
there at all.

General rule this establishes: **a battery can only evaluate a lever whose
trigger its outputs actually produce.** Match the battery to the trigger
distribution, not to the topic. Check `trims > 0` in the treatment arm before
reading any arm comparison — twice now a green "no regression" was really
"the lever never ran".

So PLAN's premise — "the `soft` battery is what should settle the
`DGQ_COMMIT_CONF_HARD` decision" — is FALSE as built, and a green
`soft` run must NOT be read as evidence for that decision. To settle it,
soft probes need LONG-FORM answers (multi-sentence, answer region well past
16 rows); the `doc_tokens` field is already in the fixture schema for the
depth ladder that would come with them.

**Instrument audit — RESOLVED 2026-07-19, and my concern was WRONG on the
point that mattered.** census `dup` counts every committed row below `kept`;
the trim scans only `(MIN_CONF_KEEP..region_end)`. I found that on two small
samples 100% of census-counted dup rows sat in the eos/pad tail and worried
the shipped dup-tier sign-off had ranked arms on padding. Campaign A
(`smoke` x 4 arms x 3 seeds, ~13.7k rows/arm) settles it. It first reproduced
the original table to two decimals (off 1.31 / dup 1.03 / hard 0.88 / both
0.80 — the engine is deterministic and the sign-off is replicable), then:

    arm    census cont/1k    region-only cont/1k_r   in-region contested
    off        1.31               2.80               8  (5 hard + 3 dup)
    dup        1.03               1.06               3
    hard       0.88               0.33               1
    both       0.80               0.00               0

**The ORDERING is preserved under both metrics** (off > dup > hard > both),
so the dup-tier sign-off is vindicated and needs no re-litigation.

Two corrections to what I claimed earlier:
- "The dup component is entirely eos padding" was a SMALL-SAMPLE artifact
  (one seed, 3 dup rows). At 3 seeds the `off` arm has 3 genuine in-region
  dup rows out of 13. Padding does inflate the absolute counts; it does not
  dominate them.
- The correction makes the levers look STRONGER, not weaker: excluding
  padding shrinks the denominator to rows the trim can actually act on, so
  `off` rises 1.31 -> 2.80 and `both` falls 0.80 -> 0.00. The dup tier goes
  from a 21% reduction on the reported metric to 8 -> 3 in-region rows.

Remaining (optional, no longer urgent): mirroring the trim's region rule in
`scan_trace` would sharpen effect sizes and make `contested_per_1k` mean what
its name implies. It is now a legibility improvement, not a correctness fix,
and it would still move a published number — so it stays a deliberate call.

**Campaign plan (GPU session 2026-07-19), predictions pre-registered.**
Validity check first in each case: confirm `trims > 0` in the treatment arm,
because the `soft` lesson is that a lever which never fires produces a green
that means nothing.

- **A — dup-tier metric re-audit.** `smoke` x {off, dup, hard, both} x 3
  seeds, scoring `contested_per_1k` BOTH ways (as census computes it, and
  restricted to the region the trim scans). Prediction: the corrected figure
  is far lower for every arm, and the ARM ORDERING changes — i.e. the dup
  tier's shipped 1.31 -> 1.03 margin shrinks or inverts once eos padding is
  excluded. If the ordering holds, the sign-off stands and this is settled.
- **B — hard-tier regression on LONG-FORM soft.** 6 long-form probes added
  (answers must clear the 16-row floor), `soft` x {off, hard} x 3 seeds.
  Prediction: `trims > 0` at last, and NO retrieval regression.
- **C — hard-tier regression on CODE.** `programmatic` x {off, dup, hard,
  both} x 3 seeds. Code has ~60-row answer regions, the best lever
  engagement available, and is the most fragile output type we generate.
  Prediction: the hard arm costs steps but does NOT raise `compile_fail` —
  a trimmed tail is re-denoised against committed context, not corrupted.

Why this before the next big campaign: the `DGQ_COMMIT_CONF_HARD` decision
is blocked on a metric mismatch (below). Both existing signals are proxies;
neither says whether trimming makes answers BETTER.

**`programmatic` — what three rounds of probes established.** Predictions
were pre-registered each round and are kept here because the FALSIFIED ones
are the useful part: nobody should re-derive these premises.

Falsified: "models emit markdown fences despite instructions" (`fenced` has
been 0 across all 14 probes, every round); "compile failures concentrate in
rust because generation truncates"; "pass rate under 50%"; "bash arithmetic
stumbles"; "RPN operand order (`2 3 -` → -1) is a trap"; "a `.`/`*` regex
matcher needs backtracking the model won't write" (it passed 8/8, including
`a*`/"" and `a*a`/"aaa"). Confirmed: `py_text_wrap` passes; the long probes
need more than one 256-token block. Roughly one prediction in four survived.

**Current state: 12/14 probes, 36/44 cases (81.8%), compile_fail 8,
wrong_output 0, fenced 0** (`--arm 'default:' --seeds 7`).

The finding that matters is the SHAPE of the two failures, not the rate.
`wrong_output` is ZERO across all 44 cases: this model never computes the
wrong thing. Both failures are a correct algorithm carrying one unparseable
token — `bash_stdin_and_argv` wrote `*$SEARCH"*` (misplaced quote, unbalanced
`"`), `rust_base_convert` wrote `let final: String = ...` (`final` is a Rust
reserved word, a Java/JS habit). Code failure here is LEXICAL, not
algorithmic. No text-judged gate can see this — both replies read as good
programs — which is precisely what the compile_fail/wrong_output split was
built to expose, and it is the battery's first real earned result.

`rust_base_convert` initially failed as "unclosed delimiter" and that was
OUR ceiling, not the model: the 512-token `SMOKE_GEN_CAP` cut a 60-line
program mid-expression. The battery was one head-only preview away from
reporting a harness artifact as a model defect. Fixed by giving the battery
its own budget (`PROG_GEN_CAP` 1536, `PROG_MAX_SEQ` 4096) and by printing
the rejected source HEAD AND TAIL, since truncation is only visible in the
tail. Any future probe that gets harder must re-check this ceiling first.

**Why the bash probe passes at seed 123 and fails at seed 7 — traced to the
STEP, no GPU needed.** The divergence is a denoise-trajectory difference, not
a layer defect, and the per-step `argmax`/`pmax` already in the trace shows it
directly. At the `[[ "$line" == *"$SEARCH"* ]]` site:

    seed 7   step 3:  '*"'(0.87)  '"$'(0.89)   -> *""$SEARCH   (DOUBLED quote)
    seed 7   step 4:  '*' (0.97)  '$' (0.95)   -> *$SEARCH     (ZERO quotes)
    seed 123 step 3:  '*' (0.98)  '"$'(0.99)   -> *"$SEARCH    (correct, never revised)

Seed 7 transiently landed in an over-quoted state, and the next step revised
BOTH adjacent positions at once — each dropping its own quote — overshooting
from two quotes to none. Seed 123 never entered that state. Correct output
needs EXACTLY ONE of the two adjacent tokens to carry a quote.

Hypothesis for the WHY, fitting but NOT established (n=1 harmful instance):
diffusion updates positions from independent per-position marginals, and
"exactly one quote across this pair" is a JOINT constraint that independent
marginals cannot represent — so both positions correct the same visible
doubling simultaneously. This would be a failure mode unavailable to a
sequential decoder, which sees its own previous emission. It also explains
the earlier observations: high commit confidence (post-correction each token
is individually plausible; only the PAIR is wrong), trajectory dependence
(only paths through the doubled state are at risk), and the compiler naming
a downstream site.

Surrounding evidence, from a detector over 52 generations looking for
adjacent pairs whose non-quote content is identical but whose quote count
changed: only 5 such edits exist. Two are seed 42 ADDING quotes
(`$` -> `"$`, both p_max 1.00; seed 42 is the clean 14/14 run); three are
seed 7 REMOVING them (p_max 0.83-1.00), of which one was harmless
(`[ "$#" ...` -> `[ $# ...`, still valid) and one is the bug. So the harmful
event is rare and the trajectory-level quote-edit direction differed between
a passing and a failing seed.

To CONFIRM or kill the joint-constraint hypothesis it needs many more harmful
instances than one: sweep `programmatic` over ~20 seeds, collect every
surgical pair edit, and test whether harmful double-removals occur
disproportionately when both positions update in the SAME step rather than
in different steps. That is the discriminating measurement; do not treat the
mechanism as established until it is run.

**Campaign C — the FIRST non-vacuous arm comparison of the session.** The
lever actually fired on `programmatic` (`trims`: off 0, hard 1, dup 2, both
2), because code generation produces the low-confidence rows retrieval never
does. Result over 4 arms x 3 seeds x 14 probes:

    arm    probes pass   cases pass   compile_fail   cont/1k
    off       37/42       116/132         15          0.84
    hard      37/42       116/132         15          0.46
    dup       38/42       121/132         10          0.23
    both      38/42       121/132         10          0.23

**The entire difference is ONE probe at ONE seed**: `rust_base_convert` at
seed 42, which `off`/`hard` fail and `dup`/`both` pass. Every other
probe-seed cell is identical across all four arms. So:

- The dup tier's apparent +3.8pp is a single probe flip. Direction is
  favourable and the mechanism is plausible (the trim cuts the bad row, the
  re-denoise picks a different identifier), but n=1 is not evidence. A
  powered sweep is running (campaign D: off vs dup x 10 fresh seeds).
- The hard tier did NOTHING: identical outcomes to `off` despite firing
  once. That is the first genuine (if thin) regression signal for
  `DGQ_COMMIT_CONF_HARD` — the lever ran and did no harm across 42
  probe-runs. It is not evidence of BENEFIT. The decision stays parked.

**Metric caution I introduced myself: case-level percentages amplify single
probe flips.** `compile_fail` fans out to every case of an unbuildable probe
(correct — it keeps the three states summing to `cases`), so one flipped
5-case probe moves the case metric by 3.8pp while the probe metric moves by
1/42. **Use probe-level counts for arm comparisons; case-level is for
diagnosing WHAT failed, not HOW MUCH better an arm is.**

**Campaign D (off vs dup x 10 fresh seeds) — the headline result: the WART
PROXY DOES NOT PREDICT EXECUTABLE CORRECTNESS.**

Probe level over 13 seeds (C+D): `off` 159/182, `dup` 161/182. Two flips,
both favourable, none against — weak but consistent, and not significant on
its own (p~0.25 as a coin flip). The dup tier is not harmful and may help
slightly; that is the whole quality claim the data supports.

The wart columns move enormously by comparison — `off` cont/1k 5.88 vs `dup`
0.41, hard rows 262 vs 12 — but the effect is TAIL RISK, not a uniform
improvement, and it does not reach the outcome:

    seed   off hard rows   dup hard rows   probe outcomes
      3        190              1          IDENTICAL (same 3 failures)
      2         64              6          IDENTICAL (same 2 failures)
      1          0              0          DIFFER (dup fixes rust_regex_lite)
    other 8   0-3             0-2          identical

Eight of ten seeds are near-identical; two degenerate badly under `off` and
the dup tier rescues them. But **seed 3 committed 190 low-confidence rows and
failed exactly the same probes as the arm that committed 1**, while the only
quality difference appeared at a seed with ZERO hard rows. Contested-committed
rows and executable correctness are close to decoupled here.

This speaks directly to the blocked `DGQ_COMMIT_CONF_HARD` question, whose
premise is "both existing signals are proxies; neither says whether trimming
makes answers BETTER". Now measured: trimming improves the PROXY by ~14x and
the OUTCOME by 2 probes in 182. **Do not size a lever decision on
`contested_per_1k`.** Caveats: 13 seeds, one battery, one model, and
executable correctness is a coarse per-probe binary — contested rows could
still affect prose quality this battery cannot see.

**Joint-constraint double-removal: REPLICATED.** Scanning 245 off-arm
generations for adjacent pairs whose non-quote content is unchanged but whose
quote count moved by >=2 with BOTH positions changing in the same step yields
just 3 events. Two are byte-identical and independent:

    seed  7 step 4:  ('*"', '"$')(2) -> ('*', '$')(0)   bash_stdin_and_argv FAILS
    seed 17 step 4:  ('*"', '"$')(2) -> ('*', '$')(0)   bash_stdin_and_argv FAILS
    seed  5 step 6:  ('(', '(')(0)  -> ('("', '("')(2)  benign addition

Same token pair, same direction, same step, same resulting probe failure, at
two unrelated seeds. The pattern is real and reproducible. The MECHANISM
(independent per-position marginals cannot encode "exactly one quote across
this pair", so both positions correct the same visible doubling) remains an
inference rather than a controlled result: a clean test would compare
same-step resolutions of a doubled state against across-step ones, which
needs more harmful instances than 2.

**Controlled test of the joint-constraint mechanism — RAN, and it refined the
hypothesis rather than confirming it (campaign E: 4 delimiter-stress probes,
18 fresh seeds, 407 generations).**

Two analysis bugs were caught before reporting, both of which INFLATED the
result: (a) arms share trajectories until a trim fires, so the same
generation was counted up to 4x; (b) a dropped `pend=[]` made every commit
re-scan the whole step history. Uncorrected they read 8-vs-6 and then
14-vs-14; corrected, the prior data held 4 events, not 14. Dedup by
(battery, seed, generation) and reset per generation.

Pre-registered predictions scored: event rate >=3x -> 2.7x (NARROWLY
FALSIFIED); overshoot >=30% if real / <10% under a coordinated null -> 37.5%
(supports); glob probes fail in correlated fashion -> weak, 3+1 of 18 seeds.

Pooled result, n=16 doubled-state resolutions: **6 overshoot (37.5%), 10
correct.** Note the perfect "both positions moved <-> overshoot" association
is ARITHMETIC, not evidence — with one quote per position a pure removal
gives 0 iff both moved. The informative quantity is the SIMULTANEOUS
double-update rate. Against a coordinated null (overshoot rare, p=0.10),
P(X>=6) = 0.0033, REJECTED. Against weak coordination (p=0.25) or full
independence (p=0.50), not rejected. So: **the model demonstrably does not
coordinate the pair, but n=16 cannot separate weak coordination from
independence.**

**The contradiction that matters, and the real finding.** The seeds with
overshoot events (103, 107, 137, 173) and the seeds that actually FAILED
(139, 151, 157, 181) have ZERO overlap, and every failure is an unbalanced
quote. So the adjacent-pair double-removal does NOT account for the observed
failures. Inspecting them:

    seed 139 FAIL:  if [ $#" -lt 1 ]; then     <- spurious CLOSING quote (11, odd)
    seed 103 pass:  if [ $#  -lt 1 ]; then
    seed 103 pass:  if [ "$#" -ne 1 ]; then

Both quoting conventions are valid (`[ $# ]` and `[ "$#" ]`); seed 139 landed
on a BLEND of the two, valid under neither. That is the same shape as seed
7's `*$SEARCH"*` (blend of `*$X*` and `*"$X"*`) but in the opposite
direction. **The general mechanism is not "double removal" — it is that the
model holds two competing valid surface conventions and independent
per-position updates can emit a MIXTURE of them.** My double-removal
hypothesis was one special case of this and explained only the subtractive
direction.

Status: the broader blend hypothesis now covers every quote failure observed
(seeds 7, 17, 139, 151, 157, 181) in both directions. It is NOT yet
quantified — a detector for "final answer region has odd quote count" run
against convention-ambiguous constructs would do it, and is the obvious next
step if this is worth more time.

Open work on this battery:
- **Judge it by the multi-seed aggregate, never one seed.** Seed 42 scores
  14/14 and seed 7 scores 12/14 on identical code; a single-seed run of this
  battery is close to meaningless, exactly as for the smoke gate.
- **Whether it can discriminate ARMS is still unknown** — every run so far
  is one arm. That, not the pass rate, decides whether it is usable for any
  lever decision. The per-seed spread (12–14/14) suggests the noise floor is
  ~2 probes, so an arm comparison needs several seeds to see past it.
- Prediction that failed, logged so it is not retried as-is:
  `bash_stdin_and_argv` was predicted to fail at ≥2 of 3 seeds as a stable
  habit. It failed at exactly one.

**SETTLED (2026-07-19, 3 seeds): code failures are TRAJECTORY-dependent but
CONFIDENTLY committed — no confidence-trim tier can reach them.**

Aggregate over `--seeds 7,42,123`: 42 probes, 132 cases, **121 pass (91.7%)**,
compile_fail 10, wrong_output 1. Per seed 12/14, **14/14**, 12/14 — and a
different pair of probes fails each time:

- seed 7: `bash_stdin_and_argv` (wrote `*$SEARCH"*`), `rust_base_convert`
  (`let final:` — a Rust reserved word)
- seed 42: nothing fails
- seed 123: `bash_underscores` (**wrong_output** — `$*` under default IFS
  printed `red green blue`), `rust_primes` (compile_fail)

Two claims from the single-seed round were WRONG and are corrected here:

- "The model never computes the wrong thing" — false. Seed 123's
  `bash_underscores` is a well-formed program that computes the wrong
  result. `wrong_output` has now fired against the model, so the three-state
  split is validated by live evidence, not only by the fixture control.
- "This is a knowledge defect, not a sampling defect" — false, and it was a
  single-seed overclaim. `bash_stdin_and_argv` is CORRECT at seed 123
  (`*"$SEARCH"*`) and wrong at seed 7. The model has no fixed inability
  here; different denoise trajectories land on different code.

What DOES survive, and is the useful part: within a trajectory the wrong
token is committed at high confidence — seed 7's `"*` at 0.9993 and ` final`
at 1.0000, seed 123's failing rows at 0.83–0.98 — while committed rows below
0.9 number 9/4490, 5/4343, 12/4352 across the three seeds. Dup-stutter
commits live at 0.40–0.86. No `DGQ_COMMIT_CONF_*` threshold separates a
confidently-wrong token from a confidently-right one, so `programmatic` is
NOT an instrument for the `DGQ_COMMIT_CONF_HARD` decision; `soft` remains
the right one.

Mechanism worth keeping: seed 7 emitted `*` `$` `SEARCH` `"*` where seed 123
emitted `*"$` `SEARCH` `"*`. The SAME closing token `"*` at ~0.998 is right
or wrong depending on an upstream omission — the model confidently closed a
quote it never opened. The defect is an upstream token choice, not a bad
token at the visible error site.

(Not established, n=3: the perfect seed had the fewest sub-0.9 committed rows
(5) and the worst had the most (12). Three points is not a correlation — do
not build on it without a real sweep.)

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
- **Confidence trim: dup tier SHIPPED default-ON at 0.9 (sign-off
  2026-07-19); hard tier SPLIT OUT and still OFF.** The two tiers were one
  flag; `DGQ_COMMIT_CONF_HARD` (unconditional p_max floor) is now
  independent of `DGQ_COMMIT_CONF_TRIM` (dup-conjunctive at tau), because
  the census showed they behave nothing alike. Isolation run
  (`census --arm off/hard/dup/both --battery smoke --seeds 7,42,123`,
  ~13.7k committed rows/arm):

      arm   pass  hard<0.5  dup  trims  cont/1k  steps  retry
      off   yes        5     13     0     1.31    362     24
      dup   yes        3     11     1     1.03    363     24
      hard  NO         0     12     4     0.88    376     24
      both  NO         0     11     4     0.80    377     24

  DUP is nearly free: passes every gate, +1 step total, contested 1.31 ->
  1.03. It also lowers HARD-class commits (5 -> 3) without a hard tier —
  truncating at a dup row keeps later contested rows out of KV and
  re-denoises them, so the tiers are not independent in effect. Golden 8/8
  byte-identical (the trim never fires on golden cases, no re-bless).
  Note: with it default-on, the `max_blocks *= 2` headroom that trimming
  enables is now always active.

  HARD is the tier that actually kills the insertion/omission class
  (5 -> 0) but stays OFF: it fires 4x as often, needs 58 blocks instead of
  57, burns +14 steps (~4%), and tips `transformer` over its 39-step
  convergence budget at seed 7 (16/17). Retry steps are IDENTICAL (24)
  across all arms — the cost is committed steps, not re-rolls.
  BLOCKED ON A METRIC MISMATCH, not on the tier being bad: that budget was
  ratcheted on a NON-trimming baseline, so a lever that deliberately
  commits partial blocks will always read as a convergence regression to
  it. Resolve with the `soft` battery (does trimming make answers better?)
  before deciding; if soft retrieval holds under `hard`, re-baseline
  `transformer` rather than shelving the tier. Caveats: 3 seeds, smoke
  only, and the failure is ONE step over (40 vs 39).
  PREDICTION LOG (worth keeping — it was wrong): I predicted hard-only
  would be free and dup-only would own the failure. Exactly inverted. Then
  predicted "any trim causes it" — also wrong; dup trims once and costs +1
  step. The trim RATE is what matters, not the tier.
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

- ~~**Hard-kill flag validation**~~ **DONE (2026-07-19)**: a set-but-
  invalid `DGQ_*` value now exits(2) naming every offender, instead of
  silently running the default. `DGQ_PREFIX_EXIT=1` — which used to
  DISABLE the lever it read like it was enabling — is fatal with
  `expected a number in [0, 0.5]`. Implemented in the shared parse
  helpers (`on_if_one`/`on_unless_zero` via `bool_value`, `parse_usize`,
  `gib_bytes`, new `ranged_f32`) rather than a per-flag table, so it
  cannot drift from the parsing; rejections accumulate in a thread-local
  and `from_env` drains + kills, so ONE run reports EVERY bad flag.
  Bools now accept 1/0, true/false, yes/no, on/off and reject the rest
  (previously a typo silently meant OFF under opt-in flags and ON under
  opt-out ones — opposite failure modes for the same typo). Unset is
  untouched: still the documented default, still silent. Tests use a
  test-only FAKE_ENV (no `set_var`, which is unsafe/racy under edition
  2024). REMAINING: the ~29 flags parsed inline via bespoke `var(...)`
  chains (enums, u32 ranges, paths) still swallow bad values — convert
  them to checked helpers as they are touched; the 4 shared helpers plus
  the 3 ranged floats cover the bulk and the known footgun.
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
  max|delta| growing 0.5 -> 9.3 with depth.
  **CULPRIT PINNED: E20 top-k sparse attention (`DGQ_ATTN_TOPK`, default
  ON).** `kv_lineage_fork_attention_bisect` runs the live case under each
  attention lever; ONLY `attn_topk=0` comes back clean (gemm_attn=0 /
  attn_mma=0 / attn_mma_full=0 all still FORK). Mechanism: top-k's
  parameters come from the DISPATCH's `t_total = kv_len + canvas`, not
  from the query's own causal context, so the same absolute position gets
  a different approximation depending on how the prefill was chunked.
  `attn_topk_k_for` = `(t_total/128).clamp(64,512)`, and in the live case
  one query gets **k=114 fresh vs k=115 reused** (then 116/115, 116/117)
  — a different NUMBER OF ATTENDED KEYS for identical Q and identical KV.
  That is a discrete approximation change, not a rounding difference,
  which is why it amplifies into a different trajectory. CAVEAT: case 2
  forks even where k is EQUAL in both arms, so the `t_total`-dependent
  selection/tiling is a SECOND phase-dependent input, not yet isolated.
  Layer-6 onset is now explained (and an earlier "sub-ULP seed" reading
  RETRACTED): top-k runs on FULL layers (5,11,17,23,29) and a layer
  writes KV BEFORE its attention, so layer 5's KV is clean while its
  attention OUTPUT differs — layer 6 is "first top-k layer + 1", not a
  visibility threshold. Don't hunt a tiny seed in layers 0-5.
  **FIXED 2026-07-19**: `attn_topk_softmax` now derives k per row from
  that row's own `n_valid` (causal key count = absolute position + 1), a
  phase-invariant quantity; the host ships the DIVISOR
  (`attn_topk_k_cfg` = fixed_k, dyn_divisor, k_min, k_max) instead of a
  resolved k, and the old host-side `attn_topk_k_for` is DELETED so it
  can't be reused. Non-causal (denoise) is unchanged by construction —
  there `n_valid == t_total` already — so the decode arm
  (`DGQ_ATTN_TOPK_DECODE`) needed no change. The lineage gate is green
  3/3 including the live shape and is now UN-IGNORED. The earlier
  "second phase-dependent input" caveat is resolved: per-row k also
  closed the case that forked at equal-k positions (a wrong k anywhere
  in a full layer perturbs the shared hidden stream).
  The per-row count is rounded UP to the prefill CHUNK GRID (passed
  explicitly by the host, NOT inferred from the dispatch row count), which
  is the `t_total` an aligned fresh chunk would have had for that
  position — so k is phase-invariant AND numerically unchanged.
  GATE STATUS: golden **8/8 byte-identical**, smoketest 17/17 x{42,7,123},
  longctx 4/4 + retrieval 8/8 + drift 0.0%. No re-bless needed: fresh
  prefill is bit-identical, only reused-KV paths move (into agreement).
  Deriving k from the RAW per-row count instead is also phase-invariant
  but shrinks k for early rows and broke golden 7/8 — the round-up is
  load-bearing, don't "simplify" it.
- ~~**Long-context generation is NOT run-to-run reproducible**~~
  **RETRACTED 2026-07-19, same day it was filed — the engine IS
  deterministic.** The "same binary, same seed, different answers"
  observation was an INVOCATION error: `smoketest --longctx` defaults to
  seed **7**, not 42, so a no-`--seed` run compared against `--seed 42`
  runs is comparing two different seeds. Verified deterministic three
  ways: in-process back-to-back generations are identical at 512/2048/
  8192/13300/20600 (`generation_determinism_vs_context_length`); cold vs
  warm Metal pipeline archive (isolated XDG_CACHE_HOME) identical; and
  re-running with the seed made explicit reproduces each earlier output
  exactly. The E20 fix is confirmed numerics-preserving end-to-end (v3
  matches baseline at seed 7 AND seed 42), which is stronger than the
  golden-only evidence. FOOTGUN WORTH KEEPING: never mix defaulted and
  explicit-seed invocations in one A/B — the tool prints `seed N`, read
  it. And run the cheapest control (re-run the exact failing command)
  before hypothesising about GPU races or codegen.
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

### Output-token classification (in progress 2026-07-20)

`fit-token-probe` SHIPPED and working: 44-sample labelled spec
(`fixtures/token_probe/code_prose.json`) -> LOO accuracy 1.000, effect size
2.59 at L9, ~100 s on an M3 Pro. Deliberately NOT a baked artifact — hidden
space is checkpoint-specific, so an end user refits locally after swapping or
requantizing a model. Measured layer ranking on that spec: L6 1.000/2.75,
L9 1.000/2.59, L29 0.977/0.97. Default is L9, because L6 is marginally better
for TASK INTENT while L9 is the better tracker of the CURRENTLY EMITTED mode,
which is what token colouring needs (and L9 gives up almost nothing on
intent). Compare layers on EFFECT SIZE; accuracy saturates.

Remaining: `DGQ_TOKEN_CLASS=<probe.json>` -> capture layer-L hidden into a
side buffer during the denoise forward -> classify at block commit ->
per-token labels on `ChatEvent::BlockCommit` -> colour in `chat_ui`. The
engine capture is the real work: the arena holds hidden for the CURRENT layer
only and overwrites it per layer, so reading L9 at commit needs an encoder
change to copy that plane aside.

**PERF IS A REQUIREMENT, NOT AN AFTERTHOUGHT — two separate obligations:**
1. **Exactly zero cost when OFF.** The capture must not be ENCODED at all
   when disabled — not "encoded and discarded", not a per-step flag read.
   Flag parsed once into `RuntimeConfig` ([[flags-config]] discipline), and
   the dispatch omitted entirely.
2. **Measured cost when ON.** Expected ~1.44 MiB device copy per step
   (256 x 2816 bf16), which should be single-digit microseconds against a
   ~700 ms step — but this project's own rule is that the guess does not
   count. Classification happens at COMMIT, so there is one readback per
   BLOCK, not per step.

Measurement protocol (per [[perf-timing-methodology]]): `bench-step-kernel
--profile-steps N`, off vs on, ADJACENT within one session, several
iterations, nothing else on the GPU. Report per-step deltas, not totals.
Acceptance: OFF must be indistinguishable from the pre-change baseline; ON
must be justified by what it buys.

### WAY OUT THERE — can the generation LANGUAGE be read off the model's own state?

Speculative, not planned, nobody is convinced. Recorded because the core
question is interesting on its own terms and because a cheap experiment can
kill it in an afternoon.

**The question.** This is a 128-expert MoE, top-8 per position per layer.
Expert specialisation by domain is a real effect in MoE models generally. Does
the routing pattern carry a legible signal for *which language this position
is being written in* — bash vs python vs rust vs prose? Same question for a
linear probe on layer-N hidden states.

**Why it would be useful.** Every structural-repair idea we have (delimiter
balance, AST validation) needs to know WHICH SPANS ARE CODE, and in what
language, before it can safely fire. Token heuristics answer that well for a
whole reply but get fuzzy exactly at span boundaries — which is where a
delimiter check most wants precision. Routing is PER-POSITION, so it is a
segmentation rather than a document label, and it is already computed every
step at no extra cost.

**The honest counter-argument.** At commit time the text is already decoded
(we need it to parse anyway), and `#!/bin/bash` / `fn main()` / `def ` are
ruthlessly discriminative. Internal state has to beat a twenty-line token
heuristic that costs nothing. It only clearly wins for (a) mixed replies with
several inline spans, and (b) a signal that exists BEFORE tokens resolve —
and (b) currently has no consumer, since in-step grammar constraint does not
transfer to diffusion (no well-defined prefix state to constrain against).

**Probe the CANVAS HIDDEN STATE, not the KV cache.** These are different
tensors and only one of them answers the question. The KV cache holds
COMMITTED context (prompt + committed blocks) and persists; the canvas hidden
state is the residual stream at the 256 canvas positions, recomputed from
scratch every denoise step and never cached. During denoise the canvas
attends causally to KV and BIDIRECTIONALLY to itself.

That asymmetry is why the canvas is the right target. "This is Rust" is
plausibly encoded diffusely across many KV positions (the instruction, `fn`,
`};`), each weak on its own, and a single K/V vector is a compressed local
slice of it. A canvas position after several attention layers is an
AGGREGATE of those positions, and aggregation is what makes distributed
information locally decodable. The bidirectional canvas self-attention
amplifies this further: once a few positions resolve to `fn`/`let`, EVERY
canvas position can absorb it, including ones to the left of the evidence and
ones whose own token is a bare newline. An AR decoder cannot do that.

**TWO CONTROLS, without which a high number means nothing.** Both exist to
stop the probe from being an expensive tokenizer:
1. **Language-neutral positions.** A probe on almost any layer will score
   highly by reading TOKEN IDENTITY — `def`, `fn`, `};` are trivially
   diagnostic. Evaluate on positions whose own token is neutral (bare
   newline, indent run, plain identifier, `}`). If accuracy survives there,
   the language state is genuinely carried in context; if it collapses, the
   probe was reading the tokens.
2. **Prompts that do NOT name the language.** Every `programmatic` probe says
   "Write a complete Rust program", so the instruction alone settles it and a
   probe would just be re-reading the prompt. Continue-this-snippet prompts
   are needed to test whether the model tracks language from the code it is
   WRITING.

**The measurement is a gradient, not a yes/no** — which makes it falsifiable.
At a language-neutral canvas position, probe accuracy should RISE with layer
depth (aggregation accumulates) and RISE with denoise step (the canvas
resolves, so self-attention has more evidence to pool). Flat-and-high from
step 1 layer 1 = reading the prompt instruction. Flat-and-low = no signal.
A rising curve is the interesting result.

**If the KV cache IS probed, probe V in the GLOBAL layers.** Note first that
"the KV signal is weak" is an INFERENCE, not a measurement — nothing here has
been probed yet, so do not go looking for why it is weak before establishing
that it is. Precision is ruled out on the numbers: KV is f16, range-checked,
max|KV| ~ 22 against an f16 max of 65504, and linear probes for coarse
categorical features survive int8. Two architectural reasons are real,
though:
- **Cached K is POST-RoPE** (`QKV -> RoPE -> attention`; Negative Knowledge
  discusses un-RoPE-ing pre-cache). Position-dependent rotation smears a
  fixed linear direction across positions, so a position-agnostic probe on K
  underperforms for reasons unrelated to whether the feature exists. **V is
  un-rotated** — `values = keys` happens BEFORE k_norm/rope — so V is the
  strictly better target.
- **25 of 30 layers are sliding-window**; a cached position there is only
  attended by nearby queries. Long-range context state should concentrate in
  the **5 global layers**.
Everything else (GQA compression, the distributed-fragment argument) is about
ACCESSIBILITY rather than presence and would persist even in a hypothetical
f32 full-attention cache.

**FIRST RESULTS (2026-07-19, `step-layer-probe`, 18 prompts, seed 42).**
Probed the CANVAS hidden state at position 129 across all 30 layers. The tool
gives a free version of the control we wanted: the canvas token at that
position is seeded noise, IDENTICAL (id 71153) in every condition, and at the
`after_preamble` checkpoint every condition is cosine 1.000 — so all structure
below is attention-derived from the prompt, at a position whose own token
carries no language. Analysis is mean-centred cosine across conditions.

**The dominant axis at L6 is TASK (write-code vs prose), not topic.** Perfect
linear separation, no overlap, over 12 prompts — neighbours of `rust_write`:

    py_write +0.73  rust_fenced +0.61  rosetta_p2r +0.57  rosetta_r2p +0.52
    bash_write +0.42  py_fenced +0.34  |  rust_talk -0.19  py_snake -0.33
    py_talk -0.64  rust_chem -0.82  prose -0.87

**Token identity is ruled out**, and not by the homonyms (capitalisation
differs, so "Rust"/"rust" may be different tokens). The decisive case is
`rust_talk`: it contains the IDENTICAL "Rust" token as `rust_write` yet sits
at -0.19 on the prose side, while `py_write` — a different language — is the
nearest neighbour at +0.73. Supporting homonym evidence: `rust_chem ~ prose`
+0.93 while `rust_chem ~ rust_write` -0.82; `py_snake ~ py_write` -0.47.

**A language sub-axis exists, is weaker, and INVERTS with depth.** A properly
crossed 2 languages x 3 contents design (only the language word differs across
rows) gives same-language-vs-same-content:

    L3 +0.445   L6 +0.563   L9 -0.207   L18 -0.394   L29 -0.529

So language dominates content at L3-L6 and content dominates language from L9
on — plausibly because the late residual encodes the specific computation and
next tokens rather than the dialect.

**Rosetta (both languages in one prompt)** lands in the code region between
the two: `rosetta_p2r` rust +0.57 / py +0.40, `rosetta_r2p` rust +0.52 / py
+0.51, both ~-0.66 vs prose. Translation DIRECTION barely registers — the two
rosetta prompts are +0.847 similar to each other.

**Starting place if this is ever pursued: L6, canvas hidden state.** That is
where both axes are strongest.

**Caveats — this is similarity structure, not a classifier.** One probe
position, one seed, 12 + 6 prompts, no trained probe and no held-out
accuracy. The centring mean is over only ~12 conditions. A methodological
trap worth recording: an earlier pass reported "max language separation at
L29 = 0.618" by averaging the two within-language pairs — side by side they
had OPPOSITE SIGNS at L15/18/21/27/29 (L29: rust -0.31, python +0.92).
Averaging disagreeing pairs manufactured the result; the crossed design above
replaced it. The within-language pairs in that first design also differed by a
"markdown code block" instruction, a bigger perturbation than the language
itself.

**L6 PROBE BUILT AND TESTED (2026-07-19). A linear "about to write code"
direction exists in the canvas hidden state, and survives four confounds.**
44 prompts (22 code across rust/python/bash/js/c/go/java, 22 prose of which 11
are HARD negatives — prose ABOUT programming: compilers, garbage collection,
memory safety, recursion). Estimator is difference-of-class-means with
leave-one-out, deliberately NOT logistic regression: at d=2816, n=44 any
fitted linear separator hits 100% on train and means nothing.

    layer   LOO acc   hard-neg   effect size (gap / within-class scatter)
    pre       --        --         0.0000     <- control: bit-identical, refuses to guess
    L3       1.000     1.000       1.51
    L6       1.000     1.000       2.05       <- PEAK
    L9       1.000     1.000       1.91
    L21      1.000     1.000       0.50
    final    1.000     1.000       0.59

Accuracy SATURATES; effect size does not. L6 and L21 both score 1.000 but L6
is 4x more separated — report effect size, not accuracy, when comparing
layers.

**Confounds excluded, each by construction:**
- *Own token*: probe position holds identical seeded noise (id 71153) in every
  condition; `after_preamble` is bit-identical with effect size exactly 0.
- *Token identity of the topic word*: `rust_talk` has the SAME "Rust" token as
  `rust_write` yet lands on the prose side; `py_write` (different language) is
  the nearest neighbour.
- *Technical vocabulary*: 11/11 hard negatives correct — "explain what garbage
  collection is" reads as PROSE despite dense programming vocabulary. The axis
  is "am I about to WRITE code", not "is this about programming".
- *Instruction verb*: a held-out control of 3 prose prompts that SAY "Write"
  and 3 code prompts that never do scores **6/6** at both L6 and L9. "Write a
  paragraph explaining photosynthesis" -> prose; "Show me a Rust program" ->
  code.
- *Prompt length*: length alone classifies at only 0.659. *Norm*: all vectors
  L2-normalised first.

**METHOD TRAP worth remembering: leave-one-out leaked the label through
floating-point rounding.** The first run scored 1.000 at `after_preamble`,
where all vectors are BIT-IDENTICAL and separation is impossible. Cause: class
means are summed over different-sized subsets, and LOO changes which class
loses a member, so rounding differs systematically with the held-out sample's
class. On a zero-signal layer that noise is the only signal and it encodes the
answer. Fix: compute an effect size and refuse to predict when ||w|| is
degenerate. Any LOO harness with per-class means has this bug.

**Still open after this:** the LANGUAGE sub-axis (rust vs python) is much
weaker than the code/prose axis and inverts with depth, so it is NOT
established as a usable classifier. All prompts are ~22 tokens, so the
sliding-window/global distinction was never exercised. And this is the CANVAS
residual, which costs a forward pass — whether a single KV vector is linearly
readable is untested, and blocked on tooling: **V is not captured in any dump**
(`step-attn-dump` emits hidden_in/q_*/attn_out/k_samples only). K IS dumped;
note RoPE is orthogonal so it PRESERVES inner products and does not obstruct
same-position comparisons — it only breaks a probe direction shared ACROSS
positions. A K probe therefore needs length-matched prompts, not an un-RoPE.

**LANGUAGE PROBE (rust vs python), n=60 matched pairs, cold AND committed
canvas.** 30 task descriptions x 2 languages, so content is matched pairwise
and only the language word differs. `--warm-steps N` was added to
`step-layer-probe` (opt-in; 0 reproduces the original step-1 behaviour) to run
N real denoise steps before the instrumented forward.

    layer      COLD acc  COLD eff   WARM acc  WARM eff
    pre          0.00      0.00       0.37      0.13     <- cold control is exact
    L6           1.00      0.75       0.65      0.20
    L7 (peak)    1.00      1.12         --        --
    L12          0.92      0.63       0.52      0.17
    L27          1.00      0.40       0.57      0.23
    L28 (peak)     --        --       0.77      0.31
    final        1.00      0.55       0.70      0.31

**The cold signal is REAL and reproduces at scale.** An earlier n=12 run gave
peak eff 1.33; at n=60 it is 1.12 with accuracy 1.00 across L6-L9. I had
predicted this was small-sample inflation and was WRONG — it held. The cold
canvas token is identical in all 60 dumps (1 distinct value), so the control
is exact and `pre` correctly reports 0.00.

**Counter-intuitively the signal is WEAKER on the committed canvas** (peak eff
0.31 vs 0.13 noise floor) than on the seeded-noise canvas (1.12 vs 0.00).
Mechanism, plausible but not proven: with a noise token the probe position's
hidden state is a PURE SUMMARY OF THE PROMPT, so context reads cleanly. Once
the position holds resolved content its own token dominates and dilutes the
prompt-derived signal. Caveat that bounds this: position 129 is unresolved
TAIL FILLER (`kaart`/`modern`/`junior` junk), so the warm probe measures a
position holding garbage, not committed program text. A position inside the
generated program is the stronger test and loses the token control.

**ONE regime, not two — the mid-layer "dip" was a DENOMINATOR artifact.**
Decomposing effect = gap/scatter over the n=60 matched pairs:

    layer   lang GAP   scatter   effect   LOO acc   content-gap
    L0       0.0059    0.0169     0.35     0.58      0.0171
    L6       0.0508    0.0673     0.75     1.00      0.0627
    L12      0.2349    0.3717     0.63     0.92      0.3776
    L18      0.3121    0.8550     0.36     0.95      0.8673
    L24      0.2554    0.8213     0.31     0.93      0.8154
    final    0.4555    0.8210     0.55     1.00      0.8086

The language GAP never shrinks: it grows monotonically, ~77x from L0 to
final, and is LARGEST at the output layer. The apparent dip at L18-L24 is
scatter peaking (0.86), and `content-gap` tracks scatter almost exactly
(0.867 vs 0.855) — the scatter IS content variation. Accuracy never falls
below 0.92 anywhere after L3.

This kills the natural three-phase story ("knows the language, diffuses
language-agnostically, then crystallises"): nothing goes agnostic. Language
identity is established early and monotonically SHARPENS to the output; the
middle layers are merely busy with content, whose variance temporarily swamps
the language component in any ratio statistic. An earlier note here claiming
"two regimes with a dip" was reading the ratio and inventing structure in the
denominator — retracted.

Practical consequence, which INVERTS the earlier advice: for a trained
fixed-direction probe use the FINAL layer (largest gap). L6 is only the best
target if you are using raw cosine similarity, where content noise matters.

Practical: for reading "which language" out of the engine, probe the CANVAS
residual around L6-L9 BEFORE the position resolves. Language separability is
about half the code-vs-prose axis (1.12 vs 2.05) and tracks language distance
(rust-vs-bash 1.99 > python-vs-bash 1.87 > rust-vs-python 1.12-1.33).

**OUTPUT-MODE probe (2026-07-19): the L6 axis is CONTENT TYPE, not output
FORMAT.** 5 modes x 8 items, cold canvas, exact token control (1 distinct
canvas token across all 40). Centroid cosines at L6:

                  code   fenced    prose    mdgen  toolask
    code          1.00     0.85    -0.72    -0.82    -0.15
    fenced        0.85     1.00    -0.86    -0.70    -0.10
    prose        -0.72    -0.86     1.00     0.66    -0.32
    mdgen        -0.82    -0.70     0.66     1.00    -0.32
    toolask      -0.15    -0.10    -0.33    -0.32     1.00

Adding "put the code in a fenced markdown code block" leaves the prompt GLUED
to plain code (+0.85) and OPPOSED to markdown prose (-0.70). The direction is
almost blind to formatting instructions — which is the useful outcome, since
"wrap it in a markdown block" is exactly the instruction most likely to fool a
naive code detector.

Two unplanned observations: markdown formatting perturbs PROSE more than
fencing perturbs CODE (mdgen~prose 0.66 vs code~fenced 0.85); and by L29 the
format distinction collapses further (code~fenced 0.91, prose~mdgen 0.90) —
four modes become two clusters.

`toolask` is near-ORTHOGONAL to all four modes (-0.10 to -0.33), suggesting
tool-calling is its own region rather than a flavour of code. **HEAVILY
CAVEATED**: this condition is a natural-language request for a tool action,
NOT the real thing. Genuine tool-emission mode needs the system block with
`<|tool>declaration:...<tool|>` special ids, and `--raw-prompt` BPEs that
markup as literal text (48 tokens vs 22), so it is unreachable from
`step-layer-probe` today. Probing it properly needs tools support added to
that command — a separate small engine change.

**MODE TRACKS THE MOST RECENT COMMITTED CONTENT — the cheap output-time
classifier looks VIABLE (2026-07-19).** Ran entirely on the working COLD path
(exact token control), sidestepping the broken warm probe: partial content is
placed in the prompt AFTER the model-turn marker, which reproduces the
output-time situation where already-committed tokens live in KV. 4 conditions
x 6 items.

                  purecode   endcode  pureprose  endprose   (L29)
    purecode          1.00      0.74      -0.90     -0.86
    endcode           0.74      1.00      -0.89     -0.84
    pureprose        -0.90     -0.89       1.00      0.75
    endprose         -0.86     -0.84       0.75      1.00

Both MIXED contexts follow their ENDING content, not their instruction:
`endcode` (prose intro then code) sits with pure code (+0.74/-0.89), and
`endprose` (code then prose) sits with pure prose (+0.75/-0.86). A probe
labelled by trailing content scores LOO acc 0.96 (L6) / 1.00 (L9) / 1.00
(L29), effects 1.43 / 1.22 / 0.92.

Two refinements this forces:
- **Late layers are better for SEGMENTATION**, inverting the earlier L6
  advice. The split: **L6 reads task INTENT** ("what was I asked to do"),
  **L29 reads current MODE** ("what am I emitting now"). Per-token
  classification wants the late layer.
- **Mode transitions look ASYMMETRIC.** At L6 `endcode` locks onto code at
  once (+0.75) while `endprose` only weakly returns to prose (+0.08), needing
  L29 to fully reassert (+0.75). Code context is "stickier"; code->prose may
  be the harder transition to detect. n=6 per group — treat as a lead.

**Scope boundary, important:** the committed content here sits in the PROMPT
(KV), which models the CROSS-BLOCK case (earlier blocks already committed).
Within a block the current content lives in the CANVAS, which canvas positions
see bidirectionally — and testing THAT was exactly what the broken warm path
was for. So: cross-block mode tracking is DEMONSTRATED; intra-block per-token
classification remains UNTESTED and is blocked on repairing the diag denoise
path.

**RETRACTION (2026-07-19): every "warm / committed canvas" result is VOID.**
The `--warm-steps` probe does NOT denoise toward the prompt. Calling
`run_denoise_step()` directly after `reset_block` churns noise instead of
converging: after 6 steps, 60 DIFFERENT prompts produce only 4 distinct
canvases, filled with multilingual garbage
(`𝔸 রক্তের ley ...`). A canvas actually resolving would give 60
distinct, prompt-specific canvases containing Python and prose. It evidently
needs state the generate loop configures and this path does not.

Retracted as a result: "the language signal is weaker on the committed canvas
(effect 1.12 -> 0.31)" — that compared cold against a DIFFERENTLY-NOISY
canvas, not a resolved one; and the positions-8/20/35/55 experiment, which
probed junk rather than generated code. **"Probe when ready to commit" is
UNTESTED, not answered.**

UNAFFECTED: everything on the cold path (`warm=0`), which uses the original
working probe and shows 1 distinct canvas across all 60 prompts (exact
control) — the L6 code/prose axis, the rust-vs-python language probe, the
gap/scatter decomposition, and the markdown/fenced/tool-mode results.

Fix path if resumed: find what `run_forward_once(StepFinishMode::Full)` needs
that `build_step_runtime` + `reset_block` does not supply (accept/commit
thresholds, step schedule, or SC state are the candidates), or drive the
canvas through the real generate loop and snapshot it. The `--warm-steps`
flag stays UNCOMMITTED — it is not fit for its stated purpose.

**Note for tree-sitter rerolling**: this bug is the point. "Advance the canvas
under control, then inspect" is exactly what a reroll needs, and it is NOT
readily available today. The prerequisite is therefore the **P5 `Refine`
primitive**, not tree-sitter and not the classifier. Also worth recording:
tree-sitter rerolling does NOT depend on the mode/language classifier at all
for single-language generations, where the language is already known from the
prompt or the fence. The classifier only earns its keep on MIXED content.

**TOOL-CALLING IS ITS OWN MODE (2026-07-19), with the control that proves it.**
No engine change was needed: `--raw` (NOT `--raw-prompt`, which is silently
ignored — unknown flags do not error) round-trips the chat template
byte-exactly, so the full `<|tool>declaration:...<tool|>` system block can be
hand-built and encodes to real special ids. 4 conditions x 8 items, cold
canvas, exact token control.

                toolreal  toolidle     prose      code     (L6)
    toolreal        1.00     -0.37     -0.82      0.18
    toolidle       -0.37      1.00      0.43     -0.84
    prose          -0.82      0.43      1.00     -0.54
    code            0.18     -0.84     -0.54      1.00

The decisive control is `toolreal` vs `toolidle`: IDENTICAL system block (same
tool declaration), differing only in whether the user asks to use the tool.
They separate at -0.37 (L6) and -0.72 (L29), so the state is tool-EMISSION
intent, not "a tool declaration is in context". Tools merely present washes
out with depth (`toolidle ~ prose` 0.43 -> 0.92 by L29).

Tool mode is NOT a flavour of code (+0.18 at L6, -0.40 at L29) despite both
being structured grammars. By L29 four conditions collapse to THREE clusters:
{toolreal}, {toolidle, prose}, {code}.

**Consequence for a cheap output-time token classifier** (the reason this
matters): prompt-context mode is cleanly separable into at least 3-4 classes,
and a linear probe is one dot product per position — 256 x 2816 ~ 720K MACs
per direction, sub-microsecond against a ~700ms step, output 256 floats that
folds into the EXISTING 36.1 KiB/step rowstats readback. No new sync. As pure
instrumentation it is trajectory-neutral, so golden stays byte-identical and
it can ship observational behind a `DGQ_` flag.

**BUT the blocking question is unanswered**: every probe so far reads ONE
position on a COLD canvas. Output-time classification reads EVERY position on
a RESOLVED one, and the single warm test showed the signal drop hard (effect
1.12 -> 0.31) — plausibly because once a position holds content, its own token
dominates and dilutes the context signal. That test probed position 129, which
held junk TAIL FILLER, not committed program text. Whether the signal survives
at positions holding REAL generated code is the experiment that decides the
whole idea, and it must run before any kernel work.

**Cheapest experiment that could kill it, in order:**
1. `step-moe-route-dump` on a handful of known-bash / known-python /
   known-prose generations. Do the expert histograms separate AT ALL? No
   probe training needed to answer this. If they do not separate, stop.
2. If they separate: linear probe on canvas hidden states (`step-layer-probe`
   dumps per-layer activations at a position) for per-position language,
   WITH both controls above, scored as the layer x step gradient.
3. Only then consider an on-device implementation.

**If it ever gets built, it should be a KERNEL, not a readback.** Router state
is on-device; pulling it back every step is a pipeline stall, and this
project's history is that instrumentation can dominate a step. The hot path
already does 1.00 syncs/step and 36.1 KiB readback for sample rowstats — a
detector that folds its output into THAT existing transfer adds no sync and
no stall.

**Adjacent, more tractable, same machinery** (worth doing first if any of
this is done at all): a purely NUMERIC defect detector on device.
- Per-token delimiter-delta table built once at load (each token's net quote
  and bracket contribution), then delimiter balance at every canvas position
  is a PREFIX SCAN — a primitive this repo already has from the SC sparse
  compaction fix. That also settles the interior-vs-tail question directly:
  you can see where balance breaks and whether it recovers before the block
  edge.
- A repair-partner table (delimiter-bearing token -> its corrected form) lets
  the same kernel gather both logits and emit the likelihood delta for the
  indifference test. Note this is the ONLY practical form of that test:
  reading logits back to compare on CPU is 256 x 262144 x 4 = ~268 MB/step.
- Two tiers: GPU numeric gate per block (~free) -> CPU decode/parse/repair
  (rare). Parsing is irregular control flow with dynamic allocation; it does
  not belong on a GPU and does not need to.

**Why the sequencing is nice:** a detector that only COMPUTES and never steers
is trajectory-neutral, so golden stays 8/8 byte-identical with no re-bless and
no live A/B. It could ship observational behind a `DGQ_` flag and produce the
blend RATE that is currently the top open item above — at zero risk to output.
Wiring it to an actual repair is trajectory-affecting and needs the full
ritual (golden re-bless, live A/B, not a replay sim).

Obligations if built: a `cpu.rs` oracle under the unified kernel tree, and the
dispatch cost MEASURED rather than assumed.

**Blocked on nothing, but pointless before:** the 20-minute specimen test
(take the six known blend specimens, construct the corrected canvas, compare
joint log-prob under one forward pass; the hypothesis predicts |delta| ~ 0).
A clean negative there collapses the indifference gate and shrinks all of the
above to a bare balance scan.

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
