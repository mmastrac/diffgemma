# OpenCode newline/structure collapse — root-caused to the serve KV lineage

The PLAN.md "span handles" phase-1 target (newline/structure collapse on
agent repair loops), investigated 2026-07-16/17. **The axis is named: the
collapse requires serve's incrementally-built (cross-turn-reuse) KV
lineage; fresh prefill of the identical token stream at the identical seed
is clean, everywhere we tested.**

## Deterministic repro (no OpenCode needed)

`oc_sim.py` is a minimal OpenCode simulacrum: it drives `serve` through the
regex-lite task with REAL `write`/`bash`(rustc)/`edit` tools and nudges the
model back into the loop if it gives up while tests still fail.

```bash
target/release/diffgemma serve -m model/diffusiongemma-q4emb \
  --addr 127.0.0.1:8100 --ctx 100000 --log-dir /tmp/logsX &
python3 oc_sim.py 8100 /tmp/logsX          # NO seed arg — see below
```

Unseeded requests all run serve's default seed 42 (`per_request_cfg`:
`job.seed.unwrap_or(base_cfg.seed)`), so the whole session is
deterministic: two independent runs reproduced the SAME trajectory
(write → rustc E0382 → edit → rustc E0308 → edit → failing tests →
turn-7 monologue) and the SAME collapse — strained blocks 5, 12, 13, 14,
15 of turn 7, late-window entropies identical to 3 decimals
(0.437/1.147/0.491/1.131/0.456). Blocks 13-16 are a `}\n` flood; block 16
*converges cleanly onto the flood* (35 steps, ent 0.045) — committed
garbage is a self-consistent attractor.

## The evidence matrix (all seed 42, matched tokens)

| context (exact rendered prompt) | fresh KV (ask --raw / fresh serve) | session-lineage KV (serve) |
|---|---|---|
| user session turn 6 (13,980 tok) | clean (also seeds 7/123/1/2; also 4096-tok budget) | COLLAPSED at block 1 |
| sim turn 7 (8,285 tok) | clean ×2 (ask seed 7; fresh-serve seeds 7 and 42) | COLLAPSED blocks 5,12-15 |

**The smoking gun (run A vs run B):** the identical unseeded sim against
serve with `DGQ_KV_REUSE=0` matches the collapsing run token-for-token
through turn 5, then at turn 6 — SAME 8,126-token prompt, SAME seed —
produces a DIFFERENT reply (edit vs bash). Reuse-lineage KV and
fresh-prefill KV are behaviorally divergent on identical inputs. The
reuse=0 session then went DEEPER into repair-loop territory (3,091-token
monologues, persistent FAIL feedback) and stayed clean: 0 strained blocks
in 50.

## Mechanism (three layers)

1. **KV lineage patchwork.** A finalized serve conversation's KV mixes
   provenances: prompt tokens from the batched fast-prefill path, reply
   tokens re-encoded through `finalize`'s truncate-to-common-prefix +
   `extend_kv` (small-chunk extend path), tool responses through response
   extends. Each path is causally correct but they are not bit-identical
   to each other or to one fresh prefill — accepted drift, per design.
2. **Chaos amplification.** The denoise loop amplifies sub-1e-4
   differences into different accept decisions (long-documented). By
   turn 6 of the repro the drift flips the reply outright. Most forks are
   benign; repair-loop contexts (near-duplicate code copies + failing-test
   grids) put a strain attractor in range.
3. **The commit amplifier.** A block that ends its 48-step schedule at
   ~45% accepted with high late-window entropy is COMMITTED anyway. The
   committed garble poisons every later block (the `}\n` flood converges
   *cleanly* once seeded). This is what turns one strained block into a
   destroyed turn.

## What it is NOT (corrections to earlier notes)

- **The 2026-07-16 "top-k family exonerated by fully-dense collapse"
  bisect was INVALID**: `ask --raw` then encoded special-token literals as
  plain BPE, so those replays fed a structurally corrupted transcript
  (prompt_len 14,555 vs serve's 13,980); the "collapse" they reproduced
  was an artifact. Fixed in `commands/common.rs` (`--raw` now uses
  `encode_with_specials`; golden 8/8 after).
- Attention config is neither implicated nor fully exonerated: flipping
  `DGQ_ATTN_TOPK_DECODE=0` (reuse ON) forked the trajectory and that path
  didn't collapse (0/66 blocks) — same ambiguity as any perturbation
  (reuse=0 also clean, different seeds also clean). The collapse class
  predates the top-k ships (PLAN documented it before 523d065).
- Not a ring/KV-machinery memory-corruption bug in the classic sense: the
  turn-6 flip shows *numeric* lineage divergence, not clobbered rows.
  (A byte-level layer-cos of lineage vs fresh KV is the natural follow-up
  to quantify where the drift concentrates.)

## Fixes, ranked

1. **Commit policy — BUILT, default OFF (`DGQ_BLOCK_COMMIT_MAX_ENT=0.2`
   to enable; `DGQ_BLOCK_COMMIT_RETRY`, default 1). A stopgap for if the
   collapse class bites again — the KV-lineage drift is the real bug:** a block that burns the whole
   step schedule with late-window mean entropy above the floor is re-rolled
   with fresh noise, and if still non-converged the turn ends WITHOUT
   committing it. Validated on the deterministic repro: run D matches the
   collapsing run A turn-for-turn, then at turn-7 block 5 the guard fires
   (re-roll → still 0.258 > 0.2 → turn ends with blocks 1-4's 1024 tokens
   kept); the `}\n` flood never forms; 0 committed-strained blocks in 64.
   Healthy paths untouched: golden 8/8 byte-identical, suite 585/0,
   smoketest 17/17 (validated at 0.2). Interacts with E7 (confidence
   accept).
2. **Mitigation also available:** `DGQ_KV_REUSE=0` on serve for agent
   workloads — costs a full prefill per turn (~25-40 s at 8-14k with
   dyn-topk prefill) and removed the collapse on this session.
3. **Longer term:** quantify + shrink lineage drift (re-ground
   conversations with a fresh prefill every N turns; or measure which
   extend path contributes most drift); the span-handles concept attacks
   the same repair-loop context shape from the content side. The model
   still fails INTO the strain on repair-loop content (run D turns 7-10
   ramble in thought and hit the length cap) — the guard contains the
   damage; it does not make the model good at the task.

## Files

- `oc_sim.py` — the simulacrum client (also in the session scratchpad).
- `repro-prompt-14k.txt`, `serve-00006.json`, `model-00006-0000{1,2}.json`
  — the original user-session turn-6 artifacts.
- Full runs on this machine: `/tmp/logs` (original session), `/tmp/logs2`
  (first sim collapse), `/tmp/logs7` (run A, deterministic replay),
  `/tmp/logs8` (run B, reuse=0), `/tmp/logs9` (run C, topk-decode=0).
