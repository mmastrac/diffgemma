# OpenCode newline/structure collapse — root-caused to the serve KV lineage

The collapse requires serve's incrementally-built (cross-turn-reuse) KV
lineage; fresh prefill of the identical token stream at the identical seed
is clean, everywhere tested. Root cause: serve's KV mixes provenances
(batched fast-prefill, `finalize`'s truncate + small-chunk `extend_kv`, tool
response extends) that are causally correct but not bit-identical to one
fresh prefill. The denoise loop amplifies that sub-1e-4 drift into
different accept decisions; in repair-loop contexts (near-duplicate code +
failing-test grids) this flips a block outright, and a non-converged block
is COMMITTED anyway, poisoning every later block (a `}\n` flood converges
*cleanly* once seeded).

## Deterministic repro (no OpenCode needed)

`oc_sim.py` is a minimal OpenCode simulacrum: it drives `serve` through the
regex-lite task with REAL `write`/`bash`(rustc)/`edit` tools and nudges the
model back into the loop if it gives up while tests still fail.

```bash
target/release/diffgemma serve -m model/diffusiongemma-q4emb \
  --addr 127.0.0.1:8100 --ctx 100000 --log-dir /tmp/logsX &
python3 oc_sim.py 8100 /tmp/logsX          # NO seed arg — see below
```

Unseeded requests all run serve's default seed 42, so the whole session is
deterministic: two independent runs reproduced the SAME trajectory and the
SAME collapse (strained blocks 5, 12-15 of turn 7, late-window entropies
identical to 3 decimals).

## The evidence matrix (all seed 42, matched tokens)

| context (exact rendered prompt) | fresh KV (ask --raw / fresh serve) | session-lineage KV (serve) |
|---|---|---|
| user session turn 6 (13,980 tok) | clean (also seeds 7/123/1/2; also 4096-tok budget) | COLLAPSED at block 1 |
| sim turn 7 (8,285 tok) | clean ×2 (ask seed 7; fresh-serve seeds 7 and 42) | COLLAPSED blocks 5,12-15 |

## Fix, validated here

`DGQ_BLOCK_COMMIT_MAX_ENT=0.2` (default OFF; `DGQ_BLOCK_COMMIT_RETRY`,
default 1): a block that burns the whole step schedule with late-window
mean entropy above the floor is re-rolled with fresh noise, and if still
non-converged the turn ends WITHOUT committing it. Validated on this repro:
run D matches the collapsing run turn-for-turn, then at turn-7 block 5 the
guard fires (re-roll → still 0.258 > 0.2 → turn ends, blocks 1-4's 1024
tokens kept); the `}\n` flood never forms; 0 committed-strained blocks in
64. Healthy paths untouched: golden 8/8 byte-identical, suite 585/0,
smoketest 17/17. A stopgap — the KV-lineage drift is the real bug.

## Files

- `oc_sim.py` — the simulacrum client (also in the session scratchpad).
- `repro-prompt-14k.txt`, `serve-00006.json`, `model-00006-0000{1,2}.json`
  — the original user-session turn-6 artifacts.
- Full runs on this machine: `/tmp/logs` (original session), `/tmp/logs2`
  (first sim collapse), `/tmp/logs7` (run A, deterministic replay),
  `/tmp/logs8` (run B, reuse=0), `/tmp/logs9` (run C, topk-decode=0).
