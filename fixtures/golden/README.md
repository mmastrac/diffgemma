# Golden byte-identity pack

The Tier-1 refactoring gate (task #73). `golden.json` holds a fixed matrix of
generation cases, each with its recorded **byte-identical** output: the exact
`token_ids`, a hash of the causal-KV state, and the denoise metadata. Because
generation is deterministic per seed, a refactor that is meant to change nothing
must reproduce every byte — far stronger and faster evidence than a quality gate.

## Run

```
# Check (the gate): every case must be byte-identical to its blessed record.
diffgemma-mps golden -m model/diffusiongemma-q4emb

# Re-record (only for an INTENDED numeric change, after the Tier-2/3 ritual):
diffgemma-mps golden -m model/diffusiongemma-q4emb --bless

# One case:
diffgemma-mps golden -m model/diffusiongemma-q4emb --filter longctx
```

## What each case pins

| id                | path made load-bearing                                   |
|-------------------|----------------------------------------------------------|
| `engine_prefill`  | ≤256-token prompt → engine prefill floor; determinism double-run |
| `early_stop_off`  | `no_early_stop` → the max-steps denoise path             |
| `cross_turn_reuse`| two turns, no reset → delta prefill / cross-turn KV reuse |
| `multi_block_reply`| >256-token reply → multiple committed blocks + fast block extend |
| `tool_call`       | tool declaration rendered → tool-call grammar emission   |
| `fast_prefill_3k` | >256-token prompt → fast prefill + batched super-chunk   |
| `ring_wrap_2p5k`  | >2048-token prompt → the sliding KV ring wraps           |
| `longctx_13k`     | 13k prompt → full-layer attention at depth               |

## Discipline

- **Byte-identity is brittle by design.** Any float reassociation (loop reorder,
  accumulator change) breaks it even when harmless — that is the point: it forces
  the "was this supposed to change numerics?" question on every refactor.
- **Re-bless is a ritual, not a reflex.** `--bless` overwrites the blessed
  records; only run it once an intended change has cleared the multi-seed
  smoketest (`{7,42,123}`) + `--longctx` + wart census + sign-off.
- **Machine class.** Fixtures are blessed on the 36GB M3 Pro dev box. The
  session runs f16 KV at `max_seq=16384` (≈20 GiB resident, below the q8-auto
  threshold). On a smaller Mac the auto policy could pick q8 KV → different bytes
  → a false regression. See `src/golden.rs` for the format-pin follow-up.
- The `longdoc.md` fixture under `../smoketest/` is frozen; do not regenerate it
  (the doc-token offsets these cases slice into would move).
