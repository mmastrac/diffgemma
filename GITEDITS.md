# GITEDITS — commit-message gaps found during the 2026-07-12 doc reduction

Audit method: for every historical claim removed from
SPEC/NOTES/KERNELS/ROADMAP/PLAN, the cited commit's message body was checked
for coverage (or `git log -S`/`--grep` located the landing commit). Note
that the doc deletions themselves preserve every removed word in history —
this file only tracks claims that deserve to be FINDABLE via commit
messages and are not.

Result: **coverage is essentially complete.** The repo's commit bodies
(12–40 lines on every load-bearing change) already carry the measurements
and verdicts. Spot-verified: 2b0d12b (E15 root cause), a259d8c (no-freeze),
696ef2e (SC seed), 3285ebe (ring clobber + stale-shader holes), 1c843a9
(tiny-M closure incl. the 0.12 ms hazard-serialization rule), 711f324
(GEMM TF/s ceiling numbers), f8e66a3 (headroom exoneration list), d6ff66c
(attention occupancy + tested-and-slower variants), cd0567a/78317c2 (q8 KV
E4 + auto-enable), 6551fef (E2 disproof), eb0edba (batched prefill).

Entries that would be nice-to-have in history but were message-omitted:

1. **Wall-clock head-to-head table (ours vs MLX-4bit, 2026-07-05 era)** —
   the probe numbers (capital 2.9 vs 3.7 s; sky ~410 tok 22.0 vs 27.6 s;
   transformer ~840 tok 50.2 vs 59.2 s) landed via docs-only commits and no
   commit message carries the full table + methodology (matched config,
   temp 0, natural finish, serialized runs). If history is re-edited,
   attach the table to the entropy-early-stop default commit (1bc47cc) or
   the nearest bench commit.

2. **Legacy (pre-tunable) hot-path budget table** — the per-bucket ms
   breakdown of the 1.22 s step existed only in KERNELS.md (now in its
   deletion commit's parent). Superseding numbers are committed (tunable
   phases); the legacy table is baseline-archaeology only. Optional.

Nothing else met the "really needed" bar.
