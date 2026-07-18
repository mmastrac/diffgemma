# Prefix-exit deliberation-depth exhibit (2026-07-18)

Same request (clean 6.9k-token OpenCode history, regex_lite task), same
seed (123), same binary — the only delta is `DGQ_PREFIX_EXIT=0.05`.

- `base.json`: 2 blocks, 141 thinking tokens (a plan restatement), clean
  tool call, 51.0s.
- `prefix_exit.json`: 12 blocks, 2,040 thinking tokens — glob-vs-regex
  `*` semantics deliberation, UTF-8 slicing hazard caught, the full
  matcher drafted and refined in-thought, 14 test cases enumerated —
  clean tool call, zero stutters, 168.6s.

Reading: the "hard-stuck tails" (mean entropy 0.7–1.5 nats) that
prefix-exit defers are MID-THOUGHT states; whole-canvas stop pressure
steamrolls them, prefix exits honor them. Single sample — the
quality-mode experiment in PLAN (matched token budgets, census
multi-seed) is the decision gate.
