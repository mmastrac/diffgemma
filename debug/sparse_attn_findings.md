# Sparse Attention Investigation — Measurement Results (2026-07-14)

## Goal
The attention stage is the only large prefill stage NOT at the half-MMA compute
wall (~1.25 TF/s vs ~3.8 TF/s ceiling), and it grows with kv. Find sparse-
attention levers.

## Method
Dumped attention probability distributions (per-head, over all KV positions) for
full layer 5 (hd=512, the E17-target layer class) at three canvas positions
(0, 129, 200) with a ~10k-token prompt (kv_len=9933). Analyzed mass
concentration vs locality, vs top-k, and vs top-k ∪ prompt-anchor.

## Findings

### Locality (block-sparse by distance from query) — DEAD
Mass within ±2k of the query = 39% (at pos 129, kv=10k). A 2k-wide local band
captures only 39% of mass — would drop 61% of attention mass. The model has a
strong "prompt-start anchor" (first 64 prompt tokens carry 19% of mass) plus
retrieval-from-the-end (far half of prompt carries 40%). A fixed-distance band
mask misses both. **Not quality-safe; lever disproven by measurement.**

### Top-k sparsity — REAL, quality-gated
| k | mass (pos 0) | mass (pos 129) | mass (pos 200) | % of KV |
|---|---|---|---|---|
| 128 | 85% | 71% | 98% | 1.26% |
| 256 | 88% | 77% | 98% | 2.51% |
| 512 | 90% | 82% | 99% | 5.03% |
| 1024 | 93% | 87% | 99% | 10.05% |

top-512 retains 82-99% of mass while processing only 5% of KV. Pattern robust
across query positions (entropy ratio 0.29-0.57, all <1.0 = non-uniform).

### Anchor union — marginal
Adding a deterministic "always include first N prompt positions" anchor on top
of top-k barely helps: top-128 (71.3% @ pos129) vs top-128+anchor64 (71.3%).
The anchor positions are ALREADY in the top-k for most heads — the anchor is a
subset of top-k, not additive. Pure anchor (no top-k) carries only 19-34%.
**The anchor is not a separate lever; top-k subsumes it.**

### FLOPs at long context — k is nearly free
Dense attention FLOPs = 2 × canvas × T × hd (QK + PV equal).
Top-k FLOPs = canvas × hd × (T + k) — QK unchanged, PV shrinks.
At T=10k:  k=128 → 50.6%, k=1024 → 55.0% of dense.
At T=100k: k=128 → 50.1%, k=1024 → 50.5% of dense.
**At long context all k values converge to ~50% FLOPs** (QK dominates, PV
becomes negligible). So pick the largest k that fits the memory/tile budget —
larger k = more mass retained at near-identical FLOP cost.

## Projected speedup (top-k=512)
Attention FLOPs ≈ halve at any context length. Attention is 26% of prefill at
kv=2k, growing with kv. So:
- Short context (kv=2k): ~13% prefill speedup (48% of 26%)
- Long context (kv=100k): ~25%+ prefill speedup (attention dominates)

Well above the bar. But non-bit-identical — needs the multi-seed wart census +
doc-QA ladder gate (AGENTS.md §8).

## Recommendation
Proceed with a top-k prototype. Because k is nearly free at long context, the
binding constraint is implementation cost (sparse gather + per-head top-k
selection), not FLOPs. Suggest:
1. Prototype at k=512 (90% mass avg, clean quality signal).
2. If quality passes, push k higher (1024, 2048) — FLOPs barely move at long
   context and mass approaches 99%.
3. The anchor is not a separate optimization — top-k subsumes it.

---

## Prototype results (2026-07-14) — SELECTED ALGORITHM TOO SLOW

Built `attention_topk` (E20) with a binary-search-on-float-bits selection
(32-pass threshold scan + atomic emit). Parity GREEN vs `cpu::topk_causal`
(7/7 tests pass: 4 CPU oracle + 3 GPU parity including f32 side + k=1 argmax).

Bench at model shape (canvas=256, 16Q/2KV, hd=512, GQA group 8), k=64:

| kv     | E17 dense (ms) | top-k (ms) | ratio |
|--------|----------------|------------|-------|
| 8192   | 59.4           | 153.6      | 0.39x |
| 30000  | 139.6          | 519.9      | 0.27x |
| 60000  | 349.8          | 1203.3     | 0.29x |
| 100000 | 552.9          | 1090.1     | 0.51x |

**Top-k is 2-3.7x SLOWER at moderate context; only narrows to 2x slower at
100k.** Root cause: the 32-iteration binary search reads the S plane 33 times
(32 search passes + 1 emit). At T=100k the S plane is 100 MB/row; 33 passes =
~52 GB of reads per the compute estimate — the selection dominates.

**The lever is real but the selection algorithm is wrong.** Theoretical floor:
- QK unchanged (same as E17 QK cost).
- topk_softmax: minimum 1-2 passes over S (mandatory read + selection).
- topk_pv: canvas × K_PAD × hd = trivially small.

At 100k: 2-pass selection ≈ 1ms (200 MB / 200 GB/s). Total top-k ≈ E17_QK + 1ms.
If E17_QK ≈ 270ms (half of 553ms total), top-k ≈ 271ms vs E17 553ms = ~2x faster.

**Fix needed: replace the 32-pass binary search with a 2-pass histogram
selection** (1 pass to build a coarse histogram of float-bit patterns, prefix-
sum to find the threshold bin, 1 pass to emit). Estimated 15-30x reduction in
topk_softmax cost.

---

## V2: histogram selection + f16 path — 2.2-2.4x FASTER (2026-07-14)

Two fixes applied:
1. **2-level radix histogram selection** (256-bin × 2 levels, 3 passes over S
   instead of 33). Replaced the binary search entirely.
2. **f16 KV path for the bench** (apples-to-apples with E17's production bench;
   the v1 bench was accidentally using the f32 side-ring path which is 2x
   slower for the QK GEMM).

Parity GREEN: 7/7 tests pass (4 CPU oracle + 3 GPU parity including f32 side
+ k=1 argmax with relaxed tolerance for ties).

Isolated attention stage bench (model shape, hc=4, k=64, f16 KV):

| kv     | E17 dense (ms) | top-k (ms) | ratio  | qk    | sm    | pv   |
|--------|----------------|------------|--------|-------|-------|------|
| 8192   | 27.9           | 12.9       | 2.17x  | 8.7   | 3.1   | 0.9  |
| 30000  | 103.2          | 43.7       | 2.36x  | 31.1  | 11.6  | 0.9  |
| 60000  | 206.0          | 86.3       | 2.39x  | 62.0  | 23.2  | 0.9  |
| 100000 | 345.7          | 142.9      | 2.42x  | 101.6 | 36.9  | 0.9  |

Stage breakdown confirms: QK is 71% of top-k time (unchanged from E17), softmax
is 26% (the 3-pass histogram — next optimization target), PV is <1% (scalar
fallback is fine; no need for MMA-based sparse PV).

## End-to-end prefill speedup (bench-prefill-super, kv=15k, n_subs=4)

| path        | ms/super-chunk |
|-------------|----------------|
| E17 dense   | 3368           |
| E20 topk=64 | 2845           |

**1.18x end-to-end prefill speedup.** Attention is ~26% of prefill, so 2.4x on
attention → 1.18x overall. The math checks out.

## Quality gate (2026-07-14)

- Smoketest 17/17 × 3 seeds {7, 42, 123}: ALL PASS
- Long-context doc-QA ladder (4/4 docs up to 20k tokens, 8/8 keywords): PASS
- Golden 3/8 (expected non-bit-identical — trajectory diverges; not a regression
  signal, just non-identity)

**Awaiting human sign-off to ship default-on.** Currently default-OFF
(`DGQ_ATTN_TOPK` opt-in).

## Next optimization (if needed)

The softmax (26% of top-k time) can be cut further by replacing the
atomic-based histogram with per-thread local histograms + a single merge
(eliminates atomic contention). Estimated ~2x reduction in softmax cost →
top-k would be ~2.6-2.8x faster than E17 (vs current 2.2-2.4x).
