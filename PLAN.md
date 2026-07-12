# diffgemma-mps — plan

Open work only. Design + contract: **ARCHITECTURE.md** (incl. Negative
Knowledge — check it before planning perf/quality work). Working
discipline + commands: **AGENTS.md**. Everything done lives in git history.

## State (for orientation, not history)

Engine is near-done: wall-clock beats MLX-4bit on short/medium chat,
denoise convergence parity-class, long context works to 105k (needle-exact,
doc-QA-grounded to 20.6k), OpenAI-compatible `serve` with multi-conversation
KV reuse + tool calling, gates green (smoketest 17/17, golden 8/8, suite
green, census signed off).

## Open items

| Item | Notes |
|---|---|
| **E16 token fusion / KV merging** (IN PROGRESS) | The only unexplored long-context denoise SPEED lever (cuts token count, not bytes — fewer score rows; ~4.5 s/step at 105k is the target). Oracle so far: fusion is gist-preserving/verbatim-lossy; residual-gated r=2 ≈ control quality but 1.4× (under bar). Next: min-pairwise/outlier gates, mass keep-lists, multi-seed, non-English. MUST gate on the doc-QA ladder, not needles |
| **Long-ctx re-validation debt** | Post-root-cause-fix: re-run needle 33k/105k and the 100k field-incident repro on the uncapped fast path |
| **E5 fragment-tile attention** (timebox ≤1 d) | Fewer, larger MMA fragments per stream on `attention_mma_full`; bar = ≥1.5× at kv 32k on the bench row, else record + move on. Full multi-seed gate if shipped (f16 math order changes) |
| **E7 confidence-threshold sampler** | MLX's alternate accept rule (`diffusion_threshold=0.9`); also the candidate fix for the early-stop creative-tail warts. Ship only if gate-neutral-or-better |
| **E3 canvas shrink near max_tokens** | Close divergence #5 (MLX shrinks to max(remaining, 64)); minor tail win; trajectory-affecting → multi-seed gate |
| **v1 productization** | README quickstart (none exists), fetch/quantize UX one-liner, benchmark page with the MLX methodology, release tagging + `--version` with `.dgq` manifest-version gate |
| **CI completion** | Nightly model-gated tiers are scaffolded; wire fully (smoketest + golden + longctx + perf floors), weekly multi-seed aggregate + census |
| **Broader eval** | The 17-prompt gate is sensitive but narrow; add a ~100-prompt adherence set, weekly, non-blocking |

## Code-health refactor backlog (2026-07-12 audit; execute in order)

1. Warning zero + CI deny-warnings gate: `cargo fix` the mechanical ~48, triage the 57 dead_code (delete stale bench scaffolding / `#[allow]` kept diagnostics).
2. Dead-residue basket: dead encoder imports in generate.rs, `block2` dep (0 refs), `div_up` ×5 → `gpu_common`, 7 stale pre-refactor path comments, main.rs duplicated FAILURE dispatch arms, `flags::Config` → `RuntimeConfig`. Flag removal RESOLVED: no genuinely-dead `DGQ_*` flags exist (the disproven ones are still wired A/B toggles, kept per the ledger). NOTE: items 1-2's non-test dead-code pruning over-cut test-only re-exports (`metal::DgqGpuBlob`, `metal::build_offsets_from_store`, `sample::ARGMAX_HIST_MAX`, `LivenessCtx.first_step`) — restored test-gated so `cargo check --tests` is clean again while the non-test build stays warning-zero.
3. Extract inline tests to sibling files: step_kv.rs (~1.5k, 50% test), step_generate.rs (~0.9k, 78% test), server.rs (~0.6k), step_kernel.rs (~0.8k).
4. Split step_kernel.rs along impl seams: step_types / step_pipelines (50-field struct) / step_dispatch (StepEnc) / step_runtime; move the ~2.6k trailing bench/capture fns + step_kv's ~1k `run_step_*` audits into diagnostic modules.
5. ~~Split main.rs (5.4k)~~ **SPLIT DONE (split-only)**: `cli.rs` (Cli/Command + parse_cli) + `commands/{mod,common,step_debug,step_gate,bench,gen_cmd,chat,smoketest,golden_cmd,model_ops}.rs`; main.rs now 50 lines. Every top-level item moved verbatim (no deletions, no MoE collapse). **Prune deferred**: the "cold" dump subcommands each pair with a `python/scripts/{dump,compare}_*.py` oracle (manual layer/attn/MoE/embed parity-debug harness) — not dead, so retiring them + collapsing the 4 `step_moe_*_dump` is its own sign-off decision, not a mechanical refactor.
6. `GemmCompileConfig` struct in device.rs (collapse the 4 preset wrappers + 4-bool threading; the KernelVariant pattern already in-file).
7. `crate::Error`: move `safetensors::Error` (de-facto crate error across 124 files) to src/error.rs + add Gpu/Runtime variants (kill the `Format(&str)` catch-all).
8. `metal/oracle/` quarantine: nest the decode-oracle family (decoder, decoder_attention, lm_head, sampler, attention-scalar, self_conditioning, kv_cache; fold kernels.rs + sampler_kernels.rs into engine.rs) — mirrors shaders/oracle.
9. Judgment-call items (need sign-off): parametrize the 4 near-copy oracle GEMM fixtures (824→~250 lines); finish the 8 renaming aliases (148 sites; KEEP pass-through flat aliases); dedup the two CPU MoE routers + RoPE/rms twins (parity-gated, exact tie-break); retire `pack/` (confirm no iris.pack deployments); delete or keep model/decoder.rs full-stack CPU forward (test-only).
10. `serve/` cluster split (server.rs → worker/wire/sse/mapper + conversation + toolcompact; check `lcp()` duplication) — last, it's the serve hot path.

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
