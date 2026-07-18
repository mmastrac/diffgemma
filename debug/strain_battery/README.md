# Strain battery — collapse repro harness + preserved artifacts

`battery.py <serve_bin> <model_dir> <prompts_dir> <outdir> [ctx]` — drives
`serve` over a matched prompt pair × seeds × env-arms and scores each run:
`nc` (non-convergence: `denoise_stop == "max_steps"` and the late-window
min of `mean_ent_per_step` stays > 0.2) and `flood_score` (top-unigram
fraction). Results land in `<outdir>/results.jsonl`. Lesson baked into v2:
single-trajectory A/Bs are meaningless (any perturbation forks the
trajectory) — only the deterministic cell (seed 42, collapse prompt)
reproduces the live failure; and give the run a 14-block runway
(MAX_TOKENS=3584) or the turn ends before the strain onset.

- `prompts/` — the matched pair (6.9k tokens each, from real OpenCode
  serve dumps): `clean.json` (never collapsed) and `collapse.json` (the
  seed-42 flood cell).
- `collapse_seed42/` — the preserved flood trajectory from a fresh-prefill
  serve run: `ops.jsonl` (replayable via `diffgemma-mps replay`) and the
  serve log. This is the input for the open "MLX matched-canvas dig" item
  in PLAN — can MLX's sampler survive the same conditioning?

Causal chain and fix: see `debug/opencode_collapse/README.md` and commit
`abb5153` (defer default OFF + ToolRepairStage).
