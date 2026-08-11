# MTP / step-simulator experiment infra (program cut 2026-08-10)

Training and corpus tooling for the speculative-drafting program — both
lines (free-run MTP drafting, step-simulator canvas seeding) were measured
dead; see ARCHITECTURE.md Negative Knowledge for the disproofs and the
numbers. Kept because the harnesses, the teacher recipe, and the capture
formats outlive the verdict (quality-injection experiments reuse them).

Everything is env-configured (no CLI args beyond epoch count); defaults
point at scratch/alan paths, so set the `MTP_*` / `STEP_*` vars.

- `mtp_train.py` — rollout fine-tune (K=4, own hiddens at depth) of the
  Gemma-4 assistant head on engine-state dumps (`mtp-dump` output).
  `MTP_DATA` colon-separated dump dirs, `MTP_OUT`, `MTP_ASSIST`, `MTP_DIFF`
  (HF snapshots), `MTP_STRIDE`, `MTP_POS_CAP`, `MTP_HOLDOUT_GROUPS`.
  Saves bf16 for the Rust loaders — copy the ORIGINAL assistant snapshot's
  config.json over the checkpoint before loading (newer transformers writes
  configs older parsers misread).
- `step_train.py` — step-simulator heads over a `mtp_step_dump_dry` dump
  (`STEP_DATA`): `STEP_ARCH=mlp|attn`, `STEP_LABEL=next|final`. Reports
  changed-position accuracy vs the copy-through baseline (the only honest
  metric; overall accuracy is dominated by copy-through).
- `step1_train.py` / `step1_emit.py` — step-1-from-prefill head (256 queries
  cross-attending prompt prefill hiddens) and the canvas emitter for the
  Rust `mtp_step1_seed_probe` (which carries the oracle + majority-prior
  control arms).
- `or_corpus.py` — OpenRouter teacher generation: 31B thinks (reasoning
  param), DeepSeek renders the relay answer from prompt+thought, difflib
  agreement filter. `OPENROUTER_API_KEY` env, `OR_SPEND_CAP`.
- `hf_import.py` — streaming HF dataset importer (OpenCodeInstruct, tulu,
  smoltalk) into the corpus prompt format.
- `make_self_prompts.py` / `make_teacher_prompts.py` — prompt sets.
- `drift_capture.py` / `drift_test.py` — the original zero-shot drift test
  of Google's stock head on our backbone hiddens (manual layer-streaming
  loader: accelerate disk-offload leaves encoder layers as meta tensors).

Corpus shards live in `/corpus` (gitignored; formats documented in
`corpus/README.md`). Checkpoints from the cut program: `head_v1{,_e0..e2}`
and `step*_head*` on alan `~/mtp/`.
