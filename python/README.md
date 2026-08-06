# diffgemma-parity

Reproducible Python reference checks for **diffgemma**: tokenizer/template
parity, layer-by-layer and generation-level numerical parity against HF and
MLX references, and prefill-tuning tooling.

Uses [uv](https://docs.astral.sh/uv/) for a locked virtualenv. Weights are read from `../model/transformer/` (symlinked Hugging Face cache).

## Setup

```bash
cd python
uv sync
```

## Run tests

```bash
uv run pytest -q
```

Covers tokenizer parity, chat-template rendering, canvas-RNG parity (`diffgemma_parity/canvas_rng.py` vs Rust `sample::initialize_canvas`), and the denoise-stats helpers shared by the dump scripts below — no full model download.

## Tokenizer & chat-template parity

`tests/test_rust_tokenizer.py`, `tests/test_chat_template.py`, `tests/test_tokenizer_parity.py` — `transformers.AutoTokenizer` / `tokenizers.Tokenizer` against the local checkpoint's `tokenizer.json` + config.

Rust cross-check:

```bash
cargo run --release -- tokenize "Hello" --raw          # raw BPE (legacy)
cargo run --release -- tokenize "Why is the sky blue?" # chat template (default)
uv run pytest -q tests/test_rust_tokenizer.py tests/test_chat_template.py
```

## Denoise-trace parity (HF / MLX / rust)

Full-model parity needs optional deps:

```bash
uv sync --extra model   # torch, loads ~48 GiB bf16 — may OOM on laptop
uv run python scripts/dump_denoise_trace.py -p Hello --seed 42 --steps 4 -o /tmp/hf_trace.json
uv run python scripts/compare_denoise_trace.py /tmp/hf_trace.json ../fixtures/generate/denoise_trace_hello_layers3_steps4_seed42.json
```

See `fixtures/generate/README.md` for the matching Rust `--write-trace` command.

For a low-RAM reference, convert the cached HF checkpoint to MLX once (`.dgq` is Rust-only):

```bash
uv sync --extra mlx
uv run python scripts/quantize_mlx.py --hf-path google/diffusiongemma-26B-A4B-it -o ../model/mlx-mxfp4
uv run python scripts/dump_mlx_denoise_trace.py --model ../model/mlx-mxfp4 -p Hello --seed 42 --steps 8 --no-early-stop -o /tmp/mlx_trace.json
uv run python scripts/compare_denoise_trace.py /tmp/mlx_trace.json /tmp/mono_trace.json
```

Rust `.dgq` for inference: `diffgemma quantize -o /tmp/quantized-weights` (different format).

## Layer-level parity dumps

Each pair dumps MLX-reference vs rust-monolithic checkpoints at one point in the forward pass and diffs them. All share `diffgemma_parity/canvas_rng.py` so both sides denoise the same canvas rows.

| dump | compare | rust subcommand |
|---|---|---|
| `dump_layer_attn.py` | `compare_layer_attn.py` | `step-attn-dump` |
| `dump_layer_hidden.py` | `compare_layer_hidden.py` | `step-layer-probe` |
| `dump_layer_moe.py` | `compare_layer_moe.py` | `step-moe-dump` |
| `dump_layer_moe_single.py` | `compare_layer_moe_single.py` | `step-moe-single-dump` |
| `dump_preamble_hidden.py` | `compare_preamble_hidden.py` | `step-preamble-dump` |
| `dump_embed_row.py` | `compare_embed_row.py` | `embed-row-dump` |
| `dump_step1_logits.py` | `compare_step1_logits.py` | `step-logits-dump` |
| `dump_step1_entropies.py` | `compare_step1_entropy.py`, `compare_step1_full.py` | `--write-trace` w/ `DGQ_TRACE_ENTROPY=full` |

## Generation-level comparison

`mlx_generate.py` (ground-truth MLX reply) + `compare_generation.py` (runs both engines and diffs decoded text, tokens, steps, latency).

**Never run two model-loading processes at once** — each holds a ~15 GiB+ resident model. Serialize these against any concurrent rust `ask`/`generate`/`serve` run.

## Prefill benchmarking & tuning

- `mlx_prefill_bench.py` — MLX prompt-processing throughput, content-independent above ~32k tokens; feeds the top-level `README.md` prefill throughput table (A/B target: rust `ask --prompt-len N`).
- `tune_prefill_attn.py` — Optuna TPE search over the E17 prefill-attention tile config (task #87), driving rust `bench-prefill-attn`.
- `holistic_sweep.sh` — 10-dim BO sweep (attn tiles, HC, softmax TPG, dense-GEMM tile, MoE-sparse/prefill tiles) against `bench-prefill-super`, one kv bracket at a time; writes `holistic_v2.db`. One process at a time — see the warning above.

## One-off oracles

- `tool_format_oracle.py` — renders the reference chat template's tool-call grammar from the HF `chat_template.jinja`; the exact strings are the oracle `src/tools/mod.rs` and `src/tools/tests.rs` are unit-tested against.
- `e22_block_mass.py` — E22 block-granular attention selection oracle (task #101, KILLED), consumes `step-attn-qk-dump` output offline.
