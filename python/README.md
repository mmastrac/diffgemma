# diffgemma-parity

Reproducible Python reference checks for **diffgemma** (tokenizer parity today; model hooks later).

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

## Hugging Face reference

Tests use `transformers.AutoTokenizer` and `tokenizers.Tokenizer` against the local checkpoint — no full model download, only `tokenizer.json` + config.

### Denoise trace (P1.6)

Full-model parity needs optional torch deps (loads ~48 GiB bf16 — may OOM on laptop):

```bash
uv sync --extra model
uv run python scripts/dump_denoise_trace.py -p Hello --seed 42 --steps 4 -o /tmp/hf_trace.json
uv run python scripts/compare_denoise_trace.py /tmp/hf_trace.json ../fixtures/generate/denoise_trace_hello_layers3_steps4_seed42.json
```

See `fixtures/generate/README.md` for the matching Rust `--write-trace` command.

### MLX quantization (low-RAM Python reference)

`.dgq` is Rust-only. To run DiffusionGemma in Python without loading bf16, convert the
cached HF checkpoint to MLX (~15 GiB mxfp4):

```bash
uv sync --extra mlx
uv run python scripts/quantize_mlx.py --hf-path google/diffusiongemma-26B-A4B-it -o ../model/mlx-mxfp4
```

Rust `.dgq` for inference: `diffgemma quantize -o /tmp/quantized-weights` (different format).

### MLX denoise trace (low-RAM reference @ 30L)

After `quantize_mlx.py`, dump the same trace schema as Rust/HF:

```bash
uv run python scripts/dump_mlx_denoise_trace.py \\
  --model ../model/mlx-mxfp4 -p Hello --seed 42 --steps 8 --no-early-stop \\
  -o /tmp/mlx_30L_8.json
uv run python scripts/compare_denoise_trace.py /tmp/mlx_30L_8.json /tmp/mono_30L_8.json
```

## Rust cross-check (optional)

After building the Rust CLI:

```bash
# Raw BPE (legacy)
cargo run --release -- tokenize "Hello" --raw

# Chat template (default for generate/chat)
cargo run --release -- tokenize "Why is the sky blue?"

uv run pytest -q tests/test_rust_tokenizer.py tests/test_chat_template.py
```
