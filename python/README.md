# diffgemma-parity

Reproducible Python reference checks for **diffgemma-mps** (tokenizer parity today; model hooks later).

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

## Rust cross-check (optional)

After building the Rust CLI:

```bash
# Raw BPE (legacy)
cargo run --release -- tokenize "Hello" --raw

# Chat template (default for generate/chat)
cargo run --release -- tokenize "Why is the sky blue?"

uv run pytest -q tests/test_rust_tokenizer.py tests/test_chat_template.py
```
