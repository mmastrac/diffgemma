# diffgemma

A low-dependency Rust + Metal inference engine for
[DiffusionGemma 26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
(Gemma-4 26B-A4B MoE, discrete block diffusion) on Apple Silicon.

Unlike a traditional LLM that generates one token at a time. DiffusionGemma
denoises a 256-token canvas in parallel, trading memory bandwidth for compute.
This engine ports that loop straight to Metal (ie: no Python, no PyTorch, no
MLX at run time)

---

## Requirements

- **macOS on Apple Silicon** (M-series). Metal is the only backend — there is
  no CPU or CUDA fallback, and the binary refuses to build on non-macOS
  targets. Developed and measured on M3 Pro; other M-series configs should
  work but are not independently validated yet.
- **36 GB unified memory** minimum. The `q4` model is ~19 GiB of weights; the
  rest covers the KV cache and denoise working set. A single 36 GB machine
  reaches ~105k-token context without swapping. Below 36 GB is out of scope
  for v1.
- **Text-only in v1.** The ~550M vision tower (SigLIP encoder + image
  splicing) is not ported; only the text decoder runs. Image input is a v2
  item.
- **Rust** (stable), via [rustup](https://rustup.rs).
- A **quantized `.dgq` pack** (~19 GiB for `q4`). Download a ready-made one with
  the built-in `download` command, or fetch the ~50 GB bf16 checkpoint from
  Hugging Face and quantize it locally. Both are below.

## Quickstart

The recommended path is using `cargo install` and `diffgemma download` to get
the binary and model. If you are familiar with the huggingface tools, `hf download`
may be a faster way to download it.

### 1. Install

`cargo install` builds the release binary and drops it in `~/.cargo/bin` (on
your `PATH`):

```bash
cargo install --git https://github.com/mmastrac/diffgemma
```

Now run `diffgemma` from anywhere.

### 2. Get the model

The preferred model is `mmastrac/diffgemma-26b-a4b-it-q4` on huggingface. The `-q4`
variant is designed to balance memory and generation performance on consumer hardware.

**Option A: download a ready-to-run pack (recommended).** The `download` command
fetches a pre-quantized `.dgq` pack from huggingface and verifies it:

```bash
target/release/diffgemma download
# defaults to mmastrac/diffgemma-26b-a4b-it-q4 -> model/diffgemma-26b-a4b-it-q4
```

Pass `--repo ORG/NAME --revision REV` for a specific quantization pack.

Skip to [step 3](#3-run).

**Option B: quantize it yourself.** Fetch the bf16 weights into your Hugging
Face cache. You need the `hf` CLI (`pip install -U huggingface_hub` or `uvx hf`):

```bash
hf download google/diffusiongemma-26B-A4B-it
```

That is ~50 GB and lands in `~/.cache/huggingface/hub/`. The quantizer reads the checkpoint:

```bash
target/release/diffgemma quantize \
  -m google/diffusiongemma-26B-A4B-it \
  -o model/diffgemma-26b-a4b-it-q4 \
  --profile q4
```

`-m` takes a local directory or an `org/name` huggingface repo id.

This writes a self-contained `model/diffgemma-26b-a4b-it-q4/`
directory (`model.dgq.json` manifest, `model.dgq.bin` blob, plus the
copied tokenizer and config) in a few minutes.

Profiles (embeddings, router, and norms always stay bf16, since they are
precision-sensitive and small):

| Profile        | MoE experts       | Attention + dense FFN | ~weights |
|----------------|-------------------|-----------------------|----------|
| `q4` (default) | 4-bit affine (5.0b) | bf16                | ~19 GiB  |
| `nvfp4x`       | NVFP4 (~4.5b)     | bf16                  | ~18 GiB  |
| `nvfp4`        | NVFP4             | NVFP4                 | ~16 GiB  |
| `q6`           | 6-bit affine (7.0b) | bf16                | ~24 GiB  |

`--set class=format` overrides one tensor class for finer control (for example
`--set experts=nvfp4`, which is what `nvfp4x` expands to). See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full class/format matrix.

### 3. Run

`-m` is optional. The engine auto-discovers models (either a
`model/diffgemma-*` pack, or a `diffgemma-*` pack already in your huggingface
cache).

One-shot prompt:

```bash
diffgemma ask \
  -p "Explain block diffusion decoding in two sentences."
```

Interactive chat:

```bash
diffgemma chat -m model/diffgemma-26b-a4b-it-q4
```

OpenAI-compatible HTTP server (defaults to `127.0.0.1:8080`, 128k-token
context):

```bash
diffgemma serve
# then POST to http://127.0.0.1:8080/v1/chat/completions
```

The chat template is applied automatically. Pass `--raw` to send bare tokenizer
input, `--ctx N` to change the context budget, `--seed N` for a fixed seed
(default 42).

### Drive it from opencode

`serve` speaks the OpenAI API, so [opencode](https://opencode.ai) can use it as a
provider. `serve` exposes the model under the **basename of the pack directory**.
`-m model/diffgemma-26b-a4b-it-q4` serves the model id `diffgemma-26b-a4b-it-q4`
at `http://127.0.0.1:8080/v1` (check it with `curl 127.0.0.1:8080/v1/models`).
Start it:

```bash
target/release/diffgemma serve --ctx 100000
```

opencode has to know that endpoint as a provider, but you **don't need to edit a
config file**. opencode reads a full config inline from the
`OPENCODE_CONFIG_CONTENT` environment variable, so you can register the provider
and pick the model in one shot (`-m provider/model`):

```bash
OPENCODE_CONFIG_CONTENT='{
  "provider": {
    "diffgemma": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "diffgemma (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "unused" },
      "models": { "diffgemma-26b-a4b-it-q4": { "name": "DiffGemma 26B-A4B q4" } }
    }
  }
}' opencode -m diffgemma/diffgemma-26b-a4b-it-q4
```

The model id must match what `serve` reports. Append
`:think` or `:think=false` to force thinking on or off (for example
`diffgemma-26b-a4b-it-q4:think=false`).

For a permanent setup, drop that same `"provider"` block into `opencode.json`
(project) or `~/.config/opencode/opencode.json` (global) and run
`opencode -m diffgemma/diffgemma-26b-a4b-it-q4`.

## Local Development

To hack on the engine or build a specific revision, clone and build in place:

```bash
git clone https://github.com/mmastrac/diffgemma diffgemma
cd diffgemma
cargo build --release
```

The binary is generated in `target/release/diffgemma`.

## Performance

Head-to-head against **MLX-4bit** (`mlx-community/diffusiongemma-26B-A4B-it-4bit`,
mlx-vlm 0.6.3, Apple's fastest published config), on one M3 Pro / 36 GB.

**Prefill throughput** (tokens/sec processing the prompt), matched context
length, mean of two interleaved runs on an idle machine:

| Context | This engine | MLX-4bit | Ratio |
|--------:|------------:|---------:|:------|
| 8k   | ~370 tok/s | 402 tok/s | MLX 1.09× |
| 32k  | 313 tok/s  | 342 tok/s | MLX 1.09× |
| 100k | 233 tok/s  | 238 tok/s | **parity (1.02×)** |

MLX-4bit is a bit faster at short and medium context, and it is a dead heat at
100k. The gap does **not** widen as context grows (earlier, pre-optimization
versions collapsed to ~2.4× at 100k). Both engines drive chunked prefill, so
neither OOMs on the 36 GB machine.

**Recall** (needle-in-haystack, corpus-unique marker, retrieval verified): this
engine is **exact at 8k / 32k / 64k / 100k**, with the marker ~40k tokens deep at
100k. MLX matches at the 32k cross-check. Long-context KV uses a sliding-window
ring, so a single 36 GB machine reaches ~105k tokens without swapping (MLX's
full-precision KV would not fit that far).

Method notes: prefill throughput is content-independent only above ~32k. At 8k it
varies ~25% with prompt content (MoE routing locality), so the 8k row uses real
text. The benchmark harness lives in `python/scripts/` (`mlx_prefill_bench.py`,
`mlx_generate.py`). All numbers come from a single M3 Pro / 36 GB machine and one
MLX release. Treat them as one-configuration measurements until corroborated on
other hardware and refreshed against MLX's current build.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: the conceptual model, the implemented
  generation contract (every deliberate divergence from the MLX/HF reference,
  with evidence), the engineering design, and a "Negative Knowledge" ledger of
  approaches that were built, measured, and disproven on this hardware.
- **[AGENTS.md](AGENTS.md)**: how to work in this repo.
- **[PLAN.md](PLAN.md)**: open work, split into v1 and v2.
- Commit history is the changelog.
