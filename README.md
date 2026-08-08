# diffgemma

A low-dependency Rust + Metal inference engine for
[DiffusionGemma 26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
(Gemma-4 26B-A4B MoE, discrete block diffusion) on Apple Silicon.

<p align="center">
  <img src="docs/isprime-demo.gif" width="820"
       alt="diffgemma's Rust harness writing an is_prime function: denoising a 256-token canvas, calling tools, and running cargo test">
</p>

## Diffusion for Text?!?

A traditional LLM emits one token per forward pass and spends most of its time
waiting on memory bandwidth. DiffusionGemma does the opposite: it denoises a
256-token canvas in parallel, trading bandwidth for compute. This engine is a
port of the original paper's architecture to Apple Silicon.

## Requirements

This is still a work-in-progress. Expanded platform support (CUDA, optimizations
for smaller memory-class models, M4/M5-specific optimizations) are planned (PRs
welcome!).

- **macOS on Apple Silicon.** Metal is currently the only backend.
- **36 GB unified memory**, minimum. The `q4` pack is ~19 GiB of weights; the
  rest is KV cache and denoise working set. On a 36 GB machine you can reach
  ~105k tokens of context without swapping.
- **Text only** The ~550M vision tower (SigLIP + image splicing) is not ported.
  Image input is a v2 item.
- **Rust** (stable), via [rustup](https://rustup.rs).
- A **quantized `.dgq` pack** (~19 GiB for `q4`). Grab a ready-made one with
  `diffgemma download`, or fetch the ~50 GB bf16 checkpoint from huggingface and
  quantize it yourself. Both paths are below.

## Quickstart

The recommended path is using `cargo install` and `diffgemma download` to get
the binary and model. If you are familiar with the huggingface tools,
`hf download` may be a faster way to download it.

### 1. Install

`cargo install` builds the release binary and drops it on your `PATH` under
`~/.cargo/bin`:

```bash
cargo install --git https://github.com/mmastrac/diffgemma
```

After that, `diffgemma` should just work from anywhere.

### 2. Get the model

The preferred pack is
[`mmastrac/diffgemma-26b-a4b-it-q4`](https://huggingface.co/mmastrac/diffgemma-26b-a4b-it-q4)
on Hugging Face. The `-q4` variant is the one tuned for memory vs. generation
speed on consumer hardware.

The `download` command fetches a pre-quantized `.dgq` pack and verifies it. If
the connection is interrupted, it will resume where it left off:

```bash
diffgemma download
# defaults to mmastrac/diffgemma-26b-a4b-it-q4 -> model/diffgemma-26b-a4b-it-q4
```

Pass `--repo ORG/NAME --revision REV` for a specific quantization pack.

You can also [quantize the model yourself](#custom-quantization) for more
control.

### 3. Run

`-m` is optional. The engine will auto-discover a `model/diffgemma-*` pack, or a
`diffgemma-*` pack already sitting in your huggingface cache.

One-shot:

```bash
diffgemma ask \
  -p "Explain block diffusion decoding in two sentences."
```

Interactive chat:

```bash
# Start chat (first response takes a bit longer while the model is loading)
diffgemma chat
```

OpenAI-compatible HTTP server (defaults to `127.0.0.1:8080`, 128k context):

```bash
diffgemma serve
# then POST to http://127.0.0.1:8080/v1/chat/completions
```

The chat template is applied automatically. Pass `--raw` for bare tokenizer
input, `--ctx N` to change the context budget, `--seed N` for a fixed seed
(default 42).

### Drive it from opencode

`serve` speaks the OpenAI API, so [opencode](https://opencode.ai) can treat it
as a provider. The model id is the **basename of the pack directory** —
`-m model/diffgemma-26b-a4b-it-q4` shows up as `diffgemma-26b-a4b-it-q4` at
`http://127.0.0.1:8080/v1` (confirm with `curl 127.0.0.1:8080/v1/models`).

```bash
diffgemma serve --ctx 100000
```

You don't need to edit a config file. opencode will take a full config from
`OPENCODE_CONFIG_CONTENT`, so you can register the provider and pick the model
in one shot:

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

The model id has to match what `serve` reports. Append `:think` or
`:think=false` to force thinking on or off (e.g.
`diffgemma-26b-a4b-it-q4:think=false`).

For something permanent, drop that same `"provider"` block into `opencode.json`
(project) or `~/.config/opencode/opencode.json` (global) and run
`opencode -m diffgemma/diffgemma-26b-a4b-it-q4`.

## Custom Quantization

Pull the bf16 weights into your huggingface cache
(`pip install -U huggingface_hub`, or `uvx hf`):

```bash
hf download google/diffusiongemma-26B-A4B-it
```

That will download ~50 GB into `~/.cache/huggingface/hub/`. Then point the
quantizer at it:

```bash
diffgemma quantize \
  -m google/diffusiongemma-26B-A4B-it \
  -o model/diffgemma-26b-a4b-it-q4 \
  --profile q4
```

`-m` accepts a local directory or an `org/name` repo id. You get a
self-contained `model/diffgemma-26b-a4b-it-q4/` directory — manifest, blob,
tokenizer, config — in a few minutes.

Profiles (embeddings, router, and norms always stay bf16; they're
precision-sensitive and small):

| Profile        | MoE experts         | Attention + dense FFN | ~weights |
| -------------- | ------------------- | --------------------- | -------- |
| `q4` (default) | 4-bit affine (5.0b) | bf16                  | ~19 GiB  |
| `nvfp4x`       | NVFP4 (~4.5b)       | bf16                  | ~18 GiB  |
| `nvfp4`        | NVFP4               | NVFP4                 | ~16 GiB  |
| `q6`           | 6-bit affine (7.0b) | bf16                  | ~24 GiB  |

`--set class=format` overrides one tensor class (e.g. `--set experts=nvfp4`,
which is what `nvfp4x` expands to). The full class/format matrix is available in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Local Development

To hack on the engine or pin a specific revision, clone and build in place:

```bash
git clone https://github.com/mmastrac/diffgemma diffgemma
cd diffgemma
cargo build --release
```

The binary will be written to `target/release/diffgemma`.

## Performance

Head-to-head against MLX-4bit (`mlx-community/diffusiongemma-26B-A4B-it-4bit`)
on one M3 Pro / 36 GB machine. Not quite the same quant: our default `q4` keeps
attention, dense FFN, and embed at bf16 and only packs the MoE experts as
group-32 affine (~5.0 bpw). MLX uses group-64 affine 4-bit across more of the
model (~4.5 bpw on those tensors, with a few left at 8-bit). `diffgemma`'s
default quantization keeps more precision outside of the MoE experts.

**Prefill throughput** (tokens/sec processing the prompt), matched context
length:

| Context | This engine |  MLX-4bit | Ratio              |
| ------: | ----------: | --------: | :----------------- |
|      8k |  ~370 tok/s | 402 tok/s | MLX 1.09×          |
|     32k |   313 tok/s | 342 tok/s | MLX 1.09×          |
|    100k |   233 tok/s | 238 tok/s | **parity (1.02×)** |

MLX is a bit faster at short and medium context; at 100k it's a dead heat. The
gap does **not** widen as context grows.

**Recall** (needle-in-haystack, corpus-unique marker, retrieval verified): this
engine is exact at 8k / 32k / 64k / 100k, with the marker ~40k tokens deep at
100k. Long-context KV uses a sliding-window ring, letting us reach ~105k tokens
without swapping.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the conceptual model, the implemented
  generation contract (every deliberate divergence from the MLX/HF reference,
  with evidence), the engineering design, and a Negative Knowledge ledger of
  approaches that were built, measured, and disproven on this hardware.
- **[AGENTS.md](AGENTS.md)** — how to work in this repo.
- **[PLAN.md](PLAN.md)** — open work, split into v1 and v2.
- Commit history is the changelog.
