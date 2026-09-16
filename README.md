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

This is still a work-in-progress. Expanded platform support, optimizations
for smaller memory-class models, and M4/M5-specific optimizations are planned
(PRs welcome!).

- **macOS on Apple Silicon** for the DiffusionGemma engine. Metal is currently
  its only backend. The portable kernel layer and the `nanogpt` example also
  run on CUDA (see [CUDA](#cuda)).
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

### Structured decisions

A system message that is a JSON question schema turns a chat request into
a few scored forwards. No text is generated: the answer template is seeded
into the canvas with each label slot as noise, and each question's
distribution is read from the logits at its slot. All questions in a
request share each forward. One read conditions on one noise draw and is
sharper than the model's marginal, so the reply averages 4 reads by
default and reports the standard error.

Request: exactly two messages. `system` is the schema, `user` is the state
as JSON.

```bash
curl -s 127.0.0.1:8080/v1/chat/completions -d '{
  "messages": [
    {"role": "system", "content": "{\"questions\": [
       {\"id\": \"urgent\", \"type\": \"noul\", \"instructions\": \"Does this ticket need a reply today?\"},
       {\"id\": \"bucket\", \"type\": \"choice\", \"instructions\": \"Which team owns it?\",
        \"options\": [{\"name\": \"billing\"}, {\"name\": \"support\"}, {\"name\": \"engineering\", \"description\": \"a defect or outage\"}]},
       {\"id\": \"tone\", \"type\": \"score\", \"instructions\": \"How angry is the customer?\",
        \"levels\": [\"calm\", \"annoyed\", \"furious\"]}]}"},
    {"role": "user", "content": "{\"ticket\": \"Since this morning the dashboard shows a blank page after login. Console says 500 from /api/session.\"}"}
  ]
}'
```

Question types: `noul` (yes/no, reported as the probability of yes),
`choice` (one of `options`), `score` (one of the ordered `levels`, plus
the expected level). Labels are single tokens: `yes`/`no`, `A`/`B`/…,
`1`/`2`/…. A schema whose labels tokenize to more than one token each is
refused.

Reply: `content` is JSON.

```json
{"answers": {
   "urgent": {"type": "noul", "noul": 0.60, "label": "yes", "confidence": 0.60,
              "stderr": 0.16, "agreement": 0.75,
              "probabilities": {"yes": 0.60, "no": 0.40}},
   "bucket": {"type": "choice", "choice": "engineering", "label": "C", "confidence": 1.0,
              "stderr": 0.0, "agreement": 1.0,
              "probabilities": {"billing": 0.0, "support": 0.0, "engineering": 1.0}},
   "tone":   {"type": "score", "score": 1.77, "level": "annoyed", "label": "2", "confidence": 0.77,
              "stderr": 0.06, "agreement": 1.0,
              "probabilities": {"calm": 0.23, "annoyed": 0.77, "furious": 0.0}}},
 "diagnostics": {"steps": 1, "hole": "noise", "samples": {"n": 4, "tops": [...]},
                 "timing": {"prefill_ms": 720, "denoise_ms": 5060, "reused_tokens": 140, ...},
                 "questions": {"tone": {"argmax_is_label": true, "entropy": 0.6, "label_mass": 1.0, ...}}}}
```

`probabilities` is the mean over the reads of a softmax over the label
logits at temperature 1. `confidence` is the top label's mean probability,
`stderr` its standard error over the reads, and `agreement` the share of
reads that picked it. Two reads at 0.9 for opposite labels average to 0.5,
the marginal over the noise. The values are the model's own marginals. No
calibration against labelled data has been applied.

Optional schema fields:

- `instructions`: context placed before the questions.
- `samples`: reads with different hole noise, averaged. Default 4. `1` is
  a single read. `"auto"` takes one read and the rest (up to `auto_max`,
  default 4) only when some slot's first-read entropy is above
  `auto_threshold` (default 0.1 nats). On a held-out set of 20 tickets and
  60 slots the rule caught all 5 slots that moved across noise draws,
  flagged none of the 55 stable ones, and stopped 15 of the 20 tickets at
  one read.
- `fix_definite`: under `"auto"`, pin the slots that settled on the first
  read to their label for the later reads, so the uncertain slots
  condition on the settled answers. Changes what is estimated (a
  conditional rather than the marginal over the noise). On the held-out
  moving slots it changed no label and moved agreement by at most 0.25.
- `steps`: forwards per read. Default 1.
- `active`: canvas rows, a multiple of 64. Default: the smallest that
  holds the template, 64 for up to about 12 questions. `256` is the full
  canvas.
- `hole`: what fills a label slot before the read. `noise` (default),
  `pad`, `label`.
- `climb`, `climb_mode`: a diagnostic hill climb. It ratifies the first
  read.

Performance on an M3 Pro with the q4 pack, a 3-question schema, a
~190-token prompt, one server process:

| Case | Wall |
| :-- | --: |
| First request on a schema (f32 engine prefill of the schema, once) | 8 to 15 s |
| Next state, default (4 reads, 64 rows) | 3.1 s (0.7 prefill + 4 × 0.6 forward) |
| Next state, `samples: "auto"`, no slot flagged (15 of 20 held-out tickets) | 1.3 s |
| Next state, 4 reads, `active: 256` | 5.8 s |
| Next state, `samples: 1` | 1.3 s |
| Next state, `samples: 1`, `active: 256` | 2.0 s |
| 8 reads | 5.3 s |
| Generating the same three answers as JSON (19 tokens, 3 to 5 steps) | 9.6 to 13.1 s |

The schema prefix stays in the KV. Each later request prefills only its
state. `DGQ_FAST_PREFILL=1` cuts the first request to 3 s but changed a
borderline answer in our runs, so it is off. Width: 32 reads per setting
gave identical 1.00 answers on the unambiguous tickets at 64 and 256 rows
and a borderline answer within 1.3 standard errors (yes 0.56 ± 0.07
against 0.68 ± 0.06), so the narrow default reads the same as the full
canvas within noise. A borderline question still moves with the hole
noise and the prefill precision (PLAN.md has the measurements). Use
`samples` and read `stderr`.

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

## CUDA

The portable kernel layer is backend-agnostic. `crates/gpukit` owns GPU
mechanism and resolves the CUDA driver at runtime with `dlopen` (nothing is
linked against `libcuda`, so the crate still builds and type-checks on a host
with no CUDA). `crates/dgops` holds one CPU reference per op with a Metal and a
CUDA body side by side and a tier-1 test pinning both to that reference.
`crates/dgemm` is the GEMM family: one tiled body per backend behind a
problem/stride API, with weight format, structure and epilogue fusion as
compile-time axis values on that body.
`crates/nanogpt` is a tiny character-level GPT (forward, backward, AdamW)
composed entirely from those ops, checked against an independent CPU forward.

The 26B DiffusionGemma engine is still Metal-only; porting its own kernels to
the CUDA backend is open work (PLAN.md).

Requirements: Linux, an NVIDIA GPU, and the CUDA toolkit (`nvcc`, found on
`PATH` or at `/usr/local/cuda/bin/nvcc`). Build with `--features cuda`.

```bash
cargo run --release -p nanogpt --features cuda -- --check       # GPU forward vs CPU
cargo run --release -p nanogpt --features cuda -- --gradcheck   # d(loss)/d(param)
cargo run --release -p nanogpt --features cuda -- --train --steps 2000
cargo run --release -p nanogpt --features cuda -- --sample --tokens 400
cargo test --release -p dgemm --features cuda                   # GEMM parity
cargo test --release -p dgops --features cuda                   # per-op parity
```

`DGQ_CUDA_ARCH` (default: detected with `nvidia-smi`, else `native`) picks
the `-arch` nvcc compiles for; `DGQ_NVCC` overrides the compiler path and
`DGQ_CUDA_DRIVER` the driver library name.

Verified end to end on an NVIDIA GB10 (sm_121, CUDA 13.0). From a machine with
SSH access to a CUDA box, `scripts/verify-cuda.sh <user@@host>` mirrors the tree
and runs the driver smoke test, per-op parity, forward parity, the gradient
check, a short training run, and a sample.

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
