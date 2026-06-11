# diffgemma-mps — implementation plan

Roadmap for a low-dependency Rust inference engine for [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it), inspired by [Iris](../flux2.c) (mmap weights, CPU reference path first, then Metal/MPS acceleration).

See `ARCHITECTURE.md` for model semantics (block diffusion, entropy sampler, causal vs bidirectional attention).

---

## Goals

| Goal | Notes |
|------|--------|
| **Single binary** | No Python runtime at inference time |
| **Low deps** | `memmap2`, `serde`; add deps only when clearly worth it (e.g. `objc2-metal` for GPU) |
| **Correctness first** | CPU/BLAS reference implementation; Metal must match byte-for-byte or within tight fp tolerance |
| **mmap weights** | ~48 GiB BF16 shards; page in on demand, no full-RAM copy at load |
| **Apple Silicon Metal** | Full accelerated path: bf16 GEMM, attention, MoE hot paths on GPU |
| **Text-first** | Decoder + entropy sampler before vision/multimodal |

**Non-goals (initially):** training, fine-tuning, LoRA, multi-user serving, CUDA/Linux GPU.

**Reality check (Apple Silicon):** DiffusionGemma wins on compute-bound datacenter GPUs. Unified memory Macs may remain bandwidth-bound; Metal still matters for usability and parity with Iris-style local inference, but speedups may be modest vs H100 numbers in the model card.

---

## Iris patterns to mirror

| Iris | diffgemma-mps |
|------|----------------|
| `iris_safetensors.c` | `src/safetensors.rs` + `src/weights.rs` ✅ |
| `parse_transformer_config` | `src/config.rs` (from `config.json`) |
| `iris_kernels.c` | `src/kernels/` (CPU + later Metal dispatch) |
| `iris_transformer_*.c` | `src/model/decoder.rs`, `src/model/encoder.rs` |
| `iris_sample.c` | `src/sample.rs` (entropy-bound block loop) |
| `iris_qwen3_tokenizer.c` | `src/tokenizer.rs` (Gemma/SentencePiece or BPE) |
| `iris_metal.m` + `iris_shaders.metal` | `src/metal/` (Rust `objc2-metal` + `.metal` shaders) |
| `make generic` / `make blas` / `make mps` | Cargo features: `default`, `blas`, `metal` |
| Python/diffusers parity scripts | `debug/` one-off compare scripts (not committed deps) |

**Rule:** read dimensions from `config.json`, never hardcode when config provides them.

---

## Model surface (from weights + config)

```
model/
  transformer/
    config.json
    model.safetensors.index.json
    model-00001-of-00011.safetensors …
    tokenizer.json (or spiece.model)   # TBD download
```

**Weight groups (1047 tensors, all BF16):**

| Prefix | Count | Role |
|--------|------:|------|
| `model.decoder.layers.*` | 655 | Block-diffusion decoder (MoE Gemma 4) |
| `model.encoder.vision_tower.*` | 355 | Vision encoder (multimodal) |
| `model.encoder.language_model.*` | 30 | Causal prefill / KV builder |
| `model.decoder.self_conditioning.*` | 4 | Self-conditioning projections |
| `model.decoder.embed_tokens` | 1 | Shared embedding (tied to LM head) |
| `model.encoder.embed_vision` | 1 | Vision token embed |

**Text config (reference):** 30 layers, hidden 2816, 16 Q heads / 8 KV heads, head_dim 256, MoE 128 experts / top-8 active, sliding window 1024 + periodic full attention, vocab 262144, canvas 256.

---

## Build targets (Cargo features)

```
cargo build                    # CPU generic (naive matmul)
cargo build --features blas    # Accelerate/OpenBLAS SGEMM on CPU
cargo build --features metal   # Apple Silicon GPU (macOS arm64 only)
```

`metal` implies `blas` for CPU fallbacks and pre-GPU validation. CI can run `blas` on any host; `metal` only on macOS.

---

## Phase 0 — Weight loading ✅

**Status:** done (`55ec41a`).

- [x] mmap single safetensors shard
- [x] Parse HF `model.safetensors.index.json`
- [x] Open all 11 shards, global tensor index
- [x] `diffgemma-mps` binary prints summary stats

**Verify:** `cargo run` → 11 shards, 1047 tensors, ~48 GiB, all BF16.

---

## Phase 1 — Config + tensor views ✅

**Deliverable:** `cargo run -- config` prints parsed architecture; typed weight accessors.

| Task | Files |
|------|-------|
| Deserialize `config.json` into `ModelConfig` | `src/config.rs` |
| BF16 slice views over mmap (`&[u16]` + shape) | `src/tensor.rs` |
| Named weight resolver (`WeightStore::tensor("…")`) | extend `src/weights.rs` |
| Per-layer weight struct stubs (layer 0) | `src/model/layer_weights.rs` |

**Exit criteria:**
- Print hidden_size, num_layers, num_experts, canvas_length, layer_types.
- Load `model.decoder.layers.0.self_attn.q_proj.weight` as bf16 `[4096, 2816]` without copy.

**Commands:**
```bash
cargo run -- config
cargo run -- weights model.decoder.layers.0.self_attn.q_proj.weight
cargo run -- layer0    # validate all 22 layer-0 tensors vs config shapes
```

---

## Phase 2 — CPU kernels (reference math) ✅

**Deliverable:** unit-tested primitives used by every layer.

| Kernel | Notes |
|--------|-------|
| `rms_norm` | Gemma-style, `eps=1e-6` |
| `linear` | `y = x @ W^T`, optional bias |
| `silu`, `gelu_pytorch_tanh` | activation from config |
| `softmax` | stable, per-row |
| `rope` | sliding + full attention variants, proportional rope for full layers |
| `matmul` | naive loop + `blas` feature via Accelerate |

**Files:** `src/kernels/cpu.rs`, `src/kernels/matmul.rs`, `src/kernels/mod.rs`, `build.rs`

**Build:** `cargo test` (default `blas` → Accelerate on macOS); `cargo test --no-default-features` for generic matmul.

**Exit criteria:** `cargo test` — RMSNorm and RoPE vs small golden vectors (hand-computed or Python reference in `/tmp`).

---

## Phase 3 — Single decoder layer (CPU) ✅

**Deliverable:** one forward pass of `model.decoder.layers.0` on random data.

| Component | Weight keys |
|-----------|-------------|
| Input RMSNorm | `input_layernorm.weight` |
| Q/K/V/O proj | `self_attn.{q,k,v,o}_proj.weight` |
| Q/K RMSNorm | `self_attn.{q,k}_norm.weight` |
| Attention | sliding window or full (per `layer_types[0]`) |
| Post-attn norm | `post_attention_layernorm.weight` |
| MoE FFN | `router.*`, `experts.gate_up_proj`, `experts.down_proj` |
| Shared MLP | `mlp.{gate,up,down}_proj.weight` (plus extra norms) |

**Hard parts:**
- MoE tensor layout: `[128, …]` expert stacks — route top-8, weighted sum.
- Sliding-window mask vs full attention (layers 5, 11, 17, 23, 29 are `full_attention`).
- `layer_scalar` — per-layer scaling (read from weights).

**Files:** `src/model/attention.rs`, `src/model/moe.rs`, `src/model/decoder_layer.rs`

**Exit criteria:** `cargo run -- layer0` completes without panic; output shape `[seq, 2816]`.

**Run:** `cargo run --release -- layer0` (default seq=16).

---

## Phase 4 — Full decoder stack (CPU) ✅

**Deliverable:** 30-layer decoder forward, bidirectional mask over canvas region.

| Task | Notes |
|------|-------|
| Stack 30 layers | Reuse layer weights per index |
| Attention mask builder | Canvas tokens: bidirectional within block; causal to KV prefix |
| Final norm | `model.decoder.norm` |
| LM head | tied embeddings → logits `[seq, 262144]` |
| Self-conditioning | `model.decoder.self_conditioning.*` wired per paper/model |

**Exit criteria:** Forward on `seq=256` random token ids + dummy KV cache; logits shape correct.

**Files:** `src/model/decoder.rs`, `src/model/mask.rs`, `src/model/kv_cache.rs`, `src/model/embed.rs`, `src/model/self_conditioning.rs`

**Run:** `cargo run --release -- decoder` (canvas=256, dummy kv=128, ~7 min on Apple Silicon).

---

## Phase 5 — Causal encoder + KV cache (CPU) ✅

**Deliverable:** prefill path that fills KV cache for prompt tokens.

| Task | Notes |
|------|-------|
| `model.encoder.language_model` layers | 30 weights — may share architecture with decoder or subset |
| Causal attention only | Standard Gemma 4 sliding/full schedule |
| KV cache layout | Per-layer `K`, `V` buffers append-only |
| Cache position offsets | RoPE with absolute positions |

**Exit criteria:** Prefill 128 prompt tokens → KV cache size matches expected `(layers, seq, kv_heads, head_dim)`.

**Files:** `src/model/encoder.rs`, `src/model/kv_cache.rs` (extend), encoder path in `attention.rs` / `decoder_layer.rs`

**Note:** Encoder layers use tied `model.decoder.layers.*` weights; only `model.encoder.language_model.layers.*.layer_scalar` buffers are separate in the checkpoint.

**Run:** `cargo run --release -- prefill`

---

## Phase 6 — Block diffusion sampler (CPU)

**Deliverable:** end-to-end text generation on CPU (slow but correct).

| Step | Implementation |
|------|----------------|
| 1. Prefill prompt | Phase 5 |
| 2. Initialize canvas | 256 random token IDs |
| 3. Denoise loop (≤48 steps) | Decoder bidirectional forward |
| 4. Entropy selection | Keep low-entropy positions; re-randomize rest |
| 5. Temperature schedule | Linear 0.8 → 0.4 |
| 6. Early stop | avg entropy < 0.005 AND argmax stable 2 steps |
| 7. Block commit | Append committed tokens; extend KV; new canvas |

**Files:** `src/sample.rs`, `src/generate.rs`

**Config defaults (model card):** `entropy_bound=0.1`, `max_steps=48`, temperature 0.8→0.4.

**Exit criteria:** Greedy-ish run on tiny prompt produces deterministic output (fixed seed); compare token ids to Python reference on 1–2 steps.

**Status:** ✅ Done — `src/sample.rs`, `src/generate.rs`, `encoder::extend_prefill`, `generate` CLI.

**Run:**
```bash
cargo run --release -- generate --seed 42 --steps 2 --prompt-len 8 --max-new-tokens 256
```

---

## Phase 7 — Tokenizer

**Deliverable:** prompt string → token ids without Python.

| Option | Tradeoff |
|--------|----------|
| Port Gemma tokenizer from HF `tokenizer.json` | Larger parser, zero runtime Python |
| Minimal BPE from `tokenizer.json` | serde parse + merge table |

**Files:** `src/tokenizer.rs`

**Exit criteria:** Encode `"Hello"` matches Python `AutoTokenizer` ids.

**Status:** ✅ Done — `src/tokenizer.rs`, `tokenize` CLI, `python/` uv parity tests.

**Run:**
```bash
cargo run --release -- tokenize "Hello"
cd python && uv sync && uv run pytest -q
```

---

## Phase 8 — Metal bootstrap

**Deliverable:** GPU device init + bf16 GEMM + buffer pool.

| Task | Iris analogue |
|------|----------------|
| `objc2-metal` device + command queue | `iris_metal_init` |
| `MTLBuffer` pool for activations | `iris_gpu_tensor_*` |
| Embed `shaders.metal` at compile time (`include_str!` + runtime compile) | `iris_shaders_source.h` |
| bf16 GEMM (MPSGraph or custom kernel) | `iris_metal_sgemm_bf16` |
| Weight upload / use mmap + `MTLResourceStorageModeShared` | Apple Silicon unified memory |

**Files:**
```
src/metal/mod.rs
src/metal/device.rs
src/metal/buffer.rs
src/metal/gemm.rs
shaders/gemm.metal
```

**Deps:** `objc2`, `objc2-metal`, `objc2-foundation` (macOS only, behind `metal` feature).

**Exit criteria:** `cargo run --features metal -- gemm` — bf16 matmul matches CPU within tolerance on 512×512.

**Status:** ✅ Done — `src/metal/*`, `shaders/gemm.metal`, `gemm` CLI.

**Run:**
```bash
cargo run --release --features metal -- gemm --size 512
```

---

## Phase 9 — Metal attention

**Deliverable:** GPU attention for sliding, full, and bidirectional masks.

| Task | Notes |
|------|-------|
| RoPE on GPU | fuse with Q/K projection where possible |
| Sliding-window attention | bounded KV, window 1024 |
| Full attention | global KV heads (2 global + local) |
| Bidirectional canvas mask | block-causal to prefix, full within canvas |
| GQA | 16 Q heads, 8 KV heads |

**Files:** `shaders/attention.metal`, `src/metal/attention.rs`

**Exit criteria:** Layer 0 attention output matches CPU forward on same inputs.

**Status:** Done.

| Added | Notes |
|-------|-------|
| `shaders/attention.metal` | `apply_rope_heads` + `gqa_attention` (causal/sliding, encoder extend, decoder bitmap) |
| `src/metal/attention.rs` | `GpuAttention`: RoPE, GQA softmax, `rope_and_gqa` |
| `src/model/attention.rs` | `GqaMask`, `gqa_attention`, `prepare_qkv_pre_rope`, `forward_to_attn_out` |
| CLI `attention` | Layer 0 RoPE+GQA parity vs CPU at seq=16 |

```bash
cargo test --features metal
cargo run --release --features metal -- attention
```

Verified: layer 0, seq=16, `max_abs_diff: 0.000001`.

---

## Phase 10 — Metal MoE + full decoder ✅

**Deliverable:** 30-layer decoder on GPU.

| Task | Notes |
|------|-------|
| Router top-k (8 of 128) | CPU router for top-k parity; GPU router path in `route_gpu` (needs top-k tie-break parity) |
| Expert GEMM batching | MoE arena: all active experts in 2 GPU syncs/layer (`gate_up` batch + `down` batch) |
| Shared MLP path | GPU RMSNorm + `f32_bf16_gemm` dense MLP; CPU GELU/SwiGLU |
| Layer norm + activations on GPU | `shaders/decoder.metal` elementwise kernels; 256-wide threadgroups (Metal 65535 TG limit) |
| Batched dispatches | `GpuBatch`: one `commit`+`wait` per independent section; **no chained readback on same buffer** |
| Weight cache | `GpuDecoderWeightCache`: pre-transposed dense bf16, cached router f32, lazy expert transpose |

**Exit criteria:** Full decoder forward matches CPU on `seq=256` (bf16 tolerance ~1e-2). `decoder-gpu --seq 16 --kv 8 --layers 3` passes with `max_abs_diff ≤ 1e-2`; ~2× faster than CPU at small scale.

**Run:**
```bash
cargo run --release --features metal -- decoder-gpu --seq 16 --kv 8 --layers 3
cargo run --release --features metal -- bench-decoder --seq 16 --kv 8 --layers 3
```

---

## Phase 11 — Metal encoder + sampler integration (in progress)

**Deliverable:** full accelerated generation.

| Task | Status | Notes |
|------|--------|-------|
| `generate-gpu` / `generate-parity` CLI | ✅ | Shared loop via `DecoderBackend`; `-p` prompt |
| `bench-decoder` | ✅ | `--seq`, `--kv`, `--layers`, `--iters` |
| `decoder-gpu` scalable parity | ✅ | `--seq`, `--kv`, `--layers`; memory estimate printed |
| GPU decoder attention | ✅ | GPU Q/K/V/o_proj + batched RoPE/GQA via shared engine pool |
| `--layers` on generate commands | ✅ | smoke-test subset of decoder stack |
| KV cache on GPU | — | persist across prefill + denoise |
| CPU↔GPU only at boundaries | partial | GPU norms/GEMM/MoE/attention; CPU QKV prep, router, sampler |
| Entropy + sampling on CPU | ✅ | same as Phase 6 |
| Block loop on GPU | ✅ | `generate-parity` token match (full 30 layers, steps=1) |

**Exit criteria:** `cargo run --features metal -- generate-gpu -p "Hello" --seed 42` prints coherent text; matches CPU path tokens for same seed.

**Verified (2025-06):**
```bash
cargo run --release --features metal -- generate-gpu -p "Hello" --seed 42 --steps 1
# prefill ~3s, denoise ~118s → 2.17 tok/s (30 layers, canvas=256)
cargo run --release --features metal -- generate-parity -p "Hello" --seed 42 --steps 1 --layers 3
# GPU-only vs golden ~15s (3 fixtures in fixtures/generate/)
```

---

## Phase 12 — Performance, memory & FastSlice

**Deliverable:** lower peak RAM, faster hot paths, foundation for unchecked inner loops.

### Memory (partial ✅)

| Task | Status | Notes |
|------|--------|-------|
| Two reusable layer scratches | ✅ | sliding + full attention shapes (~400 MiB vs 30×) |
| Per-layer MoE expert cache eviction | ✅ | `clear_expert_cache()` after each layer |
| GPU-first parity + drop GPU state | ✅ | avoids CPU+GPU peak in `decoder-gpu` |
| Buffer pool trim | ✅ | `BufferPool::trim()` at forward end |
| Memory estimates | ✅ | `src/metal/memory.rs`; printed by `decoder-gpu` |
| Per-layer weight cache paging | ✅ | one `GpuLayerWeightCache` resident; load/release per layer |
| Skip logits clone in generate loop | ✅ | reuse `sample_logits` / `sc_logits` buffers |
| Zero-copy logits in generate forward | ✅ | `logits_out` + reusable `logits_buf`; skip lm_head in bench |
| Generate golden fixtures | ✅ | `fixtures/generate/*.json`; GPU-only `generate-parity` |
| GPU KV without CPU mirrors | partial ✅ | persistent prefix buffers; canvas suffix patched per forward |

**OOM note:** full `decoder-gpu` (30 layers, seq=256) needs ~2.3 GiB single-path (estimate). Prior OOM was caused by 30 persistent attention scratches + unbounded expert transpose cache + CPU∥GPU peak.

### Performance ideas

| Idea | Impact | Notes |
|------|--------|-------|
| Wire `GpuAttention` into decoder layer | high | ✅ GPU Q/K/V/o_proj + batched RoPE/GQA |
| Share Metal pool for attention | medium | ✅ `attention_batch.rs`; no second `MetalContext` |
| Generate tok/s reporting | process | ✅ denoise throughput in `generate-gpu` output |
| Per-layer weight paging | high RAM / medium speed | transpose once per layer per session |
| Keep MoE arena batching | medium | already 2 syncs/layer; don't re-sync per expert |
| Fuse independent GPU norms/GEMMs | medium | only when buffer dependencies allow readback |
| Avoid per-layer `pool.trim` | medium | caused 3× slowdown; trim at section boundaries only |
| `bench-decoder` regression table | process | always record before/after per change |
| Optional expert transpose disk cache | low | only if re-transpose dominates after eviction |

### FastSlice — remove bounds checks in hot loops

Today `Bf16Slice::get()` bounds-checks every element; `to_f32_vec()` and MoE transpose loops call it millions of times per forward.

**Goal:** introduce `FastSlice` / `FastBf16Slice` in `src/tensor.rs` — pointer+length views with unchecked access behind a validated construction boundary.

| Task | Files | Notes |
|------|-------|-------|
| `FastSlice<'a, T>` | `src/fast_slice.rs` | raw `ptr + len` (not `&[T]`); construct after `TensorView` shape check |
| `FastBf16Slice<'a>` | `src/fast_slice.rs` | `get_u16_unchecked` / `to_f32_unchecked` over mmap bytes |
| `FastSliceMut<'a, T>` | `src/fast_slice.rs` | mutable ptr+len for transpose/dequant writes |
| `Buffer<T>` | `src/buffer.rs` | owned allocation; hot loops use `as_fast_slice()` only |
| Safety contract | docs + `debug_assert!` | only construct from validated `TensorView`, `Vec`, or pool buffer |
| Migrate matmul/GEMM inputs | `kernels/cpu.rs`, `metal/linear.rs` | ✅ inner loops |
| Migrate MoE transpose | `metal/weights.rs` | ✅ expert weight gather |
| Migrate embed gather | `model/embed.rs` | ✅ `embed_tokens`, `lm_head_tied_bf16`; `sc_probs` buffer |
| Migrate embed gather | `model/moe.rs` | CPU `linear_bf16_slice` via FastBf16Slice (partial) |
| Benchmark gate | `bench-decoder` | document ns/op or ms/fwd before and after each migration |

**Exit criteria:**
- `decoder-gpu --seq 16 --kv 8 --layers 3` still within 1e-2 tolerance after each migration step. ✅
- Measurable `bench-decoder` improvement on at least one config (record numbers in commit).
- Full `decoder-gpu` (30 layers) completes without OOM on 16 GiB unified memory. ✅ (`max_abs_diff ≈ 0.008`, GPU ~169s)

**Bench reference (seq=16, kv=8, layers=3, iters=3):**
| Milestone | per_fwd |
|-----------|---------|
| Post FastSlice commit | ~2.26s |
| Post embed migration | ~2.13s |
| Post weight paging (layers=3) | ~2.16s (RAM ~1 GiB → ~35 MiB weights) |
| Post logits reuse + skip lm_head in bench | ~1.89s |

---

## Phase 13 — CLI + ergonomics

**Deliverable:** usable local tool.

```
diffgemma-mps -p "Write a haiku" --seed 42 --max-steps 48
diffgemma-mps --inspect-weights model/transformer
diffgemma-mps --benchmark --features metal
```

| Task | Notes |
|------|-------|
| `clap` for args | only CLI dep |
| Model path `-m` | default `model/transformer` |
| Timing breakdown | prefill / denoise step / sampler |
| Optional REPL | later, Iris-style |

---

## Phase 14 — Vision / multimodal (deferred)

355 vision tensors; image → soft tokens → encoder.

| Task | Notes |
|------|-------|
| Image load (PNG) | minimal decoder or `image` crate if worth it |
| `model.encoder.vision_tower` | 27 layers, patch embed |
| `embed_vision` + soft tokens | 280 tokens per image |
| Bidirectional vision attention | per `use_bidirectional_attention: vision` |

**Exit criteria:** Image + prompt → generation matches Python on one fixture.

---

## Parity & testing strategy

| Level | What |
|-------|------|
| Unit tests | kernels, RoPE, RMSNorm, router math |
| Layer tests | layer 0 vs Python hook (`debug/compare_layer0.py` in /tmp) |
| Forward tests | full decoder logits slice vs reference |
| Generation tests | 2-step denoise, fixed seed, compare token ids |
| Metal regression | CPU vs GPU max abs diff per tensor type |

**Reference workflow:** Locked Python env in `python/` via `uv` (`uv.lock` committed). Parity tests in `python/tests/`; one-off scripts in `debug/` if needed.

**Do not commit:** `model/`, `venv/`, downloaded weights.

---

## Proposed file layout (steady state)

```
diffgemma-mps/
  Cargo.toml
  PLAN.md
  ARCHITECTURE.md
  shaders/
    gemm.metal
    attention.metal
    norm.metal
  src/
    main.rs
    config.rs
    tensor.rs
    fast_slice.rs
    buffer.rs
    safetensors.rs
    weights.rs
    tokenizer.rs
    sample.rs
    generate.rs
    kernels/
      mod.rs
      cpu.rs
    model/
      mod.rs
      attention.rs
      moe.rs
      decoder_layer.rs
      decoder.rs
      encoder.rs
    metal/          # #[cfg(feature = "metal")]
      mod.rs
      device.rs
      buffer.rs
      memory.rs
      gemm.rs
      attention.rs
      decoder.rs
  debug/            # parity scripts (optional, gitignored or checked in)
```

---

## Risk register

| Risk | Mitigation |
|------|------------|
| MoE expert layout mismatch | Inspect shapes early (Phase 1); unit test router on layer 0 |
| Sliding vs full RoPE differs | Separate code paths per `layer_types[i]`; test layers 0 and 5 |
| Bidirectional mask bugs | Small 16-token hand-crafted mask test |
| 48 GiB mmap + GPU residency | Unified memory: shared buffers; don’t duplicate weights; page layer caches |
| Hot-loop bounds checks | `FastSlice` with validated construction; migrate incrementally |
| Metal graph compile latency | Cache MPSGraph executables per shape (Iris lesson) |
| Tokenizer complexity | Defer to Phase 7; use fixed token ids until then |
| Apple Silicon bandwidth ceiling | Set expectations; profile before over-optimizing |

---

## Milestone summary

| # | Milestone | Backend |
|---|-----------|---------|
| 0 | Weight summary binary | — ✅ |
| 1 | Config + tensor views | — ✅ |
| 2 | CPU kernels tested | generic/blas ✅ |
| 3 | Decoder layer 0 | generic/blas ✅ |
| 4 | Full decoder forward | blas ✅ |
| 5 | Encoder + KV cache | blas ✅ |
| 6 | Entropy sampler + blocks | blas |
| 7 | Tokenizer | blas |
| 8 | Metal GEMM + buffers | metal |
| 9 | Metal attention | metal |
| 10 | Metal full decoder | metal ✅ |
| 11 | End-to-end generation | metal (partial) |
| 12 | Performance + FastSlice | metal (in progress) |
| 13 | CLI | metal |
| 14 | Vision (optional) | metal |

---

## Immediate next step

**Phase 11:** finish GPU generation path — KV on GPU, optional `--steps 2+` parity gate.

1. **GPU KV cache** — avoid CPU mirrors across denoise steps (largest remaining Phase 11 item).
2. **GPU QKV + o_proj** — move attention bookends off CPU; share Metal context/pool with decoder engine.
3. **Fuse attention submits** — RoPE + GQA in one command buffer per layer.

**Phase 12 (ongoing):**
```bash
cargo run --release --features metal -- bench-decoder --seq 16 --kv 8 --layers 3 --iters 3
cargo run --release --features metal -- decoder-gpu --seq 16 --kv 8 --layers 3
cargo run --release --features metal -- generate-parity -p "Hello" --seed 42 --steps 1 --layers 3
```

**Phase 11 smoke test (full stack):** `generate-parity -p "Hello" --seed 42 --steps 1` (30 layers, ~10–15 min).
