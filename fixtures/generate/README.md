# Generate golden fixtures

Regression token streams for `generate-parity` (GPU vs checked-in golden, no CPU run).

Fixtures are tagged by `weights_profile` in the JSON (`safetensors` for bf16, `dgq_q4` for quantized). Legacy bf16 files omit the field and default to `safetensors`.

## bf16 (`model/transformer`)

| File | Config | ~GPU denoise |
|------|--------|--------------|
| `hello_steps1_full.json` | `-p Hello --seed 42 --steps 1` (30 layers) | ~125s |
| `hello_steps1_layers3.json` | `--steps 1 --layers 3` | ~15s |
| `hello_steps2_full.json` | `-p Hello --seed 42 --steps 2` (30 layers) | ~290s |
| `hello_steps2_layers3.json` | `--steps 2 --layers 3` | ~78s |

## `.dgq` q4 (`-m /tmp/quantized-weights` or your quant dir)

| File | Config | ~GPU denoise |
|------|--------|--------------|
| `dgq_hello_steps1_full.json` | same as bf16 full | ~20s |
| `dgq_hello_steps1_layers3.json` | `--steps 1 --layers 3` | ~2s |
| `dgq_hello_steps2_full.json` | `--steps 2` full | ~40s |
| `dgq_hello_steps2_layers3.json` | `--steps 2 --layers 3` | ~5s |

## Monolithic step kernel (`profile: monolithic`)

Forward-only regression limits vs engine (`step-parity`). Use `DGQ_MPS_Q4=0` for deterministic Q4.

| File | Config |
|------|--------|
| `monolithic_parity_layers3_seed42.json` | `step-parity --layers 3 --seed 42` |
| `monolithic_parity_layers30_seed42.json` | `step-parity --layers 30 --seed 42` |
| `monolithic_sampler_layers3.json` | `step-verify` sampler goldens (3 seeds × 4 steps) |
| `monolithic_hello_steps4_layers3.json` | `generate-monolithic-parity -p hello --raw --layers 3 --steps 4 --seed 42 --no-early-stop` (`DGQ_MPS_Q4=0`) |
| `chat_quality_hello_layers3.json` | Templated `-p Hello`, 48 steps, early stop — quality gate (min real tokens, max pad ratio); run via `step-ci` |

**Prompts:** `-p` uses the Gemma 4 chat template by default (`<bos><|turn>user\n…<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`). Legacy goldens use bare BPE text — pass **`--raw`** for parity/regression commands that match `Hello` / `hello` fixtures.

Refresh after intentional monolithic/kernel changes:

```bash
DGQ_MPS_Q4=0 cargo run --release -- -m /tmp/quantized-weights step-parity --layers 3 --seed 42

# CI regression (config validate + step-verify + generate-monolithic-parity)
DGQ_MPS_Q4=0 cargo run --release -- -m /tmp/quantized-weights step-ci --layers 3

# Golden parity only
DGQ_MPS_Q4=0 cargo run --release -- -m /tmp/quantized-weights generate-monolithic-parity -p hello --raw --layers 3 --steps 4 --seed 42 --no-early-stop
```


```bash
# bf16
cargo run --release -- generate-gpu -p "Hello" --seed 42 --steps 1 --write-golden hello_steps1_full

# .dgq (quantize first if needed)
cargo run --release -- quantize -o /tmp/quantized-weights
cargo run --release -- -m /tmp/quantized-weights generate-gpu -p "Hello" --seed 42 --steps 1 --layers 3 --write-golden dgq_hello_steps1_layers3
cargo run --release -- -m /tmp/quantized-weights generate-parity -p "Hello" --raw --seed 42 --steps 1 --layers 3
```

Use `--compare-cpu` on `generate-parity` for full CPU vs GPU on the same weights (slow).

## Denoise traces (P1.6 canvas convergence)

Compact per-step telemetry for localizing sampler vs forward divergence. Schema: `src/denoise_trace.rs` (`schema_version: 1`).

| File | Config |
|------|--------|
| `denoise_trace_hello_layers3_steps4_seed42.json` | Templated `-p Hello`, `--layers 3 --steps 4 --seed 42 --no-early-stop`, monolithic |

**Rust dump:**

```bash
cargo run --release -- -m /tmp/quantized-weights generate-monolithic \
  -p Hello --seed 42 --layers 3 --steps 4 --no-early-stop \
  --write-trace fixtures/generate/denoise_trace_hello_layers3_steps4_seed42.json
```

**MLX reference** (local `model/mlx-mxfp4`; run **alone**, not parallel with Rust generate):

```bash
# 1) Rust trace first
cargo run --release -- -m /tmp/quantized-weights generate-monolithic \
  -p 'How can I get from Calgary to Namibia?' --seed 42 --layers 30 --steps 12 --no-early-stop \
  --write-trace /tmp/calgary_mono_trace.json

# 2) Extract canvas array for MLX (dump script expects a JSON list, not the full trace)
python3 -c "import json; json.dump(json.load(open('/tmp/calgary_mono_trace.json'))['initial_canvas_ids'], open('/tmp/calgary_canvas_ids.json','w'))"

# 3) MLX trace (sequential; do not run HF 26B on laptop — OOM)
cd python && uv run python scripts/dump_mlx_denoise_trace.py --model ../model/mlx-mxfp4 \
  -p 'How can I get from Calgary to Namibia?' --seed 42 --steps 12 --no-early-stop \
  --canvas-ids /tmp/calgary_canvas_ids.json -o /tmp/calgary_mlx_trace.json

uv run python scripts/compare_denoise_trace.py /tmp/calgary_mlx_trace.json /tmp/calgary_mono_trace.json
```

Calgary @ 12 steps (matched canvas, 22-tok prefill): argmax agrees at pos 0 from step 1; MLX `min_entropy` drops faster and accepts ramp to 110/step vs mono ~1/step — forward/quant gap, not sampler. HF `dump_denoise_trace.py` needs discrete GPU (26B); `device_map=auto` meta-tensor failure on MPS-only laptop.

**HuggingFace reference** (requires GPU + full weights; optional `uv sync --extra model`):

```bash
cd python && uv sync --extra model
uv run python/scripts/dump_denoise_trace.py -p Hello --seed 42 --steps 4 -o /tmp/hf_trace.json
uv run python/scripts/compare_denoise_trace.py /tmp/hf_trace.json /tmp/rust_trace.json
```

Traces store per-step `accept_count`, `min_entropy`, `low_entropy_positions`, and `argmax_prefix[0..16]` — not full 262K×256 logits (too large for git). For layer-level probes use `step-probe` on a single forward.
