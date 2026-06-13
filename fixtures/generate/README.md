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

Refresh after intentional monolithic/kernel changes:

```bash
DGQ_MPS_Q4=0 cargo run --release --features metal -- -m /tmp/quantized-weights step-parity --layers 3 --seed 42
```


```bash
# bf16
cargo run --release --features metal -- generate-gpu -p "Hello" --seed 42 --steps 1 --write-golden hello_steps1_full

# .dgq (quantize first if needed)
cargo run --release --features metal -- quantize -o /tmp/quantized-weights
cargo run --release --features metal -- -m /tmp/quantized-weights generate-gpu -p "Hello" --seed 42 --steps 1 --layers 3 --write-golden dgq_hello_steps1_layers3
cargo run --release --features metal -- -m /tmp/quantized-weights generate-parity -p "Hello" --seed 42 --steps 1 --layers 3
```

Use `--compare-cpu` on `generate-parity` for full CPU vs GPU on the same weights (slow).
