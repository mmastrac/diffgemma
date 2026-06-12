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

Refresh after intentional sampler/decoder changes:

```bash
# bf16
cargo run --release --features metal -- generate-gpu -p "Hello" --seed 42 --steps 1 --write-golden hello_steps1_full

# .dgq (quantize first if needed)
cargo run --release --features metal -- quantize -o /tmp/quantized-weights
cargo run --release --features metal -- -m /tmp/quantized-weights generate-gpu -p "Hello" --seed 42 --steps 1 --layers 3 --write-golden dgq_hello_steps1_layers3
cargo run --release --features metal -- -m /tmp/quantized-weights generate-parity -p "Hello" --seed 42 --steps 1 --layers 3
```

Use `--compare-cpu` on `generate-parity` for full CPU vs GPU on the same weights (slow).
