# Generate golden fixtures

Regression token streams for `generate-parity` (GPU vs checked-in golden, no CPU run).

| File | Config | ~GPU denoise |
|------|--------|--------------|
| `hello_steps1_full.json` | `-p Hello --seed 42 --steps 1` (30 layers) | ~125s |
| `hello_steps1_layers3.json` | `--steps 1 --layers 3` | ~15s |
| `hello_steps2_full.json` | `-p Hello --seed 42 --steps 2` (30 layers) | ~290s |
| `hello_steps2_layers3.json` | `--steps 2 --layers 3` | ~78s |

Refresh after intentional sampler/decoder changes:

```bash
cargo run --release --features metal -- generate-gpu -p "Hello" --seed 42 --steps 1 --write-golden hello_steps1_full
cargo run --release --features metal -- generate-gpu -p "Hello" --seed 42 --steps 2 --write-golden hello_steps2_full
cargo run --release --features metal -- generate-parity -p "Hello" --seed 42 --steps 1 --layers 3
cargo run --release --features metal -- generate-parity -p "Hello" --seed 42 --steps 1
cargo run --release --features metal -- generate-parity -p "Hello" --seed 42 --steps 2
```

Use `--compare-cpu` on `generate-parity` for full CPU vs GPU (slow).
