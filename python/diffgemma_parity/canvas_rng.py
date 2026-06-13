"""Canvas initialization helpers (match Rust `sample::initialize_canvas` or MLX)."""

from __future__ import annotations

from typing import Literal

CanvasRng = Literal["rust", "mlx"]


def rust_initialize_canvas(seed: int, canvas_len: int, vocab_size: int) -> list[int]:
    """Same LCG stream as Rust `Rng::new(seed)` + `initialize_canvas`."""
    state = (seed + 1) & ((1 << 64) - 1)
    out: list[int] = []
    for _ in range(canvas_len):
        state = (state * 6_966_169_279 + 1_039_523_323) & ((1 << 64) - 1)
        out.append(int((state >> 32) % max(vocab_size, 1)))
    return out


def mlx_initialize_canvas(seed: int, canvas_len: int, vocab_size: int) -> list[int]:
    import mlx.core as mx

    mx.random.seed(seed)
    ids = mx.random.randint(0, vocab_size, (canvas_len,))
    mx.eval(ids)
    return [int(x) for x in ids.tolist()]


def initialize_canvas(
    seed: int,
    canvas_len: int,
    vocab_size: int,
    *,
    rng: CanvasRng = "mlx",
) -> list[int]:
    if rng == "rust":
        return rust_initialize_canvas(seed, canvas_len, vocab_size)
    return mlx_initialize_canvas(seed, canvas_len, vocab_size)
