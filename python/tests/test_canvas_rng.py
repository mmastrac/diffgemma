"""Canvas RNG parity with Rust `sample::initialize_canvas`."""

from diffgemma_parity.canvas_rng import rust_initialize_canvas

# From `cargo test` / manual: Rng::new(42) first 16 canvas ids @ vocab=262144.
RUST_CANVAS_42_HEAD = [
    69,
    171994,
    13470,
    145834,
    187385,
    150612,
    136754,
    58067,
    136985,
    56736,
    256539,
    123397,
    200670,
    61309,
    194031,
    39695,
]


def test_rust_canvas_seed_42():
    got = rust_initialize_canvas(42, 256, 262_144)[:16]
    assert got == RUST_CANVAS_42_HEAD
