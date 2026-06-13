"""Tests for denoise trace stats helpers (no torch required)."""

from diffgemma_parity.denoise_stats import (
    cur_step_from_index,
    step_stats_from_entropies,
)


def test_cur_step_countdown():
    assert cur_step_from_index(1, 48) == 48
    assert cur_step_from_index(48, 48) == 1


def test_step_stats_uniform_low_entropy():
    ent = [0.05] * 256
    mask = [True] * 20 + [False] * 236
    stats = step_stats_from_entropies(ent, mask)
    assert stats.accept_count == 20
    assert stats.low_entropy_positions == 256
    assert stats.min_entropy == 0.05
