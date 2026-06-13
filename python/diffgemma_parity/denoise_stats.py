"""Shared denoise-step statistics (matches Rust `sample::step_entropy_stats`)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ENTROPY_BOUND = 0.1


@dataclass
class StepStats:
    accept_count: int
    low_entropy_positions: int
    min_entropy: float
    mean_entropy: float
    max_entropy: float


def step_stats_from_entropies(
    entropies: list[float], accept_mask: list[bool], threshold: float = ENTROPY_BOUND
) -> StepStats:
    low = sum(1 for e in entropies if e < threshold)
    accept_count = sum(1 for a in accept_mask if a)
    if not entropies:
        return StepStats(0, 0, 0.0, 0.0, 0.0)
    return StepStats(
        accept_count=accept_count,
        low_entropy_positions=low,
        min_entropy=min(entropies),
        mean_entropy=sum(entropies) / len(entropies),
        max_entropy=max(entropies),
    )


def cur_step_from_index(step_index: int, max_denoise_steps: int) -> int:
    """HF countdown step (max .. 1) from 1-based step_index."""
    return max_denoise_steps - step_index + 1


def trace_step_record(
    *,
    block: int,
    step_index: int,
    max_denoise_steps: int,
    stats: StepStats,
    argmax_prefix: list[int],
    early_stop: bool,
) -> dict[str, Any]:
    return {
        "block": block,
        "step_index": step_index,
        "cur_step": cur_step_from_index(step_index, max_denoise_steps),
        "accept_count": stats.accept_count,
        "low_entropy_positions": stats.low_entropy_positions,
        "min_entropy": stats.min_entropy,
        "mean_entropy": stats.mean_entropy,
        "max_entropy": stats.max_entropy,
        "argmax_prefix": argmax_prefix[:16],
        "early_stop": early_stop,
    }


def stats_dict(stats: StepStats) -> dict[str, Any]:
    return asdict(stats)
