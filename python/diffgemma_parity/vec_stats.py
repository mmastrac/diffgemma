"""Vector comparison helpers for MLX vs Rust parity dumps."""

from __future__ import annotations

import math
import struct


def bf16_bits_to_f32(bits: int) -> float:
    """Decode bf16 stored in the low 16 bits of a u16."""
    u = (int(bits) & 0xFFFF) << 16
    return struct.unpack(">f", struct.pack(">I", u))[0]


def fp16_bits_to_f32(bits: int) -> float:
    """Decode IEEE fp16 bit pattern (Metal `half` / Rust router weights)."""
    return struct.unpack("<e", struct.pack("<H", int(bits) & 0xFFFF))[0]


def vec_l2(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def vec_stats(a: list[float], b: list[float]) -> dict[str, float]:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    cos = dot / (na * nb) if na > 1e-12 and nb > 1e-12 else float("nan")
    diff = [b[i] - a[i] for i in range(n)]
    l2 = math.sqrt(sum(d * d for d in diff))
    max_abs = max(abs(d) for d in diff) if diff else 0.0
    rel = l2 / na if na > 1e-12 else float("nan")
    return {
        "cosine": cos,
        "l2": l2,
        "max_abs": max_abs,
        "rel_l2": rel,
        "l2_ref": na,
        "l2_cand": nb,
    }
