#!/usr/bin/env python3
"""E22-M0 block-mass oracle (task #101).

Consumes `step-attn-qk-dump` output (full canvas Q plane + layer K cache, raw
f32) and answers, offline: can BLOCK-granular (128-key) pre-QK selection with
per-64-row-tile shared top-B block lists retain the attention mass that
row-level top-k retains?

Per (kv depth, layer):
  1. True probs: softmax(Q·K^T) per (canvas row, head) over the PROMPT keys
     (canvas keys excluded — E22 always forces the frontier).
  2. Row top-k reference mass, k in {128, 512} (the findings-table baseline).
  3. Block selection: K block centroids (mean over 128 rows, per kv-head);
     tile centroid of Q (mean over 64 rows, per q-head); block scores =
     qc · centroid; per (tile, head) select top-B blocks; coverage = prob
     mass of the selected blocks' keys, averaged over rows/heads.
  4. Selection fidelity: Spearman rank corr between centroid block score and
     (a) the row-averaged true block mass, (b) true block max score.

KILL (per PLAN E22): union-top-32 < ~90% mass at the deepest kv, or
materially under row-top-512.

Usage:
  uv run python scripts/e22_block_mass.py --dump-dir DIR [--block 128] [--tile 64]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BLOCK = 128
TILE = 64


def load_case(dump_dir: Path, layer: int, kv_len: int):
    meta = json.loads((dump_dir / f"meta_L{layer}_kv{kv_len}.json").read_text())
    canvas, nh, nkv, hd = (
        meta["canvas"],
        meta["n_heads"],
        meta["n_kv_heads"],
        meta["head_dim"],
    )
    q = np.fromfile(dump_dir / f"q_L{layer}_kv{kv_len}.f32", dtype=np.float32)
    k = np.fromfile(dump_dir / f"k_L{layer}_kv{kv_len}.f32", dtype=np.float32)
    q = q.reshape(canvas, nh, hd)
    k = k.reshape(meta["total_kv"], nkv, hd)
    return meta, q, k


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def analyze(dump_dir: Path, layer: int, kv_len: int, blocks_b: list[int]):
    meta, q, k = load_case(dump_dir, layer, kv_len)
    canvas, nh, nkv, hd = q.shape[0], q.shape[1], k.shape[1], q.shape[2]
    group = nh // nkv
    t_prompt = meta["kv_len"]  # prompt keys only; canvas keys are forced
    n_blocks = t_prompt // BLOCK  # drop the ragged tail block (also forced)
    t_used = n_blocks * BLOCK
    kp = k[:t_used]  # [T, nkv, hd]

    # probs[row, head, t] over prompt keys (denoise queries see all prompt keys)
    # scores: q[row,h,:] . k[t, h//group, :]
    kh = kp[:, [h // group for h in range(nh)], :]  # [T, nh, hd]
    scores = np.einsum("rhd,thd->rht", q.astype(np.float32), kh, optimize=True)
    scores -= scores.max(axis=2, keepdims=True)
    np.exp(scores, out=scores)
    probs = scores / scores.sum(axis=2, keepdims=True)  # [row, head, T]

    out = {"layer": layer, "kv": t_prompt, "n_blocks": n_blocks}

    # Row top-k reference mass (mean over rows x heads)
    for kk in (64, 128, 512):
        kk_eff = min(kk, t_used)
        part = np.partition(probs, t_used - kk_eff, axis=2)[:, :, t_used - kk_eff :]
        out[f"row_top{kk}"] = float(part.sum(axis=2).mean())

    # Block structures
    pb = probs.reshape(canvas, nh, n_blocks, BLOCK)
    block_mass = pb.sum(axis=3)          # [row, head, B] true mass per block
    centroids = kp.reshape(n_blocks, BLOCK, nkv, hd).mean(axis=1)  # [B, nkv, hd]

    # Oracle ceiling: TRUE top-B blocks per (row, head) — what perfect
    # block-level selection could retain (upper bound for any block scheme).
    for b in blocks_b:
        b_eff = min(b, n_blocks)
        part = np.partition(block_mass, n_blocks - b_eff, axis=2)[:, :, n_blocks - b_eff :]
        out[f"oracle_block_top{b}"] = float(part.sum(axis=2).mean())

    # Tile-shared centroid selection (the E22 mechanism)
    n_tiles = canvas // TILE
    qt = q.reshape(n_tiles, TILE, nh, hd).mean(axis=1)  # [tile, head, hd]
    ch = centroids[:, [h // group for h in range(nh)], :]  # [B, nh, hd]
    sel_scores = np.einsum("uhd,bhd->uhb", qt, ch, optimize=True)  # [tile,head,B]

    fid_mass = []
    tile_block_mass = block_mass.reshape(n_tiles, TILE, nh, n_blocks).mean(axis=1)
    for b in blocks_b:
        b_eff = min(b, n_blocks)
        sel = np.argpartition(sel_scores, n_blocks - b_eff, axis=2)[:, :, n_blocks - b_eff :]
        # coverage: for each (tile, head), mass of selected blocks, averaged
        # over the tile's rows
        cov = 0.0
        for u in range(n_tiles):
            for h in range(nh):
                cov += block_mass[u * TILE : (u + 1) * TILE, h, sel[u, h]].sum()
        out[f"tile_top{b}"] = float(cov / (canvas * nh))
        # E22-M3 decode v0 question: ONE list shared by ALL heads (mma_full
        # range-dispatch, zero kernel changes). Union of the 16 heads'
        # per-tile top-B -> report the union's SIZE (block budget actually
        # spent) and its coverage.
        ucov, usz = 0.0, 0
        for u in range(n_tiles):
            ublocks = np.unique(sel[u].ravel())
            usz += len(ublocks)
            ucov += block_mass[u * TILE : (u + 1) * TILE, :, ublocks].sum()
        out[f"union_top{b}"] = float(ucov / (canvas * nh))
        out[f"union_top{b}_nblocks"] = float(usz / n_tiles)
    # fidelity at the (tile, head) level
    for u in range(n_tiles):
        for h in range(nh):
            fid_mass.append(spearman(sel_scores[u, h], tile_block_mass[u, h]))
    out["fidelity_spearman_mean"] = float(np.mean(fid_mass))
    out["fidelity_spearman_p10"] = float(np.percentile(fid_mass, 10))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--blocks", default="8,16,32,64")
    args = ap.parse_args()
    d = Path(args.dump_dir)
    blocks_b = [int(x) for x in args.blocks.split(",")]

    cases = sorted(d.glob("meta_L*_kv*.json"))
    if not cases:
        print("no meta_L*_kv*.json in", d)
        return 1
    for m in cases:
        parts = m.stem.replace("meta_L", "").split("_kv")
        layer, kv = int(parts[0]), int(parts[1])
        r = analyze(d, layer, kv, blocks_b)
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
