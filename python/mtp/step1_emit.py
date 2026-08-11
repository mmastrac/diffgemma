"""Emit the step1 head's predicted step-1 canvases for the held blocks, plus
the majority-prior canvas (negative control), as raw u32 token files for the
Mac engine's seeded-canvas eval."""

import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open

DATA = os.path.expanduser("~/mtp/corpus/steps_v0")
DIFF_SNAP = os.path.expanduser("~/mtp/models/diff")
HID, CANVAS = 2816, 256
torch.manual_seed(7)
random.seed(7)

blocks = []
for f in sorted(glob.glob(f"{DATA}/blk*_final.json")):
    meta = json.load(open(f))
    si = meta["id"]
    pf = f"{DATA}/blk{si}_prefill_hidden.bin"
    if not os.path.exists(pf):
        continue
    p = len(meta["prompt_ids"])
    blocks.append(
        {
            "id": si,
            "prefill": torch.from_numpy(np.fromfile(pf, dtype=np.float32).reshape(p, HID)),
            "arg0": torch.from_numpy(
                np.fromfile(f"{DATA}/blk{si}_step0_argmax.bin", dtype=np.uint32).astype(np.int64)
            ),
        }
    )

# Reproduce step1_train.py's holdout split exactly (seed 7, HOLDOUT=8... the
# 130-block run used STEP_HOLDOUT=15).
ids = sorted(b["id"] for b in blocks)
random.shuffle(ids)
held_ids = set(ids[:15])
train_b = [b for b in blocks if b["id"] not in held_ids]
prior = torch.mode(torch.stack([b["arg0"] for b in train_b]), dim=0).values

idx = json.load(open(f"{DIFF_SNAP}/model.safetensors.index.json"))["weight_map"]
with safe_open(f"{DIFF_SNAP}/{idx['model.decoder.embed_tokens.weight']}", framework="pt") as ef:
    embed = ef.get_tensor("model.decoder.embed_tokens.weight").to(torch.bfloat16).cuda()


class Step1Head(nn.Module):
    def __init__(self, width=512, layers=3, heads=8):
        super().__init__()
        self.kv_proj = nn.Linear(HID, width)
        self.q = nn.Embedding(CANVAS, width)
        self.cross = nn.ModuleList(nn.MultiheadAttention(width, heads, batch_first=True) for _ in range(layers))
        self.self_ = nn.ModuleList(nn.MultiheadAttention(width, heads, batch_first=True) for _ in range(layers))
        self.mlp = nn.ModuleList(
            nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.RMSNorm(width) for _ in range(3 * layers))
        self.out = nn.Linear(width, HID)
        self.norm = nn.RMSNorm(HID)

    def forward(self, prefill):
        kv = self.kv_proj(prefill)[None]
        x = self.q.weight[None]
        for i, (ca, sa, mlp) in enumerate(zip(self.cross, self.self_, self.mlp)):
            a = self.norms[3 * i](x)
            x = x + ca(a, kv, kv, need_weights=False)[0]
            a = self.norms[3 * i + 1](x)
            x = x + sa(a, a, a, need_weights=False)[0]
            x = x + mlp(self.norms[3 * i + 2](x))
        return self.norm(self.out(x[0]))


head = Step1Head().cuda()
head.load_state_dict(torch.load(os.path.expanduser("~/mtp/step1_head_130.pt")))
head.eval()

out_dir = os.path.expanduser("~/mtp/step1_pred")
os.makedirs(out_dir, exist_ok=True)
with torch.no_grad():
    for b in blocks:
        if b["id"] not in held_ids:
            continue
        logits = (head(b["prefill"].cuda()).to(torch.bfloat16) @ embed.T).float()
        pred = logits.argmax(dim=-1).cpu().numpy().astype(np.uint32)
        pred.tofile(f"{out_dir}/blk{b['id']}_pred.bin")
        agree = float((torch.from_numpy(pred.astype(np.int64)) == b["arg0"]).float().mean())
        print(f"blk{b['id']}: agree-with-real-step1 {100 * agree:.1f}%")
prior.numpy().astype(np.uint32).tofile(f"{out_dir}/prior.bin")
print("held ids:", sorted(held_ids))
