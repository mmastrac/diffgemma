"""Step-1-from-prefill pilot: predict the FIRST denoise step's argmax from
prompt prefill hiddens alone (the noise canvas carries no information, so
step 1 is a pure function of the prompt — the "less inputs" variant).

Model: 256 learned canvas-position queries cross-attend over the prompt's
layer-29 prefill hiddens (projected), a couple of self-attn/MLP blocks, then
project to 2816 and score against the frozen tied embedding table.

Baselines reported:
  - per-position majority token over the train blocks (a prompt-independent
    prior; the head must beat this to prove it reads the prompt)
"""

import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

DATA = os.environ.get("STEP_DATA", os.path.expanduser("~/mtp/corpus/steps_v0"))
OUT = os.environ.get("STEP_OUT", os.path.expanduser("~/mtp/step1_head_v0"))
DIFF_SNAP = os.environ.get("MTP_DIFF", os.path.expanduser("~/mtp/models/diff"))
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
HOLDOUT = int(os.environ.get("STEP_HOLDOUT", "8"))
LR = float(os.environ.get("STEP_LR", "3e-4"))
DEV = os.environ.get("STEP_DEV", "cuda" if torch.cuda.is_available() else "cpu")
HID = 2816
CANVAS = 256

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

ids = sorted(b["id"] for b in blocks)
random.shuffle(ids)
held_ids = set(ids[:HOLDOUT])
train_b = [b for b in blocks if b["id"] not in held_ids]
held_b = [b for b in blocks if b["id"] in held_ids]
print(f"{len(blocks)} blocks: train {len(train_b)} held {len(held_b)}", flush=True)

# Prompt-independent prior: per-position majority token over train blocks.
stack = torch.stack([b["arg0"] for b in train_b])  # [N, 256]
prior = torch.mode(stack, dim=0).values  # [256]
for name, split in (("train", train_b), (" held", held_b)):
    acc = torch.stack([(b["arg0"] == prior).float().mean() for b in split]).mean()
    print(f"majority-prior baseline {name}: {100 * acc:.2f}%", flush=True)

idx = json.load(open(f"{DIFF_SNAP}/model.safetensors.index.json"))["weight_map"]
emb_key = "model.decoder.embed_tokens.weight"
with safe_open(f"{DIFF_SNAP}/{idx[emb_key]}", framework="pt") as ef:
    embed = ef.get_tensor(emb_key).to(torch.bfloat16).to(DEV)


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
        # prefill [P, 2816] (single prompt; prompts vary in length)
        kv = self.kv_proj(prefill)[None]
        x = self.q.weight[None]
        for i, (ca, sa, mlp) in enumerate(zip(self.cross, self.self_, self.mlp)):
            a = self.norms[3 * i](x)
            x = x + ca(a, kv, kv, need_weights=False)[0]
            a = self.norms[3 * i + 1](x)
            x = x + sa(a, a, a, need_weights=False)[0]
            x = x + mlp(self.norms[3 * i + 2](x))
        return self.norm(self.out(x[0]))


head = Step1Head().to(DEV)
print(f"head params {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M", flush=True)


@torch.no_grad()
def eval_split(split, name):
    head.eval()
    hit = tot = 0
    for b in split:
        logits = (head(b["prefill"].to(DEV)).to(torch.bfloat16) @ embed.T).float()
        hit += int((logits.argmax(dim=-1) == b["arg0"].to(DEV)).sum())
        tot += CANVAS
    head.train()
    print(f"  {name}: acc {100 * hit / max(tot, 1):.2f}%", flush=True)


print("pre-training:", flush=True)
eval_split(held_b, "held")

opt = torch.optim.AdamW(head.parameters(), lr=LR)
for epoch in range(EPOCHS):
    t0 = time.time()
    random.shuffle(train_b)
    total = 0.0
    for b in train_b:
        logits = (head(b["prefill"].to(DEV)).to(torch.bfloat16) @ embed.T).float()
        loss = F.cross_entropy(logits, b["arg0"].to(DEV))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.detach())
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch}: loss {total / len(train_b):.4f} ({time.time() - t0:.0f}s)", flush=True)
        eval_split(held_b, "held")

print("post-training:", flush=True)
eval_split(train_b, "train")
eval_split(held_b, "held")
torch.save(head.state_dict(), f"{OUT}.pt")
print(f"saved {OUT}.pt", flush=True)
