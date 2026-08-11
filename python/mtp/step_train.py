"""Step-simulator head pilot.

Predict the diffusion denoiser's NEXT-step argmax at each canvas position
from the CURRENT step's (argmax token, layer-29 hidden) at that position.
No recurrence: hidden_t[i] carries the full-context attention state from the
real step that produced it. Serving shape: real step -> k cheap head steps ->
real step (native verification).

Most labels are copy-through (the canvas mostly stops changing after a few
steps), so the whole game is the CHANGED positions: report accuracy there
separately, plus the false-change rate on stable positions. Copy-through is
the baseline the head must beat.

Labels: STEP_LABEL=next  argmax_{t+1}[i]   (t = 0..S-2)
        STEP_LABEL=final final[i]          (t = 0..S-1)

Model (STEP_ARCH=mlp): concat(embed(tok)*sqrt(2816), hidden) -> MLP -> 2816,
residual add of the input token's embed row (identity/copy is the init
behavior), logits via the frozen tied embedding table.

Model (STEP_ARCH=attn): same input per position, but a small transformer
self-attends over the 256 canvas positions before the output projection.
Rationale: a change at position i is usually driven by NEIGHBORS resolving in
the current step, and hidden_t[i] only encodes attention over the previous
canvas — cross-position mixing is the information the MLP structurally lacks.
A step-index embedding is added to every position (early steps churn, late
steps are near-fixed-point).
"""

import glob
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

DATA = os.environ.get("STEP_DATA", os.path.expanduser("~/mtp/corpus/steps_v0"))
OUT = os.environ.get("STEP_OUT", os.path.expanduser("~/mtp/step_head_v0"))
DIFF_SNAP = os.environ.get("MTP_DIFF", os.path.expanduser("~/mtp/models/diff"))
LABEL = os.environ.get("STEP_LABEL", "next")  # next | final
ARCH = os.environ.get("STEP_ARCH", "mlp")  # mlp | attn
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BS = int(os.environ.get("STEP_BS", "16"))  # canvases per batch (256 pos each)
HOLDOUT = int(os.environ.get("STEP_HOLDOUT", "6"))
CHANGED_W = float(os.environ.get("STEP_CHANGED_W", "1.0"))
LR = float(os.environ.get("STEP_LR", "3e-4"))
DEV = os.environ.get("STEP_DEV", "cuda" if torch.cuda.is_available() else "cpu")
HID = 2816
CANVAS = 256
EMBED_SCALE = float(torch.tensor(HID**0.5).to(torch.bfloat16))

torch.manual_seed(7)
random.seed(7)

# ---------------------------------------------------------------- data
blocks = []
for f in sorted(glob.glob(f"{DATA}/blk*_final.json")):
    meta = json.load(open(f))
    si = meta["id"]
    steps = meta["steps"]
    hid = [
        torch.from_numpy(
            np.fromfile(f"{DATA}/blk{si}_step{t}_hidden.bin", dtype=np.float32).reshape(CANVAS, HID)
        )
        for t in range(steps)
    ]
    arg = [
        torch.from_numpy(np.fromfile(f"{DATA}/blk{si}_step{t}_argmax.bin", dtype=np.uint32).astype(np.int64))
        for t in range(steps)
    ]
    final = torch.tensor(meta["final"], dtype=torch.int64)
    if len(final) < CANVAS:
        # Trimmed commit: positions past the commit keep the converged
        # canvas (the last step's argmax) as their label.
        final = torch.cat([final, arg[-1][len(final):]])
    blocks.append({"id": si, "hid": hid, "arg": arg, "final": final})

ids = sorted(b["id"] for b in blocks)
random.shuffle(ids)
held_ids = set(ids[:HOLDOUT])
print(f"{len(blocks)} blocks, held out {sorted(held_ids)}; label={LABEL} arch={ARCH}", flush=True)


def make_canvases(blist):
    """Canvas-major: hs [N,256,2816], ts/ys [N,256], st [N]."""
    hs, ts, ys, st = [], [], [], []
    for b in blist:
        s = len(b["hid"])
        for t in range(s):
            if LABEL == "next":
                if t == s - 1:
                    continue
                y = b["arg"][t + 1]
            else:
                y = b["final"]
            hs.append(b["hid"][t])
            ts.append(b["arg"][t])
            ys.append(y)
            st.append(t)
    return torch.stack(hs), torch.stack(ts), torch.stack(ys), torch.tensor(st)


train_c = make_canvases([b for b in blocks if b["id"] not in held_ids])
held_c = make_canvases([b for b in blocks if b["id"] in held_ids])
n_tr, n_he = len(train_c[3]), len(held_c[3])
chg_tr = (train_c[1] != train_c[2]).float().mean().item()
chg_he = (held_c[1] != held_c[2]).float().mean().item()
print(f"canvases: train {n_tr} (changed {100 * chg_tr:.1f}%)  held {n_he} (changed {100 * chg_he:.1f}%)", flush=True)
print(f"copy-through baseline: train {100 * (1 - chg_tr):.1f}%  held {100 * (1 - chg_he):.1f}%", flush=True)

# ---------------------------------------------------------------- embed table
idx = json.load(open(f"{DIFF_SNAP}/model.safetensors.index.json"))["weight_map"]
emb_key = "model.decoder.embed_tokens.weight"
with safe_open(f"{DIFF_SNAP}/{idx[emb_key]}", framework="pt") as ef:
    embed = ef.get_tensor(emb_key).to(torch.bfloat16).to(DEV)  # [V, 2816], frozen
print(f"embed table {tuple(embed.shape)} on {DEV}", flush=True)


class StepHeadMlp(nn.Module):
    def __init__(self, width=2048):
        super().__init__()
        self.inp = nn.Linear(2 * HID, width)
        self.mid = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.out = nn.Linear(width, HID)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.norm = nn.RMSNorm(HID)

    def forward(self, hidden, tok_embed, step_t):
        x = torch.cat([tok_embed, hidden], dim=-1)
        h = F.gelu(self.inp(x))
        h = h + self.mid(h)
        return self.norm(self.out(h) + tok_embed)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.n1 = nn.RMSNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.n2 = nn.RMSNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        a = self.n1(x)
        x = x + self.attn(a, a, a, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class StepHeadAttn(nn.Module):
    def __init__(self, width=512, layers=4, heads=8):
        super().__init__()
        self.inp = nn.Linear(2 * HID, width)
        self.pos = nn.Embedding(CANVAS, width)
        self.step = nn.Embedding(32, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.out = nn.Linear(width, HID)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.norm = nn.RMSNorm(HID)

    def forward(self, hidden, tok_embed, step_t):
        # hidden [B,256,2816], tok_embed [B,256,2816], step_t [B]
        x = self.inp(torch.cat([tok_embed, hidden], dim=-1))
        x = x + self.pos.weight[None] + self.step(step_t.clamp(max=31))[:, None]
        for b in self.blocks:
            x = b(x)
        return self.norm(self.out(x) + tok_embed)


head = (StepHeadAttn() if ARCH == "attn" else StepHeadMlp()).to(DEV)
print(f"head params {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M", flush=True)


def forward_batch(c, j):
    h = c[0][j].to(DEV)
    tk = c[1][j].to(DEV)
    te = embed[tk].float() * EMBED_SCALE
    q = head(h, te, c[3][j].to(DEV))
    return (q.to(torch.bfloat16) @ embed.T).float(), tk


@torch.no_grad()
def eval_split(c, name):
    head.eval()
    hit = tot = c_hit = c_tot = fc = s_tot = 0
    by_t = defaultdict(lambda: [0, 0, 0, 0])
    for i in range(0, len(c[3]), BS):
        j = torch.arange(i, min(i + BS, len(c[3])))
        logits, tk = forward_batch(c, j)
        y = c[2][j].to(DEV)
        pred = logits.argmax(dim=-1)
        ok = pred == y
        changed = tk != y
        hit += int(ok.sum())
        tot += ok.numel()
        c_hit += int((ok & changed).sum())
        c_tot += int(changed.sum())
        fc += int(((pred != tk) & ~changed).sum())
        s_tot += int((~changed).sum())
        for bi, t in enumerate(c[3][j].tolist()):
            b = by_t[min(t, 8)]
            b[1] += ok[bi].numel()
            b[0] += int(ok[bi].sum())
            b[3] += int(changed[bi].sum())
            b[2] += int((ok[bi] & changed[bi]).sum())
    head.train()
    print(
        f"  {name}: acc {100 * hit / max(tot, 1):.2f}%  changed-acc {100 * c_hit / max(c_tot, 1):.2f}% ({c_tot})  "
        f"false-change {100 * fc / max(s_tot, 1):.2f}%",
        flush=True,
    )
    for t in sorted(by_t):
        a, n, ca, cn = by_t[t]
        tag = f"t={t}" if t < 8 else "t>=8"
        print(f"    {tag:<5} acc {100 * a / max(n, 1):5.1f}% ({n})  changed {100 * ca / max(cn, 1):5.1f}% ({cn})", flush=True)


print("pre-training:", flush=True)
eval_split(held_c, "held")

opt = torch.optim.AdamW(head.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * ((n_tr + BS - 1) // BS))
for epoch in range(EPOCHS):
    t0 = time.time()
    perm = torch.randperm(n_tr)
    total = nb = 0.0
    for i in range(0, n_tr, BS):
        j = perm[i : i + BS]
        logits, tk = forward_batch(train_c, j)
        y = train_c[2][j].to(DEV)
        flat_l, flat_y, flat_t = logits.flatten(0, 1), y.flatten(), tk.flatten()
        if CHANGED_W != 1.0:
            w = torch.where(flat_t != flat_y, CHANGED_W, 1.0)
            loss = (F.cross_entropy(flat_l, flat_y, reduction="none") * w).sum() / w.sum()
        else:
            loss = F.cross_entropy(flat_l, flat_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        total += float(loss.detach())
        nb += 1
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch}: loss {total / nb:.4f} ({time.time() - t0:.0f}s)", flush=True)
        eval_split(held_c, "held")
print("post-training:", flush=True)
eval_split(train_c, "train")
eval_split(held_c, "held")
torch.save(head.state_dict(), f"{OUT}.pt")
print(f"saved {OUT}.pt", flush=True)
