"""MTP head rollout fine-tune dry run.

Same dump and eval as the 1-step version, with the recipe fix that run
surfaced: steps 2..K of each training example consume the head's OWN
post_projection hidden (gradients flow through the recurrence), while input
tokens stay teacher-forced. Loss is CE + hidden-MSE at every depth. The
1-step recipe lifted first-token accuracy but collapsed multi-token drafts
(exposure bias); rollout trains the regime drafting actually runs in.

Eval reports first-token accuracy AND free-running accepted prefix (own
tokens + own hiddens, K=8) pre/post. Saves bf16 so the Rust loaders read it.
"""

import glob
import json
import random
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

SCRATCH = os.environ.get("MTP_SCRATCH", "/private/tmp/claude-501/-Users-matt-Documents-github-diffgemma-mps/7c4f2278-4d80-451e-95a8-bc39ca3229da/scratchpad")
DATA = os.environ.get("MTP_DATA", f"{SCRATCH}/mtp_data")
OUT = os.environ.get("MTP_OUT", f"{SCRATCH}/mtp_head_tuned")
ASSIST_SNAP = os.environ.get("MTP_ASSIST") or glob.glob(
    "/Users/matt/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it-assistant/snapshots/*"
)[0]
DIFF_SNAP = os.environ.get("MTP_DIFF") or glob.glob(
    "/Users/matt/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/*"
)[0]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ROLLOUT_K = 4
STRIDE = int(os.environ.get("MTP_STRIDE", "2"))
POS_CAP = int(os.environ.get("MTP_POS_CAP", "0"))
K_EVAL = 8
HOLDOUT = int(os.environ.get("MTP_HOLDOUT_GROUPS", "12"))
MSE_W = 0.1
EMBED_SCALE = float(torch.tensor(2816.0**0.5).to(torch.bfloat16))

torch.manual_seed(7)
random.seed(7)

manifest = []
for d in DATA.split(":"):
    if os.path.exists(f"{d}/manifest.jsonl"):
        for l in open(f"{d}/manifest.jsonl"):
            if l.strip():
                m = json.loads(l)
                m["dir"] = d
                manifest.append(m)
    else:
        for m in json.load(open(f"{d}/manifest.json")):
            m["dir"] = d
            manifest.append(m)


class Seq:
    def __init__(self, m):
        self.id = str(m["id"])
        self.source = m.get("source", "self")
        self.seq = m["seq"]
        self.ans_start = m["ans_start"]
        s = len(self.seq)
        d = m["dir"]
        prefix = f"{d}/seq{self.id}" if os.path.exists(f"{d}/seq{self.id}_hidden.bin") else f"{d}/{self.id}"
        load = lambda name, shape: torch.tensor(
            np.fromfile(f"{prefix}_{name}.bin", dtype=np.float32).reshape(shape)
        )
        self.hidden = load("hidden", (s, 2816))
        self.kv = {
            "sliding_attention": (
                load("k_swa", (1, 8, s, 256)),
                load("v_swa", (1, 8, s, 256)),
            ),
            "full_attention": (
                load("k_full", (1, 2, s, 512)),
                load("v_full", (1, 2, s, 512)),
            ),
        }


def base_id(sid):
    for suffix in ("_31b_think", "_31b", "_relay_think", "_relay"):
        if sid.endswith(suffix):
            return sid[: -len(suffix)]
    return sid


seqs = [Seq(m) for m in manifest]
groups = sorted({base_id(s.id) for s in seqs})
random.shuffle(groups)
held_groups = set(groups[:HOLDOUT])
held = [s for s in seqs if base_id(s.id) in held_groups]
train = [s for s in seqs if base_id(s.id) not in held_groups]
print(
    f"{len(seqs)} sequences ({len(groups)} prompt groups): train={len(train)} held={len(held)}; rollout K={ROLLOUT_K}"
)

idx = json.load(open(f"{DIFF_SNAP}/model.safetensors.index.json"))["weight_map"]
emb_key = "model.decoder.embed_tokens.weight"
emb_file = safe_open(f"{DIFF_SNAP}/{idx[emb_key]}", framework="pt")
emb_slice = emb_file.get_slice(emb_key)
emb_cache = {}


def embed_row(tok):
    if tok not in emb_cache:
        emb_cache[tok] = emb_slice[tok : tok + 1].to(torch.float32) * EMBED_SCALE
    return emb_cache[tok]


from transformers import Gemma4AssistantForCausalLM

DEV = os.environ.get(
    "MTP_DEV",
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
)
head = Gemma4AssistantForCausalLM.from_pretrained(ASSIST_SNAP, dtype=torch.float32).to(DEV)
head.train()
print(f"device: {DEV}")


def head_step(sq, p, tok_in, hidden):
    x = torch.cat([embed_row(tok_in).to(hidden.device), hidden], dim=-1)[None]
    kv_p = {k: (v[0][:, :, : p + 1].to(DEV), v[1][:, :, : p + 1].to(DEV)) for k, v in sq.kv.items()}
    return head(
        inputs_embeds=x.to(DEV),
        attention_mask=torch.ones(1, p + 1, dtype=torch.long, device=DEV),
        position_ids=torch.tensor([[p]], device=DEV),
        shared_kv_states=kv_p,
        use_cache=False,
    )


ce = torch.nn.CrossEntropyLoss()
mse = torch.nn.MSELoss()


def rollout_loss(sq, p):
    hidden = sq.hidden[p : p + 1].to(DEV)
    loss = 0.0
    steps = 0
    for j in range(ROLLOUT_K):
        if p + j + 1 >= len(sq.seq):
            break
        out = head_step(sq, p, sq.seq[p + j], hidden)
        loss = loss + ce(out.logits[0, -1][None], torch.tensor([sq.seq[p + j + 1]], device=DEV))
        loss = loss + MSE_W * mse(out.last_hidden_state[0, -1], sq.hidden[p + j + 1].to(DEV))
        hidden = out.last_hidden_state[0]
        steps += 1
    return loss / max(steps, 1)


@torch.no_grad()
def eval_split(split, name):
    from collections import defaultdict

    head.eval()
    by_src = defaultdict(lambda: [0, 0, 0, 0])
    hit = tot = 0
    acc_sum = acc_n = 0
    for sq in split:
        s = len(sq.seq)
        stat = by_src[sq.source]
        for p in range(sq.ans_start - 1, s - 1, max(STRIDE // 2, 1)):
            out = head_step(sq, p, sq.seq[p], sq.hidden[p : p + 1].to(DEV))
            ok = int(out.logits.argmax(dim=-1).item() == sq.seq[p + 1])
            hit += ok
            tot += 1
            stat[0] += ok
            stat[1] += 1
        # Free-running accepted prefix at a thinner stride (it is K forwards each).
        for p in range(sq.ans_start - 1, s - K_EVAL - 1, 4):
            hidden = sq.hidden[p : p + 1].to(DEV)
            tok = sq.seq[p]
            a = 0
            for j in range(K_EVAL):
                out = head_step(sq, p, tok, hidden)
                tok = int(out.logits.argmax(dim=-1))
                hidden = out.last_hidden_state[0]
                if tok != sq.seq[p + j + 1]:
                    break
                a += 1
            acc_sum += a
            acc_n += 1
            stat[2] += a
            stat[3] += 1
    head.train()
    print(
        f"  {name}: first-token {100.0 * hit / max(tot, 1):.1f}% ({tot})  "
        f"free-run accepted {acc_sum / max(acc_n, 1):.2f}/{K_EVAL} ({acc_n})",
        flush=True,
    )
    for src_name, (h, t, a, an) in sorted(by_src.items()):
        print(
            f"    {src_name:<20} first {100.0 * h / max(t, 1):5.1f}% ({t})  accepted {a / max(an, 1):.2f} ({an})",
            flush=True,
        )


print("pre-training:")
eval_split(train, "train")
eval_split(held, "held")

opt = torch.optim.AdamW(head.parameters(), lr=1e-4)
for epoch in range(EPOCHS):
    t0 = time.time()
    order = []
    for sq in train:
        ps = list(range(sq.ans_start - 1, len(sq.seq) - 1, STRIDE))
        if POS_CAP and len(ps) > POS_CAP:
            ps = random.sample(ps, POS_CAP)
        order.extend((sq, p) for p in ps)
    random.shuffle(order)
    total = 0.0
    max_steps = int(os.environ.get("MTP_MAX_STEPS", "0"))
    if max_steps:
        order = order[:max_steps]
    for i, (sq, p) in enumerate(order):
        loss = rollout_loss(sq, p)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.detach())
        if (i + 1) % 200 == 0:
            print(f"  epoch {epoch} step {i + 1}/{len(order)} loss {total / (i + 1):.3f}", flush=True)
    dt = time.time() - t0
    print(
        f"epoch {epoch}: mean loss {total / len(order):.3f} ({dt:.0f}s, {dt / max(len(order), 1):.2f}s/step)",
        flush=True,
    )
    head.save_pretrained(f"{OUT}_e{epoch}", safe_serialization=True)

print("post-training:")
eval_split(train, "train")
eval_split(held, "held")

head.to(torch.bfloat16)
head.save_pretrained(OUT, safe_serialization=True)
print(f"saved {OUT} (bf16)")
