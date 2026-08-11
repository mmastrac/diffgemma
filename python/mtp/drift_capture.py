"""Phase 1 (manual streaming): DiffusionGemma encoder forward, one layer resident at a time.

Encoder layer weights are TIED to `model.decoder.layers.*` in the checkpoint
(encoder keys hold only layer_scalar), so each layer is materialized from the
decoder keys + the encoder's own layer_scalar, run, and freed. Captures the
layer-29 pre-norm output and layers-28/29 K/V exactly as attention consumes
them, into drift_states.npz for drift_test.py's head-only phase.
"""

import gc
import glob
import json
import time

import numpy as np
import torch
from safetensors import safe_open

SCRATCH = "/private/tmp/claude-501/-Users-matt-Documents-github-diffgemma-mps/7c4f2278-4d80-451e-95a8-bc39ca3229da/scratchpad"
SNAP = glob.glob(
    "/Users/matt/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/*"
)[0]
PROMPT = "Write a Python function is_prime(n) that returns True if n is prime, using trial division up to sqrt(n). Only code, no explanation."
LAST_FULL, LAST_SWA = 29, 28

from transformers import AutoConfig, AutoTokenizer
import transformers.models.diffusion_gemma.modeling_diffusion_gemma as dg

cfg = AutoConfig.from_pretrained(SNAP).text_config
tok = AutoTokenizer.from_pretrained(SNAP)
reply = open(f"{SCRATCH}/target_reply.txt").read().strip()
prompt_ids = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}], add_generation_prompt=True
)
if not isinstance(prompt_ids, list):
    prompt_ids = prompt_ids["input_ids"]
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
reply_ids = tok(reply, add_special_tokens=False)["input_ids"]
seq = prompt_ids + reply_ids
S = len(seq)
print(f"seq={S} prompt={len(prompt_ids)} reply={len(reply_ids)}", flush=True)

idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
shard_handles = {}
def get_tensor(key):
    shard = idx[key]
    if shard not in shard_handles:
        shard_handles[shard] = safe_open(f"{SNAP}/{shard}", framework="pt")
    return shard_handles[shard].get_tensor(key)

embed_key = next(k for k in idx if k.endswith("embed_tokens.weight") and "vision" not in k)
print("embed key:", embed_key, flush=True)
emb_w = get_tensor(embed_key)
embed_scale = torch.tensor(float(cfg.hidden_size) ** 0.5).to(emb_w.dtype)
ids = torch.tensor(seq, dtype=torch.long)
# Scale in bf16 (faithful to ScaledWordEmbedding's downcast), then compute f32:
# fresh nn modules are f32 and load_state_dict upcasts the bf16 weights.
hidden = (emb_w[ids][None] * embed_scale).float()
print("embedded:", tuple(hidden.shape), hidden.dtype, flush=True)

rotary = dg.DiffusionGemmaTextRotaryEmbedding(cfg).eval()
position_ids = torch.arange(S)[None]
pos_emb = {
    lt: rotary(hidden, position_ids, lt) for lt in set(cfg.layer_types)
}
neg = torch.finfo(hidden.dtype).min
causal = torch.full((1, 1, S, S), neg, dtype=hidden.dtype).triu(1)

captured_kv = {}
orig_eager = dg.eager_attention_forward
def capture_eager(module, q, k, v, mask, **kw):
    if getattr(module, "layer_idx", None) in (LAST_FULL, LAST_SWA):
        captured_kv[module.layer_idx] = (k.detach().float(), v.detach().float())
    return orig_eager(module, q, k, v, mask, **kw)
dg.eager_attention_forward = capture_eager

dec_prefix = "model.decoder.layers."
enc_prefix = "model.encoder.language_model.layers."
t_all = time.time()
for i in range(cfg.num_hidden_layers):
    t0 = time.time()
    layer = dg.DiffusionGemmaEncoderTextLayer(cfg, i)
    state = {}
    for key in idx:
        if key.startswith(f"{dec_prefix}{i}."):
            state[key[len(f"{dec_prefix}{i}.") :]] = get_tensor(key)
    state["layer_scalar"] = get_tensor(f"{enc_prefix}{i}.layer_scalar")
    missing, unexpected = layer.load_state_dict(state, strict=False)
    layer.eval()
    lt = cfg.layer_types[i]
    with torch.no_grad():
        hidden = layer(
            hidden,
            position_embeddings=pos_emb[lt],
            attention_mask=causal,
            position_ids=position_ids,
        )
    if i == LAST_FULL:
        h29 = hidden.detach().float().clone()
    if i == 0:
        print(f"layer 0 missing={missing} unexpected={unexpected}", flush=True)
    print(
        f"layer {i:>2} {lt:<17} {time.time() - t0:5.1f}s |h|={hidden.float().norm():.1f}",
        flush=True,
    )
    del layer, state
    gc.collect()
dg.eager_attention_forward = orig_eager
print(f"encoder streamed in {time.time() - t_all:.0f}s", flush=True)

emb_rows = emb_w[torch.tensor(sorted(set(seq)))].float()
np.savez(
    f"{SCRATCH}/drift_states.npz",
    h29=h29.numpy(),
    k_full=captured_kv[LAST_FULL][0].numpy(),
    v_full=captured_kv[LAST_FULL][1].numpy(),
    k_swa=captured_kv[LAST_SWA][0].numpy(),
    v_swa=captured_kv[LAST_SWA][1].numpy(),
    emb_rows=emb_rows.numpy(),
    row_tokens=np.array(sorted(set(seq))),
    embed_scale=np.array(float(embed_scale)),
)
print("saved drift_states.npz", flush=True)
