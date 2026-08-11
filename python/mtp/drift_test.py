"""Zero-shot drift test: Google's Gemma 4 MTP head on DiffusionGemma prefill states.

Phase 1: run the local bf16 DiffusionGemma ENCODER (CPU + disk offload) over
[chat prompt + the target's own committed reply], capturing
  - layer-29 output (pre-final-norm; the hidden_states[-1] tap the head trained on)
  - K/V of layer 29 (last full_attention) and layer 28 (last sliding_attention)
    exactly as the attention interface consumes them.
Phase 2: teacher-forced K=8 drafting with Gemma4AssistantForCausalLM using the
SinglePositionMultiTokenCandidateGenerator recipe, scored as longest matching
prefix against the actual continuation.
"""

import gc
import glob
import json
import sys
import time

import numpy as np
import torch

SCRATCH = "/private/tmp/claude-501/-Users-matt-Documents-github-diffgemma-mps/7c4f2278-4d80-451e-95a8-bc39ca3229da/scratchpad"
DIFF_SNAP = glob.glob(
    "/Users/matt/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/*"
)[0]
ASSIST_SNAP = glob.glob(
    "/Users/matt/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it-assistant/snapshots/*"
)[0]
PROMPT = "Write a Python function is_prime(n) that returns True if n is prime, using trial division up to sqrt(n). Only code, no explanation."
K_DRAFT = 8
LAST_FULL, LAST_SWA = 29, 28

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(DIFF_SNAP)
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
ans_start = len(prompt_ids)
print(f"seq={len(seq)} prompt={ans_start} reply={len(reply_ids)}", flush=True)

state_path = f"{SCRATCH}/drift_states.npz"

if len(sys.argv) < 2 or sys.argv[1] != "head-only":
    # ---- Phase 1: encoder forward with capture ----
    from transformers import DiffusionGemmaForBlockDiffusion
    import transformers.models.diffusion_gemma.modeling_diffusion_gemma as dg_mod

    t0 = time.time()
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        DIFF_SNAP,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={"cpu": "20GiB"},
        offload_folder=f"{SCRATCH}/offload",
        offload_state_dict=True,
        attn_implementation="eager",
    )
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    enc_text = None
    for name, mod in model.named_modules():
        if type(mod).__name__ == "DiffusionGemmaEncoderTextModel":
            enc_text = mod
            print("encoder text model at:", name, flush=True)
            break
    assert enc_text is not None

    captured_kv = {}
    orig_eager = dg_mod.eager_attention_forward

    def capture_eager(module, q, k, v, mask, **kw):
        if (
            type(module).__name__ == "DiffusionGemmaEncoderTextAttention"
            and module.layer_idx in (LAST_FULL, LAST_SWA)
        ):
            captured_kv[module.layer_idx] = (
                k.detach().float().cpu(),
                v.detach().float().cpu(),
            )
        return orig_eager(module, q, k, v, mask, **kw)

    dg_mod.eager_attention_forward = capture_eager

    captured_hidden = {}
    def layer29_hook(mod, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured_hidden["h29"] = h.detach().float().cpu()

    enc_text.layers[LAST_FULL].register_forward_hook(layer29_hook)

    t0 = time.time()
    ids = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        enc_text(input_ids=ids)
    print(f"encoder forward in {time.time() - t0:.0f}s", flush=True)
    dg_mod.eager_attention_forward = orig_eager

    emb_w = model.get_input_embeddings().weight
    emb_rows = emb_w[torch.tensor(sorted(set(seq)))].detach().float().cpu()
    row_index = {t: i for i, t in enumerate(sorted(set(seq)))}
    embed_scale = torch.tensor(2816.0**0.5).to(torch.bfloat16).float().item()

    np.savez(
        state_path,
        h29=captured_hidden["h29"].numpy(),
        k_full=captured_kv[LAST_FULL][0].numpy(),
        v_full=captured_kv[LAST_FULL][1].numpy(),
        k_swa=captured_kv[LAST_SWA][0].numpy(),
        v_swa=captured_kv[LAST_SWA][1].numpy(),
        emb_rows=emb_rows.numpy(),
        row_tokens=np.array(sorted(set(seq))),
        embed_scale=np.array(embed_scale),
    )
    print("states saved:", {k: tuple(v.shape) for k, v in np.load(state_path).items()}, flush=True)
    del model
    gc.collect()

# ---- Phase 2: zero-shot head drafting ----
from transformers import Gemma4AssistantForCausalLM

data = np.load(state_path)
h29 = torch.tensor(data["h29"])
kv = {
    "full_attention": (torch.tensor(data["k_full"]), torch.tensor(data["v_full"])),
    "sliding_attention": (torch.tensor(data["k_swa"]), torch.tensor(data["v_swa"])),
}
row_index = {int(t): i for i, t in enumerate(data["row_tokens"])}
emb_rows = torch.tensor(data["emb_rows"])
embed_scale = float(data["embed_scale"])

head = Gemma4AssistantForCausalLM.from_pretrained(
    ASSIST_SNAP, dtype=torch.float32, attn_implementation="eager"
)
head.eval()
print("head loaded", flush=True)

def embed(tok_id):
    return emb_rows[row_index[tok_id]][None, None, :] * embed_scale

results = []
for p in range(ans_start - 1, len(seq) - K_DRAFT - 1):
    truth = seq[p + 1 : p + 1 + K_DRAFT]
    last_hidden = h29[:, p : p + 1]
    last_tok = seq[p]
    kv_p = {k: (v[0][:, :, : p + 1], v[1][:, :, : p + 1]) for k, v in kv.items()}
    mask = torch.ones(1, p + 1, dtype=torch.long)
    pos = torch.tensor([[p]], dtype=torch.long)
    drafted = []
    with torch.no_grad():
        for _ in range(K_DRAFT):
            inputs_embeds = torch.cat([embed(last_tok), last_hidden], dim=-1)
            out = head(
                inputs_embeds=inputs_embeds,
                attention_mask=mask,
                position_ids=pos,
                shared_kv_states=kv_p,
                use_cache=False,
            )
            last_tok = int(out.logits.argmax(dim=-1))
            last_hidden = out.last_hidden_state
            drafted.append(last_tok)
            if last_tok not in row_index:
                emb_rows = torch.cat([emb_rows, torch.zeros(1, emb_rows.shape[-1])])
                row_index[last_tok] = len(emb_rows) - 1  # placeholder; fixed below
                break
    accept = 0
    for d, t in zip(drafted, truth):
        if d != t:
            break
        accept += 1
    results.append({"pos": p, "accept": accept, "drafted": drafted, "truth": truth})

acc = [r["accept"] for r in results]
print(f"\npositions={len(acc)} K={K_DRAFT}")
print(f"mean accepted prefix: {np.mean(acc):.2f} / {K_DRAFT}")
print(f"first-token accuracy: {np.mean([a >= 1 for a in acc]) * 100:.1f}%")
print(f">=4 accepted: {np.mean([a >= 4 for a in acc]) * 100:.1f}%")
print(f"full {K_DRAFT} accepted: {np.mean([a == K_DRAFT for a in acc]) * 100:.1f}%")
json.dump(results, open(f"{SCRATCH}/drift_results.json", "w"))

for r in results[:: max(1, len(results) // 12)]:
    d = tok.decode(r["drafted"][: max(r["accept"], 1)])
    t = tok.decode(r["truth"])
    print(f"  pos {r['pos']:>4} accept={r['accept']} drafted[:acc]={d!r} truth={t!r}")
