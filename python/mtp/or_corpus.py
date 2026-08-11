"""Teacher corpus builder over OpenRouter.

Two arms per input prompt:
  teacher_31b   - Gemma 4 31B answers directly (in-family, thinking stripped).
  teacher_relay - 31B thinks, DeepSeek renders the final answer from
                  [prompt + thought]; kept only when the render agrees with
                  31B's own answer (or the pair is marked verifiable).

Reads a prompts JSONL ({"id", "prompt"}), writes a corpus JSONL consumable by
`diffgemma mtp-dump` ({"id", "prompt", "reply", "source"}). Set
OPENROUTER_API_KEY. Progress is append-only; already-emitted ids are skipped
on re-run.
"""

import difflib
import json
import os
import sys
import time
import urllib.request

API = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ["OPENROUTER_API_KEY"]
M31 = "google/gemma-4-31b-it"
DS = "deepseek/deepseek-v3.2"
THOUGHT_BUDGET = 900
ANSWER_BUDGET = 350
AGREE_MIN = 0.55

RENDER_INSTRUCTION = (
    "You are given a question and a colleague's reasoning about it. Write the "
    "final answer that follows from this reasoning. Deviate only to fix "
    "outright errors. Be concise; answer only, no preamble, no restating the "
    "reasoning."
)


SPEND_CAP = float(os.environ.get("OR_SPEND_CAP", "3.0"))
# $/M input, $/M output (openrouter listed rates).
PRICES = {M31: (0.08, 0.35), DS: (0.2072, 0.3108)}
spent = 0.0


def call(model, messages, max_tokens, reasoning=False):
    global spent
    if spent >= SPEND_CAP:
        raise SystemExit(f"spend cap ${SPEND_CAP} reached (${spent:.2f})")
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7}
    if reasoning:
        payload["reasoning"] = {"enabled": True}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            msg = out["choices"][0]["message"]
            u = out.get("usage", {})
            pi, po = PRICES[model]
            spent += u.get("prompt_tokens", 0) * pi / 1e6 + u.get("completion_tokens", 0) * po / 1e6
            return msg.get("content") or "", msg.get("reasoning")
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def split_thought(text):
    # Gemma 4 ceremony: <|channel>thought ... <channel|> answer. Providers may
    # render it slightly differently; fall back to no-thought.
    for open_tag, close_tag in [("<|channel>thought", "<channel|>")]:
        if open_tag in text and close_tag in text:
            head, rest = text.split(close_tag, 1)
            thought = head.split(open_tag, 1)[1].strip()
            return thought, rest.strip()
    return None, text.strip()


def agree(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main():
    prompts_path, out_path = sys.argv[1], sys.argv[2]
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
    out = open(out_path, "a")
    kept = dropped = 0
    for line in open(prompts_path):
        if not line.strip():
            continue
        rec = json.loads(line)
        base_id, prompt = rec["id"], rec["prompt"]
        if f"{base_id}_31b" in done:
            continue
        raw, reasoning = call(
            M31, [{"role": "user", "content": prompt}], THOUGHT_BUDGET + ANSWER_BUDGET, reasoning=True
        )
        if reasoning:
            thought, answer_31b = reasoning.strip(), raw.strip()
        else:
            thought, answer_31b = split_thought(raw)
        if len(answer_31b) < 20:
            dropped += 1
            continue
        rec_31b = {
            "id": f"{base_id}_31b",
            "prompt": prompt,
            "reply": answer_31b,
            "source": "teacher_31b",
        }
        if thought:
            rec_31b["thought"] = thought
        out.write(json.dumps(rec_31b) + "\n")
        kept += 1
        if thought:
            context = f"Question:\n{prompt}\n\nReasoning:\n{thought}"
            rendered, _ = call(
                DS,
                [
                    {"role": "system", "content": RENDER_INSTRUCTION},
                    {"role": "user", "content": context},
                ],
                ANSWER_BUDGET,
            )
            rendered = rendered.strip()
            if len(rendered) >= 20 and agree(rendered, answer_31b) >= AGREE_MIN:
                out.write(
                    json.dumps(
                        {
                            "id": f"{base_id}_relay",
                            "prompt": prompt,
                            "reply": rendered,
                            "source": "teacher_relay",
                        }
                    )
                    + "\n"
                )
                kept += 1
            else:
                dropped += 1
        out.flush()
        print(f"{base_id}: kept={kept} dropped={dropped} spent=${spent:.3f}", flush=True)
    print(f"done: {kept} records, {dropped} dropped, spent ${spent:.2f} -> {out_path}")


if __name__ == "__main__":
    main()
