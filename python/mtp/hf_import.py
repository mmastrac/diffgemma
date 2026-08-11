"""HF dataset importer: stream, filter, and sample instruction pairs into
dump-ready corpus JSONL ({"id", "prompt", "reply", "source"}).

Usage: hf_import.py <dataset-tag> <n> <out.jsonl>
Tags: opencode (nvidia/OpenCodeInstruct), tulu (allenai/tulu-3-sft-mixture),
smoltalk (HuggingFaceTB/smoltalk).

Filters: single-turn only, length caps sized to the mtp-dump sequence limit,
mostly-ASCII, no chat-markup artifacts, prompt-hash dedupe. Foreign-teacher
data: keep source tags distinct and the self arm as the acceptance anchor.
"""

import hashlib
import json
import random
import sys

from datasets import load_dataset

MIN_REPLY = 30
MAX_REPLY = 4200
MAX_PROMPT = 1600
SHUFFLE_BUFFER = 20_000
ARTIFACTS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|channel>", "[INST]", "</s>")


def ascii_ratio(s):
    if not s:
        return 0.0
    return sum(1 for c in s if ord(c) < 128) / len(s)


def clean_pair(prompt, reply):
    if not prompt or not reply:
        return None
    prompt, reply = prompt.strip(), reply.strip()
    if not (MIN_REPLY <= len(reply) <= MAX_REPLY and len(prompt) <= MAX_PROMPT):
        return None
    if ascii_ratio(prompt) < 0.95 or ascii_ratio(reply) < 0.9:
        return None
    joined = prompt + reply
    if any(a in joined for a in ARTIFACTS):
        return None
    return prompt, reply


def from_messages(row):
    msgs = row.get("messages") or []
    if len(msgs) == 2 and msgs[0].get("role") == "user" and msgs[1].get("role") == "assistant":
        return msgs[0].get("content"), msgs[1].get("content")
    return None, None


DATASETS = {
    "opencode": {
        "path": "nvidia/OpenCodeInstruct",
        "split": "train",
        "extract": lambda r: (r.get("input") or r.get("question"), r.get("output") or r.get("response")),
    },
    "tulu": {
        "path": "allenai/tulu-3-sft-mixture",
        "split": "train",
        "extract": from_messages,
    },
    "smoltalk": {
        "path": "HuggingFaceTB/smoltalk",
        "name": "all",
        "split": "train",
        "extract": from_messages,
    },
}


def main():
    tag, n, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    spec = DATASETS[tag]
    ds = load_dataset(
        spec["path"],
        spec.get("name"),
        split=spec["split"],
        streaming=True,
    ).shuffle(seed=7, buffer_size=SHUFFLE_BUFFER)
    seen = set()
    kept = 0
    scanned = 0
    random.seed(7)
    with open(out_path, "w") as out:
        for row in ds:
            scanned += 1
            if scanned > n * 200 and kept == 0:
                sys.exit(f"no rows extracted after {scanned}; check field mapping: {list(row)}")
            pair = clean_pair(*spec["extract"](row))
            if pair is None:
                continue
            prompt, reply = pair
            h = hashlib.sha1(prompt.encode()).hexdigest()[:16]
            if h in seen:
                continue
            seen.add(h)
            out.write(
                json.dumps(
                    {
                        "id": f"{tag}_{kept:05}",
                        "prompt": prompt,
                        "reply": reply,
                        "source": f"hf_{tag}",
                    }
                )
                + "\n"
            )
            kept += 1
            if kept % 200 == 0:
                print(f"{kept}/{n} (scanned {scanned})", flush=True)
            if kept >= n:
                break
    print(f"kept {kept} of {scanned} scanned -> {out_path}")


if __name__ == "__main__":
    main()
