#!/usr/bin/env python3
"""Render the reference diffusiongemma chat template for tool-use cases.

The quantized package ships an empty `chat_template`, so the ground-truth tool
grammar lives only in the HF bf16 model's `chat_template.jinja`. This script
renders representative cases; the exact strings are the oracle that `src/tools.rs`
is unit-tested against (keep them in sync if the template ever changes).

Usage: python3 python/scripts/tool_format_oracle.py [path/to/chat_template.jinja]
Default path: the cached google/diffusiongemma-26B-A4B-it snapshot.
"""
import glob
import os
import sys

import jinja2

DEFAULT = sorted(
    glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/"
            "snapshots/*/chat_template.jinja"
        )
    )
)


def render(tmpl_path):
    env = jinja2.Environment()
    env.globals.update(bos_token="<bos>")
    t = env.from_string(open(tmpl_path).read())

    list_dir = {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Lists all files and subdirectories in a specified directory path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The absolute path to the directory to list (e.g., /tmp)",
                    }
                },
                "required": ["path"],
            },
        },
    }
    print("=== CASE 1: tools + user, gen prompt ===")
    print(
        t.render(
            messages=[{"role": "user", "content": "List files in /tmp"}],
            tools=[list_dir],
            add_generation_prompt=True,
        )
    )
    print("\n=== CASE 2: assistant tool_call + tool response ===")
    print(
        t.render(
            messages=[
                {"role": "user", "content": "List files in /tmp"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "list_dir", "arguments": {"path": "/tmp"}}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "a.txt\nb.txt"},
            ],
            tools=[list_dir],
            add_generation_prompt=True,
        )
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else (DEFAULT[-1] if DEFAULT else None)
    if not path or not os.path.exists(path):
        sys.exit("chat_template.jinja not found; pass the path explicitly")
    render(path)
