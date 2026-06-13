"""Chat template parity: Rust formatter vs HuggingFace apply_chat_template."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model" / "transformer"
BINARY = ROOT / "target" / "release" / "diffgemma-mps"


def _cargo_release_binary() -> Path | None:
    if shutil.which("cargo") is None:
        return None
    if not BINARY.exists():
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=ROOT,
            check=True,
        )
    return BINARY


@pytest.fixture(scope="module")
def hf_tokenizer():
    if not (MODEL_DIR / "tokenizer.json").exists():
        pytest.skip("model/transformer/tokenizer.json not present")
    return AutoTokenizer.from_pretrained(MODEL_DIR)


@pytest.mark.parametrize(
    "text",
    [
        "Why is the sky blue?",
        "Hello",
        "Tell me about DiffusionGemma.",
    ],
)
def test_rust_chat_tokenize_matches_hf(text: str, hf_tokenizer) -> None:
    binary = _cargo_release_binary()
    if binary is None:
        pytest.skip("cargo not available")

    messages = [{"role": "user", "content": text}]
    expected = hf_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]

    proc = subprocess.run(
        [str(binary), "tokenize", text],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["chat_template"] is True
    assert payload["ids"] == expected
