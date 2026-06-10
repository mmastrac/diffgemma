"""Optional cross-check: Rust `tokenize` CLI vs HuggingFace."""

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
    return AutoTokenizer.from_pretrained(MODEL_DIR)


@pytest.mark.parametrize("text", ["Hello", "Hello world", "Why is the sky blue?"])
def test_rust_cli_matches_hf(text: str, hf_tokenizer) -> None:
    binary = _cargo_release_binary()
    if binary is None:
        pytest.skip("cargo not available")

    proc = subprocess.run(
        [str(binary), "tokenize", text],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    expected = hf_tokenizer.encode(text, add_special_tokens=False)
    assert payload["ids"] == expected
