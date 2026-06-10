"""Reference tokenizer parity against HuggingFace checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer
from transformers import AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parents[2] / "model" / "transformer"


@pytest.fixture(scope="module")
def hf_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def raw_tokenizer():
    return Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))


@pytest.mark.parametrize(
    "text",
    [
        "Hello",
        "hello",
        "Hello world",
        "Why is the sky blue?",
        "  spaced  ",
        "café",
    ],
)
def test_hf_matches_raw_tokenizer(text: str, hf_tokenizer, raw_tokenizer) -> None:
    hf_ids = hf_tokenizer.encode(text, add_special_tokens=False)
    raw_ids = raw_tokenizer.encode(text).ids
    assert hf_ids == raw_ids


def test_hello_expected_id(hf_tokenizer) -> None:
    assert hf_tokenizer.encode("Hello", add_special_tokens=False) == [9259]
