"""Emit the self-arm corpus JSONL: varied prompts, no replies (the engine
generates them during mtp-dump)."""

import json
import sys

LANGS = ["Python", "Rust", "JavaScript", "Go", "C"]
CODE_TASKS = [
    "checks if a string is a palindrome",
    "computes the factorial of n iteratively",
    "returns the maximum value in a list without using builtins",
    "counts vowels in a string",
    "merges two sorted lists into one sorted list",
    "computes the greatest common divisor of two integers",
    "reverses the words in a sentence",
    "returns the n-th triangular number",
    "removes duplicate values from a list preserving order",
    "converts a temperature between celsius and fahrenheit",
    "computes the dot product of two vectors",
    "finds the longest word in a sentence",
    "checks whether parentheses in a string are balanced",
    "computes the running mean of a stream of numbers",
    "flattens a nested list one level deep",
    "returns the binary representation of an integer",
    "computes the Levenshtein distance of two short strings",
    "rotates a list left by k positions",
    "sums the digits of an integer",
    "finds the first non-repeating character in a string",
    "implements binary search over a sorted array",
    "converts a roman numeral string to an integer",
    "checks if two strings are anagrams",
    "generates the first n Fibonacci numbers",
    "splits a string into chunks of size k",
]
EXPLAIN_TOPICS = [
    "why binary search needs a sorted input",
    "the difference between a stack and a queue",
    "what a hash collision is",
    "why floating point addition is not associative",
    "what tail recursion is",
    "the two's complement representation of negative integers",
    "what a race condition is",
    "why caching improves latency",
    "what big-O notation measures",
    "the difference between TCP and UDP",
    "what a deadlock is and one way to avoid it",
    "the difference between processes and threads",
    "what an index does in a database",
    "why passwords are hashed rather than encrypted",
    "what garbage collection does",
    "the difference between latency and throughput",
    "what a memory leak is",
    "how DNS resolves a hostname",
    "what idempotency means for an API",
    "the difference between compilation and interpretation",
    "what a pure function is",
    "why integer overflow happens",
    "what mutual exclusion protects against",
    "the difference between unit and integration tests",
    "what a bloom filter trades away",
]
QA = [
    "What is the capital of Japan?",
    "How many bits are in a byte?",
    "What year did the first moon landing happen?",
    "What does CPU stand for?",
    "Name the largest planet in the solar system.",
    "What is the chemical symbol for gold?",
    "How many degrees are in a right angle?",
    "What language has the most native speakers?",
    "What is the boiling point of water at sea level in celsius?",
    "Which ocean is the deepest?",
]
DEBUG_SNIPPETS = [
    ("Python", "def mean(xs):\n    return sum(xs) / len(xs)", "an empty list"),
    ("Python", "def last(xs):\n    return xs[len(xs)]", "any list"),
    ("Rust", "fn div(a: i32, b: i32) -> i32 { a / b }", "b equal to zero"),
    ("JavaScript", "function inc(x) { return x + '1'; }", "a numeric argument"),
    ("Python", "def get(d, k):\n    return d[k]", "a missing key"),
]

out = []
for i, t in enumerate(CODE_TASKS):
    for lang in (LANGS[i % len(LANGS)], LANGS[(i + 2) % len(LANGS)]):
        out.append(
            f"Write a {lang} function that {t}. Only code, no explanation."
        )
for t in EXPLAIN_TOPICS:
    out.append(f"Explain {t} in two or three sentences.")
    out.append(f"Explain {t} to a beginner in one short paragraph.")
for q in QA:
    out.append(f"{q} Answer in one sentence.")
for lang, snippet, cond in DEBUG_SNIPPETS:
    out.append(
        f"This {lang} code fails on {cond}:\n\n{snippet}\n\nShow the fixed version. Only code, no explanation."
    )
for t in CODE_TASKS[:15]:
    out.append(
        f"Write a Python function that {t}, then add one usage example. Keep it short."
    )

path = sys.argv[1] if len(sys.argv) > 1 else "corpus_self.jsonl"
with open(path, "w") as f:
    for i, p in enumerate(out):
        f.write(json.dumps({"id": f"self_{i:04}", "prompt": p, "source": "self"}) + "\n")
print(f"{len(out)} prompts -> {path}")
