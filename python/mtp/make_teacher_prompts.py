"""Emit the teacher-arm prompt JSONL: wider and harder than the self set,
weighted toward tasks where 31B thinking does real work."""

import json
import sys

LANGS = ["Python", "Rust", "JavaScript", "Go", "C", "TypeScript"]
CODE_TASKS = [
    "parses a duration string like '2h15m30s' into total seconds",
    "computes the median of a list without sorting the original",
    "returns all pairs in a list that sum to a target value",
    "validates an IPv4 address string",
    "implements a bounded LRU cache with get and put",
    "topologically sorts a small dependency graph given as an adjacency map",
    "computes the longest common prefix of a list of strings",
    "converts an integer to its English words form up to 9999",
    "finds the majority element of a list if one exists",
    "implements run-length encoding and decoding",
    "evaluates a postfix arithmetic expression",
    "merges overlapping intervals",
    "returns the k most frequent words in a text",
    "checks if a binary tree given as nested tuples is height-balanced",
    "implements integer square root without floating point",
    "diffs two flat dictionaries into added, removed, changed keys",
    "parses a CSV line respecting quoted fields",
    "computes moving averages over a window of size k",
    "converts snake_case identifiers in a string to camelCase",
    "implements exponentiation by squaring",
    "detects a cycle in a linked list",
    "generates all balanced parentheses strings of length 2n",
    "implements a simple token bucket rate limiter",
    "normalizes a file path without using library path functions",
    "finds the single missing number in a permutation of 0..n",
    "implements binary search returning the insertion point",
    "counts islands in a small 0/1 grid",
    "reverses the bits of a 32-bit integer",
    "computes the edit distance between two short strings",
    "groups anagrams from a list of words",
]
REASONING = [
    "A bat and a ball cost $1.10 together, and the bat costs $1.00 more than the ball. What does the ball cost?",
    "If a clock shows 3:15, what is the angle between the hour and minute hands?",
    "A snail climbs 3 meters each day and slips back 2 meters each night. How many days to climb a 10 meter wall?",
    "You have a 3-liter jug and a 5-liter jug. How do you measure exactly 4 liters?",
    "Three people split a restaurant bill of $75 with a 20% tip. How much does each pay?",
    "A train travels 240 km at 80 km/h, then 120 km at 60 km/h. What is its average speed for the whole trip?",
    "If today is Wednesday, what day of the week is it 100 days from now?",
    "Two dice are rolled. What is the probability the sum is 7 or 11?",
    "Water doubles in a pond every day and fills it on day 30. On which day was it half full?",
    "A rope ladder hangs off a boat with rungs 30 cm apart. The tide rises 90 cm. How many rungs go underwater?",
    "What is 15% of 240, and what is 240 increased by 15%?",
    "A rectangle's length is twice its width and its perimeter is 36. What is its area?",
    "How many trailing zeros does 25 factorial have?",
    "If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?",
    "A car's odometer reads 199,999. How many kilometers until all digits are the same again?",
]
EXPLAIN = [
    "why quicksort's worst case is quadratic and how randomized pivots help",
    "the difference between optimistic and pessimistic locking",
    "how a bloom filter can give false positives but never false negatives",
    "why TCP needs both sequence numbers and acknowledgments",
    "the difference between symmetric and asymmetric encryption",
    "what eventual consistency means for a distributed cache",
    "why tail-call optimization matters for recursive functions",
    "how copy-on-write makes process forking cheap",
    "the difference between row-oriented and column-oriented storage",
    "why floating point equality comparisons are dangerous",
    "how virtual memory lets processes exceed physical RAM",
    "what backpressure is in a streaming system",
    "why database transactions need isolation levels",
    "how consistent hashing reduces rebalancing on node changes",
    "the difference between concurrency and parallelism",
    "why immutable data structures simplify concurrent code",
    "what a memory barrier does and when you need one",
    "how HTTPS certificate verification works at a high level",
    "why premature optimization is discouraged, with one concrete exception",
    "the CAP theorem tradeoffs with a concrete example",
]
TRANSFORMS = [
    "Rewrite this loop as a list comprehension:\n\nresult = []\nfor x in items:\n    if x > 0:\n        result.append(x * 2)",
    "Convert this callback code to async/await:\n\nfetchUser(id, (err, user) => {\n  if (err) return handle(err);\n  fetchPosts(user, (err, posts) => {\n    if (err) return handle(err);\n    render(posts);\n  });\n});",
    "Add error handling to this function:\n\ndef read_config(path):\n    with open(path) as f:\n        return json.load(f)",
    "Write a docstring for this function:\n\ndef window(xs, k):\n    return [xs[i:i+k] for i in range(len(xs) - k + 1)]",
    "Simplify this boolean expression and explain each step:\n\nif not (a == b or not c):",
]

out = []
for i, t in enumerate(CODE_TASKS):
    for lang in (LANGS[i % len(LANGS)], LANGS[(i + 3) % len(LANGS)]):
        out.append(f"Write a {lang} function that {t}. Only code, no explanation.")
    out.append(
        f"Write a Python function that {t}, with a brief comment on the approach."
    )
for q in REASONING:
    out.append(f"{q} Show your reasoning briefly, then give the answer.")
    out.append(f"{q} Answer concisely.")
for t in EXPLAIN:
    out.append(f"Explain {t} in a short paragraph.")
    out.append(f"Explain {t} in exactly three sentences.")
for t in TRANSFORMS:
    out.append(t)

path = sys.argv[1] if len(sys.argv) > 1 else "prompts_teacher.jsonl"
with open(path, "w") as f:
    for i, p in enumerate(out):
        f.write(json.dumps({"id": f"tc{i:04}", "prompt": p}) + "\n")
print(f"{len(out)} prompts -> {path}")
