#!/usr/bin/env python3
"""OpenCode simulacrum: drive serve through the regex-lite repair loop with
real write/bash tools, reproducing the session KV lifecycle (multi-turn
reuse + finalize + extends). Usage: oc_sim.py <port> <outdir> [seed]"""
import json, re, subprocess, sys, urllib.request

PORT = sys.argv[1]
OUT = sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else None
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
S = "/private/tmp/claude-501/-Users-matt-Documents-github-diffgemma-mps/2a74c759-e315-41fb-a746-4a68f5d5ca7d/scratchpad"

SYSTEM = open(f"{S}/oc_system.txt").read()
TASK = ('Write a test Rust file to /tmp/ where we write a function for "regex-lite" '
        'along with a test harness and run it. It should accept "*" for repetition, '
        'and "." for wildcards. No external crates and a dozen test cases.')

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Executes a given bash command in a persistent shell session with optional timeout.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command to execute"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "write", "description": "Writes content to a file, creating it if needed.", "parameters": {"type": "object", "properties": {"content": {"type": "string", "description": "File content"}, "filePath": {"type": "string", "description": "Absolute path"}}, "required": ["content", "filePath"]}}},
    {"type": "function", "function": {"name": "read", "description": "Reads a file and returns its contents.", "parameters": {"type": "object", "properties": {"filePath": {"type": "string", "description": "Absolute path"}}, "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Replaces oldString with newString in a file.", "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"}}, "required": ["filePath", "oldString", "newString"]}}},
]

ALLOWED = re.compile(r"^\s*(rustc\b|/tmp/\S+$|cd /tmp && )")

def run_tool(name, args):
    if name == "write":
        path = args.get("filePath", "/tmp/oc_sim_out.rs")
        if not path.startswith("/tmp/"):
            return "error: only /tmp paths allowed"
        open(path, "w").write(args.get("content", ""))
        return "Wrote file successfully."
    if name == "read":
        path = args.get("filePath", "")
        try:
            return open(path).read()[:4000]
        except OSError as e:
            return f"error: {e}"
    if name == "edit":
        path = args.get("filePath", "")
        try:
            t = open(path).read()
            if args.get("oldString", "") not in t:
                return "error: oldString not found"
            open(path, "w").write(t.replace(args["oldString"], args.get("newString", ""), 1))
            return "Edited file successfully."
        except OSError as e:
            return f"error: {e}"
    if name == "bash":
        cmd = args.get("command", "")
        if not ALLOWED.match(cmd):
            return "error: command not allowed in sim"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120, cwd="/tmp")
        out = (r.stdout + r.stderr)[:2000]
        return out or f"(exit {r.returncode})"
    return "error: unknown tool"

def post(payload):
    req = urllib.request.Request(URL, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)

log = open(f"{OUT}/sim_transcript.jsonl", "w")
messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
for turn in range(1, 12):
    payload = {"model": "diffusiongemma", "messages": messages, "tools": TOOLS, "max_tokens": 4096}
    if SEED is not None:
        payload["seed"] = SEED + turn
    d = post(payload)
    msg = d["choices"][0]["message"]
    fin = d["choices"][0]["finish_reason"]
    usage = d.get("usage", {})
    content = msg.get("content") or ""
    calls = msg.get("tool_calls") or []
    print(f"turn {turn}: finish={fin} prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} calls={[c['function']['name'] for c in calls]} content[:80]={content[:80]!r}", flush=True)
    log.write(json.dumps({"turn": turn, "finish": fin, "usage": usage, "msg": msg}) + "\n")
    log.flush()
    messages.append({k: v for k, v in msg.items() if v})
    if not calls:
        # Real-user behavior: if the model stops while the last tool result
        # still showed an error/failure, nudge it to continue the repair loop.
        last_tool = next((m["content"] for m in reversed(messages) if m.get("role") == "tool"), "")
        if re.search(r"error|FAIL|Passed \d+/", last_tool) and turn < 10:
            print("assistant stopped with failing state; nudging", flush=True)
            messages.append({"role": "user", "content": "The tests still fail. Please fix the code and re-run until all tests pass."})
            continue
        print("assistant final reply; ending loop", flush=True)
        break
    for c in calls:
        try:
            args = json.loads(c["function"]["arguments"])
        except json.JSONDecodeError:
            args = {}
        result = run_tool(c["function"]["name"], args)
        print(f"   tool {c['function']['name']} -> {result[:100]!r}", flush=True)
        messages.append({"role": "tool", "tool_call_id": c.get("id", "call_0"), "content": result})
print("sim done", flush=True)
