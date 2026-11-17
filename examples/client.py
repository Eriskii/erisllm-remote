#!/usr/bin/env python3
"""A minimal client for erisllm-remote's local HTTP API (standard library only).

Start the proxy first, which connects to the server over iroh:

    erisllm-remote proxy --server SERVER_ID          # http://127.0.0.1:8799

Then:

    python client.py                                  # a sample DM, streamed with logprobs
    python client.py --jlens 24,32,40                 # also what those layers hold, per token

From your own code: `generate(tokens, **params)` yields the server's events as dicts;
`tokenize(text)` and `detokenize(tokens)` are there for convenience. The server never
changes your tokens: build the prompt, BOS included, exactly as the adapter expects it.
"""
import argparse
import datetime
import json
import urllib.request

API = "http://127.0.0.1:8799/v1"


def _post(path, body):
    req = urllib.request.Request(f"{API}/{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req)


def generate(tokens, **params):
    """Events for one generation, as they arrive. Stop iterating to cancel it."""
    with _post("generate", {"tokens": tokens, **params}) as response:
        for line in response:
            yield json.loads(line)


def tokenize(text, add_special_tokens=False):
    with _post("tokenize", {"text": text, "add_special_tokens": add_special_tokens}) as response:
        return json.load(response)["tokens"]


def detokenize(tokens):
    with _post("detokenize", {"tokens": tokens}) as response:
        return json.load(response)["text"]


def info():
    with urllib.request.urlopen(f"{API}/info") as response:
        return json.load(response)


def sample_prompt(fmt, name="alice", text="hey what are you up to"):
    """A one-message DM in the adapter's training layout (see its README)."""
    header = f"[Discord DM with {name} · {datetime.date.today()}]"
    if fmt == "gemma4":
        return f"<bos><|turn>system\n{header}<turn|>\n<|turn>user\n{name}: {text}<turn|>\n<|turn>model\n"
    return f"<bos><start_of_turn>user\n{header}\n{name}: {text}<end_of_turn>\n<start_of_turn>model\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jlens", help="comma-separated layers to read through the Jacobian lens, e.g. 24,32,40")
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()

    server = info()
    print(f"{server['model']} + {server['adapter']} ({server['engine']})")
    tokens = tokenize(sample_prompt(server["format"]))
    params = dict(max_tokens=120, temperature=0.9, top_p=0.95, logprobs=3, decode=True, lora_strength=args.strength)
    if args.jlens:
        params["jlens"] = {"layers": [int(l) for l in args.jlens.split(",")], "top_k": 5}
    for event in generate(tokens, **params):
        if event["event"] == "token":
            alternatives = ", ".join(f"{t[2]!r} {t[1]:.2f}" for t in event["top"])
            print(f"{event['text']!r:>18} {event['logprob']:7.3f}   [{alternatives}]")
            for layer, top in event.get("jlens", {}).get("top", {}).items():
                print(f"{'':>20}L{layer}: {' '.join(repr(t[2]) for t in top)}")
        elif event["event"] in ("done", "error"):
            print(event)
