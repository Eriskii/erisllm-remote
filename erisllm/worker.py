#!/usr/bin/env python3
"""The engine as a child process of `erisllm-remote serve`.

Usage: .venv/bin/python -m erisllm.worker [--adapter ADAPTER_FOLDER] [--jlens auto|none|PATH]

Requests arrive on stdin and events leave on stdout, one JSON object per line. Every
request carries an "id" that the events answering it repeat. A request's "op" is
"generate" (the default), "info", "tokenize", "detokenize", or "cancel" (stops the
generation with that id). Generations run one at a time on the thread that loaded
the model; the quick ops are answered at once, even while one runs. A generation
ends with a "done" or "error" event, a quick op with a "result" or "error" event.
The first line out is a "ready" event carrying what "info" returns.
"""
import argparse
import json
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="base model (default: the one the adapter was trained on)")
    ap.add_argument("--adapter", type=Path, default=ROOT / "adapter", help="adapter folder (default ./adapter; plain Gemma is lora_strength 0)")
    ap.add_argument("--jlens", default="auto", help='"auto" (the known lens for the base model), "none", or a .pt/.safetensors file')
    ap.add_argument("--cache-gb", type=float, default=2.0, help="MLX buffer cache cap")
    ap.add_argument("--lora-parts", default="all", choices=["all", "mlp", "attention", "none"],
                    help="where the adapter applies when a request does not say")
    args = ap.parse_args()
    if not (args.adapter / "adapters.safetensors").exists():
        sys.exit(f"no adapter at {args.adapter}: pass --adapter FOLDER (it holds adapters.safetensors and adapter_config.json)")

    # stdout belongs to the protocol. Anything a library prints goes to stderr instead.
    out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def send(obj):
        line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        with lock:
            out.write(line + "\n")
            out.flush()

    from erisllm.engine import Engine, RequestError

    engine = Engine(args.adapter, args.model, args.jlens, args.cache_gb, args.lora_parts)
    send({"event": "ready", **engine.info()})

    jobs: queue.Queue = queue.Queue()
    cancels: dict = {}

    def reader():
        for line in iter(sys.stdin.buffer.readline, b""):
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise ValueError("a request is a JSON object")
            except ValueError as e:
                send({"id": None, "event": "error", "message": f"bad request: {e}"})
                continue
            rid, op = req.get("id"), req.get("op", "generate")
            if op == "cancel":
                flag = cancels.get(rid)
                if flag:
                    flag.set()
            elif op == "generate":
                cancels[rid] = threading.Event()
                jobs.put(req)
            else:
                try:
                    send({"id": rid, "event": "result", **engine.call(op, req)})
                except Exception as e:
                    send({"id": rid, "event": "error", "message": str(e) if isinstance(e, RequestError) else f"{type(e).__name__}: {e}"})
        jobs.put(None)  # stdin closed: the server is gone

    threading.Thread(target=reader, daemon=True).start()
    while (req := jobs.get()) is not None:
        rid = req.get("id")
        flag = cancels.get(rid) or threading.Event()
        try:
            for event in engine.generate(req, flag.is_set):
                send({"id": rid, **event})
        except RequestError as e:
            send({"id": rid, "event": "error", "message": str(e)})
        except Exception as e:
            traceback.print_exc()
            send({"id": rid, "event": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            cancels.pop(rid, None)


if __name__ == "__main__":
    main()
