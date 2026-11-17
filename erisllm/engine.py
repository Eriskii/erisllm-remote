"""The inference engine behind erisllm-remote: one model, one request at a time.

Runs on the hand-written Gemma forward of `erisllm.fastlora` (its KV-cache inference
path); mlx-lm is only used to load the weights and tokenizer.

A request carries token ids, which are used exactly as given: no chat template, no
BOS, nothing added or changed. For each token it generates, the engine reports the
id, its log-probability, optionally the most likely alternatives, and optionally
what chosen layers hold, read through the Jacobian lens (`erisllm.jlens`) and
decomposed into J-space codes. It can report the same for the prompt's own tokens.

The adapter is applied at a per-request strength (0 is plain Gemma, as in the chat
server), optionally in the MLP or attention matrices only.

The KV cache of the previous request is kept. A request that starts with the same
tokens, at the same adapter setting, only runs its new tokens.

Conventions, for both prompt and generated tokens: a token's `logprob` and `top`
come from the model's own distribution, before penalties, temperature or
truncation; `sample_logprob` is the token's log-probability under the distribution
it was actually drawn from. A token's `jlens` and `jspace` read the residual stream
at the position that predicted it, the position just before it.
"""
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p, apply_xtc, make_logits_processors
from mlx_lm.tuner.lora import LoRALinear

from erisllm import jlens as jlens_lib
from erisllm.fastlora import FastLoRA, _rms

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mlx-community/gemma-3-12b-it-qat-4bit"

MAX_TOKENS = 32768
MAX_LOGPROBS = 100
MAX_LENS_TOP = 200
MAX_PROBES = 64
MAX_JSPACE_K = 64
# Rows of full-vocabulary scores held at once (each row is about 1 MB in float32).
ROW_BLOCK = 256
# Prompt tokens per forward pass: the first chunk (banded attention), and later ones.
FIRST_CHUNK = 4096
LATER_CHUNK = 1024


class RequestError(ValueError):
    """Something wrong with a request; reported to the client, the server carries on."""


def _r(x, digits: int = 4):
    x = float(x)  # numpy and mlx scalars are not JSON
    return round(x, digits) if math.isfinite(x) else None


class Options:
    """A validated generate request."""

    def __init__(self, req: dict, engine: "Engine"):
        def num(key, default, lo, hi, kind=float):
            v = req.get(key, default)
            if v is None:
                return None
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and not isinstance(v, int)):
                raise RequestError(f"{key}: expected {'an integer' if kind is int else 'a number'}")
            if not (lo <= v <= hi) or (isinstance(v, float) and not math.isfinite(v)):
                raise RequestError(f"{key}: must be between {lo} and {hi}")
            return kind(v)

        def ids(key, value):
            if not isinstance(value, list) or not all(type(t) is int and 0 <= t < engine.vocab_size for t in value):
                raise RequestError(f"{key}: expected a list of token ids in [0, {engine.vocab_size})")
            return value

        self.tokens = ids("tokens", req.get("tokens"))
        if not self.tokens:
            raise RequestError("tokens: must not be empty")
        self.max_tokens = num("max_tokens", 256, 0, MAX_TOKENS, int)
        self.temperature = num("temperature", 1.0, 0.0, 100.0)
        self.top_p = num("top_p", 1.0, 0.0, 1.0)
        self.top_k = num("top_k", 0, 0, engine.vocab_size - 1, int)
        self.min_p = num("min_p", 0.0, 0.0, 1.0)
        self.xtc_probability = num("xtc_probability", 0.0, 0.0, 1.0)
        self.xtc_threshold = num("xtc_threshold", 0.1, 0.0, 0.5)
        self.seed = num("seed", None, 0, 2**63 - 1, int)
        self.lora_strength = num("lora_strength", 1.0, -8.0, 8.0)
        self.lora_parts = req.get("lora_parts", engine.lora_parts)
        if self.lora_parts not in ("all", "mlp", "attention", "none"):
            raise RequestError("lora_parts: one of all, mlp, attention, none")
        self.logprobs = num("logprobs", 0, 0, MAX_LOGPROBS, int)
        self.prompt_logprobs = bool(req.get("prompt_logprobs", False))
        self.decode = bool(req.get("decode", False))
        self.reuse_cache = bool(req.get("cache", True))

        stop = req.get("stop_tokens")
        self.stop_tokens = set(engine.eos if stop is None else ids("stop_tokens", stop))
        seqs = req.get("stop_sequences") or []
        if not isinstance(seqs, list):
            raise RequestError("stop_sequences: expected a list of token id lists")
        self.stop_sequences = [ids("stop_sequences", s) for s in seqs if s]

        bias = req.get("logit_bias") or {}
        if not isinstance(bias, dict):
            raise RequestError("logit_bias: expected an object of token id to bias")
        try:
            bias = {int(k): float(v) for k, v in bias.items()}
        except (TypeError, ValueError):
            raise RequestError("logit_bias: keys are token ids, values numbers") from None
        if any(not 0 <= k < engine.vocab_size or not math.isfinite(v) for k, v in bias.items()):
            raise RequestError("logit_bias: token id out of range or bias not finite")
        penalties = {
            key: num(key, None, -100.0 if key != "repetition_penalty" else 0.0, 100.0)
            for key in ("repetition_penalty", "presence_penalty", "frequency_penalty")
        }
        sizes = {key: num(key, 20, 1, 1 << 20, int) for key in ("repetition_context_size", "presence_context_size", "frequency_context_size")}
        self.processors = make_logits_processors(logit_bias=bias or None, **penalties, **sizes)

        # Truncations, in the order mlx-lm's sampler applies them.
        self.filters = []
        if 0 < self.top_p < 1:
            self.filters.append(lambda x, p=self.top_p: apply_top_p(x, p))
        if self.min_p > 0:
            self.filters.append(lambda x, p=self.min_p: apply_min_p(x, p))
        if self.xtc_probability > 0:
            special = sorted(engine.eos)
            self.filters.append(lambda x: apply_xtc(x, self.xtc_probability, self.xtc_threshold, special))
        if self.top_k > 0:
            self.filters.append(lambda x, k=self.top_k: apply_top_k(x, k))

        self.jlens = LensSpec(req["jlens"], engine) if req.get("jlens") else None


class LensSpec:
    """What to read through the lens: which layers, how, and how much of it."""

    def __init__(self, spec: dict, engine: "Engine"):
        if not isinstance(spec, dict):
            raise RequestError("jlens: expected an object")
        self.mode = spec.get("mode", "jacobian")
        if self.mode not in ("jacobian", "logit"):
            raise RequestError("jlens.mode: jacobian or logit")
        final = engine.n_layers - 1
        if self.mode == "jacobian" and engine.lens is None:
            raise RequestError("jlens: no Jacobian lens is loaded on this server; use mode 'logit'")
        readable = set(range(engine.n_layers)) if self.mode == "logit" else set(engine.lens.layers) | {final}
        layers = spec.get("layers", "all")
        if layers == "all":
            layers = sorted(readable)
        if not isinstance(layers, list) or not layers or not all(type(l) is int for l in layers):
            raise RequestError('jlens.layers: a list of layer indices, or "all"')
        layers = [l + engine.n_layers if l < 0 else l for l in layers]
        bad = [l for l in layers if l not in readable]
        if bad:
            raise RequestError(f"jlens.layers: {bad} cannot be read in {self.mode} mode (layers are 0..{final})")
        self.layers = list(dict.fromkeys(layers))
        self.top_k = spec.get("top_k", 10)
        self.probe = spec.get("probe") or []
        self.jspace_k = spec.get("jspace_k", 0)
        for key, value, hi in (("top_k", self.top_k, MAX_LENS_TOP), ("jspace_k", self.jspace_k, MAX_JSPACE_K)):
            if type(value) is not int or not 0 <= value <= hi:
                raise RequestError(f"jlens.{key}: an integer from 0 to {hi}")
        if (not isinstance(self.probe, list) or len(self.probe) > MAX_PROBES
                or not all(type(t) is int and 0 <= t < engine.vocab_size for t in self.probe)):
            raise RequestError(f"jlens.probe: at most {MAX_PROBES} token ids")
        self.prompt = bool(spec.get("prompt", False))


class Engine:
    def __init__(self, adapter: Path, model: str | None = None, lens: str = "auto", cache_limit_gb: float = 2.0,
                 lora_parts: str = "all"):
        if lora_parts not in ("all", "mlp", "attention", "none"):
            raise ValueError(f"lora_parts: one of all, mlp, attention, none (not {lora_parts!r})")
        self.lora_parts = lora_parts  # what a request gets when it does not say
        record = adapter / "adapter_config.json"
        trained = json.loads(record.read_text()) if record and record.exists() else {}
        self.model_name = model or trained.get("model", DEFAULT_MODEL)
        self.adapter = adapter
        self.format = trained.get("format") or "gemma3"

        # Prompts vary in length, so freed buffers are rarely reused; without a cap MLX
        # keeps them all and the process can grow until the machine runs out of memory.
        mx.set_cache_limit(int(cache_limit_gb * 2**30))
        self.model, self.tokenizer = load(self.model_name, adapter_path=str(adapter))
        self.model.eval()
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        mx.set_wired_limit(info["max_recommended_working_set_size"])  # this process is the machine's model server
        self.stream = mx.new_stream(mx.default_device())

        self.fl = FastLoRA(self.model)
        self.lm = self.fl.text
        self.inner = self.fl.lm
        self.n_layers = len(self.fl.layers)
        self.d_model = self.fl.args.hidden_size
        self.dtype = self.inner.embed_tokens(mx.zeros((1,), mx.int32)).dtype
        self.tied = bool(getattr(self.lm, "tie_word_embeddings", False)) or "lm_head" not in self.lm
        self.vocab_size = self.fl.args.vocab_size
        self.eos = sorted(self.tokenizer.eos_token_ids)
        # (module, scale it was trained at, whether it is an attention matrix)
        self.lora = [(m, m.scale, ".self_attn." in f".{name}") for name, m in self.model.named_modules() if isinstance(m, LoRALinear)]
        self.lins, self.lins_key = None, None

        self.lens = jlens_lib.load(lens, self.model_name)
        if self.lens is not None:
            if self.lens.d_model != self.d_model or self.lens.layers[-1] >= self.n_layers:
                raise ValueError(f"lens {self.lens.source} does not fit this model (width {self.lens.d_model}, layers up to {self.lens.layers[-1]})")
        # The final norm's gain: norm(ones) = gain / sqrt(1 + eps).
        self.gain = _rms(self.inner.norm, mx.ones((1, self.d_model), mx.float32)).astype(mx.float32)[0]
        self._atom_norms = {}

        self.cache = None
        self.cache_tokens: list[int] = []
        self.cache_key = None

    # ----- the model, opened up -----

    def _forward(self, ids: mx.array, want=()) -> tuple[mx.array, dict[int, mx.array]]:
        """Runs ids [1, n] through the model and its cache: (residual after the last layer
        [1, n, d], {layer: residual after it [1, n, d]} for each layer in want). Lazy."""
        return self.fl.forward(ids, self.lins, self.cache, frozenset(want))

    def head(self, h: mx.array) -> mx.array:
        """float32 logits from residuals after the last layer: final norm, unembedding, soft cap."""
        return self.fl.logits(h)

    def _unembed(self, x: mx.array) -> mx.array:
        """x @ W_U^T, with no norm and no soft cap."""
        return self.inner.embed_tokens.as_linear(x) if self.tied else self.lm.lm_head(x)

    def _unembed_rows(self, ids: mx.array) -> mx.array:
        """Rows of W_U for these token ids."""
        if self.tied:
            return self.inner.embed_tokens(ids)
        head = self.lm.lm_head
        if hasattr(head, "scales"):
            return mx.dequantize(head.weight[ids], head.scales[ids], head.get("biases")[ids] if "biases" in head else None,
                                 head.group_size, head.bits)
        return head.weight[ids]

    def _set_lora(self, strength: float, parts: str):
        if self.lins_key == (strength, parts):
            return
        for layer, trained, attention in self.lora:
            off = parts == "none" or (parts == "mlp" and attention) or (parts == "attention" and not attention)
            layer.scale = 0.0 if off else trained * strength
        self.lins, self.lins_key = self.fl.adapters(), (strength, parts)

    def _reuse(self, tokens: list[int], key, allowed: bool) -> int:
        """Keeps what the cache holds of this prompt; the number of prompt tokens already in it."""
        if allowed and self.cache is not None and key == self.cache_key:
            common = 0
            for a, b in zip(self.cache_tokens, tokens):
                if a != b:
                    break
                common += 1
            common = min(common, len(tokens) - 1)  # the last prompt token always runs, for its logits
            extra = len(self.cache_tokens) - common
            if common > 0 and all(kv.can_trim(extra) for kv in self.cache):
                for kv in self.cache:
                    kv.trim(extra)
                del self.cache_tokens[common:]
                return common
        self.cache = self.fl.new_cache()
        self.cache_tokens = []
        self.cache_key = key
        return 0

    # ----- requests -----

    def info(self) -> dict:
        return {
            "engine": "fastlora",
            "model": self.model_name,
            "adapter": str(self.adapter.relative_to(ROOT)) if self.adapter.is_relative_to(ROOT) else str(self.adapter),
            "format": self.format,
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "eos_token_ids": self.eos,
            "bos_token_id": self.tokenizer.bos_token_id,
            "lora_layers": len(self.lora),
            "default_lora_parts": self.lora_parts,
            "jlens": None if self.lens is None else {
                "source": self.lens.source,
                "layers": self.lens.layers + [self.n_layers - 1],
                "n_prompts": self.lens.n_prompts,
            },
            "limits": {"max_tokens": MAX_TOKENS, "logprobs": MAX_LOGPROBS, "jlens_top_k": MAX_LENS_TOP,
                       "jlens_probe": MAX_PROBES, "jspace_k": MAX_JSPACE_K},
        }

    def call(self, op: str, req: dict) -> dict:
        """The quick requests, which do not touch the model and can run beside a generation."""
        if op == "info":
            return self.info()
        if op == "tokenize":
            text = req.get("text")
            if not isinstance(text, str):
                raise RequestError("text: expected a string")
            return {"tokens": self.tokenizer.encode(text, add_special_tokens=bool(req.get("add_special_tokens", False)))}
        if op == "detokenize":
            tokens = req.get("tokens")
            if not isinstance(tokens, list) or not all(type(t) is int and 0 <= t < self.vocab_size for t in tokens):
                raise RequestError("tokens: expected a list of token ids")
            return {"text": self.tokenizer.decode(tokens), "pieces": self.tokenizer.convert_ids_to_tokens(tokens)}
        raise RequestError(f"unknown op {op!r}")

    def generate(self, req: dict, cancelled=lambda: False):
        """Events for one request: start, then prompt/progress/token events, then done."""
        o = Options(req, self)
        try:
            with mx.stream(self.stream):
                yield from self._generate(o, cancelled)
        except Exception:
            self.cache = None  # it may hold a half-run step
            raise

    def _text(self, token: int) -> str:
        return self.tokenizer.decode([token])

    def _pairs(self, ids: list, vals: list, decode: bool) -> list:
        if decode:
            return [[i, _r(v), self._text(i)] for i, v in zip(ids, vals)]
        return [[i, _r(v)] for i, v in zip(ids, vals)]

    @staticmethod
    def _top(lp: mx.array, k: int) -> tuple[mx.array, mx.array]:
        """The k largest entries of each row, largest first: (ids, values)."""
        ids = mx.argpartition(-lp, kth=k - 1, axis=-1)[..., :k]
        vals = mx.take_along_axis(lp, ids, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        return mx.take_along_axis(ids, order, axis=-1), mx.take_along_axis(vals, order, axis=-1)

    def _decide(self, h: mx.array, hist, o: Options) -> dict:
        """The next token from the last layer's residual h [1, d], and what is reported about it. Lazy."""
        logits = self.head(h)
        raw = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        proc = logits
        for p in o.processors:
            proc = p(hist, proc)
        if o.temperature == 0:
            y = mx.argmax(proc, axis=-1)
            sample_lp = mx.zeros((1,), mx.float32)
        else:
            lp = proc - mx.logsumexp(proc, axis=-1, keepdims=True)
            for f in o.filters:
                lp = f(lp)
            scaled = lp * (1.0 / o.temperature)
            scaled = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
            y = mx.random.categorical(scaled)
            sample_lp = mx.take_along_axis(scaled, y[:, None], axis=-1)[:, 0]
        out = {"y": y, "lp": mx.take_along_axis(raw, y[:, None], axis=-1)[:, 0], "slp": sample_lp}
        if o.logprobs:
            out["top_ids"], out["top_lps"] = self._top(raw, o.logprobs)
        return out

    def _generate(self, o: Options, cancelled):
        tokens, n = o.tokens, len(o.tokens)
        lens = o.jlens
        prompt_lens = lens is not None and lens.prompt
        prompt_out = o.prompt_logprobs or prompt_lens
        key = (o.lora_strength, o.lora_parts)
        self._set_lora(o.lora_strength, o.lora_parts)
        reused = self._reuse(tokens, key, o.reuse_cache and not prompt_out)
        if o.seed is not None:
            mx.random.seed(o.seed)
        yield {"event": "start", "prompt_tokens": n, "reused_tokens": reused}
        if prompt_out:
            yield {"event": "prompt", "pos": 0, "token": tokens[0], **({"text": self._text(tokens[0])} if o.decode else {})}

        started = time.perf_counter()
        gen_layers = lens.layers if lens else []
        feed = tokens[reused:]
        last_h, last_rows = None, {}
        start = 0
        while start < len(feed):
            if cancelled():
                yield self._done("cancelled", n, reused, 0, started, None)
                return
            step = FIRST_CHUNK if self.cache_tokens == [] else LATER_CHUNK
            chunk = feed[start:start + step]
            final = start + len(chunk) >= len(feed)
            want = (set(gen_layers) if prompt_lens or final else set())
            h, taps = self._forward(mx.array(chunk)[None], want)
            self.cache_tokens.extend(chunk)
            if prompt_out:
                yield from self._prompt_events(h, taps, tokens, reused + start, len(chunk), o)
            if final:
                last_h = h[:, -1, :]
                last_rows = {l: taps[l][0, -1:] for l in gen_layers}
            else:
                mx.eval([kv.state for kv in self.cache])
            mx.clear_cache()
            start += len(chunk)
            if len(feed) > len(chunk) or start < len(feed):
                yield {"event": "progress", "processed": reused + start, "total": n}

        if o.max_tokens == 0:
            mx.eval([kv.state for kv in self.cache])
            yield self._done("length", n, reused, 0, started, None)
            return

        hist = mx.array(tokens) if o.processors else None
        cur = self._decide(last_h, hist, o)
        cur_lens = self._lens_build(last_rows, lens) if lens and (lens.top_k or lens.probe) else None
        mx.async_eval(list(cur.values()) + self._arrays(cur_lens))
        cur_rows = last_rows
        generated: list[int] = []
        first_at = None
        while True:
            fed = False
            if len(generated) + 1 < o.max_tokens and not cancelled():
                y = cur["y"]
                h, taps = self._forward(y[None], gen_layers)
                if hist is not None:
                    hist = mx.concatenate([hist, y.astype(hist.dtype)])
                nxt = self._decide(h[:, -1, :], hist, o)
                nxt_rows = {l: taps[l][0, -1:] for l in gen_layers}
                nxt_lens = self._lens_build(nxt_rows, lens) if cur_lens is not None else None
                mx.async_eval(list(nxt.values()) + self._arrays(nxt_lens))
                fed = True
            tok = cur["y"].item()
            if first_at is None:
                first_at = time.perf_counter()
            if fed:
                self.cache_tokens.append(tok)
            event = {"event": "token", "index": len(generated), "token": tok,
                     "logprob": _r(cur["lp"].item()), "sample_logprob": _r(cur["slp"].item())}
            if o.decode:
                event["text"] = self._text(tok)
            if o.logprobs:
                event["top"] = self._pairs(cur["top_ids"][0].tolist(), cur["top_lps"][0].tolist(), o.decode)
            if cur_lens is not None:
                event["jlens"] = self._lens_collect(cur_lens, o.decode)[0]
            if lens and lens.jspace_k:
                event["jspace"] = self._jspace(cur_rows, lens, o.decode)[0]
            generated.append(tok)
            yield event

            if tok in o.stop_tokens:
                reason = "stop_token"
            elif any(generated[-len(s):] == s for s in o.stop_sequences):
                reason = "stop_sequence"
            elif len(generated) >= o.max_tokens:
                reason = "length"
            elif not fed:
                reason = "cancelled"
            else:
                cur, cur_lens, cur_rows = nxt, nxt_lens, nxt_rows
                continue
            yield self._done(reason, n, reused, len(generated), started, first_at)
            return

    def _done(self, reason, n, reused, generated, started, first_at):
        now = time.perf_counter()
        prefill = (first_at or now) - started
        decode = now - first_at if first_at else 0.0
        return {"event": "done", "finish_reason": reason, "prompt_tokens": n, "reused_tokens": reused,
                "generated_tokens": generated, "prefill_seconds": round(prefill, 3), "decode_seconds": round(decode, 3),
                "decode_tokens_per_second": round((generated - 1) / decode, 2) if generated > 1 and decode > 0 else None}

    def _prompt_events(self, h, taps, tokens, base, c, o: Options):
        """Events for the prompt tokens that positions base..base+c-1 predict."""
        m = min(c, len(tokens) - 1 - base)
        if m <= 0:
            return
        lens = o.jlens if o.jlens and o.jlens.prompt else None
        chosen = top = None
        if o.prompt_logprobs:
            chosen, top = [], ([], []) if o.logprobs else None
            for a in range(0, m, ROW_BLOCK):
                b = min(a + ROW_BLOCK, m)
                logits = self.head(h[0, a:b])
                lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                targets = mx.array(tokens[base + 1 + a:base + 1 + b])
                picked = mx.take_along_axis(lp, targets[:, None], axis=-1)[:, 0]
                arrays = [picked]
                if o.logprobs:
                    arrays += list(self._top(lp, o.logprobs))
                mx.eval(arrays)
                chosen += picked.tolist()
                if o.logprobs:
                    top[0].extend(arrays[1].tolist())
                    top[1].extend(arrays[2].tolist())
                del logits, lp
        rows = {l: taps[l][0, :m] for l in lens.layers} if lens else {}
        readouts = self._lens_collect(self._lens_build(rows, lens, eager=True), o.decode) if lens and (lens.top_k or lens.probe) else None
        codes = self._jspace(rows, lens, o.decode) if lens and lens.jspace_k else None
        for j in range(m):
            pos = base + 1 + j
            event = {"event": "prompt", "pos": pos, "token": tokens[pos]}
            if o.decode:
                event["text"] = self._text(tokens[pos])
            if chosen is not None:
                event["logprob"] = _r(chosen[j])
                if top:
                    event["top"] = self._pairs(top[0][j], top[1][j], o.decode)
            if readouts:
                event["jlens"] = readouts[j]
            if codes:
                event["jspace"] = codes[j]
            yield event

    # ----- Jacobian lens and J-space -----

    def _J(self, layer: int, mode: str):
        """J_l, or None where the transport is the identity (the last layer, or the logit lens)."""
        if mode == "logit" or layer == self.n_layers - 1:
            return None
        return self.lens.J[layer]

    def _transport(self, x: mx.array, layer: int, mode: str) -> mx.array:
        J = self._J(layer, mode)
        return x if J is None else x.astype(mx.float32) @ J.T

    @staticmethod
    def _arrays(plan) -> list:
        return [] if plan is None else [a for block in plan[2] for a in block.values()]

    def _lens_build(self, rows: dict[int, mx.array], spec: LensSpec, eager: bool = False):
        """Lens readouts of rows {layer: [m, d]}: (layers, m, blocks of lazy arrays)."""
        layers = spec.layers
        m = rows[layers[0]].shape[0]
        x = mx.concatenate([self._transport(rows[l], l, spec.mode).astype(self.dtype) for l in layers])
        probe = mx.array(spec.probe) if spec.probe else None
        blocks = []
        for a in range(0, x.shape[0], ROW_BLOCK):
            logits = self.head(x[a:a + ROW_BLOCK])
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            block = {}
            if spec.top_k:
                block["ids"], block["vals"] = self._top(lp, spec.top_k)
            if probe is not None:
                pv = lp[:, probe]
                block["probe"] = pv
                block["rank"] = mx.stack([(lp > pv[:, j:j + 1]).sum(axis=-1) for j in range(len(spec.probe))], axis=-1)
            if eager:
                mx.eval(list(block.values()))
            blocks.append(block)
        return layers, m, blocks

    def _lens_collect(self, plan, decode: bool) -> list[dict]:
        """Per position: {"top": {layer: [[id, logprob], ...]}, "probe": {layer: [[logprob, rank], ...]}}."""
        layers, m, blocks = plan
        lists = {key: [row for block in blocks for row in block[key].tolist()] for key in blocks[0]}
        out = []
        for j in range(m):
            entry = {}
            if "ids" in lists:
                entry["top"] = {str(l): self._pairs(lists["ids"][i * m + j], lists["vals"][i * m + j], decode) for i, l in enumerate(layers)}
            if "probe" in lists:
                entry["probe"] = {str(l): [[_r(v), r] for v, r in zip(lists["probe"][i * m + j], lists["rank"][i * m + j])]
                                  for i, l in enumerate(layers)}
            out.append(entry)
        return out

    def _atom_norm(self, layer: int, mode: str) -> mx.array:
        """|v_t| for every token's J-lens vector v_t = J_l^T (g * u_t) at this layer (computed once)."""
        key = (layer, mode)
        if key not in self._atom_norms:
            J = self._J(layer, mode)
            rows = self.inner.embed_tokens.weight.shape[0] if self.tied else self.lm.lm_head.weight.shape[0]
            parts = []
            for a in range(0, rows, 8192):
                v = self._unembed_rows(mx.arange(a, min(a + 8192, rows))).astype(mx.float32) * self.gain
                if J is not None:
                    v = v @ J
                parts.append(mx.sqrt((v * v).sum(axis=-1)))
                mx.eval(parts[-1])
            self._atom_norms[key] = mx.maximum(mx.concatenate(parts), 1e-6)
        return self._atom_norms[key]

    def _jspace(self, rows: dict[int, mx.array], spec: LensSpec, decode: bool) -> list[dict]:
        """Per position, {layer: {"atoms": [[token, coef, share], ...], "explained": x}}: the nearest
        sparse non-negative combination of spec.jspace_k J-lens vectors to each residual."""
        layers = spec.layers
        m = rows[layers[0]].shape[0]
        out = [{} for _ in range(m)]
        # Rows run layer by layer; blocks of them are pursued together.
        flat = [(l, j) for l in layers for j in range(m)]
        for a in range(0, len(flat), ROW_BLOCK):
            part = flat[a:a + ROW_BLOCK]
            segments = []
            for i, (l, _) in enumerate(part):
                if segments and segments[-1][0] == l:
                    segments[-1][2] = i + 1
                else:
                    segments.append([l, i, i + 1])
            H = mx.concatenate([rows[l][part[s][1]:part[e - 1][1] + 1] for l, s, e in segments]).astype(mx.float32)
            for (l, j), result in zip(part, self._pursue(H, segments, spec.jspace_k, spec.mode, decode)):
                out[j][str(l)] = result
        return out

    def _pursue(self, H: mx.array, segments, k: int, mode: str, decode: bool) -> list[dict]:
        """Non-negative orthogonal matching pursuit of the rows of H [R, d] over their layers' J-lens
        vectors: pick the vector best aligned with what is left, refit all coefficients >= 0, repeat."""
        R = H.shape[0]
        g = self.gain

        def per_layer(X, fn):
            return mx.concatenate([fn(X[s:e], l) for l, s, e in segments])

        def toward_vocab(r, l):  # v_t . r for every t, divided by |v_t|
            J = self._J(l, mode)
            w = (r if J is None else r @ J.T) * g
            return self._unembed(w.astype(self.dtype)).astype(mx.float32) / self._atom_norm(l, mode)

        def atoms(t, l):  # v_t = J_l^T (g * u_t) as rows
            J = self._J(l, mode)
            u = self._unembed_rows(t).astype(mx.float32) * g
            return u if J is None else u @ J

        r = H
        chosen, vectors = [], []
        alive = np.ones(R, bool)
        coef = np.zeros((R, 0))
        for _ in range(k):
            scores = per_layer(r, toward_vocab)
            if chosen:
                scores = mx.put_along_axis(scores, mx.array(np.stack(chosen, 1)), mx.array(-np.inf, mx.float32), axis=-1)
            best = mx.argmax(scores, axis=-1)
            gain = mx.take_along_axis(scores, best[:, None], axis=-1)[:, 0]
            mx.eval(best, gain)
            alive &= np.array(gain) > 0
            if not alive.any():
                break
            t = np.array(best)
            chosen.append(t)
            v = per_layer(mx.array(t), atoms) * mx.array(alive[:, None].astype(np.float32))
            vectors.append(v)
            A = mx.stack(vectors, axis=1)  # [R, i, d]
            G = A @ A.transpose(0, 2, 1)
            b = (A @ H[:, :, None])[:, :, 0]
            mx.eval(G, b)
            coef = _nnls(np.array(G, np.float64), np.array(b, np.float64))
            r = H - (mx.array(coef.astype(np.float32))[:, None, :] @ A)[:, 0, :]
        h2 = (H * H).sum(axis=-1)
        r2 = (r * r).sum(axis=-1)
        mx.eval(h2, r2)
        h2, r2 = np.array(h2), np.array(r2)
        norms = {l: self._atom_norm(l, mode) for l, _, _ in segments}
        layer_of = [l for l, s, e in segments for _ in range(s, e)]
        results = []
        for i in range(R):
            picked = [(int(chosen[j][i]), float(coef[i, j])) for j in range(len(chosen)) if coef[i, j] > 0]
            picked.sort(key=lambda p: -p[1])
            size = math.sqrt(h2[i]) or 1.0
            sizes = norms[layer_of[i]][mx.array([p[0] for p in picked])].tolist() if picked else []
            entries = []
            for (tok, c), vnorm in zip(picked, sizes):
                entry = [tok, _r(c, 6), _r(c * vnorm / size)]
                if decode:
                    entry.append(self._text(tok))
                entries.append(entry)
            results.append({"atoms": entries, "explained": _r(1.0 - r2[i] / h2[i]) if h2[i] > 0 else None})
        return results


def _nnls(G: np.ndarray, b: np.ndarray, iters: int = 300) -> np.ndarray:
    """argmin_{c >= 0} c^T G c - 2 b^T c for each row of a batch: G [R, k, k], b [R, k]."""
    R, k, _ = G.shape
    ridge = 1e-9 * np.trace(G, axis1=1, axis2=2)[:, None, None] * np.eye(k) + 1e-12 * np.eye(k)
    try:
        exact = np.linalg.solve(G + ridge, b[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        exact = np.full((R, k), -1.0)
    ok = (exact >= 0).all(axis=1)
    c = np.where(ok[:, None], exact, np.maximum(exact, 0) if np.isfinite(exact).all() else 0.0)
    todo = ~ok
    if todo.any():  # accelerated projected gradient where the unconstrained answer goes negative
        Gt, bt, x = G[todo], b[todo], c[todo]
        step = 1.0 / np.maximum(np.linalg.eigvalsh(Gt)[:, -1], 1e-30)[:, None]
        y, t = x.copy(), 1.0
        for _ in range(iters):
            grad = (Gt @ y[:, :, None])[:, :, 0] - bt
            x_new = np.maximum(y - step * grad, 0.0)
            t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
            y = x_new + ((t - 1) / t_new) * (x_new - x)
            x, t = x_new, t_new
        c[todo] = x
    return c
