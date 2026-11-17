# erisllm-remote

Serve a LoRA-adapted Gemma 3 or Gemma 4 12B from an Apple Silicon Mac and use it
from any other machine. The client sends token ids, which are used exactly as given
(no template, no BOS added, nothing changed), and gets back a stream of events:
every generated token with its log-probability and alternatives, and on request
what chosen layers hold, read through the
[Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html) and
decomposed into J-space codes. Sampling settings and the adapter's strength are set
per request. Client and server find each other with [iroh](https://iroh.computer) by
their ids, through NAT, with no ports to open; only clients the server's owner has
allowed get in.

Built for ErisLLM, a persona adapter. The adapter is not included: the server takes
any mlx-lm LoRA adapter for Gemma 3 or Gemma 4 12B.

| Piece | What it is |
|---|---|
| `src/` | the Rust program: `serve` puts the model process behind an iroh endpoint; `request`, `info` and `proxy` are the client |
| `erisllm/engine.py` | the engine: requests, sampling, logprobs, prefix reuse, J-lens and J-space |
| `erisllm/fastlora.py` | the forward pass it runs on: a hand-written Gemma forward (custom Metal kernels, banded sliding-window attention) with a KV cache |
| `erisllm/jlens.py` | finds, downloads and loads pre-fitted Jacobian lenses (reads torch files without torch) |
| `erisllm/worker.py` | the engine as a child process speaking JSON lines on stdin/stdout |
| `examples/client.py` | a standard-library Python client for the proxy |

## Using a server (any machine)

Install Rust ([rustup.rs](https://rustup.rs)), then:

```
cargo install --git https://github.com/Eriskii/erisllm-remote
erisllm-remote id                        # send this to the server's owner once
export ERISLLM_SERVER=<the server's id>  # Windows: set ERISLLM_SERVER=...
erisllm-remote info
erisllm-remote request req.json          # one request, events printed as JSON lines
erisllm-remote proxy                     # local HTTP API on http://127.0.0.1:8799/v1/
```

It runs on macOS, Linux and Windows; the client needs no GPU and no Python. Your
key is kept in the system's config folder (`erisllm-remote id` prints where);
keeping it keeps your id.

The proxy is the flexible way in, from any language:

| Endpoint | In | Out |
|---|---|---|
| `POST /v1/generate` | a generate request (below) | events as NDJSON, streamed; closing the connection cancels |
| `POST /v1/tokenize` | `{"text": "...", "add_special_tokens": false}` | `{"tokens": [...]}` |
| `POST /v1/detokenize` | `{"tokens": [...]}` | `{"text": "...", "pieces": [...]}` |
| `GET /v1/info` | | model, adapter, format, vocabulary size, layers, EOS ids, lens, limits |

```
curl -N localhost:8799/v1/generate -d '{"tokens": [2, 105, 2364, 107], "max_tokens": 50, "logprobs": 5}'
python examples/client.py --jlens 24,32,40
```

On a loopback address the proxy refuses browsers (requests with an `Origin`
header) and unexpected `Host` names, so web pages cannot use it. If iroh's address
lookup is blocked on a network (some firewalls inspect TLS), give the server's
address directly with `--addr IP:PORT` (or `ERISLLM_SERVER_ADDR`); the server then
needs a known port (`serve --port`).

Rust programs can use the client directly: `erisllm_remote::Client::connect(id)`,
then `client.request(&json)` returns the events.

## Running a server (an Apple Silicon Mac)

```
git clone https://github.com/Eriskii/erisllm-remote && cd erisllm-remote
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cargo build --release
./target/release/erisllm-remote serve --adapter PATH/TO/ADAPTER
./target/release/erisllm-remote allow <a client's id> name
```

The adapter folder holds mlx-lm's `adapters.safetensors` and `adapter_config.json`.
The base model is read from that file's `model` (default
`mlx-community/gemma-3-12b-it-qat-4bit`) and downloads on first use, about 8 GB;
set `"format": "gemma4"` there for a Gemma 4 adapter. The server prints its id
(`serving as …`) and answers only clients in its allowlist, which is re-read on
every connection. It needs about 10 GB of memory with the lens.

| Option | |
|---|---|
| `--adapter DIR` | the adapter (default `./adapter`) |
| `--lora-parts all\|mlp\|attention\|none` | where the adapter applies when a request does not say |
| `--jlens auto\|none\|FILE` | `auto` fetches the pre-fitted lens for the base model once (about 1.4 GB) into `data/jlens/` |
| `--model NAME` | another base model than the adapter's own |
| `--port N` | a fixed UDP port |
| `--allow-any` | answer every client (testing only) |

## Generate requests

Only `tokens` is required.

| Field | Default | Meaning |
|---|---|---|
| `tokens` | | prompt token ids, used verbatim. Gemma's tokenizers do not always add `<bos>`; put it in yourself (id 2) |
| `max_tokens` | 256 | 0 runs the prompt only (for `prompt_logprobs`) |
| `temperature` | 1.0 | 0 is greedy |
| `top_p`, `top_k`, `min_p` | 1.0, 0, 0.0 | truncation, applied in mlx-lm's order (top-p, min-p, XTC, top-k) |
| `xtc_probability`, `xtc_threshold` | 0.0, 0.1 | XTC sampling |
| `repetition_penalty`, `presence_penalty`, `frequency_penalty` | off | with `*_context_size` (default 20 tokens) |
| `logit_bias` | | `{"token id": bias}` |
| `seed` | | seeds the sampler for this request |
| `stop_tokens` | the model's EOS ids | ids that end generation (the stop token is still sent); `[]` for none |
| `stop_sequences` | | lists of ids that end generation |
| `lora_strength` | 1.0 | adapter strength: 0 is plain Gemma, above 1 exaggerates, negative inverts (−8 to 8) |
| `lora_parts` | `"all"` | `"mlp"` or `"attention"` keeps the adapter in those matrices only; `"none"` |
| `logprobs` | 0 | how many top alternatives to send with each token (up to 100) |
| `prompt_logprobs` | false | also send an event per prompt token with its log-probability |
| `jlens` | | lens readouts, below |
| `decode` | false | add each token's text next to its id (display only) |
| `cache` | true | reuse the previous request's KV cache for a shared prefix |

`jlens` is an object:

| Field | Default | Meaning |
|---|---|---|
| `layers` | `"all"` | layer indices (0 to 47; negative counts from the end) or `"all"` |
| `mode` | `"jacobian"` | `"logit"` reads layers through the plain logit lens instead |
| `top_k` | 10 | top lens tokens per layer (0 for none, up to 200) |
| `probe` | | token ids whose lens log-probability and rank to report at every layer |
| `jspace_k` | 0 | J-space code size per layer (up to 64; 0 for none) |
| `prompt` | false | also read every prompt position |

## Events

One JSON object per line. Always `start` first, then `progress` while a long
prompt runs, `prompt` events if asked for, `token` events, and `done`; or `error`
at any point.

From a real run (a DM, `"logprobs": 2, "decode": true`, lens on layers 32 and 40,
`jspace_k` 4; lists shortened):

```json
{"event":"start","prompt_tokens":36,"reused_tokens":0}
{"event":"token","index":1,"token":93982,"text":" messed","logprob":-7.2405,"sample_logprob":-7.4685,
 "top":[[2844,-2.9905," working"],[3490,-3.2405," doing"]],
 "jlens":{"top":{"32":[[69706,-0.6081," chatting"],[3978,-1.8581," getting"]],
                 "40":[[11293,-0.7128," eating"],[10685,-1.2128," watching"]]}},
 "jspace":{"40":{"atoms":[[11293,157.728975,0.0237," eating"],[68308,123.368463,0.017," 공부"]],"explained":0.0016}}}
{"event":"done","finish_reason":"stop_token","prompt_tokens":36,"reused_tokens":0,"generated_tokens":15,
 "prefill_seconds":0.298,"decode_seconds":3.251,"decode_tokens_per_second":4.31}
```

(That run is slow because of the J-space pursuit; without it the same request runs at about 20 tokens/s.)

- `logprob` and `top` are the model's own distribution, before penalties,
  temperature and truncation. `sample_logprob` is the token's log-probability
  under the distribution it was actually drawn from (0 when greedy).
- `prompt` events carry `pos`, `token`, `logprob` (none for position 0) and `top`.
- `jlens` and `jspace` on a token read the residual stream at the position that
  predicted it (the position before it), after the named layer. Layer 47, the last,
  reads out exactly the model's own prediction.
- `jlens.top[layer]` is `[[token, log-probability], ...]` under
  `softmax(unembed(norm(J_l h)))`; `jlens.probe[layer]` is `[[log-probability, rank], ...]`
  in the order of `probe`, rank 0 being the top token.
- `jspace[layer].atoms` is `[[token, coefficient, share], ...]`: the residual's nearest
  non-negative combination of `jspace_k` J-lens vectors `v_t = J_lᵀ (g ⊙ u_t)` (u_t the
  token's unembedding row, g the final norm's gain), found by non-negative orthogonal
  matching pursuit; `share` is coefficient × |v_t| / |h|. `explained` is the fraction of
  |h|² the combination accounts for. It is small (well under 1%) because a few huge
  dimensions dominate Gemma's residual norm; the atoms are still the meaningful part.
- `finish_reason`: `stop_token`, `stop_sequence`, `length` or `cancelled`.
- With `decode`, each `[id, value]` pair gains the token's text as a third element.

## The lenses

| Base model | Lens | Fitted on |
|---|---|---|
| Gemma 3 12B (`gemma-3-12b-it-qat-4bit`) | `neuronpedia/jacobian-lens`, `gemma-3-12b-it/jlens/Salesforce-wikitext` | `google/gemma-3-12b-it` in bfloat16, 844 wikitext prompts |
| Gemma 4 12B (`gemma-4-12B-it-qat-OptiQ-4bit`) | `gghfez/gemma-4-it-qat-jacobian-lens` | `google/gemma-4-12B-it-qat-q4_0-unquantized` (CC BY-NC 4.0) |

Both were fitted on the base model, not the adapted one, and the Gemma 3 lens on
the original weights rather than the QAT build used here. They read the adapted
model well enough to be useful, but they are approximations of its own Jacobian.
The Gemma 3 lens is the sharper of the two: on Anthropic's example "the currency
used in the country shaped like a boot is", layers 36 to 44 read "euro", and the
layer-32 J-space code holds "called", "欧元", "Италия" and "करेंसी".

## Performance (M5 Pro, Gemma 3 12B with a rank-32 adapter)

| | |
|---|---|
| decode | 22 tokens/s (the stock mlx-lm path: 18) |
| prefill, 4,096 tokens | 870 tokens/s (stock: 670) |
| decode with the lens on all 48 layers, top 10 | 13 tokens/s |
| decode with J-space, k = 16 on 4 layers | about 3 tokens/s (k rounds of pursuit per token) |
| memory | about 10 GB with the lens, plus the KV cache (about 0.4 MB per token of context) |

One request runs at a time; others wait their turn. Prefix reuse means a
conversation that grows turn by turn only runs its new tokens.

## Correctness checks (all run on the real model)

- The cached forward against a fresh full forward: greedy tokens are the argmax
  of the fresh pass, log-probabilities within 0.07. The banded first chunk, masked
  later chunks, sliding-cache compaction past 3,000 tokens and trimmed reuse all
  agree with the stock mlx-lm forward to bfloat16 noise (Gemma 3: mean |Δ log p|
  0.03; Gemma 4: within the stock path's own cached-versus-uncached difference).
- Requests with the same seed and the same path are bit-identical. A request
  served from the prefix cache computes its first step differently, so a seeded
  sample can differ from the uncached run; send `"cache": false` for exact replays.
- The training kernel's exact gradient checks still pass after the inference
  path was added (1.2e-5 from autograd on the CPU).
- Over iroh, server and client on one machine with a direct address: refusal of
  unknown clients, allow, generate with lens and J-space, the proxy's endpoints,
  and cancellation by hanging up. Connecting by id alone through iroh's address
  lookup and relays (the normal case between two networks) was not tested here.
