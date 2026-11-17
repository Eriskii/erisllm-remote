"""Hand-scheduled LoRA forward/backward for Gemma 3 and Gemma 4 on MLX.

The stock mlx-lm trainer runs autograd over the whole network with per-layer
gradient checkpointing. On a 4-bit 12B model at 4k tokens that spends more than
half of each step outside the big matmuls. This module replaces it with an
explicit per-layer forward and backward that:

  * keeps only the activations the backward pass needs (q/k/v, the MLP gate and
    up outputs, and the residual stream), so most or all layers fit in memory
    without re-running their forward pass;
  * backpropagates through frozen quantized weights with a transient dense
    dequantize + matmul, which is faster than the transposed quantized matmul;
  * runs the LoRA path in the activation dtype with a fused addmm;
  * fuses GELU(gate) * up and its VJP into single Metal kernels;
  * computes sliding-window attention on a band instead of a masked T x T
    matrix, and recomputes attention probabilities in the backward pass
    instead of storing them;
  * computes logits only at positions that carry loss.

It also runs inference (`new_cache`, `adapters`, `forward`): the same per-layer
forward, extending a KV cache, with the banded kernel for a long first chunk.

Gradients match autograd on the stock model to float32 rounding on the CPU. On
the GPU, MLX's matmul itself carries ~1e-3 relative error, so the two paths only
agree to within the stock path's own noise there (see scripts/check_fastlora.py).
LoRA dropout and DoRA are not supported.

Gemma 4 (the dense text models; no MoE block, per-layer inputs or shared KV layers)
differs from Gemma 3 in ways handled per layer: its norms store the gain directly,
values are normalised, full-attention layers have wider heads and take their values
from the key projection, each layer's output is multiplied by a scalar, and the
final logits are soft-capped.
"""

from __future__ import annotations

import math
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear

_GA = math.sqrt(2.0 / math.pi)
_GB = 0.044715

_LIN_NAMES = ("q", "k", "v", "o", "gate", "up", "down")


@mx.compile
def _act(g, u):
    x = g.astype(mx.float32)
    t = mx.tanh(_GA * (x + _GB * x * x * x))
    return (0.5 * x * (1 + t) * u.astype(mx.float32)).astype(g.dtype)


_act_vjp_kernel = mx.fast.metal_kernel(
    name="gelu_mul_vjp",
    input_names=["g", "u", "c"],
    output_names=["p", "dg", "du"],
    source=f"""
        uint i = thread_position_in_grid.x;
        float x = float(g[i]);
        float uu = float(u[i]);
        float cc = float(c[i]);
        float x2 = x * x;
        float t = metal::precise::tanh({_GA:.9f}f * (x + {_GB:.9f}f * x2 * x));
        float gel = 0.5f * x * (1.0f + t);
        float dgel = 0.5f * (1.0f + t)
            + 0.5f * x * (1.0f - t * t) * {_GA:.9f}f * (1.0f + {3 * _GB:.9f}f * x2);
        p[i] = T(gel * uu);
        dg[i] = T(cc * uu * dgel);
        du[i] = T(cc * gel);
    """,
)


def _act_vjp_cpu(g, u, c):
    x, uu, cc = g.astype(mx.float32), u.astype(mx.float32), c.astype(mx.float32)
    x2 = x * x
    t = mx.tanh(_GA * (x + _GB * x2 * x))
    gel = 0.5 * x * (1 + t)
    dgel = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * _GA * (1 + 3 * _GB * x2)
    return [a.astype(g.dtype) for a in (gel * uu, cc * uu * dgel, cc * gel)]


def _act_vjp(g, u, c):
    """Returns (gelu(g) * u, d/dg, d/du) for cotangent c in one pass."""
    if mx.default_device() != mx.gpu:
        return _act_vjp_cpu(g, u, c)
    return _act_vjp_kernel(
        inputs=[g, u, c],
        template=[("T", g.dtype)],
        grid=(g.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[g.shape] * 3,
        output_dtypes=[g.dtype] * 3,
    )


# Sliding-window attention on a band. Scores arrive as [B * nC, kv_heads, R * C, S]:
# nC chunks of C queries (the R query heads that share a kv head are stacked along
# the row axis), each against the S = C + W - 1 keys its window can reach. In row r
# of chunk c the valid keys are t in [max(r, W - 1 - c * C), r + W - 1]; the lower
# bound also excludes the left padding of the first chunks.
#
# Probabilities are never materialised. The kernel writes e = exp(s - max) (zero
# outside the band) and the row sums z; p = e / z is applied to the small arrays on
# either side of the matmuls that consume it.
_band_exp_kernel = mx.fast.metal_kernel(
    name="band_exp",
    input_names=["s"],
    output_names=["e", "z"],
    source="""
        uint row = thread_position_in_grid.x;
        uint S = s_shape[3];
        uint RC = s_shape[2];
        uint r = (row % RC) % C;
        uint c = (row / (s_shape[1] * RC)) % NC;
        int lo = int(W) - 1 - int(c * C);
        uint a = lo > int(r) ? uint(lo) : r;
        uint b = r + W - 1;
        size_t base = size_t(row) * S;
        float m = -3.0e38f;
        for (uint t = a; t <= b; t++) m = metal::max(m, float(s[base + t]));
        float zz = 0.0f;
        for (uint t = 0; t < S; t++) {
            float ev = (t >= a && t <= b) ? metal::exp(float(s[base + t]) - m) : 0.0f;
            zz += ev;
            e[base + t] = T(ev);
        }
        z[row] = zz;
    """,
)

# ds = p * (dp - sum_t p * dp) with p = e * invz and the row sum passed in as `dot`.
_band_ds_kernel = mx.fast.metal_kernel(
    name="band_ds",
    input_names=["e", "dp", "invz", "dot"],
    output_names=["ds"],
    source="""
        uint i = thread_position_in_grid.x;
        uint row = i / e_shape[3];
        ds[i] = T(float(e[i]) * invz[row] * (float(dp[i]) - dot[row]));
    """,
)


def _band_exp(s, C, W, nC, dtype, mask):
    """Banded exp(s - rowmax) in ``dtype`` and its float32 row sums [..., 1]."""
    if mx.default_device() != mx.gpu:
        s = mx.where(mask, s.astype(mx.float32), -mx.inf)
        e = mx.exp(s - s.max(-1, keepdims=True))
        return e.astype(dtype), e.sum(-1, keepdims=True)
    return _band_exp_kernel(
        inputs=[s], template=[("T", dtype), ("C", C), ("W", W), ("NC", nC)],
        grid=(s.size // s.shape[-1], 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[s.shape, s.shape[:-1] + (1,)], output_dtypes=[dtype, mx.float32],
    )


def _band_ds(e, dp, invz, dot):
    if mx.default_device() != mx.gpu:
        return (e.astype(mx.float32) * invz * (dp.astype(mx.float32) - dot)).astype(e.dtype)
    return _band_ds_kernel(
        inputs=[e, dp, invz, dot], template=[("T", e.dtype)],
        grid=(e.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[e.shape], output_dtypes=[e.dtype],
    )[0]


def _rms(norm, x):
    # Gemma 3's norm stores its gain as an offset from one; Gemma 4 uses MLX's RMSNorm.
    w = norm.weight if isinstance(norm, nn.RMSNorm) else 1.0 + norm.weight
    return mx.fast.rms_norm(x, w, norm.eps)


def _rms_vjp(norm, x, c):
    return mx.vjp(lambda x_: _rms(norm, x_), [x], [c])[1][0]


def _flat(x):
    return x.reshape(-1, x.shape[-1])


class _Lin:
    """A frozen (quantized or dense) linear plus, if it is adapted, its LoRA
    factors cast to the activation dtype."""

    def __init__(self, module, dtype):
        # A scale of exactly 0 is handled as "no adapter": mx.addmm(alpha=0.0) returns
        # c + a @ b instead of c (MLX 0.32.4), so it must never be called with alpha 0.
        if isinstance(module, LoRALinear) and float(module.scale) != 0.0:
            self.base = module.linear
            self.a = module.lora_a.astype(dtype)
            self.b = module.lora_b.astype(dtype)
            self.s = float(module.scale)
        else:
            self.base = module.linear if isinstance(module, LoRALinear) else module
            self.a, self.b, self.s = None, None, 0.0
        if "bias" in self.base:
            raise NotImplementedError("linear layers with bias are not supported")

    def _mm(self, x):
        w = self.base
        if not isinstance(w, nn.QuantizedLinear):
            return x @ w.weight.T
        return mx.quantized_matmul(
            x, w.weight, scales=w.scales, biases=w.biases, transpose=True,
            group_size=w.group_size, bits=w.bits, mode=w.mode,
        )

    def _dense(self):
        """The [out, in] weight as a dense matrix. For quantized weights this is a
        ~1 ms transient that makes the backward matmul ~25% faster than
        quantized_matmul(transpose=False)."""
        w = self.base
        if not isinstance(w, nn.QuantizedLinear):
            return w.weight
        return mx.dequantize(
            w.weight, scales=w.scales, biases=w.biases,
            group_size=w.group_size, bits=w.bits, mode=w.mode,
        )

    def fwd(self, x):
        """Returns (y, x @ A). The second is tiny and saved for the B gradient."""
        y = self._mm(x)
        if self.a is None:
            return y, None
        xa = x @ self.a
        return mx.addmm(y, xa, self.b, alpha=self.s), xa

    def bwd_x(self, c, need_dx=True):
        """Input cotangent for output cotangent c. Also returns c @ B^T for bwd_params."""
        dx = c @ self._dense() if need_dx else None
        if self.a is None:
            return dx, None
        cb = c @ self.b.T
        if need_dx:
            dx = mx.addmm(dx, cb, self.a.T, alpha=self.s)
        return dx, cb

    def bwd_params(self, x, xa, c, cb):
        """(dA, dB) given the layer input x, saved x @ A, output cotangent c and c @ B^T."""
        if self.a is None:
            return None, None
        da = self.s * (_flat(x).T @ _flat(cb))
        db = self.s * (_flat(xa).T @ _flat(c))
        return da.astype(mx.float32), db.astype(mx.float32)


class FastLoRA:
    """Loss and LoRA gradients for a Gemma 3 or Gemma 4 model whose linears were
    wrapped by ``mlx_lm.tuner.utils.linear_to_lora_layers``.

    Args:
        model: the mlx-lm model (multimodal wrapper or text-only).
        chunk: query chunk size for banded sliding-window attention.
        recompute_layers: number of layers (the lowest trained ones) that keep only
            their input and re-run their forward pass during backward. Storing a
            layer costs ~0.1 MB per token; recomputing it costs ~35% of its step
            time. ``None`` picks the smallest number that fits ``memory_budget``
            for each batch.
        memory_budget: bytes the step may use when ``recompute_layers`` is None.
            Defaults to 80% of Metal's recommended working set.
        head_chunk: loss positions scored per logits matmul (bounds the size of
            the [positions, vocab] logits).
    """

    def __init__(self, model, chunk: int = 512, recompute_layers: int | None = None,
                 memory_budget: int | None = None, head_chunk: int = 1024):
        self.model = model
        lm = model.language_model if hasattr(model, "language_model") else model
        self.text = lm
        self.lm = lm.model
        self.args = self.lm.args if "args" in vars(self.lm) else self.lm.config
        for unsupported in ("enable_moe_block", "hidden_size_per_layer_input", "num_kv_shared_layers"):
            if getattr(self.args, unsupported, None):
                raise NotImplementedError(f"models with {unsupported} are not supported")
        self.layers = self.lm.layers
        self.chunk = chunk
        self.recompute_layers = recompute_layers
        self.head_chunk = head_chunk
        # Optional ``mlp_hook(i, act, m)``: called in every forward pass with layer i's
        # neuron activations [B, T, intermediate] and its MLP output before the norm.
        self.mlp_hook = None
        self._masks = {}

        prefix = "language_model.model." if hasattr(model, "language_model") else "model."
        self._keys = {}
        for i, layer in enumerate(self.layers):
            for name, mod, path in self._linears(layer):
                if isinstance(mod, LoRALinear):
                    if getattr(mod.dropout, "_p_1", 1.0) != 1.0:
                        raise NotImplementedError("LoRA dropout is not supported")
                    self._keys[(i, name)] = f"{prefix}layers.{i}.{path}"
                elif not isinstance(mod, (nn.Linear, nn.QuantizedLinear)):
                    raise NotImplementedError(f"unsupported adapter type {type(mod).__name__}")
        if not self._keys:
            raise ValueError("model has no LoRA layers; call linear_to_lora_layers first")
        # Nothing below the lowest adapted layer needs a backward pass.
        self._lowest = min(i for i, _ in self._keys)

        if memory_budget is None:
            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
            limit = info.get("max_recommended_working_set_size")
            if not limit:  # CPU device: fall back to 3/4 of physical memory
                limit = 0.75 * os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            memory_budget = int(0.8 * limit)
        self.memory_budget = memory_budget
        self._weight_bytes = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
        a = self.args
        # Floats kept per token by a stored layer: q, k, v, o, h, m, gate and up outputs.
        self._stored_floats = (
            (a.num_attention_heads + 2 * a.num_key_value_heads) * a.head_dim
            + 3 * a.hidden_size + 2 * a.intermediate_size
        )

    def plan(self, tokens: int, itemsize: int = 2) -> int:
        """How many layers must recompute for a batch of ``tokens`` to fit the budget."""
        if self.recompute_layers is not None:
            return self.recompute_layers
        trained = len(self.layers) - self._lowest
        per_layer = tokens * self._stored_floats * itemsize
        # ~1 MB per token of transient working memory (one layer's backward, the head).
        free = self.memory_budget - self._weight_bytes - tokens * 1_000_000 * itemsize // 2
        return max(0, min(trained, trained - int(free // per_layer)))

    @staticmethod
    def _linears(layer):
        at, mlp = layer.self_attn, layer.mlp
        # Gemma 4's full-attention layers have no value projection: values are the
        # key projection's output, normalised separately.
        value = (("v", at.v_proj, "self_attn.v_proj"),) if "v_proj" in at else ()
        return (
            ("q", at.q_proj, "self_attn.q_proj"),
            ("k", at.k_proj, "self_attn.k_proj"),
            *value,
            ("o", at.o_proj, "self_attn.o_proj"),
            ("gate", mlp.gate_proj, "mlp.gate_proj"),
            ("up", mlp.up_proj, "mlp.up_proj"),
            ("down", mlp.down_proj, "mlp.down_proj"),
        )

    # ---- attention ---------------------------------------------------------

    def _band_mask(self, L, B):
        """Boolean band mask [B * nC, 1, R * C, S]; only the CPU fallback needs it."""
        if mx.default_device() == mx.gpu:
            return None
        key = (L, B)
        if key not in self._masks:
            a = self.args
            C, W, R = self.chunk, a.sliding_window, a.num_attention_heads // a.num_key_value_heads
            nC, S = -(-L // C), C + W - 1
            r = np.arange(C)[:, None]
            t = np.arange(S)[None, :]
            band = (t >= r) & (t - r <= W - 1)
            j = (np.arange(nC) * C)[:, None, None] - (W - 1) + t[None]
            m = (band[None] & (j >= 0))[:, None]  # [nC, 1, C, S]
            self._masks = {key: mx.array(np.tile(m, (B, 1, R, 1)))}
        return self._masks[key]

    def _banded(self, i, L):
        return self.layers[i].self_attn.is_sliding and L > self.args.sliding_window

    def _prep(self, i, q, k, v, offset=0):
        """Projected [B, L, heads * d] -> per-head, normed and rotated [B, heads, L, d].
        ``offset`` is the position of the first token (non-zero after a KV cache)."""
        at = self.layers[i].self_attn
        H, Hk, d = at.n_heads, at.n_kv_heads, at.head_dim  # per layer in Gemma 4
        B, L, _ = q.shape
        q = at.rope(at.q_norm(q.reshape(B, L, H, d).transpose(0, 2, 1, 3)), offset=offset)
        k = at.rope(at.k_norm(k.reshape(B, L, Hk, d).transpose(0, 2, 1, 3)), offset=offset)
        v = v.reshape(B, L, Hk, d).transpose(0, 2, 1, 3)
        if "v_norm" in at:
            v = at.v_norm(v)
        return q, k, v

    def _attn(self, i, q, k, v):
        """Full causal attention through MLX's fused kernel. Returns [B, L, H * d]."""
        B, L, _ = q.shape
        q, k, v = self._prep(i, q, k, v)
        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.layers[i].self_attn.scale, mask="causal"
        )
        return o.transpose(0, 2, 1, 3).reshape(B, L, -1)

    # Banded sliding-window attention. Each chunk of C queries only sees the
    # C + W - 1 keys its window can reach, so scores are L x (C + W - 1), not L x L.

    def _chunk(self, x):
        """[B, H, L, d] -> [B * nC, Hk, R * C, d], zero-padding L up to nC * C."""
        a = self.args
        Hk, C = a.num_key_value_heads, self.chunk
        B, H, L, d = x.shape
        nC = -(-L // C)
        if nC * C != L:
            x = mx.pad(x, [(0, 0), (0, 0), (0, nC * C - L), (0, 0)])
        x = x.reshape(B, Hk, H // Hk, nC, C, d).transpose(0, 3, 1, 2, 4, 5)
        return x.reshape(B * nC, Hk, (H // Hk) * C, d)

    def _unchunk(self, x, B, L):
        """Inverse of _chunk: [B * nC, Hk, R * C, d] -> [B, H, L, d]."""
        C = self.chunk
        N, Hk, RC, d = x.shape
        nC, R = N // B, RC // C
        x = x.reshape(B, nC, Hk, R, C, d).transpose(0, 2, 3, 1, 4, 5)
        return x.reshape(B, Hk * R, nC * C, d)[:, :, :L]

    def _window(self, x):
        """[B, Hk, L, d] -> [B * nC, Hk, S, d]: the keys each query chunk can reach."""
        C, W = self.chunk, self.args.sliding_window
        B, Hk, L, d = x.shape
        nC, S = -(-L // C), C + W - 1
        x = mx.pad(x, [(0, 0), (0, 0), (W - 1, nC * C - L), (0, 0)])
        x = mx.stack([x[:, :, c * C : c * C + S] for c in range(nC)], axis=1)
        return x.reshape(B * nC, Hk, S, d)

    def _unwindow(self, x, B, L):
        """Adjoint of _window: overlap-add [B * nC, Hk, S, d] back to [B, Hk, L, d]."""
        C, W = self.chunk, self.args.sliding_window
        N, Hk, S, d = x.shape
        nC, m = N // B, -(-S // C)
        x = x.reshape(B, nC, Hk, S, d).transpose(0, 2, 1, 3, 4)
        if m * C != S:
            x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, m * C - S), (0, 0)])
        x = x.reshape(B, Hk, nC, m, C, d)
        # Rows [j * C, (j + 1) * C) of chunk c land on block c + j of the padded sequence.
        out = None
        for j in range(m):
            piece = mx.pad(x[:, :, :, j], [(0, 0), (0, 0), (j, m - 1 - j), (0, 0), (0, 0)])
            out = piece if out is None else out + piece
        return out.reshape(B, Hk, (nC + m - 1) * C, d)[:, :, W - 1 : W - 1 + L]

    def _band_fwd(self, i, q, k, v, prepped=None):
        at = self.layers[i].self_attn
        C, W = self.chunk, self.args.sliding_window
        B, L, _ = q.shape
        nC = -(-L // C)
        qr, kr, vr = prepped if prepped is not None else self._prep(i, q, k, v)
        qc, kc, vc = self._chunk(qr * at.scale), self._window(kr), self._window(vr)
        # bfloat16 would round the logits to ~0.1; float16 keeps 8x more precision
        # at the same matmul speed.
        sd = mx.float16 if q.dtype == mx.bfloat16 else q.dtype
        e, z = _band_exp(
            qc.astype(sd) @ kc.astype(sd).swapaxes(-1, -2), C, W, nC, q.dtype, self._band_mask(L, B)
        )
        invz = 1.0 / z
        o = ((e @ vc) * invz).astype(q.dtype)
        att = self._unchunk(o, B, L).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return att, (qc, kc, vc, e, invz, o)

    def _band_bwd(self, i, q, k, v, datt):
        """Re-runs the banded forward, then returns (att, dq, dk, dv)."""
        at = self.layers[i].self_attn
        B, L, _ = q.shape
        att, (qc, kc, vc, e, invz, o) = self._band_fwd(i, q, k, v)
        do = self._chunk(datt.reshape(B, L, self.args.num_attention_heads, -1).transpose(0, 2, 1, 3))
        # sum_t p * dp = do . o, so the softmax VJP needs no reduction over the band.
        dot = (do.astype(mx.float32) * o.astype(mx.float32)).sum(-1, keepdims=True)
        ds = _band_ds(e, do @ vc.swapaxes(-1, -2), invz, dot)
        dqr = self._unchunk(ds @ kc, B, L) * at.scale
        dkr = self._unwindow(ds.swapaxes(-1, -2) @ qc, B, L)
        dvr = self._unwindow(e.swapaxes(-1, -2) @ (do * invz).astype(q.dtype), B, L)
        dq, dk, dv = mx.vjp(lambda q, k, v: self._prep(i, q, k, v), [q, k, v], [dqr, dkr, dvr])[1]
        return att, dq, dk, dv

    def _cached_attn(self, i, q, k, v, kv: "KV"):
        """Attention for the positions after those ``kv`` holds, which it then also holds.
        A first chunk takes the training paths (banded for long sliding-window layers);
        later chunks and single tokens attend to the cache, masked where needed."""
        at = self.layers[i].self_attn
        B, n, _ = q.shape
        p = kv.end
        W = self.args.sliding_window if at.is_sliding else None
        qr, kr, vr = self._prep(i, q, k, v, offset=p)
        kv.append(kr, vr)
        if p == 0 and n > 1:
            if W is not None and n > W:
                return self._band_fwd(i, q, k, v, prepped=(qr, kr, vr))[0]
            o = mx.fast.scaled_dot_product_attention(qr, kr, vr, scale=at.scale, mask="causal")
        else:
            # The first new query, at position p, sees back to p - W + 1.
            keys, values, first = kv.view(p - W + 1 if W is not None else 0)
            mask = None
            if n > 1:
                qpos = mx.arange(p, p + n)[:, None]
                kpos = mx.arange(first, first + keys.shape[2])[None, :]
                mask = kpos <= qpos
                if W is not None:
                    mask = mask & (qpos - kpos < W)
            o = mx.fast.scaled_dot_product_attention(qr, keys, values, scale=at.scale, mask=mask)
        return o.transpose(0, 2, 1, 3).reshape(B, n, -1)

    # ---- one transformer block --------------------------------------------

    def _fwd(self, i, x, lins, save, kv=None):
        L = self.layers[i]
        xn = _rms(L.input_layernorm, x)
        q, qa = lins["q"].fwd(xn)
        k, ka = lins["k"].fwd(xn)
        v, va = lins["v"].fwd(xn) if "v" in lins else (k, None)
        if kv is not None:
            att = self._cached_attn(i, q, k, v, kv)
        elif self._banded(i, x.shape[1]):
            att = self._band_fwd(i, q, k, v)[0]
        else:
            att = self._attn(i, q, k, v)
        o, oa = lins["o"].fwd(att)
        h = x + _rms(L.post_attention_layernorm, o)
        hn = _rms(L.pre_feedforward_layernorm, h)
        g, ga = lins["gate"].fwd(hn)
        u, ua = lins["up"].fwd(hn)
        act = _act(g, u)
        m, ma = lins["down"].fwd(act)
        if self.mlp_hook is not None:
            self.mlp_hook(i, act, m)
        out = h + _rms(L.post_feedforward_layernorm, m)
        if "layer_scalar" in L:
            out = out * L.layer_scalar.astype(out.dtype)
        if not save:
            return out, {"x": x}
        st = dict(x=x, q=q, k=k, v=v, o=o, h=h, g=g, u=u, m=m)
        st["xa"] = {n: a for n, a in zip(_LIN_NAMES, (qa, ka, va, oa, ga, ua, ma)) if a is not None}
        return out, st

    def _bwd(self, i, st, lins, dout, need_dx):
        L = self.layers[i]
        xa = st["xa"]
        grads = {}

        def put(name, x, c, cb):
            da, db = lins[name].bwd_params(x, xa.get(name), c, cb)
            if da is not None:
                grads[(i, name)] = (da, db)

        if "layer_scalar" in L:
            dout = dout * L.layer_scalar.astype(dout.dtype)
        # MLP: out = h + norm(down(gelu(gate(hn)) * up(hn)))
        dm = _rms_vjp(L.post_feedforward_layernorm, st["m"], dout)
        dp, cb = lins["down"].bwd_x(dm)
        p, dg, du = _act_vjp(st["g"], st["u"], dp)
        put("down", p, dm, cb)
        hn = _rms(L.pre_feedforward_layernorm, st["h"])
        dhn_g, cb = lins["gate"].bwd_x(dg)
        put("gate", hn, dg, cb)
        dhn_u, cb = lins["up"].bwd_x(du)
        put("up", hn, du, cb)
        dh = dout + _rms_vjp(L.pre_feedforward_layernorm, st["h"], dhn_g + dhn_u)

        # Attention: h = x + norm(o(attn(q(xn), k(xn), v(xn)))). The attention core is
        # re-run here instead of storing its probabilities.
        do = _rms_vjp(L.post_attention_layernorm, st["o"], dh)
        datt, cb = lins["o"].bwd_x(do)
        if self._banded(i, datt.shape[1]):
            att, dq, dk, dv = self._band_bwd(i, st["q"], st["k"], st["v"], datt)
        else:
            (att,), (dq, dk, dv) = mx.vjp(
                lambda q, k, v: self._attn(i, q, k, v), [st["q"], st["k"], st["v"]], [datt]
            )
        put("o", att, do, cb)
        xn = _rms(L.input_layernorm, st["x"])
        dxn = None
        # Where values come from the key projection, its output gets both cotangents.
        outputs = (("q", dq), ("k", dk), ("v", dv)) if "v" in lins else (("q", dq), ("k", dk + dv))
        for name, c in outputs:
            d, cb = lins[name].bwd_x(c, need_dx=need_dx)
            put(name, xn, c, cb)
            if need_dx:
                dxn = d if dxn is None else dxn + d
        dx = dh + _rms_vjp(L.input_layernorm, st["x"], dxn) if need_dx else None
        return dx, grads

    # ---- head --------------------------------------------------------------

    def _logits(self, sel):
        """float32 logits for final-normed states [n, hidden], soft-capped where the model is."""
        if getattr(self.text, "tie_word_embeddings", True):
            logits = self.lm.embed_tokens.as_linear(sel)
        else:
            logits = self.text.lm_head(sel)
        logits = logits.astype(mx.float32)
        cap = getattr(self.text, "final_logit_softcapping", None)
        return mx.tanh(logits / cap) * cap if cap else logits

    def _head(self, h, pos, targets, weights, grad=True):
        """Loss over the scored positions and (if grad) its gradient w.r.t. the final hidden state."""
        total = weights.sum() if weights is not None else pos.size

        def f(h_, p_, t_, w_):
            ce = nn.losses.cross_entropy(self._logits(_flat(_rms(self.lm.norm, h_))[p_]), t_, reduction="none")
            return (ce if w_ is None else ce * w_).sum() / total

        loss, dh = None, None
        for a in range(0, pos.size, self.head_chunk):
            sl = slice(a, a + self.head_chunk)
            chunk = (h, pos[sl], targets[sl], None if weights is None else weights[sl])
            l, d = mx.value_and_grad(f)(*chunk) if grad else (f(*chunk), None)
            loss = l if loss is None else loss + l
            if grad:
                dh = d if dh is None else dh + d
            mx.eval(loss, dh)
        return loss, dh

    # ---- full step ---------------------------------------------------------

    def _embed(self, ids):
        h = self.lm.embed_tokens(ids)
        if "embed_scale" in vars(self.lm):  # Gemma 4
            h = h * self.lm.embed_scale
        else:
            h = h * mx.array(self.args.hidden_size**0.5, mx.bfloat16).astype(h.dtype)
        if h.dtype == mx.float16:
            raise NotImplementedError("float16 activations need residual clipping; use bfloat16")
        return h

    def hidden(self, ids, hook=None):
        """Forward only: the residual stream after the last layer. ``hook(i, h)``, if
        given, is called with each layer's output and may return a replacement."""
        h = self._embed(ids)
        for i in range(len(self.layers)):
            lins = {name: _Lin(mod, h.dtype) for name, mod, _ in self._linears(self.layers[i])}
            h, _ = self._fwd(i, h, lins, save=False)
            if hook is not None:
                out = hook(i, h)
                h = h if out is None else out
            mx.eval(h)
        return h

    # ---- inference -----------------------------------------------------------

    def new_cache(self) -> list["KV"]:
        """An empty KV cache, one entry per layer."""
        W = self.args.sliding_window
        return [KV(W if layer.self_attn.is_sliding else None) for layer in self.layers]

    def adapters(self) -> list[dict]:
        """Every layer's linears with its adapters at their modules' current scales
        (set ``LoRALinear.scale`` first). Build again after changing a scale."""
        dtype = self.lm.embed_tokens(mx.zeros((1,), mx.int32)).dtype
        lins = [{name: _Lin(mod, dtype) for name, mod, _ in self._linears(layer)} for layer in self.layers]
        mx.eval([[l.a, l.b] for layer in lins for l in layer.values() if l.a is not None])
        return lins

    def forward(self, ids, lins, cache, want=()):
        """Runs ids [B, n] after the positions ``cache`` holds, extending it. Returns the
        residual stream after the last layer [B, n, d] and {i: residual after layer i}
        for each i in ``want``. Lazy: evaluate what you need."""
        h = self._embed(ids)
        taps = {}
        for i in range(len(self.layers)):
            h, _ = self._fwd(i, h, lins[i], save=False, kv=cache[i])
            if i in want:
                taps[i] = h
        return h, taps

    def logits(self, h):
        """float32 next-token logits from residuals after the last layer, [..., vocab]."""
        return self._logits(_rms(self.lm.norm, h))

    def loss(self, ids, pos, targets, weights=None):
        """Forward only: the loss ``loss_and_grads`` would report, for evaluation."""
        return self._head(self.hidden(ids), pos, targets, weights, grad=False)[0]

    def token_losses(self, ids, pos, targets, hook=None):
        """Forward only: cross-entropy of each scored target separately, shape [n]."""
        sel = _flat(_rms(self.lm.norm, self.hidden(ids, hook)))[pos]
        out = []
        for a in range(0, pos.size, self.head_chunk):
            logits = self._logits(sel[a : a + self.head_chunk])
            out.append(nn.losses.cross_entropy(logits, targets[a : a + self.head_chunk], reduction="none"))
            mx.eval(out[-1])
        return mx.concatenate(out) if out else mx.zeros((0,))

    def loss_and_grads(self, ids, pos, targets, weights=None):
        """One forward/backward pass.

        Args:
            ids: [B, T] input token ids (right-padded if B > 1).
            pos: [n] flat indices into B * T of positions whose next token is scored.
            targets: [n] the next-token ids at those positions.
            weights: optional [n] per-target loss weights.

        Returns:
            (loss, grads) where grads is a flat dict keyed like
            ``tree_flatten(model.trainable_parameters())``.
        """
        h = self._embed(ids)
        dtype = h.dtype
        n, lowest = len(self.layers), self._lowest
        first_stored = lowest + self.plan(ids.size, dtype.size)

        states, all_lins = [], []
        for i in range(n):
            lins = {name: _Lin(mod, dtype) for name, mod, _ in self._linears(self.layers[i])}
            h, st = self._fwd(i, h, lins, save=i >= first_stored)
            if i < lowest:
                st = None
            mx.eval(h, st)
            all_lins.append(lins)
            states.append(st)

        loss, dh = self._head(h, pos, targets, weights)
        del h

        out = {}
        for i in reversed(range(lowest, n)):
            st, states[i] = states[i], None
            if "g" not in st:
                _, st = self._fwd(i, st["x"], all_lins[i], save=True)
            dh, grads = self._bwd(i, st, all_lins[i], dh, need_dx=i > lowest)
            del st
            for (li, name), (da, db) in grads.items():
                key = self._keys[(li, name)]
                out[key + ".lora_a"] = da
                out[key + ".lora_b"] = db
            mx.eval(dh, grads)
        return loss, out


class KV:
    """One layer's keys and values (normed and rotated) for positions start..end-1.

    Grows in steps of 256 positions. A sliding-window layer drops what lies beyond its
    window once it holds ``SLACK_MAX`` more than that, keeping ``SLACK`` extra so the
    cache can still be cut back that far (the next request may share only part of it).
    """

    STEP = 256
    SLACK = 1024
    SLACK_MAX = 2048

    def __init__(self, window: int | None):
        self.window = window
        self.keys = self.values = None
        self.start = self.end = 0

    @staticmethod
    def _sized(x, used, capacity):
        out = mx.zeros(x.shape[:2] + (capacity,) + x.shape[3:], x.dtype)
        if used:
            out[..., :used, :] = x[..., :used, :]
        return out

    def append(self, k, v):
        n, used = k.shape[2], self.end - self.start
        if self.keys is None:
            cap = -(-n // self.STEP) * self.STEP
            self.keys, self.values = self._sized(k, 0, cap), self._sized(v, 0, cap)
        elif used + n > self.keys.shape[2]:
            cap = -(-(used + n) // self.STEP) * self.STEP
            self.keys, self.values = self._sized(self.keys, used, cap), self._sized(self.values, used, cap)
        self.keys[..., used:used + n, :] = k
        self.values[..., used:used + n, :] = v
        self.end += n
        used += n
        if self.window is not None and used > self.window + self.SLACK_MAX:
            drop = used - (self.window + self.SLACK)
            keep = used - drop
            cap = -(-keep // self.STEP) * self.STEP + self.STEP
            self.keys = self._sized(self.keys[..., drop:used, :], keep, cap)
            self.values = self._sized(self.values[..., drop:used, :], keep, cap)
            self.start += drop

    def view(self, first=0):
        """(keys, values, position of the first) from position ``first`` (or the oldest held) on."""
        a = max(self.start, first) - self.start
        b = self.end - self.start
        return self.keys[..., a:b, :], self.values[..., a:b, :], self.start + a

    def can_trim(self, n: int) -> bool:
        end = self.end - n
        if end < self.start:
            return False
        # The next token at position ``end`` must still see its whole window.
        return self.window is None or self.start == 0 or end - self.window + 1 >= self.start

    def trim(self, n: int):
        self.end -= n

    @property
    def state(self):
        return [] if self.keys is None else [self.keys, self.values]
