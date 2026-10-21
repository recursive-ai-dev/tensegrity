"""Building blocks: RMSNorm, RoPE, grouped-query attention (global + banded-local), SwiGLU, Block.

Parameter *names* are identical to the original single-file release for every
legacy feature, so old ``state_dict``s load unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  (explicit: `import torch` alone does not guarantee it)

from .config import TensegrityConfig

KV = Tuple[torch.Tensor, torch.Tensor]


# =============================================================================
# SDPA capability probing
# =============================================================================
#
# The original code decided whether SDPA supports `enable_gqa` with
# `inspect.signature(F.scaled_dot_product_attention)`. On real torch builds that
# raises ValueError (it is a C builtin with no introspectable signature), so the
# flag was *always* False. We probe by calling the function instead.

def _probe_enable_gqa() -> bool:
    try:
        q = torch.zeros(1, 2, 1, 2)
        kv = torch.zeros(1, 1, 1, 2)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        return True
    except (TypeError, RuntimeError, ValueError):
        return False


SDPA_HAS_ENABLE_GQA: bool = _probe_enable_gqa()

_NATIVE_GQA_CACHE: Dict[Tuple[int, torch.dtype], bool] = {}


def native_gqa_ok(t: torch.Tensor) -> bool:
    """Use SDPA's native ``enable_gqa`` only where it hits the fused FlashAttention kernel.

    PyTorch's memory-efficient kernel (the only fused kernel on a Turing T4, sm75) does
    not implement GQA; with ``enable_gqa=True`` SDPA would silently fall back to the
    *math* backend and materialise the full (T, T) score matrix. So: native GQA only on
    CUDA, compute capability >= 8, fp16/bf16. Everywhere else K/V are expanded instead,
    which keeps the fused kernels engaged.
    """
    if not SDPA_HAS_ENABLE_GQA or not t.is_cuda or t.dtype not in (torch.float16, torch.bfloat16):
        return False
    key = (t.device.index if t.device.index is not None else torch.cuda.current_device(), t.dtype)
    ok = _NATIVE_GQA_CACHE.get(key)
    if ok is None:
        ok = torch.cuda.get_device_capability(t.device)[0] >= 8
        _NATIVE_GQA_CACHE[key] = ok
    return ok


def _expand_kv(k: torch.Tensor, v: torch.Tensor, n_rep: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_rep == 1:
        return k, v
    B, Hkv, T, D = k.shape
    return (k[:, :, None, :, :].expand(B, Hkv, n_rep, T, D).reshape(B, Hkv * n_rep, T, D),
            v[:, :, None, :, :].expand(B, Hkv, n_rep, T, D).reshape(B, Hkv * n_rep, T, D))


# =============================================================================
# Normalisation, RoPE, small helpers
# =============================================================================

_HAS_FUSED_RMS_NORM: bool = hasattr(F, "rms_norm")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_FUSED_RMS_NORM:
            w = self.weight if x.dtype == self.weight.dtype else self.weight.to(x.dtype)
            return F.rms_norm(x, self.weight.shape, w, self.eps)
        in_dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms * self.weight.float()).to(in_dtype)


def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0,
                    device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_seq_len, head_dim // 2).

    The angles are computed in float64 and rounded once. Streaming generation runs far past ``block_size``
    and a float32 ``position * frequency`` product loses precision linearly with position (measured max
    phase error: ~2e-4 rad at 10^4 positions, ~1e-3..5e-3 at 10^5, ~2e-2..5e-2 at 10^6, for head_dim 32-128),
    which is noise in the relative rotation between a query and its keys."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float64) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float64)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). cos/sin: (>=T, D/2) or pre-shaped (1, 1, T, D/2)."""
    B, H, T, D = x.shape
    half = D // 2
    if cos.ndim == 4:
        c, s = cos, sin
    else:
        assert T <= cos.shape[0], (
            f"sequence length {T} exceeds RoPE table length {cos.shape[0]}; "
            f"call precompute_rope with a larger max_seq_len")
        c = cos[:T].float().view(1, 1, T, half)
        s = sin[:T].float().view(1, 1, T, half)
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


def activation_rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean over batch/tokens of the per-token feature RMS (eps-stabilised sqrt)."""
    return torch.sqrt(x.float().pow(2).mean(dim=-1) + eps).mean()


def apply_logit_soft_cap(logits: torch.Tensor, soft_cap: float) -> torch.Tensor:
    """Gemma-2 soft-capping: ``cap * tanh(logits / cap)`` (computed in fp32)."""
    return (soft_cap * torch.tanh(logits.float() / soft_cap)).to(logits.dtype)


# =============================================================================
# Attention
# =============================================================================

_BAND_MASK_CACHE: Dict[tuple, torch.Tensor] = {}


def _band_mask(n_chunks: int, window: int, batch: int, device: torch.device) -> torch.Tensor:
    """Boolean mask (batch * n_chunks, 1, W, 2W) for chunked sliding-window attention.

    Chunk c's queries (positions cW + i, i in [0, W)) see a 2W-key strip = [chunk c-1 | chunk c]
    whose key j sits at absolute position (c-1)W + j. A query may attend iff
        key_pos <= q_pos   <=>  j <= i + W
        key_pos >  q_pos - W  <=>  j >  i
    i.e. exactly W keys (including itself) -- the same window as the dense
    ``j <= i and j > i - W`` mask. Chunk 0 has no predecessor, so keys j < W are masked out.
    """
    key = (n_chunks, window, batch, str(device))
    m = _BAND_MASK_CACHE.get(key)
    if m is None:
        i = torch.arange(window, device=device).view(-1, 1)
        j = torch.arange(2 * window, device=device).view(1, -1)
        band = (j > i) & (j <= i + window)                     # (W, 2W)
        masks = band.unsqueeze(0).repeat(n_chunks, 1, 1)       # (nC, W, 2W)
        masks[0] &= (j >= window)
        m = masks.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * n_chunks, 1, window, 2 * window)
        if len(_BAND_MASK_CACHE) >= 16:
            _BAND_MASK_CACHE.pop(next(iter(_BAND_MASK_CACHE)))
        _BAND_MASK_CACHE[key] = m
    return m


def local_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
                           n_rep: int, dropout_p: float = 0.0) -> torch.Tensor:
    """Sliding-window causal attention in ONE fused SDPA call.

    The sequence is cut into chunks of ``window`` queries; each chunk attends to the
    2W-key strip [previous chunk | itself] under a banded mask (see ``_band_mask``).
    Chunks are folded into the batch dimension, so there is no Python loop over chunks
    and no (T, T) tensor. Result equals dense masked attention exactly.

    q: (B, H, T, D); k, v: (B, Hkv, T, D). Returns (B, H, T, D).
    """
    B, H, T, D = q.shape
    Hkv = k.shape[1]
    W = window
    if T <= W:  # window covers everything -> plain causal attention
        if n_rep > 1 and native_gqa_ok(q):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True, enable_gqa=True)
        if n_rep > 1:
            k, v = _expand_kv(k, v, n_rep)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

    n_chunks = (T + W - 1) // W
    pad = n_chunks * W - T
    if pad:  # pad at the END: padded keys lie in the future of every real query, so are masked
        q = F.pad(q, (0, 0, 0, pad))
    kp = F.pad(k, (0, 0, W, pad)).view(B, Hkv, n_chunks + 1, W, D)
    vp = F.pad(v, (0, 0, W, pad)).view(B, Hkv, n_chunks + 1, W, D)
    k2 = torch.cat([kp[:, :, :-1], kp[:, :, 1:]], dim=3)       # (B, Hkv, nC, 2W, D)
    v2 = torch.cat([vp[:, :, :-1], vp[:, :, 1:]], dim=3)

    qf = q.view(B, H, n_chunks, W, D).transpose(1, 2).reshape(B * n_chunks, H, W, D)
    kf = k2.transpose(1, 2).reshape(B * n_chunks, Hkv, 2 * W, D)
    vf = v2.transpose(1, 2).reshape(B * n_chunks, Hkv, 2 * W, D)
    if n_rep > 1:
        kf, vf = _expand_kv(kf, vf, n_rep)

    mask = _band_mask(n_chunks, W, B, q.device)
    yf = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask, dropout_p=dropout_p)
    y = yf.view(B, n_chunks, H, W, D).transpose(1, 2).reshape(B, H, n_chunks * W, D)
    return y[:, :, :T] if pad else y


class CausalSelfAttention(nn.Module):
    """GQA attention. ``is_global`` layers attend causally to everything; local layers to
    the trailing ``sliding_window`` positions."""

    def __init__(self, cfg: TensegrityConfig, is_global: bool, layer_idx: int = 0):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.is_global = is_global
        self.sliding_window = cfg.sliding_window
        self.dropout_p = cfg.dropout
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        if cfg.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        if cfg.v0_mix and layer_idx > 0:
            self.v0_gain = nn.Parameter(torch.full((cfg.n_kv_head,), 0.5))
            self.v0_gain._no_decay = True  # type: ignore[attr-defined]
        else:
            self.v0_gain = None

    def _project(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 v0: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v_raw = v
        if self.v0_gain is not None and v0 is not None:
            v = v + self.v0_gain.to(v.dtype).view(1, -1, 1, 1) * v0
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v, v_raw

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                past: Optional[KV] = None, return_kv: bool = False, max_cache: Optional[int] = None,
                v0: Optional[torch.Tensor] = None, return_v0: bool = False):
        """
        Training / prefill (``past is None``): full-sequence attention starting at position 0.
        Decode (``past`` given): ``x`` must be ONE new token; its K/V are appended to ``past``.
        With ``return_kv`` the (trimmed) cache for subsequent decode steps is returned too.
        ``max_cache`` bounds a GLOBAL layer's cache to the most recent ``max_cache`` keys (streaming decode);
        local layers are always bounded by their window.
        """
        B, T, _ = x.shape
        q, k, v, v_raw = self._project(x, cos, sin, v0)
        dropout_p = self.dropout_p if self.training else 0.0

        if past is not None:
            assert T == 1, "cached decode supports a single new token per step"
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
            limit = self.sliding_window if not self.is_global else max_cache
            if limit is not None and k.shape[2] > limit:
                k = k[:, :, -limit:]
                v = v[:, :, -limit:]
            ke, ve = _expand_kv(k, v, self.n_rep)
            y = F.scaled_dot_product_attention(q, ke, ve)  # every cached key is causally visible
        elif self.is_global:
            if self.n_rep > 1 and native_gqa_ok(q):
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True,
                                                   enable_gqa=True)
            else:
                ke, ve = _expand_kv(k, v, self.n_rep)
                y = F.scaled_dot_product_attention(q, ke, ve, dropout_p=dropout_p, is_causal=True)
        else:
            y = local_causal_attention(q, k, v, self.sliding_window, self.n_rep, dropout_p)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.resid_dropout(self.o_proj(y))
        if not return_kv:
            return (y, v_raw) if return_v0 else y
        if not self.is_global and k.shape[2] > self.sliding_window:
            k, v = k[:, :, -self.sliding_window:], v[:, :, -self.sliding_window:]
        kv = (k.contiguous(), v.contiguous())
        return (y, kv, v_raw) if return_v0 else (y, kv)


# =============================================================================
# Feed-forward and block
# =============================================================================

class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int, bias: bool, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    """Pre-norm transformer block with learnable residual gains ("homeostatic" gains).

        x <- x + x0_gain * x0                      (only if cfg.x0_mix)
        x <- x + attn_gain * Attn(RMSNorm(x))
        x <- x + ffn_gain  * SwiGLU(RMSNorm(x))

    LayerDrop is decided by the *model* (one CPU draw per forward), not here: a block
    never touches RNG or the network, so it is safe under gradient checkpointing,
    torch.compile and DDP.
    """

    def __init__(self, cfg: TensegrityConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, is_global=cfg.is_global_layer(layer_idx), layer_idx=layer_idx)
        self.ln2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.n_embd, cfg.ffn_hidden, cfg.bias, cfg.dropout)
        self.attn_gain = nn.Parameter(torch.tensor(float(cfg.gain_init)))
        self.ffn_gain = nn.Parameter(torch.tensor(float(cfg.gain_init)))
        self.x0_gain = nn.Parameter(torch.zeros(())) if cfg.x0_mix else None
        if cfg.n_loops > 1:
            n_extra = cfg.n_loops - 1
            self.loop_attn_gains = nn.Parameter(torch.full((n_extra,), float(cfg.gain_init)))
            self.loop_ffn_gains = nn.Parameter(torch.full((n_extra,), float(cfg.gain_init)))
            self.loop_x0_gains = nn.Parameter(torch.zeros(n_extra)) if cfg.x0_mix else None
        else:
            self.loop_attn_gains = None
            self.loop_ffn_gains = None
            self.loop_x0_gains = None
        self.homeo_target = cfg.homeo_target
        self.track_homeo = cfg.homeo_weight > 0.0
        # depth-scaled stochastic-depth probability (consumed by TensegrityLM)
        self.drop_p = cfg.layerdrop_max * (layer_idx / (cfg.n_layer - 1)) if cfg.n_layer > 1 else 0.0

    def _homeo(self, sub_out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if self.track_homeo and torch.is_grad_enabled():
            return (activation_rms(sub_out) - self.homeo_target).pow(2)
        return ref.new_zeros((), dtype=torch.float32)

    def _gains(self, loop_idx: int):
        if loop_idx == 0 or self.loop_attn_gains is None:
            return self.attn_gain, self.ffn_gain, self.x0_gain
        k = loop_idx - 1
        xg = self.loop_x0_gains[k] if self.loop_x0_gains is not None else None
        return self.loop_attn_gains[k], self.loop_ffn_gains[k], xg

    def _forward_impl(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      x0: Optional[torch.Tensor] = None, v0: Optional[torch.Tensor] = None,
                      loop_idx: int = 0, need_v0: bool = False):
        ag, fg, xg = self._gains(loop_idx)
        if xg is not None and x0 is not None:
            x = x + xg * x0
        if need_v0:
            attn_out, v0_out = self.attn(self.ln1(x), cos, sin, v0=v0, return_v0=True)
        else:
            attn_out, v0_out = self.attn(self.ln1(x), cos, sin, v0=v0), v0
        homeo = self._homeo(attn_out, x)
        x = x + ag * attn_out
        ffn_out = self.ffn(self.ln2(x))
        homeo = homeo + self._homeo(ffn_out, x)
        x = x + fg * ffn_out
        return x, homeo, v0_out

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                x0: Optional[torch.Tensor] = None, v0: Optional[torch.Tensor] = None,
                loop_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        x, homeo, _ = self._forward_impl(x, cos, sin, x0, v0, loop_idx, need_v0=False)
        return x, homeo

    def forward_incremental(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                            x0: Optional[torch.Tensor], past: Optional[KV], max_cache: Optional[int] = None,
                            v0: Optional[torch.Tensor] = None, loop_idx: int = 0,
                            return_v0: bool = False):
        """KV-cached forward for inference (eval mode; no LayerDrop, no homeostasis loss)."""
        ag, fg, xg = self._gains(loop_idx)
        if xg is not None and x0 is not None:
            x = x + xg * x0
        if return_v0:
            attn_out, kv, v0_out = self.attn(self.ln1(x), cos, sin, past=past, return_kv=True, max_cache=max_cache,
                                             v0=v0, return_v0=True)
        else:
            attn_out, kv = self.attn(self.ln1(x), cos, sin, past=past, return_kv=True, max_cache=max_cache, v0=v0)
            v0_out = v0
        x = x + ag * attn_out
        x = x + fg * self.ffn(self.ln2(x))
        return (x, kv, v0_out) if return_v0 else (x, kv)


class ByteMix(nn.Module):
    """Learned causal FIR filter (and optional gated bilinear n-gram mixer) over the previous K-1 byte embeddings:

        y_t = e_t + sum_{j=1..K-1} (w_j + bi_j * tanh(e_t / scale)) * e_{t-j}

    A byte is a fragment of a character/word, so the first layers of a byte LM spend
    capacity just assembling local n-grams. This gives every position direct, cheap
    access to its recent bytes and multiplicative byte-pair interactions. Zero-initialised,
    so it starts as the identity and preserves function under width growth.
    """

    def __init__(self, kernel: int, n_embd: int, bilinear: bool = False,
                 init_std: float = 0.02, embed_scale: bool = False):
        super().__init__()
        self.kernel = kernel
        self.init_std = init_std
        self.embed_scale = embed_scale
        self.weight = nn.Parameter(torch.zeros(kernel - 1, n_embd))
        self.weight._no_decay = True  # type: ignore[attr-defined]
        if bilinear:
            self.bi_weight = nn.Parameter(torch.zeros(kernel - 1, n_embd))
            self.bi_weight._no_decay = True  # type: ignore[attr-defined]
        else:
            self.bi_weight = None

    def forward(self, e: torch.Tensor, n_prefix: int) -> torch.Tensor:
        """e: (B, n_prefix + T, C) embeddings whose first ``n_prefix`` (= K-1) rows are
        left context. Returns (B, T, C)."""
        T = e.shape[1] - n_prefix
        cur = e[:, n_prefix:]
        out = cur
        w = self.weight if e.dtype == self.weight.dtype else self.weight.to(e.dtype)
        if self.bi_weight is not None:
            bw = self.bi_weight if e.dtype == self.bi_weight.dtype else self.bi_weight.to(e.dtype)
            scale = self.init_std * (e.shape[-1] ** 0.5 if self.embed_scale else 1.0)
            g = torch.tanh(cur * (1.0 / scale))
            for j in range(1, self.kernel):
                prev = e[:, n_prefix - j: n_prefix - j + T]
                out = out + (w[j - 1] + bw[j - 1] * g) * prev
            return out
        for j in range(1, self.kernel):
            out = out + w[j - 1] * e[:, n_prefix - j: n_prefix - j + T]
        return out
