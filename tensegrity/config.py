"""TensegrityConfig: the single description of a model's architecture.

Backward-compatibility contract
-------------------------------
Every field that existed in the original single-file release keeps its name,
type and default. Every field added since (``norm_eps``, ``confidence_detach``,
``byte_mix_kernel``, ``x0_mix``) defaults to the value that reproduces the
original behaviour exactly. A checkpoint saved by the old code therefore loads
into the new code (via :meth:`TensegrityConfig.from_dict`) as the very same
network, and the *presets* (see ``presets.py``) are what opt into new features.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional


def ffn_hidden_dim(n_embd: int, mult: float, align: int) -> int:
    """SwiGLU hidden width: ``int(n_embd * mult)`` rounded UP to a multiple of ``align``."""
    assert n_embd > 0, f"n_embd must be positive, got {n_embd}"
    assert mult > 0, f"mult must be positive, got {mult}"
    assert align >= 1, f"align must be a positive integer, got {align}"
    hidden = int(n_embd * mult)
    hidden = align * ((hidden + align - 1) // align)
    return max(hidden, align)


@dataclass
class TensegrityConfig:
    # ------------------------------------------------------------------ shape
    vocab_size: int = 259
    block_size: int = 4096
    n_layer: int = 28
    n_embd: int = 2560
    n_head: int = 20
    n_kv_head: int = 4
    dropout: float = 0.0

    # -------------------------------------------------------------- attention
    sliding_window: int = 1024
    global_every: int = 4          # layer i is global iff (i + 1) % global_every == 0
    rope_base: float = 12000.0

    # -------------------------------------------------------------------- FFN
    ffn_mult: float = 8.0 / 3.0
    ffn_align: int = 64

    # ------------------------------------------- homeostasis / stochastic depth
    homeo_target: float = 1.0
    homeo_weight: float = 0.01
    gain_init: Optional[float] = None
    layerdrop_max: float = 0.1

    # ------------------------------------------------------- confidence head
    use_confidence_head: bool = True
    confidence_weight: float = 0.05

    # ------------------------------------------------------------------- misc
    bias: bool = False
    init_std: float = 0.02
    grad_checkpoint: bool = True
    grad_checkpoint_every: int = 1  # checkpoint every Nth block; 1 = all blocks

    use_qk_norm: bool = True        # per-head RMSNorm on q,k before RoPE
    z_loss_weight: float = 0.0      # PaLM/Gemma z-loss coefficient (0 = off)
    logit_soft_cap: Optional[float] = None  # Gemma-2 tanh soft-cap (None = off)
    embed_scale: bool = True        # multiply embeddings by sqrt(n_embd)

    # ---------------------------------------------------- added after v1 (all
    # defaults reproduce v1 behaviour exactly; see module docstring)
    norm_eps: float = 1e-6          # eps of the model-width RMSNorms (ln1/ln2/ln_f)
    confidence_detach: bool = False  # True: confidence head is a stop-gradient probe
    byte_mix_kernel: int = 0        # K>=2: learned causal FIR over the previous K-1 byte embeddings
    byte_mix_bilinear: bool = False  # gated bilinear n-gram interaction inside ByteMix
    x0_mix: bool = False            # per-layer learnable re-injection of the embedding stream
    v0_mix: bool = False            # per-layer learnable mixing of layer 0's value projection
    n_loops: int = 1                # number of recursive passes through the block stack (>=1)

    # --------------------------------------------------------------- validate
    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError(f"head_dim ({self.n_embd // self.n_head}) must be even for RoPE")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {self.n_layer}")
        if self.n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {self.n_loops}")
        if self.global_every < 1:
            raise ValueError(f"global_every must be >= 1, got {self.global_every}")
        if self.grad_checkpoint_every < 1:
            raise ValueError(f"grad_checkpoint_every must be >= 1, got {self.grad_checkpoint_every}")
        if not 0.0 <= self.layerdrop_max < 1.0:
            raise ValueError(f"layerdrop_max must be in [0, 1), got {self.layerdrop_max}")
        if self.sliding_window < 1:
            raise ValueError(f"sliding_window must be >= 1, got {self.sliding_window}")
        if self.byte_mix_kernel != 0 and self.byte_mix_kernel < 2:
            raise ValueError(f"byte_mix_kernel must be 0 (off) or >= 2, got {self.byte_mix_kernel}")
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be > 0, got {self.norm_eps}")
        if self.gain_init is None:
            self.gain_init = 1.0 / math.sqrt(2.0 * self.n_layer * self.n_loops)

    # ------------------------------------------------------------ constructors
    @classmethod
    def field_names(cls) -> set:
        return {f.name for f in dataclasses.fields(cls)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any], warn_unknown: bool = True, strict: bool = False) -> "TensegrityConfig":
        """Build from a (possibly older/newer) config dict. Missing keys take defaults
        (== legacy behaviour for fields added later).

        Unknown keys are either a typo or a field written by a NEWER release whose meaning this code
        does not know -- dropping it could silently build a different network. With ``strict=True``
        (what every checkpoint / export loader uses) they raise; otherwise they are dropped with a
        warning."""
        known = cls.field_names()
        unknown = sorted(k for k in d if k not in known)
        if unknown and strict:
            raise ValueError(f"unknown TensegrityConfig field(s) {unknown}: not a typo-free config of this release "
                             f"(known fields: {sorted(known)}). The file was probably written by a newer version of "
                             f"tensegrity; upgrade it, or load with strict=False to drop them at your own risk.")
        if unknown and warn_unknown:
            warnings.warn(f"TensegrityConfig.from_dict: ignoring unknown key(s) {unknown}", stacklevel=2)
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **overrides: Any) -> "TensegrityConfig":
        """Copy with field overrides. ``gain_init`` is recomputed unless given explicitly."""
        d = self.to_dict()
        if "gain_init" not in overrides and ("n_layer" in overrides or "n_loops" in overrides):
            d["gain_init"] = None
        d.update(overrides)
        return TensegrityConfig(**d)

    # ----------------------------------------------------------------- derived
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def gqa_ratio(self) -> int:
        return self.n_head // self.n_kv_head

    @property
    def ffn_hidden(self) -> int:
        return ffn_hidden_dim(self.n_embd, self.ffn_mult, self.ffn_align)

    def is_global_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.global_every == 0

    @property
    def n_global_layers(self) -> int:
        return sum(self.is_global_layer(i) for i in range(self.n_layer))

    def receptive_field(self) -> int:
        """The longest dependency the network CAN express, in positions (the current byte included).

        With at least one global layer a byte can read any earlier byte directly, so this is
        ``block_size``. With none, information can only hop ``sliding_window - 1`` positions per local
        layer, and ByteMix adds ``byte_mix_kernel - 1`` at the input:
        ``1 + n_layer * (sliding_window - 1) + (byte_mix_kernel - 1)`` (e.g. 4 local layers of window
        124 see at most 493 bytes of a 1024-byte context), capped at ``block_size``. This is an upper
        bound on what the architecture allows, not a measurement of what a trained model uses --
        see :func:`tensegrity.evaluation.context_probe` for that."""
        if self.n_global_layers:
            return self.block_size
        mix = max(0, self.byte_mix_kernel - 1)
        return min(self.block_size, 1 + self.n_layer * (self.sliding_window - 1) + mix)

    def n_params(self) -> int:
        """Exact parameter count (tied embedding counted once). Verified against a real
        instantiation for every preset in tests/test_presets.py."""
        d, hd, h = self.n_embd, self.head_dim, self.ffn_hidden
        q_out, kv_out = self.n_head * hd, self.n_kv_head * hd
        per_block = 2 * d                                   # ln1, ln2
        per_block += d * q_out + 2 * d * kv_out + q_out * d  # q,k,v,o
        per_block += 3 * d * h                              # SwiGLU
        per_block += 2                                      # attn_gain, ffn_gain
        if self.bias:
            per_block += q_out + 2 * kv_out + d + 2 * h + d
        if self.use_qk_norm:
            per_block += 2 * hd
        if self.x0_mix:
            per_block += 1
        if self.n_loops > 1:
            per_block += (self.n_loops - 1) * (2 + (1 if self.x0_mix else 0))
        total = self.vocab_size * d + self.n_layer * per_block + d  # embedding + blocks + ln_f
        if self.v0_mix and self.n_layer > 1:
            total += (self.n_layer - 1) * self.n_kv_head
        if self.use_confidence_head:
            total += d + 1
        if self.byte_mix_kernel:
            n_taps = (self.byte_mix_kernel - 1) * d
            total += n_taps * (2 if self.byte_mix_bilinear else 1)
        return total

    def n_params_non_embedding(self) -> int:
        return self.n_params() - self.vocab_size * self.n_embd

    def flops_per_token(self, seq_len: Optional[int] = None, training: bool = True) -> float:
        """Approximate matmul FLOPs per token (2 per multiply-add; training = 3x forward).
        Includes the lm_head and the attention score/value products at the *effective*
        number of keys each layer attends to (full causal for global layers, the window for local)."""
        T = seq_len or self.block_size
        d, hd, h = self.n_embd, self.head_dim, self.ffn_hidden
        q_out, kv_out = self.n_head * hd, self.n_kv_head * hd
        proj = d * q_out + 2 * d * kv_out + q_out * d + 3 * d * h
        W = self.sliding_window
        keys_global = (T + 1) / 2.0
        keys_local = (T + 1) / 2.0 if T <= W else W - W * (W - 1) / (2.0 * T)
        attn = 0.0
        for i in range(self.n_layer):
            keys = keys_global if self.is_global_layer(i) else keys_local
            attn += 4.0 * keys * q_out        # QK^T and AV, 2 FLOPs per MAC
        fwd = 2.0 * (self.n_loops * self.n_layer * proj + self.vocab_size * d) + self.n_loops * attn
        return fwd * (3.0 if training else 1.0)

    def describe(self) -> str:
        n = self.n_params()
        kinds = "".join("G" if self.is_global_layer(i) else "l" for i in range(self.n_layer))
<<<<<<< HEAD
        loops = f"x{self.n_loops}" if self.n_loops > 1 else ""
        return (f"{n / 1e6:.2f}M params | L={self.n_layer}{loops} d={self.n_embd} "
=======
        rf = self.receptive_field()
        reach = f"reach={rf}" + ("" if rf == self.block_size else f" (LOCAL-ONLY: sees at most {rf} of {self.block_size} bytes)")
        return (f"{n / 1e6:.2f}M params | L={self.n_layer} d={self.n_embd} "
>>>>>>> 2ce839cc872fad399fe0c8bb31dc531bcfdf000b
                f"heads={self.n_head}/{self.n_kv_head}kv (hd={self.head_dim}) ffn={self.ffn_hidden} "
                f"ctx={self.block_size} win={self.sliding_window} layers[{kinds}] {reach}")
