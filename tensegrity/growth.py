"""Function-preserving model growth (Net2Net-style) and continual-learning tools (EWC).

``grow_model`` turns a trained small model into a larger one that computes *exactly the same
function* at the moment of growth -- and, unlike v1, whose new capacity can actually learn.

Why v1 growth along ``n_embd`` was a trap
-----------------------------------------
Every new weight region was zero-filled on BOTH sides of a product (new head: q=k=v=o=0; new
hidden unit: gate=up=down=0; new residual channel: every reader AND every writer = 0). A
product of two zeros has zero gradient w.r.t. both factors, forever, so the grown model matched
the old logits yet could never use its extra width (verified: every new region had exactly 0.0
gradient). The rule used here is *one factor zero, the other random*:

  * WRITERS into the residual stream (``o_proj`` columns/rows, ``w_down`` columns/rows, the
    embedding's new channels) are zero  -> the old function is untouched, and
  * READERS (q/k/v, gate/up input columns, new-head projections) keep a small random init
    -> the zero writers receive non-zero gradient on the first step.

RMSNorm statistics are kept exact by zero residual channels + the sqrt(old/new) weight rescale
(and the matching ``norm_eps`` rescale, stored in the config so it survives a save/load).
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from .checkpoint import unwrap_model
from .config import TensegrityConfig
from .data import ByteDataset
from .model import TensegrityLM

# Fields whose change would alter the computed function (or the parameter layout) in a way
# growth cannot compensate for. Everything else (dropout, loss weights, LayerDrop, init_std,
# checkpointing knobs, ...) is free to differ between rungs.
_LOCKED = (
    "vocab_size", "sliding_window", "global_every", "rope_base", "bias", "use_qk_norm",
    "embed_scale", "logit_soft_cap", "use_confidence_head", "byte_mix_kernel", "x0_mix",
)


def _check_compatible(old: TensegrityConfig, new: TensegrityConfig) -> None:
    errs = []
    for f in _LOCKED:
        if getattr(old, f) != getattr(new, f):
            errs.append(f"{f}: {getattr(old, f)!r} -> {getattr(new, f)!r} (changes the function)")
    if old.head_dim != new.head_dim:
        errs.append(f"head_dim: {old.head_dim} -> {new.head_dim} (RoPE frequencies depend on head_dim; "
                    f"grow by adding heads, not by widening them)")
    if old.gqa_ratio != new.gqa_ratio:
        errs.append(f"GQA ratio: {old.gqa_ratio} -> {new.gqa_ratio} (would re-map query heads to KV groups)")
    for f in ("n_layer", "n_embd", "n_head", "n_kv_head", "ffn_hidden", "block_size"):
        if getattr(new, f) < getattr(old, f):
            errs.append(f"{f}: {getattr(old, f)} -> {getattr(new, f)} (growth cannot shrink)")
    if errs:
        raise ValueError("incompatible growth target:\n  " + "\n  ".join(errs))


class GrowthError(ValueError):
    """Growth produced a model that does not faithfully contain its parent (state lost or altered)."""


def _rescale_factors(oc: TensegrityConfig, nc: TensegrityConfig) -> Tuple[float, float]:
    """``(norm_scale, gain_scale)`` -- the exact factors growth applies to carried state.

    Widening multiplies RMSNorm weights by sqrt(D0/D1) (the zero-padded channels dilute the mean square);
    with ``embed_scale`` the embedding multiplier sqrt(D) also grows by sqrt(D1/D0), so every residual
    gain must follow by the same factor. Single source of truth for ``grow_model`` AND its report."""
    D0, D1 = oc.n_embd, nc.n_embd
    wide = D1 > D0
    return ((D0 / D1) ** 0.5 if wide else 1.0), ((D1 / D0) ** 0.5 if (wide and oc.embed_scale) else 1.0)


def _rescale_map(oc: TensegrityConfig, nc: TensegrityConfig) -> Dict[str, float]:
    """State-dict key -> ``a`` for carried tensors whose values are multiplied by ``a`` (theta' = a * theta)."""
    norm_scale, gain_scale = _rescale_factors(oc, nc)
    out: Dict[str, float] = {}
    if norm_scale != 1.0:
        out["ln_f.weight"] = norm_scale
    for i in range(oc.n_layer):
        if norm_scale != 1.0:
            out[f"blocks.{i}.ln1.weight"] = norm_scale
            out[f"blocks.{i}.ln2.weight"] = norm_scale
        if gain_scale != 1.0:
            out[f"blocks.{i}.attn_gain"] = gain_scale
            out[f"blocks.{i}.ffn_gain"] = gain_scale
    return out


@dataclass
class GrowthReport:
    """What a growth step did, in a form that can be stored next to the checkpoint."""
    axes: List[str]
    old_params: int
    new_params: int
    changed_fields: Dict[str, List[Any]]            # config field -> [old, new]
    norm_eps: List[float]                            # [old, new]
    rescale: Dict[str, float]                        # carried tensors that were multiplied by a (theta' = a theta)
    carried_entries: int                             # old state entries found, with their values, in the new model
    embedded_entries: int                            # ... of which were placed into a larger tensor
    carried_scalars: int
    new_only: List[str]                              # new state entries that have no old counterpart (appended blocks)
    max_carry_deviation: float                       # max |new[corner] - a * old| over every carried entry
    checks: Dict[str, Any] = field(default_factory=dict)   # filled by verification steps (preservation, trainability, ...)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        return (f"growth [{', '.join(self.axes) or 'no change'}]: {self.old_params / 1e6:.2f}M -> {self.new_params / 1e6:.2f}M "
                f"params; {self.carried_entries} state entries carried ({self.embedded_entries} embedded in larger "
                f"tensors, {len(self.rescale)} rescaled), {len(self.new_only)} new, "
                f"max carry deviation {self.max_carry_deviation:.1e}")


def verify_growth_mapping(old_model: nn.Module, new_model: nn.Module, rescale: Optional[Mapping[str, float]] = None
                          ) -> Dict[str, Any]:
    """Independent audit of a growth step, not trusting the code that did it: every entry of the old
    model's ``state_dict`` must be present in the new one with its values in the leading corner,
    multiplied by the declared factor (``rescale``; 1 for everything else). Raises :class:`GrowthError`
    naming every violation -- an entry the transplant forgot (the v1 grower dropped the Q/K-norm
    scales), altered, or whose tensor shrank. Returns counts for the :class:`GrowthReport`."""
    old_sd, new_sd = unwrap_model(old_model).state_dict(), unwrap_model(new_model).state_dict()
    rescale = dict(rescale or {})
    new_m = unwrap_model(new_model)
    problems: List[str] = []
    carried = embedded = scalars = 0
    worst = 0.0
    for key, old in old_sd.items():
        if key == "lm_head.weight":                      # tied to token_emb.weight, audited through it
            continue
        if key not in new_sd:
            problems.append(f"{key}: present in the old model, absent from the grown one (state would be lost)")
            continue
        new = new_sd[key]
        if new.dim() != old.dim() or any(n < o for n, o in zip(new.shape, old.shape)):
            problems.append(f"{key}: shape {tuple(old.shape)} -> {tuple(new.shape)} (a carried tensor may only grow)")
            continue
        corner = new[tuple(slice(0, s) for s in old.shape)]
        a = rescale.get(key, 1.0)
        want = old if a == 1.0 else old * a
        dev = float((corner.float() - want.float()).abs().max()) if old.numel() else 0.0
        worst = max(worst, dev)
        if (a == 1.0 and not torch.equal(corner, want)) or (a != 1.0 and not torch.allclose(corner, want, rtol=1e-6, atol=0)):
            problems.append(f"{key}: carried values differ from the old model's"
                            f"{'' if a == 1.0 else f' x {a:.6g}'} (max |diff| {dev:.3e})")
            continue
        carried += 1
        scalars += old.numel()
        embedded += int(tuple(new.shape) != tuple(old.shape))
    if getattr(new_m, "lm_head", None) is not None and new_m.lm_head.weight is not new_m.token_emb.weight:
        problems.append("lm_head.weight is no longer tied to token_emb.weight")
    old_layers = 1 + max((int(k.split(".")[1]) for k in old_sd if k.startswith("blocks.")), default=-1)
    new_only = sorted(k for k in new_sd if k not in old_sd)
    stray = [k for k in new_only if not (k.startswith("blocks.") and int(k.split(".")[1]) >= old_layers)]
    if stray:
        problems.append(f"new state outside the appended blocks: {stray[:6]} (a locked field changed the architecture?)")
    if problems:
        raise GrowthError("growth did not carry the old model faithfully:\n  " + "\n  ".join(problems[:12])
                          + (f"\n  ... and {len(problems) - 12} more" if len(problems) > 12 else ""))
    return {"carried": carried, "embedded": embedded, "scalars": scalars, "new_only": new_only, "max_deviation": worst}


def retarget(old: TensegrityConfig, size_from: TensegrityConfig, keep_context: bool = True) -> TensegrityConfig:
    """Config for growing ``old`` towards the *size* of ``size_from`` (e.g. the next preset).

    Takes the width / depth / heads / FFN from ``size_from`` but keeps every function-defining
    field of ``old`` (attention pattern, RoPE base, norms, ByteMix, ...), so a model trained with a
    customised window or context still grows cleanly into a rung of the ladder. ``keep_context``
    keeps ``old.block_size`` (pass False to adopt ``size_from``'s context length)."""
    keep = {f: getattr(old, f) for f in _LOCKED}
    if keep_context:
        keep["block_size"] = old.block_size
    d = size_from.to_dict()
    d.update(keep)
    d["gain_init"] = None
    return TensegrityConfig(**d)


def derive_target(old: TensegrityConfig, new_n_layer: Optional[int] = None,
                  ffn_growth_mult: Optional[float] = None, new_n_embd: Optional[int] = None,
                  new_n_head: Optional[int] = None, new_n_kv_head: Optional[int] = None,
                  new_block_size: Optional[int] = None) -> TensegrityConfig:
    """Build the target config from the v1-style keyword arguments."""
    d = old.to_dict()
    if new_n_layer is not None:
        d["n_layer"] = new_n_layer
        d["gain_init"] = None if new_n_layer != old.n_layer else old.gain_init
    if ffn_growth_mult is not None:
        if ffn_growth_mult < 1.0:
            raise ValueError("ffn_growth_mult must be >= 1.0")
        d["ffn_mult"] = old.ffn_mult * ffn_growth_mult
    if new_n_embd is not None or new_n_head is not None:
        hd = old.head_dim
        n_embd = new_n_embd if new_n_embd is not None else new_n_head * hd
        if n_embd % hd != 0:
            raise ValueError(f"new_n_embd ({n_embd}) must be a multiple of head_dim ({hd}); e.g. "
                             f"{hd * (n_embd // hd + 1)}")
        implied = n_embd // hd
        if new_n_head is not None and new_n_head != implied:
            raise ValueError(f"new_n_head={new_n_head} would change head_dim (only {implied} keeps "
                             f"head_dim={hd}); omit new_n_head to have it inferred")
        d["n_embd"], d["n_head"] = n_embd, implied
        d["n_kv_head"] = new_n_kv_head if new_n_kv_head is not None else max(1, implied // old.gqa_ratio)
    if new_block_size is not None:
        d["block_size"] = new_block_size
    return TensegrityConfig(**d)


@torch.no_grad()
def _randomize_like_init(t: torch.Tensor, std: float) -> None:
    t.normal_(mean=0.0, std=std)


@torch.no_grad()
def _grow_block(ob: nn.Module, nb: nn.Module, oc: TensegrityConfig, nc: TensegrityConfig) -> None:
    D0, D1 = oc.n_embd, nc.n_embd
    hd = oc.head_dim
    q0, kv0 = oc.n_head * hd, oc.n_kv_head * hd
    h0 = oc.ffn_hidden
    std = nc.init_std
    scale, gain_scale = _rescale_factors(oc, nc)
    wide = D1 > D0

    # ---- norms: sqrt(old/new) keeps x/rms(x) * w exact when the new channels are zero
    for name in ("ln1", "ln2"):
        w_new, w_old = getattr(nb, name).weight, getattr(ob, name).weight
        w_new[:D0].copy_(w_old * scale)
    # ---- residual gains (embed_scale makes the stream sqrt(D1/D0) larger -> branches must follow)
    nb.attn_gain.copy_(ob.attn_gain * gain_scale)
    nb.ffn_gain.copy_(ob.ffn_gain * gain_scale)
    if ob.x0_gain is not None:
        nb.x0_gain.copy_(ob.x0_gain)  # x0 and the stream scale together: no extra factor

    # ---- attention.   READERS random (already init'd by the fresh model), WRITERS zero.
    a_o, a_n = ob.attn, nb.attn
    for lin in ("q_proj", "k_proj", "v_proj"):
        out0 = q0 if lin == "q_proj" else kv0
        w = getattr(a_n, lin).weight
        w[:out0, :D0].copy_(getattr(a_o, lin).weight)       # old heads, old channels: exact copy
        if wide:                                             # old heads reading NEW channels: input is 0 at growth,
            _randomize_like_init(w[:out0, D0:], std)         # so random init is free and breaks the symmetry later
        if oc.bias:
            getattr(a_n, lin).bias[:out0].copy_(getattr(a_o, lin).bias)
    w_o = a_n.o_proj.weight
    w_o.zero_()                                              # writers: new heads -> all dims, all heads -> new dims
    w_o[:D0, :q0].copy_(a_o.o_proj.weight)
    if oc.bias:
        a_n.o_proj.bias.zero_()
        a_n.o_proj.bias[:D0].copy_(a_o.o_proj.bias)
    if a_o.q_norm is not None:                               # head_dim unchanged -> per-head norms copy 1:1
        a_n.q_norm.weight.copy_(a_o.q_norm.weight)
        a_n.k_norm.weight.copy_(a_o.k_norm.weight)

    # ---- feed-forward
    f_o, f_n = ob.ffn, nb.ffn
    for lin in ("w_gate", "w_up"):
        w = getattr(f_n, lin).weight                        # (h1, D1): readers of the stream
        w[:h0, :D0].copy_(getattr(f_o, lin).weight)
        if wide:
            _randomize_like_init(w[:h0, D0:], std)
        if oc.bias:
            getattr(f_n, lin).bias[:h0].copy_(getattr(f_o, lin).bias)
        # w[h0:, :]  = fresh random init -> new hidden units are readers, they stay random
    w_d = f_n.w_down.weight                                  # (D1, h1): writer
    w_d.zero_()
    w_d[:D0, :h0].copy_(f_o.w_down.weight)
    if oc.bias:
        f_n.w_down.bias.zero_()
        f_n.w_down.bias[:D0].copy_(f_o.w_down.bias)


@torch.no_grad()
def grow_model(old_model: TensegrityLM, old_cfg: Optional[TensegrityConfig] = None, *,
               target: Optional[TensegrityConfig] = None,
               new_n_layer: Optional[int] = None, ffn_growth_mult: Optional[float] = None,
               new_n_embd: Optional[int] = None, new_n_head: Optional[int] = None,
               new_n_kv_head: Optional[int] = None, new_block_size: Optional[int] = None,
               device: Optional[Union[str, torch.device]] = None, seed: Optional[int] = 0,
               return_report: bool = False):
    """
    Return ``(new_model, new_cfg)``: a larger model that, at the moment of growth, computes the
    same function as ``old_model`` (within float rounding) and whose new capacity is trainable.
    With ``return_report=True`` returns ``(new_model, new_cfg, GrowthReport)``.

    Every entry of the old model's state is audited against the result by
    :func:`verify_growth_mapping` before this returns -- state the transplant failed to carry (or
    carried at the wrong scale) raises :class:`GrowthError` instead of producing a model that merely
    looks right on its first test input. ``old_model`` is never modified (not even moved between devices).

    Pass either ``target`` (a full target :class:`TensegrityConfig`, e.g. the next preset), or
    the v1-style keywords (``new_n_layer``, ``new_n_embd``, ``ffn_growth_mult``, ...).

    Growth axes
      * depth   -- new blocks are appended with residual gains = 0 (exact no-op, ReZero-style start);
      * FFN     -- new hidden units: random gate/up (readers), zero down (writer);
      * width   -- more heads and residual channels, per the reader/writer rule in the module doc;
      * context -- ``block_size`` may grow (RoPE is relative; weights are unchanged).
    Optimizer state is NOT carried over -- build a fresh optimizer for the grown model.
    """
    old_model = unwrap_model(old_model)
    old_cfg = old_cfg or old_model.cfg
    if target is None:
        target = derive_target(old_cfg, new_n_layer, ffn_growth_mult, new_n_embd, new_n_head,
                               new_n_kv_head, new_block_size)
    _check_compatible(old_cfg, target)
    new_cfg = target
    device = device if device is not None else next(old_model.parameters()).device

    # RMSNorm exactness under zero-padded channels. With the stream scaled by sigma and the norm
    # weight by sqrt(D0/D1), out = x / sqrt(mean_new + eps_new) * w matches the old norm iff
    #     eps_new = eps_old * sigma^2 * D0 / D1,   sigma^2 = D1/D0 if embed_scale else 1.
    # So eps is unchanged when embed_scale is on (the sqrt(D1) embedding factor cancels the
    # dilution of the mean-square) and scales by D0/D1 when it is off. Stored in the config so a
    # saved/reloaded grown model is still exactly the same function.
    D0_, D1_ = old_cfg.n_embd, new_cfg.n_embd
    sigma2 = (D1_ / D0_) if (D1_ > D0_ and old_cfg.embed_scale) else 1.0
    new_cfg = new_cfg.replace(norm_eps=old_cfg.norm_eps * sigma2 * D0_ / D1_)

    if seed is not None:
        torch.manual_seed(seed)  # reproducible random init of the new (reader) regions
    new_model = TensegrityLM(new_cfg).to(device)
    D0, D1 = old_cfg.n_embd, new_cfg.n_embd
    scale, _ = _rescale_factors(old_cfg, new_cfg)

    # ---- embedding (tied lm_head), final norm, confidence head, byte-mix
    emb = new_model.token_emb.weight
    emb.zero_()                                              # new channels = zero writers
    emb[:, :D0].copy_(old_model.token_emb.weight)
    new_model.ln_f.weight[:D0].copy_(old_model.ln_f.weight * scale)
    if old_model.confidence_head is not None:
        ch_n, ch_o = new_model.confidence_head, old_model.confidence_head
        ch_n.weight.zero_()                                  # single-factor zero: x_new becomes non-zero via writers
        ch_n.weight[:, :D0].copy_(ch_o.weight)
        ch_n.bias.copy_(ch_o.bias)
    if old_model.byte_mix is not None:
        new_model.byte_mix.weight.zero_()
        new_model.byte_mix.weight[:, :D0].copy_(old_model.byte_mix.weight)

    # ---- blocks
    for i in range(old_cfg.n_layer):
        _grow_block(old_model.blocks[i], new_model.blocks[i], old_cfg, new_cfg)
    for i in range(old_cfg.n_layer, new_cfg.n_layer):        # appended blocks: exact no-ops at init
        blk = new_model.blocks[i]
        blk.attn_gain.zero_()
        blk.ffn_gain.zero_()
        if blk.x0_gain is not None:
            blk.x0_gain.zero_()

    rescale = _rescale_map(old_cfg, new_cfg)
    stats = verify_growth_mapping(old_model, new_model, rescale)       # raises GrowthError if anything was lost
    if not return_report:
        return new_model, new_cfg
    od, nd = old_cfg.to_dict(), new_cfg.to_dict()
    axes = [name for name, grew in (
        ("depth", new_cfg.n_layer > old_cfg.n_layer), ("ffn", new_cfg.ffn_hidden > old_cfg.ffn_hidden),
        ("width", new_cfg.n_embd > old_cfg.n_embd or new_cfg.n_head > old_cfg.n_head),
        ("context", new_cfg.block_size > old_cfg.block_size)) if grew]
    report = GrowthReport(
        axes=axes, old_params=old_model.num_parameters(), new_params=new_model.num_parameters(),
        changed_fields={k: [od[k], nd[k]] for k in nd if od.get(k) != nd[k]},
        norm_eps=[old_cfg.norm_eps, new_cfg.norm_eps], rescale=rescale, carried_entries=stats["carried"],
        embedded_entries=stats["embedded"], carried_scalars=stats["scalars"], new_only=stats["new_only"],
        max_carry_deviation=stats["max_deviation"])
    return new_model, new_cfg, report


# =============================================================================
# Verification
# =============================================================================

@torch.no_grad()
def check_function_preserved(old_model: TensegrityLM, new_model: TensegrityLM, cfg: Optional[TensegrityConfig] = None,
                             device: Optional[str] = None, n_tokens: int = 48, batch: int = 2,
                             atol: float = 1e-4, verbose: bool = True) -> float:
    """Feed identical random byte sequences through both models (eval, fp32, TF32 off) and
    require the logits (and confidence scores) to agree within ``atol``. Raises on failure.
    Each model is fed on its own device (they may differ, e.g. a CPU parent and a GPU candidate); the
    comparison is made on the CPU. ``device`` is accepted for backward compatibility."""
    cfg = cfg or old_model.cfg
    od, nd = next(old_model.parameters()).device, next(new_model.parameters()).device
    was = (old_model.training, new_model.training)
    old_model.eval()
    new_model.eval()
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False  # TF32 rounding alone can exceed a tight atol
    torch.backends.cudnn.allow_tf32 = False
    try:
        g = torch.Generator().manual_seed(0)
        x = torch.randint(0, 256, (batch, min(n_tokens, cfg.block_size)), generator=g)
        lo, _, co = old_model(x.to(od))
        ln, _, cn = new_model(x.to(nd))
        diff = float((lo.cpu() - ln.cpu()).abs().max())
        if co is not None and cn is not None:
            diff = max(diff, float((co.cpu() - cn.cpu()).abs().max()))
        ok = diff < atol
        if verbose:
            print(f"[grow] max |old - new| over logits/confidence = {diff:.3e} ({'PASS' if ok else 'FAIL'}, atol={atol:.0e})")
        if not ok:
            raise AssertionError("grow_model() did not preserve the model's function; do not train this model.")
        return diff
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
        old_model.train(was[0])
        new_model.train(was[1])


def _new_regions(old: TensegrityConfig, name: str, shape: Tuple[int, ...], layer: int) -> List[torch.Tensor]:
    """Boolean masks of the entries of ``name`` that did not exist in the old model AND are
    expected to receive gradient. Entries that *read* the new residual channels are excluded:
    their input is exactly zero at growth time (that is what keeps the function unchanged), so
    their gradient is legitimately zero until a writer has moved."""
    full = lambda: torch.ones(shape, dtype=torch.bool)  # noqa: E731
    if layer >= old.n_layer:
        return [full()] if len(shape) >= 2 else []
    hd, D0 = old.head_dim, old.n_embd
    q0, kv0, h0 = old.n_head * hd, old.n_kv_head * hd, old.ffn_hidden
    regions = []
    if name.endswith(("attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight")):
        out0 = q0 if name.endswith("q_proj.weight") else kv0
        m = torch.zeros(shape, dtype=torch.bool)
        m[out0:, :D0] = True                        # new heads reading old channels
        regions.append(m)
    elif name.endswith("attn.o_proj.weight"):
        m = torch.zeros(shape, dtype=torch.bool)
        m[:, q0:] = True                            # new heads writing
        regions.append(m)
        m = torch.zeros(shape, dtype=torch.bool)
        m[D0:, :] = True                            # writing into new channels
        regions.append(m)
    elif name.endswith(("ffn.w_gate.weight", "ffn.w_up.weight")):
        m = torch.zeros(shape, dtype=torch.bool)
        m[h0:, :D0] = True                          # new hidden units reading old channels
        regions.append(m)
    elif name.endswith("ffn.w_down.weight"):
        m = torch.zeros(shape, dtype=torch.bool)
        m[:, h0:] = True                            # new hidden units writing
        regions.append(m)
        m = torch.zeros(shape, dtype=torch.bool)
        m[D0:, :] = True                            # writing into new channels
        regions.append(m)
    return [r for r in regions if r.any()]


def check_growth_trainable(new_model: TensegrityLM, old_cfg: TensegrityConfig, x: Optional[torch.Tensor] = None,
                           y: Optional[torch.Tensor] = None, probe_step: float = 0.05,
                           verbose: bool = True) -> Dict[str, float]:
    """
    Prove the new capacity can learn. On a throw-away copy of ``new_model`` this runs two
    forward/backward passes with one normalised SGD step in between, and requires that every
    region added by growth (new heads, new hidden units, new residual channels, new blocks)
    receives a non-zero gradient in at least one pass.

    Two passes are needed because some gradients are *legitimately* zero at the instant of
    growth -- a ReZero-style block with gain 0 only wakes its inner weights once the gain has
    moved; the last block's writers into new channels only get signal once an earlier block has
    written into them. A *dead* region (the v1 width-growth failure: a product of two zeros) stays
    at exactly 0 in both passes. Raises AssertionError listing dead regions. Does not modify
    ``new_model``.
    """
    new = new_model.cfg
    probe = copy.deepcopy(new_model)
    dev = next(probe.parameters()).device
    if x is None:
        g = torch.Generator().manual_seed(1)
        x = torch.randint(0, 256, (4, min(32, new.block_size)), generator=g).to(dev)
        y = torch.randint(0, 256, (4, x.shape[1]), generator=g).to(dev)
    probe.train()
    probe._drop_ps = [0.0] * len(probe._drop_ps)
    probe.grad_checkpoint = False

    best: Dict[str, float] = {}

    def record() -> None:
        for name, p in probe.named_parameters():
            if p.grad is None:
                continue
            if name.startswith("blocks."):
                layer = int(name.split(".")[1])
                if layer >= old_cfg.n_layer and name.endswith(("attn_gain", "ffn_gain")):
                    best[name] = max(best.get(name, 0.0), float(p.grad.abs().max()))
                    continue
                for k, mask in enumerate(_new_regions(old_cfg, name, tuple(p.shape), layer)):
                    key = f"{name}#{k}"
                    best[key] = max(best.get(key, 0.0), float(p.grad[mask.to(p.grad.device)].abs().max()))
            elif name == "token_emb.weight" and new.n_embd > old_cfg.n_embd:
                best[name] = max(best.get(name, 0.0), float(p.grad[:, old_cfg.n_embd:].abs().max()))

    for pass_idx in range(2):
        probe.zero_grad(set_to_none=True)
        _, losses, _ = probe(x, y)
        losses["total_loss"].backward()
        record()
        if pass_idx == 0:
            gn = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in probe.parameters() if p.grad is not None))
            with torch.no_grad():
                for p in probe.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-probe_step / (gn + 1e-12))
    dead = sorted(k for k, v in best.items() if v == 0.0)
    if verbose:
        print(f"[grow] trainability: {len(best)} new regions probed, "
              f"min max-|grad| = {min(best.values()) if best else float('nan'):.3e}, dead = {len(dead)}")
    if dead:
        raise AssertionError("grown model has regions that can never learn (gradient stays exactly 0): "
                             + ", ".join(dead[:8]))
    return best


# =============================================================================
# Elastic Weight Consolidation (continual learning)
# =============================================================================

FISHER_KINDS = ("empirical", "sampled")


def compute_fisher_diagonal(model: TensegrityLM, data_dir: str, device: Optional[str] = None,
                            n_sequences: int = 64, seq_len: Optional[int] = None, dtype: str = "auto",
                            seed: int = 0, kind: str = "empirical"
                            ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Diagonal Fisher estimate, ``mean over sequences of (gradient of that sequence's loss)^2``, one
    backward pass per sequence (v1 squared an 8-sequence batch mean, which underestimates it by
    roughly the batch size). Returns ``(fisher, params_snapshot)``.

    Observation unit: **one sequence of ``seq_len`` bytes, loss = mean cross-entropy over its
    positions** (so values scale with 1/seq_len^2 relative to a per-sequence log-likelihood *sum*; the
    EWC strength ``lambda`` absorbs that constant). ``kind`` says which gradients are squared:

    * ``"empirical"`` (default): the gradient of the loss on the TRUE next bytes. Cheap and standard in
      EWC, but it is **not** the Fisher information -- it is the second moment of the loss gradient under
      the data distribution, which differs from the Fisher whenever the model is not already the
      data-generating distribution (Kunstner et al., 2019, arXiv:1905.12558).
    * ``"sampled"``: the Monte-Carlo true Fisher -- the next bytes are SAMPLED from the model's own
      predictive distribution and the gradient of their log-likelihood is squared. Same cost; this is the
      quantity the Laplace approximation behind EWC actually calls for.

    Gradients are computed in fp32 for stability, in eval mode (no dropout / LayerDrop).
    """
    if kind not in FISHER_KINDS:
        raise ValueError(f"kind must be one of {FISHER_KINDS}, got {kind!r}")
    model = unwrap_model(model)
    cfg = model.cfg
    device = device or str(next(model.parameters()).device)
    seq_len = seq_len or cfg.block_size
    ds = ByteDataset(data_dir, seed=seed)
    batches = ds.fixed_batches("train", n_sequences, 1, seq_len, device, seed=seed)
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed + 1)
    fisher = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in model.named_parameters() if p.requires_grad}
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            if kind == "empirical":
                _, losses, _ = model(x, y)
                loss = losses["ce_loss"]
            else:
                logits, _, _ = model(x)
                flat = logits.float().reshape(-1, logits.size(-1))
                y_s = torch.multinomial(torch.softmax(flat.detach(), -1).cpu(), 1, generator=gen).to(flat.device)
                loss = torch.nn.functional.cross_entropy(flat, y_s.view(-1))
            loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach().float().pow(2)
    for n in fisher:
        fisher[n] /= float(len(batches))
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return fisher, snapshot


@torch.no_grad()
def expand_reference_state(old_fisher: Dict[str, torch.Tensor], old_params: Dict[str, torch.Tensor],
                           new_model: nn.Module, rescale: Optional[Union[Mapping[str, float], GrowthReport]] = None
                           ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Re-map an (old_fisher, old_params) pair onto a grown model so the EWC penalty starts at 0 and
    keeps its meaning.

    * New entries (new heads / units / channels) get Fisher 0 and anchor = their current value: EWC puts
      no constraint on the new capacity.
    * Carried entries that growth RESCALED (``theta' = a * theta``: RMSNorm weights and residual gains
      under width growth) must move their anchor and importance with them: ``theta0' = a * theta0`` and
      ``F' = F / a^2`` -- the same quadratic penalty ``F (theta - theta0)^2`` in the new coordinates.
      Pass the ``rescale`` mapping (or the :class:`GrowthReport`) that ``grow_model(return_report=True)``
      returned. Without it every factor is taken as 1, which is only right for depth / FFN / context growth:
      after width growth the penalty of the *unchanged function* would be non-zero and EWC would pull the
      rescaled weights back, undoing the function preservation.

    The mapping is exact for these per-tensor rescalings; the importance of weights whose *role* changed
    (e.g. attention projections now sharing a wider residual stream) is carried over as an approximation --
    re-estimate the Fisher on the grown model (``compute_fisher_diagonal``) when that matters.
    """
    if not old_fisher or not old_params:
        return {}, {}
    factors: Mapping[str, float] = getattr(rescale, "rescale", rescale) or {}
    new_f, new_p = {}, {}
    for name, p in new_model.named_parameters():
        if name not in old_fisher:
            continue
        a = float(factors.get(name, 1.0))
        of, op = old_fisher[name].to(p.device) / (a * a), old_params[name].to(p.device, p.dtype) * a
        if of.shape == p.shape:
            new_f[name], new_p[name] = of, op
        else:
            nf = torch.zeros_like(p, dtype=torch.float32)
            nop = p.detach().clone()
            sl = tuple(slice(0, min(s, p.shape[i])) for i, s in enumerate(of.shape))
            nf[sl] = of[sl]
            nop[sl] = op[sl]
            new_f[name], new_p[name] = nf, nop
    return new_f, new_p


def ewc_penalty(model: nn.Module, fisher: Dict[str, torch.Tensor], old_params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """``sum_i F_i (theta_i - theta*_i)^2`` over parameters present in ``fisher`` (a quadratic
    approximation of the old task's loss around ``theta*``; see :func:`compute_fisher_diagonal` for what
    ``F`` is and is not)."""
    total = None
    for name, p in model.named_parameters():
        f = fisher.get(name)
        if f is None:
            continue
        term = (f * (p.float() - old_params[name].float()).pow(2)).sum()
        total = term if total is None else total + term
    return total if total is not None else torch.zeros((), device=next(model.parameters()).device)


def make_ewc_loss(fisher: Dict[str, torch.Tensor], old_params: Dict[str, torch.Tensor], lam: float):
    """Adapter for ``Trainer(extra_loss=...)``."""
    return lambda model: lam * ewc_penalty(model, fisher, old_params)
