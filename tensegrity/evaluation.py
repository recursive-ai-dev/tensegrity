"""Evaluation: exact byte accounting, named multi-domain suites, context probes, residual diagnostics.

* :func:`evaluate_batches` reports the loss in the units a byte model should be compared in. The
  cross-entropy is summed per target; targets that are real bytes (< 256) and targets that are special
  tokens (the EOS document marker) are accounted for separately, so **bits per byte = total NLL of the
  byte targets / number of byte targets / ln 2** (``bpb_bytes``) and the price of coding document
  boundaries (``bits_per_special``) is not smuggled into it. The older ``bpb`` field (loss / ln 2, i.e.
  bits per *target token*) is kept for continuity; the two agree when the targets contain no specials.
  Byte-token perplexity (``ppl``) is per token and must not be compared to a subword model's perplexity.
* :class:`EvalSuite` evaluates named domains separately (a continual-learning experiment needs "old
  corpus" and "new corpus" as different numbers) with fixed windows, and defines its aggregate explicitly.
* :func:`context_probe` measures what a model actually uses of its context with controlled copy tasks at
  increasing distances; :func:`residual_report` measures how much each sublayer changes the residual stream.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import manifest as M
from .checkpoint import unwrap_model
from .data import ByteDataset
from .hw import amp_settings, pick_dtype
from .layers import precompute_rope

ROLES = ("retain", "adapt", "monitor")
_LN2 = math.log(2.0)


# =============================================================================
# Exact accounting
# =============================================================================

@torch.no_grad()
def evaluate_batches(model: nn.Module, batches: Sequence, device: str, dtype: torch.dtype) -> Dict[str, float]:
    """Cross-entropy over fixed ``(x, y)`` batches with byte / special-token accounting.

    Returns ``loss`` (nats per target token), ``bpb`` (= loss / ln 2: bits per target token, the v1/v2
    number), ``ppl``, and the exact fields ``bpb_bytes`` (bits per raw byte, see module doc),
    ``bits_per_special`` (mean bits for special-token targets, NaN if there are none), ``n_tokens``,
    ``n_bytes``, ``n_special``, ``nll_bits_bytes``, ``nll_bits_special``."""
    amp = amp_settings(dtype, device)
    was_training = model.training
    model.eval()
    nll_b = nll_s = 0.0
    n_b = n_s = 0
    try:
        for x, y in batches:
            with torch.autocast(device_type=amp["autocast_device"], dtype=amp["amp_dtype"], enabled=amp["use_amp"]):
                logits, _, _ = model(x)
            t = y.reshape(-1)
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), t, ignore_index=-1, reduction="none")
            is_byte = (t >= 0) & (t < 256)
            is_special = t >= 256
            nll_b += float(ce[is_byte].sum())
            nll_s += float(ce[is_special].sum())
            n_b += int(is_byte.sum())
            n_s += int(is_special.sum())
    finally:
        model.train(was_training)
    n = n_b + n_s
    loss = (nll_b + nll_s) / max(n, 1)
    return {"loss": loss, "bpb": loss / _LN2, "ppl": math.exp(min(loss, 50.0)),
            "bpb_bytes": nll_b / n_b / _LN2 if n_b else float("nan"),
            "bits_per_special": nll_s / n_s / _LN2 if n_s else float("nan"),
            "n_tokens": n, "n_bytes": n_b, "n_special": n_s,
            "nll_bits_bytes": nll_b / _LN2, "nll_bits_special": nll_s / _LN2}


def evaluate(model: nn.Module, data_dir: str, cfg, device: str, dtype: Optional[torch.dtype] = None,
             n_batches: int = 20, batch_size: int = 16, split: str = "val") -> Dict[str, float]:
    """Convenience: :func:`evaluate_batches` on a fixed set of windows from ``data_dir``."""
    dtype = dtype or pick_dtype("auto", device)
    ds = ByteDataset(data_dir)
    batches = ds.fixed_batches(split, n_batches, batch_size, cfg.block_size, device)
    return evaluate_batches(model, batches, device, dtype)


# =============================================================================
# Multi-domain suite
# =============================================================================

@dataclass
class EvalDomain:
    """One named held-out set.  ``role`` documents its job in an experiment: ``"retain"`` (old knowledge
    that must not degrade), ``"adapt"`` (the new domain the model should improve on), ``"monitor"``."""
    name: str
    data_dir: str
    role: str = "retain"
    weight: float = 1.0
    split: str = "val"
    n_batches: int = 20
    batch_size: int = 16


@dataclass
class SuiteResult:
    """Per-domain metrics plus an explicit aggregate.

    * ``macro_*``: weighted mean over domains (weights = ``EvalDomain.weight``, equal by default) --
      every domain counts the same however large its corpus is;
    * ``micro_bpb_bytes``: total bits / total bytes over all domains (pooled; large evaluation sets dominate).
    ``selection_loss`` (= ``macro_loss``) is what the Trainer uses to pick the best checkpoint."""
    domains: Dict[str, Dict[str, float]]
    roles: Dict[str, str]
    macro_loss: float
    macro_bpb_bytes: float
    micro_bpb_bytes: float

    @property
    def selection_loss(self) -> float:
        return self.macro_loss

    def to_dict(self) -> Dict[str, Any]:
        return {"domains": self.domains, "roles": self.roles, "macro_loss": self.macro_loss,
                "macro_bpb_bytes": self.macro_bpb_bytes, "micro_bpb_bytes": self.micro_bpb_bytes}

    def delta(self, before: "SuiteResult") -> Dict[str, float]:
        """Per-domain change in bits per byte, ``self - before`` (negative = better)."""
        return {k: self.domains[k]["bpb_bytes"] - before.domains[k]["bpb_bytes"] for k in self.domains}

    def table(self) -> str:
        rows = [f"{'domain':16s}{'role':9s}{'bits/byte':>10s}{'nats/tok':>10s}{'bytes':>10s}"]
        for k, m in self.domains.items():
            rows.append(f"{k:16s}{self.roles[k]:9s}{m['bpb_bytes']:10.4f}{m['loss']:10.4f}{int(m['n_bytes']):10,d}")
        rows.append(f"{'macro mean':16s}{'':9s}{self.macro_bpb_bytes:10.4f}{self.macro_loss:10.4f}")
        rows.append(f"{'pooled':16s}{'':9s}{self.micro_bpb_bytes:10.4f}")
        return "\n".join(rows)


class EvalSuite:
    """A fixed evaluation set over named domains. Windows are drawn once per ``(block_size, device)`` with a
    fixed seed, so every model and every evaluation sees the same bytes; each domain's dataset is verified
    against its manifest and its fingerprint is part of the suite's identity."""

    def __init__(self, domains: Sequence[EvalDomain], seed: int = 0):
        domains = list(domains)
        if not domains:
            raise ValueError("an EvalSuite needs at least one domain")
        names = [d.name for d in domains]
        if len(set(names)) != len(names):
            raise ValueError(f"domain names must be unique, got {names}")
        for d in domains:
            if d.role not in ROLES:
                raise ValueError(f"domain {d.name!r}: role must be one of {ROLES}, got {d.role!r}")
            if d.weight <= 0 or d.n_batches < 1 or d.batch_size < 1:
                raise ValueError(f"domain {d.name!r}: weight, n_batches and batch_size must be positive")
            if d.split not in ("train", "val"):
                raise ValueError(f"domain {d.name!r}: split must be 'train' or 'val'")
        self.domains, self.seed = domains, int(seed)
        self._ds = {d.name: ByteDataset(d.data_dir, on_change="error") for d in domains}
        for ds in self._ds.values():
            ds.verify()
        self._cache: Dict[tuple, Dict[str, list]] = {}

    @classmethod
    def from_dirs(cls, dirs: Dict[str, str], roles: Optional[Dict[str, str]] = None, **domain_kw: Any) -> "EvalSuite":
        """``EvalSuite.from_dirs({"stories": dir_a, "plays": dir_b}, roles={"plays": "adapt"})``."""
        roles = roles or {}
        return cls([EvalDomain(n, d, roles.get(n, "retain"), **domain_kw) for n, d in dirs.items()])

    def batches(self, block_size: int, device: str) -> Dict[str, list]:
        key = (int(block_size), str(device))
        if key not in self._cache:
            self._cache[key] = {
                d.name: self._ds[d.name].fixed_batches(d.split, d.n_batches, d.batch_size, block_size, device,
                                                       seed=self.seed) for d in self.domains}
        return self._cache[key]

    def evaluate(self, model: nn.Module, device: Optional[str] = None, dtype: Optional[torch.dtype] = None,
                 block_size: Optional[int] = None) -> SuiteResult:
        model = unwrap_model(model)
        device = device or str(next(model.parameters()).device)
        dtype = dtype or pick_dtype("auto", device)
        T = block_size or model.cfg.block_size
        per = {name: evaluate_batches(model, b, device, dtype) for name, b in self.batches(T, device).items()}
        w = {d.name: d.weight for d in self.domains}
        tot_w = sum(w.values())
        macro_loss = sum(w[k] * per[k]["loss"] for k in per) / tot_w
        macro_bpb = sum(w[k] * per[k]["bpb_bytes"] for k in per) / tot_w
        bits = sum(per[k]["nll_bits_bytes"] for k in per)
        nbytes = sum(per[k]["n_bytes"] for k in per)
        return SuiteResult(per, {d.name: d.role for d in self.domains}, macro_loss, macro_bpb,
                           bits / nbytes if nbytes else float("nan"))

    # ------------------------------------------------------------------ identity
    def identity(self) -> Dict[str, Any]:
        """Everything that defines the evaluation set: the domain specs and the fingerprint of each dataset."""
        return {"seed": self.seed, "domains": [
            {"name": d.name, "role": d.role, "weight": d.weight, "split": d.split, "n_batches": d.n_batches,
             "batch_size": d.batch_size, "fingerprint": M.data_identity(d.data_dir)["fingerprint"]}
            for d in self.domains]}

    def fingerprint(self) -> str:
        return "sha256:" + hashlib.sha256(M.canonical_json(self.identity())).hexdigest()


# =============================================================================
# Long streams: exact windowed decoding vs the streaming cache
# =============================================================================

@torch.no_grad()
def long_stream_nll(model: nn.Module, ids: Sequence[int], modes: Sequence[str] = ("window", "streaming"),
                    device: Optional[str] = None, batch: int = 32) -> Dict[str, Dict[str, Any]]:
    """Teacher-forced bits per byte of a token stream LONGER than the context, under each decoding mode.

    Every target ``ids[t]`` (t >= 1) is scored from the context a generator would hold:

    * ``"window"``: the last ``block_size`` tokens re-encoded from scratch (what ``generate`` computes with
      ``cache_mode="window"``) -- the reference;
    * ``"streaming"``: the bounded KV cache of ``cache_mode="streaming"``, one token at a time.

    Targets up to ``block_size`` are identical for both (everything fits); beyond it the modes differ and the
    difference IS the approximation error of streaming, measured on your text. Returns per mode
    ``bpb`` (all byte targets), ``bpb_within`` (t <= block_size), ``bpb_beyond`` (t > block_size) and the
    per-position ``nll_bits`` array. Special-token targets are skipped in the means."""
    model = unwrap_model(model)
    device = device or str(next(model.parameters()).device)
    bs = model.cfg.block_size
    seq = torch.as_tensor(list(ids), dtype=torch.long, device=device)
    N = int(seq.numel())
    if N < 3:
        raise ValueError("need at least 3 tokens")
    was_training = model.training
    model.eval()
    results: Dict[str, Dict[str, Any]] = {}
    try:
        head = min(N, bs)
        logits, _, _ = model(seq[None, :head])
        head_nll = -torch.log_softmax(logits[0, :-1].float(), -1).gather(-1, seq[1:head, None]).squeeze(-1)
        for mode in modes:
            if mode not in ("window", "streaming"):
                raise ValueError(f"mode must be 'window' or 'streaming', got {mode!r}")
            nll = [head_nll]                                         # targets t = 1 .. head-1 (one prefix pass)
            if N > bs and mode == "window":
                starts = list(range(0, N - bs))                      # window s = context seq[s:s+bs] -> target t = s + bs
                for i in range(0, len(starts), batch):
                    chunk = starts[i:i + batch]
                    x = torch.stack([seq[s: s + bs] for s in chunk])
                    lg, _, _ = model(x)
                    tgt = seq[torch.tensor(chunk, device=device) + bs]
                    nll.append(-torch.log_softmax(lg[:, -1].float(), -1).gather(-1, tgt[:, None]).squeeze(-1))
            elif N > bs:
                lg, past, _ = model.forward_incremental(seq[None, :bs], None, 0, streaming=True)
                m = max(model.cfg.byte_mix_kernel - 1, 0)
                tail = seq[None, :bs][:, -m:] if m else None
                step_nll = [-torch.log_softmax(lg[0, -1].float(), -1)[seq[bs]]]       # t = bs: last prefill position
                for t in range(bs + 1, N):                           # feed seq[t-1] (stream position t-1), score seq[t]
                    lg, past, _ = model.forward_incremental(seq[None, t - 1:t], past, t - 1, tail, streaming=True)
                    step_nll.append(-torch.log_softmax(lg[0, -1].float(), -1)[seq[t]])
                    if m:
                        tail = torch.cat([tail, seq[None, t - 1:t]], dim=1)[:, -m:]
                nll.append(torch.stack(step_nll))
            all_nll = torch.cat(nll) / _LN2                          # index i scores target seq[i + 1]
            is_byte = (seq[1:] < 256)
            within = torch.arange(1, N, device=device) <= bs
            def mean(mask: torch.Tensor) -> float:
                return float(all_nll[mask].mean()) if bool(mask.any()) else float("nan")
            results[mode] = {"bpb": mean(is_byte), "bpb_within": mean(is_byte & within), "bpb_beyond": mean(is_byte & ~within),
                             "n_scored": int(is_byte.sum()), "nll_bits": all_nll.cpu().tolist()}
    finally:
        model.train(was_training)
    return results


# =============================================================================
# Context probes
# =============================================================================

_LETTERS = list(range(ord("a"), ord("z") + 1))


@torch.no_grad()
def context_probe(model: nn.Module, distances: Sequence[int] = (0, 16, 64, 256, 1024), device: Optional[str] = None,
                  key_len: int = 12, n_trials: int = 64, seed: int = 0) -> List[Dict[str, Any]]:
    """Controlled copy / retrieval test at increasing distances.

    Each trial is ``KEY . FILLER(d) . KEY`` over random lowercase letters (``KEY`` has ``key_len``
    letters, ``FILLER`` is ``d`` unrelated letters). The model is scored, teacher-forced, on predicting
    letters 2..``key_len`` of the *second* copy: they are unpredictable except by retrieving the first
    copy ``d + key_len`` positions back. A control trial replaces the first copy with unrelated letters.
    For every distance the result row has ``accuracy`` (chance = 1/26), ``nll_bits`` (copy trial),
    ``control_nll_bits`` (~log2 26 = 4.70), ``gain_bits`` (their difference: how much the earlier copy
    helped) and ``reachable`` (whether ``distance + key_len`` fits inside the architecture's
    :meth:`~tensegrity.config.TensegrityConfig.receptive_field`). Distances that do not fit in the context
    window are reported with ``fits=False`` and no measurement.

    An untrained model scores at chance everywhere; this is an instrument for trained models, to be read
    next to ``reachable`` (a local-only layout cannot beat chance beyond its receptive field, whatever
    it was trained on)."""
    model = unwrap_model(model)
    cfg = model.cfg
    device = device or str(next(model.parameters()).device)
    was_training = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    letters = torch.tensor(_LETTERS)
    rows: List[Dict[str, Any]] = []
    try:
        for d in distances:
            total = 2 * key_len + d
            row: Dict[str, Any] = {"distance": int(d), "fits": total <= cfg.block_size,
                                   "reachable": d + key_len <= cfg.receptive_field()}
            if not row["fits"]:
                rows.append(row)
                continue

            def sample(length: int) -> torch.Tensor:
                return letters[torch.randint(0, len(letters), (n_trials, length), generator=g)]
            key, filler, other = sample(key_len), sample(d), sample(key_len)
            result = {}
            for name, first in (("copy", key), ("control", other)):
                seq = torch.cat([first, filler, key], dim=1).to(device)
                logits, _, _ = model(seq[:, :-1])
                tgt = seq[:, 1:]
                pos = slice(key_len + d, total - 1)                  # predicting key[1:], given key[:-1] so far
                lp = torch.log_softmax(logits[:, pos].float(), -1)
                t = tgt[:, pos]
                nll = -lp.gather(-1, t.unsqueeze(-1)).squeeze(-1)
                result[name] = (float(nll.mean()) / _LN2, float((lp.argmax(-1) == t).float().mean()))
            row.update(accuracy=result["copy"][1], nll_bits=result["copy"][0], control_nll_bits=result["control"][0],
                       control_accuracy=result["control"][1], gain_bits=result["control"][0] - result["copy"][0])
            rows.append(row)
    finally:
        model.train(was_training)
    return rows


def format_probe(rows: Sequence[Dict[str, Any]], chance: float = 1 / 26) -> str:
    out = [f"{'distance':>9s} {'reachable':>9s} {'accuracy':>9s} {'gain (bits)':>12s}   (chance accuracy {chance:.3f})"]
    for r in rows:
        if not r["fits"]:
            out.append(f"{r['distance']:9d} {'-':>9s} {'(does not fit in the context window)':>24s}")
        else:
            out.append(f"{r['distance']:9d} {str(r['reachable']):>9s} {r['accuracy']:9.3f} {r['gain_bits']:12.2f}")
    return "\n".join(out)


# =============================================================================
# Residual-stream diagnostics
# =============================================================================

def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(x.float().pow(2).mean(dim=-1)).mean())


@torch.no_grad()
def residual_report(model: nn.Module, idx: torch.Tensor, eps: float = 1e-6) -> List[Dict[str, Any]]:
    """How much does each sublayer change the residual stream? Per block, for the attention and the FFN:

        rho = RMS(gain * f(x)) / (RMS(x) + eps)

    the size of what the branch *adds* relative to the stream it adds to (``branch_rms`` is the raw
    branch output, what the homeostasis loss regularises). The two can disagree by orders of magnitude
    -- a branch with RMS 1 behind a gain of 0.05 moves the stream 5% as much -- so instrument ``rho``
    before deciding whether regularising either helps. Eval mode, one forward pass over ``idx`` (B, T)."""
    model = unwrap_model(model)
    was_training = model.training
    model.eval()
    try:
        T = idx.size(1)
        x0 = model._embed(idx)
        x = x0
        cos_all, sin_all = model._rope_tables(idx.device)
        cos, sin = cos_all[:T], sin_all[:T]
        rows = []
        for i, b in enumerate(model.blocks):
            if b.x0_gain is not None:
                x = x + b.x0_gain * x0
            a = b.attn(b.ln1(x), cos, sin)
            rho_a = _rms(b.attn_gain * a) / (_rms(x) + eps)
            x = x + b.attn_gain * a
            f = b.ffn(b.ln2(x))
            rho_f = _rms(b.ffn_gain * f) / (_rms(x) + eps)
            x = x + b.ffn_gain * f
            rows.append({"layer": i, "kind": "global" if b.attn.is_global else "local", "stream_rms": _rms(x),
                         "attn_branch_rms": _rms(a), "attn_gain": float(b.attn_gain), "attn_rho": rho_a,
                         "ffn_branch_rms": _rms(f), "ffn_gain": float(b.ffn_gain), "ffn_rho": rho_f})
        return rows
    finally:
        model.train(was_training)
