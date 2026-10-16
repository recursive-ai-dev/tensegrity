"""Continual-learning building blocks: persistent EWC state and teacher-based distillation.

Both plug into :class:`~tensegrity.train.Trainer`:

* :class:`EWCState` -- the Fisher estimate, the anchor weights and the strength ``lambda`` as ONE object
  that can be saved next to a run, fingerprinted (so a resume can prove it is using the same
  constraint), and carried through growth with the correct transformation of anchors and importances.
  It is a loss *term*, not evidence of retention: compare it against plain replay under a matched budget
  with :mod:`tensegrity.experiment` before believing it helps.
* :class:`DistillLoss` -- KL(teacher || student) on the training batch. After growth the parent is the
  natural teacher: the candidate starts with identical outputs, and distillation keeps it near them while
  it adapts. This is an *approximate recovery procedure*, not a guarantee of function preservation.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import manifest as M
from .checkpoint import unwrap_model
from .growth import (FISHER_KINDS, GrowthReport, compute_fisher_diagonal, ewc_penalty, expand_reference_state)

_EWC_FORMAT = 1


def tensors_fingerprint(named: Mapping[str, torch.Tensor], extra: str = "") -> str:
    """``sha256:`` over names, shapes and exact values of tensors (key order independent)."""
    h = hashlib.sha256(extra.encode())
    for name in sorted(named):
        t = named[name].detach().cpu().contiguous()
        h.update(f"{name}|{tuple(t.shape)}|{t.dtype}|".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return "sha256:" + h.hexdigest()


class EWCState:
    """Elastic Weight Consolidation reference: ``penalty = lam * sum_i F_i (theta_i - theta*_i)^2``.

    ``meta`` records how ``F`` was obtained (``kind``, observation unit, sequences, data fingerprint) so a
    stored state is never mistaken for something it is not."""

    def __init__(self, fisher: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor], lam: float,
                 meta: Optional[Dict[str, Any]] = None):
        if lam < 0:
            raise ValueError(f"lam must be >= 0, got {lam}")
        if set(fisher) != set(params):
            raise ValueError("fisher and params must cover the same parameters")
        self.fisher, self.params, self.lam = fisher, params, float(lam)
        self.meta: Dict[str, Any] = dict(meta or {})
        self._fp: Optional[str] = None

    # ------------------------------------------------------------- construction
    @classmethod
    def estimate(cls, model: nn.Module, data_dir: str, lam: float, *, kind: str = "empirical", n_sequences: int = 64,
                 seq_len: Optional[int] = None, seed: int = 0, device: Optional[str] = None) -> "EWCState":
        """Estimate the Fisher of ``model`` on the TRAIN split of ``data_dir`` (see
        :func:`~tensegrity.growth.compute_fisher_diagonal` for what each ``kind`` is)."""
        if not isinstance(model, nn.Module):
            raise TypeError(f"estimate() needs a model object, got {type(model).__name__}; load a checkpoint with "
                            f"tensegrity.load_model(path) first")
        m = unwrap_model(model)
        fisher, snap = compute_fisher_diagonal(m, data_dir, device, n_sequences, seq_len, seed=seed, kind=kind)
        meta = {"kind": kind, "unit": "sequence (token-mean cross-entropy, one backward pass per sequence)",
                "n_sequences": n_sequences, "seq_len": seq_len or m.cfg.block_size, "seed": seed,
                "data_fingerprint": M.data_identity(data_dir)["fingerprint"], "n_params": m.num_parameters()}
        return cls(fisher, snap, lam, meta)

    # ------------------------------------------------------------------- the loss
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """The un-weighted penalty ``sum_i F_i (theta_i - theta*_i)^2``."""
        return ewc_penalty(unwrap_model(model), self.fisher, self.params)

    def __call__(self, model: nn.Module) -> torch.Tensor:
        """``lam * penalty`` -- usable directly as ``Trainer(extra_loss=...)``."""
        return self.lam * self.penalty(model)

    # ------------------------------------------------------------- growth + identity
    def expanded(self, new_model: nn.Module, report: Optional[Union[GrowthReport, Mapping[str, float]]] = None
                 ) -> "EWCState":
        """The same constraint on a grown model (see :func:`~tensegrity.growth.expand_reference_state`).
        Pass the :class:`GrowthReport`: without it, width growth leaves a spurious penalty on the rescaled
        weights."""
        f, p = expand_reference_state(self.fisher, self.params, unwrap_model(new_model), report)
        meta = {**self.meta, "carried_through_growth_from": self.fingerprint()}
        return EWCState(f, p, self.lam, meta)

    def fingerprint(self) -> str:
        if self._fp is None:
            core = {k: self.meta.get(k) for k in ("kind", "n_sequences", "seq_len", "seed", "data_fingerprint")}
            self._fp = tensors_fingerprint({**{"F." + k: v for k, v in self.fisher.items()},
                                            **{"P." + k: v for k, v in self.params.items()}}, f"ewc|{self.lam!r}|{core}")
        return self._fp

    # ---------------------------------------------------------------------- state
    def state_dict(self) -> Dict[str, Any]:
        return {"format": _EWC_FORMAT, "lam": self.lam, "meta": self.meta,
                "fisher": {k: v.detach().cpu() for k, v in self.fisher.items()},
                "params": {k: v.detach().cpu() for k, v in self.params.items()}}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EWCState":
        if state.get("format") != _EWC_FORMAT:
            raise ValueError(f"unsupported EWC state format {state.get('format')!r}")
        return cls(dict(state["fisher"]), dict(state["params"]), state["lam"], dict(state["meta"]))

    def save(self, path: str) -> None:
        from .checkpoint import atomic_torch_save
        atomic_torch_save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "EWCState":
        return cls.from_state_dict(torch.load(path, map_location=device, weights_only=True))

    def to(self, device: Union[str, torch.device]) -> "EWCState":
        self.fisher = {k: v.to(device) for k, v in self.fisher.items()}
        self.params = {k: v.to(device) for k, v in self.params.items()}
        return self


class DistillLoss:
    """Distillation of a frozen ``teacher`` into the model being trained, as a ``Trainer(batch_loss=...)``.

    ``weight * T^2 * KL(softmax(teacher/T) || softmax(student/T))`` averaged over the positions that have
    a target. The teacher runs in eval mode under the same autocast as the student, without gradients."""

    def __init__(self, teacher: nn.Module, weight: float = 1.0, temperature: float = 1.0):
        if weight < 0 or temperature <= 0:
            raise ValueError("weight must be >= 0 and temperature > 0")
        self.teacher = unwrap_model(teacher)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.weight, self.temperature = float(weight), float(temperature)
        self._fp: Optional[str] = None

    def to(self, device: Union[str, torch.device]) -> "DistillLoss":
        self.teacher.to(device)
        return self

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.teacher.cfg.block_size:
            raise ValueError(f"teacher context ({self.teacher.cfg.block_size}) is shorter than the training "
                             f"sequences ({x.size(1)})")
        with torch.no_grad():
            t_logits, _, _ = self.teacher(x)
        T = self.temperature
        log_s = F.log_softmax(logits.float() / T, dim=-1)
        t = F.softmax(t_logits.float() / T, dim=-1)
        kl = (t * (torch.log(t.clamp_min(1e-12)) - log_s)).sum(-1)            # (B, T)
        valid = (y != -1).float()
        return self.weight * T * T * (kl * valid).sum() / valid.sum().clamp_min(1.0)

    def fingerprint(self) -> str:
        if self._fp is None:
            self._fp = tensors_fingerprint(self.teacher.state_dict(), f"distill|{self.weight!r}|{self.temperature!r}")
        return self._fp


def hook_info(fn: Optional[Callable]) -> Optional[Dict[str, Any]]:
    """What a checkpoint records about a loss hook, so a resume can tell it is the same constraint."""
    if fn is None:
        return None
    info: Dict[str, Any] = {"type": type(fn).__name__}
    fp = getattr(fn, "fingerprint", None)
    if callable(fp):
        info["fingerprint"] = fp()
    return info
