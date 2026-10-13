"""Optimizers: AdamW (fused when available) and a single-device Muon + auxiliary AdamW.

Muon is vendored from https://github.com/KellerJordan/Muon (SingleDeviceMuonWithAuxAdam),
the reference implementation behind the modded-nanogpt speedrun. The Newton-Schulz
coefficients and the momentum / scale formulas are reproduced verbatim -- they are exactly
the numbers those results are evidence for, not values to casually retune.

Deviations from upstream (numerics only, not the algorithm):
  * Newton-Schulz runs in bf16 only on hardware with *native* bf16 (compute capability >= 8).
    On a Turing T4 it runs in fp16 (tensor-core fast); on CPU in fp32. The original release
    used `torch.cuda.is_bf16_supported()`, which is True on a T4 through emulation.
  * The Frobenius normalisation is done in fp32 before down-casting, so tiny gradients cannot
    underflow to a zero norm in fp16.
  * The gradient is not modified in place.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


# =============================================================================
# Muon
# =============================================================================

def _ns_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return torch.float32


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz iteration: approximate orthogonalisation of G (all singular
    values pushed to ~U(0.5, 1.5)), the cheap stand-in for the SVD-based polar factor."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    X = X.to(_ns_dtype(G.device))
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95,
                ns_steps: int = 5, nesterov: bool = True) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = torch.lerp(grad, momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # conv filters; unused by this model, kept for fidelity to upstream
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    return update * max(1.0, update.size(-2) / update.size(-1)) ** 0.5


def adam_update(grad: torch.Tensor, buf1: torch.Tensor, buf2: torch.Tensor, step: int,
                betas: Tuple[float, float], eps: float) -> torch.Tensor:
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    return (buf1 / (1 - betas[0] ** step)) / ((buf2 / (1 - betas[1] ** step)).sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """One optimizer object: Muon on groups with ``use_muon=True``, AdamW on the rest."""

    def __init__(self, param_groups: List[Dict[str, Any]]):
        for g in param_groups:
            if "use_muon" not in g:
                raise ValueError("every param group needs a 'use_muon' flag")
            if g["use_muon"]:
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.0)
            else:
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-10)
                g.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        return loss


# =============================================================================
# Parameter partitioning + factory
# =============================================================================

def _is_no_decay(p: nn.Parameter) -> bool:
    return p.dim() < 2 or getattr(p, "_no_decay", False)


def configure_optimizer(model: nn.Module, weight_decay: float, learning_rate: float,
                        betas: Tuple[float, float], device_type: str,
                        use_8bit: bool = False, use_muon: bool = False,
                        muon_lr: float = 0.02, muon_momentum: float = 0.95,
                        muon_weight_decay: float = None) -> torch.optim.Optimizer:
    """
    AdamW (default): >=2-D weights are decayed; norms, scalars (residual gains, x0 gains),
    biases and ByteMix taps are not.

    use_muon=True: every >=2-D weight inside ``model.blocks`` (attention + SwiGLU matrices) is
    trained with Muon; everything else -- the tied token embedding / lm_head, the confidence
    head, ByteMix, and every 1-D / scalar parameter -- stays on AdamW. This is Muon's own
    guidance (hidden matrices only). The walk covers the whole model twice (blocks, then the
    remainder), so a parameter added to the model later can never silently fall through.

    muon_weight_decay: decoupled decay for the Muon group (default: ``weight_decay``).
    """
    seen = set()

    def take(params):
        out = []
        for p in params:
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    if use_muon:
        blocks = getattr(model, "blocks", [])
        muon_params = take(p for p in blocks.parameters() if p.dim() >= 2)
        decay = take(p for p in model.parameters() if not _is_no_decay(p))
        no_decay = take(model.parameters())
        mwd = weight_decay if muon_weight_decay is None else muon_weight_decay
        groups = [
            dict(params=muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=mwd, use_muon=True),
            dict(params=decay, lr=learning_rate, betas=betas, eps=1e-10, weight_decay=weight_decay, use_muon=False),
            dict(params=no_decay, lr=learning_rate, betas=betas, eps=1e-10, weight_decay=0.0, use_muon=False),
        ]
        return SingleDeviceMuonWithAuxAdam([g for g in groups if g["params"]])

    decay = take(p for p in model.parameters() if not _is_no_decay(p))
    no_decay = take(model.parameters())
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    groups = [g for g in groups if g["params"]]

    if use_8bit and device_type == "cuda" and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(groups, lr=learning_rate, betas=betas)
        except ImportError:
            pass

    extra: Dict[str, Any] = {}
    if device_type == "cuda" and torch.cuda.is_available():
        try:
            if "fused" in inspect.signature(torch.optim.AdamW).parameters:
                extra["fused"] = True
        except (TypeError, ValueError):
            pass
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)
