"""Learning-rate schedules and run planning (token budget / wall-clock budget -> steps)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay to ``min_lr`` ("temperature annealing")."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def get_lr_wsd(step: int, warmup_steps: int, stable_steps: int, decay_steps: int,
               max_lr: float, min_lr: float) -> float:
    """
    Warmup-Stable-Decay (MiniCPM; also used by Llama 3.1 and DeepSeek-V2): linear warmup,
    hold at ``max_lr``, then linear decay to ``min_lr``. The decay only has to be as long as
    ``decay_steps`` -- the total budget need not be fixed at step 0.
    """
    assert warmup_steps >= 1, "warmup_steps must be >= 1"
    assert decay_steps >= 1, "decay_steps must be >= 1"
    decay_start = warmup_steps + stable_steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step < decay_start:
        return max_lr
    if step >= decay_start + decay_steps:
        return min_lr
    return max_lr - (step - decay_start) / decay_steps * (max_lr - min_lr)


@dataclass
class RunPlan:
    """A concrete, serialisable training plan. Persisted next to the checkpoints so a resumed
    run (even in a new Colab session) keeps exactly the same schedule."""
    tokens_per_step: int
    max_steps: int
    warmup_steps: int
    stable_steps: int
    decay_steps: int
    schedule: str = "wsd"

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_step * self.max_steps

    def multiplier(self, step: int, floor_ratio: float) -> float:
        """LR as a fraction of the peak, in [floor_ratio, 1]."""
        if self.schedule == "wsd":
            return get_lr_wsd(step, self.warmup_steps, self.stable_steps, self.decay_steps, 1.0, floor_ratio)
        return get_lr(step, self.warmup_steps, self.max_steps, 1.0, floor_ratio)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_plan(total_tokens: int, tokens_per_step: int, schedule: str = "wsd",
              warmup_frac: float = 0.02, decay_frac: float = 0.20, min_warmup: int = 20) -> RunPlan:
    """Turn a token budget into steps. For WSD the final ``decay_frac`` of steps is the cool-down."""
    if schedule not in ("wsd", "cosine"):
        raise ValueError(f"schedule must be 'wsd' or 'cosine', got {schedule!r}")
    if tokens_per_step <= 0 or total_tokens <= 0:
        raise ValueError("total_tokens and tokens_per_step must be positive")
    max_steps = max(10, total_tokens // tokens_per_step)
    warmup = min(max(min_warmup, int(warmup_frac * max_steps)), max(1, max_steps // 4))
    decay = max(1, int(decay_frac * max_steps))
    stable = max(0, max_steps - warmup - decay)
    return RunPlan(tokens_per_step, max_steps, warmup, stable, decay, schedule)


def plan_for_time_budget(tokens_per_sec: float, hours: float, tokens_per_step: int,
                         max_tokens: Optional[int] = None, safety: float = 0.92, **kw) -> RunPlan:
    """Largest plan expected to finish within ``hours`` of pure training time (``safety`` leaves
    room for evaluation, checkpointing and throughput jitter), capped at ``max_tokens``."""
    if tokens_per_sec <= 0 or hours <= 0:
        raise ValueError("tokens_per_sec and hours must be positive")
    budget = int(tokens_per_sec * hours * 3600.0 * safety)
    if max_tokens is not None:
        budget = min(budget, int(max_tokens))
    return make_plan(max(budget, tokens_per_step * 10), tokens_per_step, **kw)


def extend_plan(plan: RunPlan, extra_steps: int, at_step: int = 0) -> RunPlan:
    """Lengthen a WSD run by ``extra_steps`` (e.g. "I have two more hours"). The extra steps join the
    *stable* phase, so the cool-down still happens at the end -- the property that makes WSD
    suitable for open-ended training. Only legal before the cool-down has started."""
    if plan.schedule != "wsd":
        raise ValueError("extend_plan needs a WSD plan (a cosine schedule's shape depends on its total length)")
    if extra_steps < 1:
        raise ValueError("extra_steps must be >= 1")
    if at_step > plan.warmup_steps + plan.stable_steps:
        raise ValueError(f"the cool-down already began at step {plan.warmup_steps + plan.stable_steps}; "
                         f"continue as a NEW run initialised from the weights instead")
    return RunPlan(plan.tokens_per_step, plan.max_steps + extra_steps, plan.warmup_steps,
                   plan.stable_steps + extra_steps, plan.decay_steps, "wsd")
