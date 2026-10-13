"""Model-size presets: the **growth ladder** 1M -> 2M -> 5M -> 10M -> 50M -> 115M, plus the
original named presets (unchanged, so old checkpoints/configs keep their meaning).

Ladder design rules (so ``grow_model`` can walk the whole ladder with exact function
preservation -- verified for every adjacent pair in tests/test_presets.py):

  * ``head_dim = 32`` and GQA ratio 2:1 everywhere  (heads = d/32, kv heads = d/64);
  * ``n_embd``, ``n_layer``, FFN width and ``block_size`` are non-decreasing rung to rung;
  * the attention pattern (``global_every``, ``sliding_window``) is identical, so layer *i*
    keeps its type when new layers are appended;
  * FFN width is solved per rung so the parameter count lands within ~1% of the name.

Training defaults (batch size, learning rate, byte budget) are starting points chosen from
standard width-scaling heuristics and the small ablations in tools/ablate.py; they are not
tuned per-size on large runs. Treat them as defaults, not optima.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .config import TensegrityConfig
from .schedule import RunPlan, make_plan
from .train import TrainConfig, default_ema_decay

# --------------------------------------------------------------------------- legacy presets
BASE15M_CONFIG_KWARGS: Dict[str, Any] = dict(
    n_layer=4, n_embd=512, n_head=4, n_kv_head=2, ffn_mult=4.0, ffn_align=64,
    block_size=256, sliding_window=124, global_every=4)

BALANCED135M_CONFIG_KWARGS: Dict[str, Any] = dict(
    n_layer=12, n_embd=1024, n_head=16, n_kv_head=4, ffn_mult=8.0 / 3.0, ffn_align=64,
    block_size=2048, sliding_window=512, global_every=4)

HIGHINTEL400M_CONFIG_KWARGS: Dict[str, Any] = dict(
    n_layer=16, n_embd=1536, n_head=12, n_kv_head=4, ffn_mult=8.0 / 3.0, ffn_align=64,
    block_size=2048, sliding_window=512, global_every=4)

LEGACY_PRESETS: Dict[str, Dict[str, Any]] = {
    "base15m": BASE15M_CONFIG_KWARGS,
    "balanced135m": BALANCED135M_CONFIG_KWARGS,
    "highintel400m": HIGHINTEL400M_CONFIG_KWARGS,
}

# ------------------------------------------------------------------------------ the ladder
_HEAD_DIM = 32

# Architecture shared by every rung (and therefore frozen along the ladder).
LADDER_COMMON: Dict[str, Any] = dict(
    block_size=1024, sliding_window=256, global_every=3, ffn_align=32,
    layerdrop_max=0.0, grad_checkpoint=False,
    use_qk_norm=True, embed_scale=False, logit_soft_cap=30.0, z_loss_weight=1e-4,
    confidence_detach=True, byte_mix_kernel=4, x0_mix=False, homeo_weight=0.01,
)  # embed_scale / byte_mix_kernel / optimizer choices: see tools/ablation_*.log and the README


@dataclass(frozen=True)
class Preset:
    name: str                 # "5m"
    label: str                # "5M"
    n_embd: int
    n_layer: int
    ffn_hidden: int
    batch_tokens: int         # tokens per optimizer step
    learning_rate: float      # peak AdamW LR (embeddings, norms, scalars; whole model when optimizer="adamw")
    default_tokens: int       # default training byte budget
    min_vram_gb: float        # rough floor for comfortable training at the default micro-batch
    blurb: str

    def model_kwargs_params(self) -> int:
        """Parameter count of this rung's default configuration."""
        return TensegrityConfig(**self.model_kwargs).n_params()

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        heads = self.n_embd // _HEAD_DIM
        return {**LADDER_COMMON, "n_layer": self.n_layer, "n_embd": self.n_embd, "n_head": heads,
                "n_kv_head": heads // 2, "ffn_mult": self.ffn_hidden / self.n_embd}


LADDER: Tuple[Preset, ...] = (
    Preset("1m", "1M", 128, 5, 384, 32_768, 4.0e-3, 300_000_000, 1.0,
           "Smoke-test scale. Learns spelling, word shapes and simple sentence rhythm in minutes."),
    Preset("2m", "2M", 192, 5, 480, 32_768, 2.8e-3, 400_000_000, 1.0,
           "Fastest rung that produces locally coherent sentences on narrow corpora."),
    Preset("5m", "5M", 256, 7, 672, 65_536, 2.0e-3, 600_000_000, 2.0,
           "Sweet spot for a free-tier GPU: a few hours for a clearly coherent story model."),
    Preset("10m", "10M", 320, 9, 832, 65_536, 1.6e-3, 800_000_000, 3.0,
           "Noticeably better long-range consistency; still comfortable on a T4."),
    Preset("50m", "50M", 512, 15, 1664, 131_072, 1.0e-3, 1_200_000_000, 8.0,
           "Serious small model. T4: ~a day of compute; L4/A100 recommended."),
    Preset("115m", "115M", 768, 18, 2016, 262_144, 6.5e-4, 1_750_000_000, 12.0,
           "Top of the ladder (GPT-2-small class parameter count, byte-level). A100/L4 or patience."),
)

_BY_NAME: Dict[str, Preset] = {p.name: p for p in LADDER}
LADDER_NAMES: Tuple[str, ...] = tuple(p.name for p in LADDER)
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {**LEGACY_PRESETS, **{p.name: p.model_kwargs for p in LADDER}}


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "")


def get_preset(name: str) -> Preset:
    """Ladder preset (training defaults + shape) by name, case-insensitive: "5m", "5M"."""
    key = _norm(name)
    if key not in _BY_NAME:
        raise ValueError(f"unknown ladder preset {name!r}; available: {list(LADDER_NAMES)}")
    return _BY_NAME[key]


def get_preset_config(name: str, **overrides: Any) -> TensegrityConfig:
    """TensegrityConfig for a ladder or legacy preset, with optional field overrides
    (e.g. ``get_preset_config("5m", dropout=0.1, block_size=2048)``)."""
    key = _norm(name)
    if key not in MODEL_PRESETS:
        raise ValueError(f"unknown preset {name!r}; available presets: {sorted(MODEL_PRESETS)}")
    kwargs = dict(MODEL_PRESETS[key])
    kwargs.update(overrides)
    return TensegrityConfig(**kwargs)


def next_preset(name: str) -> Optional[Preset]:
    """The rung above ``name`` on the ladder (None at the top)."""
    i = LADDER_NAMES.index(get_preset(name).name)
    return LADDER[i + 1] if i + 1 < len(LADDER) else None


def describe_presets() -> str:
    rows = [f"{'preset':7s}{'params':>10s}{'layers':>7s}{'d_model':>8s}{'heads/kv':>9s}{'ffn':>6s}"
            f"{'ctx':>6s}{'batch tok':>10s}{'peak lr':>9s}{'byte budget':>12s}"]
    for p in LADDER:
        c = get_preset_config(p.name)
        rows.append(f"{p.label:7s}{c.n_params() / 1e6:9.2f}M{c.n_layer:7d}{c.n_embd:8d}"
                    f"{c.n_head:5d}/{c.n_kv_head:<3d}{c.ffn_hidden:6d}{c.block_size:6d}"
                    f"{p.batch_tokens:10,d}{p.learning_rate:9.1e}{p.default_tokens / 1e6:10.0f}M")
    return "\n".join(rows)


def train_setup(name: str, micro_batch: int, grad_accum: int, total_tokens: Optional[int] = None,
                optimizer: str = "muon", schedule: str = "wsd", ema: bool = True,
                plan: Optional[RunPlan] = None, **tc_overrides: Any) -> Tuple[TrainConfig, RunPlan]:
    """Preset training defaults -> ``(TrainConfig, RunPlan)`` for a concrete micro-batch / accumulation.

    ``total_tokens`` defaults to the preset's budget. Pass ``plan`` to reuse an existing plan
    (e.g. one derived from a wall-clock budget by ``plan_for_time_budget``)."""
    p = get_preset(name)
    cfg = get_preset_config(p.name)
    tokens_per_step = micro_batch * grad_accum * cfg.block_size
    plan = plan or make_plan(total_tokens or p.default_tokens, tokens_per_step, schedule)
    tc = TrainConfig(
        batch_size=micro_batch, grad_accum_steps=grad_accum, learning_rate=p.learning_rate,
        min_lr=p.learning_rate / 10.0, optimizer=optimizer,
        muon_weight_decay=0.0,   # measured: decay on the Muon matrices hurt at 500 steps (tools/ablation_round2.log)
        ema_decay=default_ema_decay(plan.max_steps) if ema else 0.0,
        eval_interval=max(50, plan.max_steps // 20), log_interval=max(1, plan.max_steps // 100))
    for k, v in tc_overrides.items():
        if not hasattr(tc, k):
            raise TypeError(f"TrainConfig has no field {k!r}")
        setattr(tc, k, v)
    return tc, plan
