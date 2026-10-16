"""Climb the size ladder: train a small model, grow it (function-preserving), keep training.

    results = climb_ladder(["1m", "2m", "5m"], data_dir, "runs/ladder", hours=[0.2, 0.4, 1.0])

Each rung is trained with its own preset hyper-parameters; between rungs ``grow_model`` carries
the previous rung's best weights into the next architecture, *verifying* that the function was
preserved and that the new capacity is trainable before spending any compute on it.

The whole climb is resumable: finished rungs are recorded (``rung_done.json``) and skipped, an
interrupted rung resumes from its own checkpoint, and per-rung plans are persisted, so re-running
the same call after a Colab disconnect simply continues.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from .checkpoint import load_checkpoint, load_model
from .config import TensegrityConfig
from .growth import check_function_preserved, check_growth_trainable, grow_model, grow_optimizer_state, retarget
from .hw import autotune_micro_batch, measure_throughput, pick_device, pick_dtype
from .model import TensegrityLM
from .presets import get_preset, get_preset_config, train_setup
from .schedule import make_plan, plan_for_time_budget
from .train import Trainer, load_run_plan, preflight_or_fallback


@dataclass
class RungResult:
    rung: str
    params: int
    status: str                   # "completed" | "time_limit"
    steps: int
    tokens_seen: int
    best_val_loss: float
    best_val_bpb: float
    run_dir: str
    growth_max_logit_diff: Optional[float] = None
    seconds: float = 0.0
    skipped: bool = False         # True when loaded from a previous session


def _per_rung(value, n: int, name: str):
    if value is None:
        return [None] * n
    if isinstance(value, (int, float)):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError(f"{name} needs {n} entries (one per rung), got {len(value)}")
    return value


def climb_ladder(rungs: Sequence[str], data_dirs, root: str, *, hours=None, tokens=None, device: str = "auto",
                 dtype: str = "auto", optimizer: str = "muon", micro_batch: Optional[int] = None,
                 model_overrides: Optional[Dict[str, Any]] = None, tc_overrides: Optional[Dict[str, Any]] = None,
                 batch_tokens: Optional[int] = None, intermediate_cooldown: Optional[float] = None, data_weights: Optional[Sequence[float]] = None, lr_scale_after_growth: float = 1.0,
                 callbacks: Optional[Sequence[Callable]] = None, seed: int = 1337, verbose: bool = True,
                 ) -> List[RungResult]:
    """
    Train every rung in ``rungs`` in order, growing between them.

    hours / tokens : per-rung budgets (a number applies to every rung; a list gives one per rung).
        ``hours`` measures this machine's throughput on each rung and sizes the run to fit;
        ``tokens`` is an explicit byte budget; neither -> each preset's default budget.
    model_overrides : architecture overrides for the FIRST rung (e.g. ``block_size``); later rungs
        inherit every function-defining field through ``retarget``.
    lr_scale_after_growth : multiply the preset peak LR for grown rungs (1.0 = the preset's own).
    batch_tokens : tokens per optimizer step for EVERY rung (default: each preset's own; use a small
        value such as 4096 on CPU).
    intermediate_cooldown : fraction of a NON-final rung's steps spent cooling the LR down (default
        None = the usual 20%). A small value (e.g. 0.03) skips most of it, because the next rung re-warms
        anyway; in tools/ladder_vs_scratch.py this recovered ~0.01 bits/byte at equal compute, at the price
        of intermediate checkpoints that are not annealed (worse as standalone models).
    """
    rungs = [get_preset(r).name for r in rungs]
    if len(set(rungs)) != len(rungs):
        raise ValueError("rungs must be distinct")
    n = len(rungs)
    hours_l = _per_rung(hours, n, "hours")
    tokens_l = _per_rung(tokens, n, "tokens")
    if hours is not None and tokens is not None:
        raise ValueError("pass hours OR tokens, not both")
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    os.makedirs(root, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)

    results: List[RungResult] = []
    prev_dir: Optional[str] = None
    prev_cfg: Optional[TensegrityConfig] = None
    for i, name in enumerate(rungs):
        preset = get_preset(name)
        run_dir = os.path.join(root, f"rung{i}_{name}")
        done_path = os.path.join(run_dir, "rung_done.json")
        if os.path.exists(done_path):
            with open(done_path) as fh:
                r = RungResult(**{**json.load(fh), "skipped": True})
            say(f"[ladder] rung {i} ({preset.label}) already finished: val {r.best_val_bpb:.3f} bpb -- skipping")
            results.append(r)
            prev_dir = run_dir
            prev_cfg = TensegrityConfig.from_dict(load_checkpoint(os.path.join(run_dir, "ckpt_best.pt"))["config"])
            continue

        t0 = time.time()
        saved = load_run_plan(run_dir)
        growth_diff: Optional[float] = None
        init_opt_fn: Optional[Callable] = None
        if i == 0:
            cfg = get_preset_config(name, **(model_overrides or {}))
            model = TensegrityLM(cfg)
        else:
            assert prev_dir is not None and prev_cfg is not None
            cfg = retarget(prev_cfg, get_preset_config(name)).replace(
                **{k: v for k, v in (model_overrides or {}).items() if k in ("dropout", "layerdrop_max", "homeo_weight")})
            if saved is not None:  # interrupted mid-rung: the checkpoint will overwrite these weights anyway
                model = TensegrityLM(saved[2])
                cfg = saved[2]
            else:
                parent, parent_cfg, _ = load_model(os.path.join(prev_dir, "ckpt_best.pt"), "cpu")
                model, cfg = grow_model(parent, parent_cfg, target=cfg, device="cpu")
                growth_diff = check_function_preserved(parent, model, parent_cfg, "cpu", verbose=verbose)
                check_growth_trainable(model, parent_cfg, verbose=verbose)
                prev_latest = os.path.join(prev_dir, "ckpt_latest.pt")
                if os.path.exists(prev_latest):
                    prev_opt_state = load_checkpoint(prev_latest, "cpu").get("optimizer_state_dict")
                    if prev_opt_state is not None:
                        init_opt_fn = lambda new_opt, pm=parent, pos=prev_opt_state, nm=model: (  # noqa: E731
                            grow_optimizer_state(pm, pos, nm, new_opt))
                say(f"[ladder] grew {parent_cfg.n_params() / 1e6:.2f}M -> {cfg.n_params() / 1e6:.2f}M "
                    f"(function preserved, new capacity trainable)")
        model.to(dev)
        T = cfg.block_size

        if saved is not None:
            plan, tc, _ = saved
            for k, v in (tc_overrides or {}).items():     # this session's settings (limits, cadence) win
                setattr(tc, k, v)
            say(f"[ladder] rung {i} ({preset.label}): resuming its saved plan ({plan.max_steps:,} steps)")
        else:
            target = batch_tokens or preset.batch_tokens
            if micro_batch is not None:
                mb, accum = micro_batch, max(1, round(target / (micro_batch * T)))
            else:
                tune = autotune_micro_batch(model, T, dev, dt, target, verbose=False)
                mb, accum = tune["micro_batch"], tune["grad_accum"]
                model.grad_checkpoint = tune["grad_checkpoint"]
            tps_step = mb * accum * T
            plan_kw = {"decay_frac": intermediate_cooldown} if (intermediate_cooldown is not None and i < n - 1) else {}
            if hours_l[i] is not None:
                tps = measure_throughput(model, mb, T, dev, dt, steps=3, warmup=1)
                plan = plan_for_time_budget(tps, float(hours_l[i]), tps_step, max_tokens=preset.default_tokens, **plan_kw)
                say(f"[ladder] rung {i} ({preset.label}): {tps / 1e3:.1f}k tok/s -> {plan.total_tokens / 1e6:.0f}M bytes "
                    f"in ~{hours_l[i]} h")
            else:
                plan = make_plan(int(tokens_l[i] or preset.default_tokens), tps_step, **plan_kw)
            tc, plan = train_setup(name, mb, accum, plan=plan, optimizer=optimizer, dtype=dtype, seed=seed + i,
                                   **(tc_overrides or {}))
            if i > 0 and lr_scale_after_growth != 1.0:
                tc.learning_rate *= lr_scale_after_growth
                tc.min_lr *= lr_scale_after_growth
            tc = preflight_or_fallback(model, tc, dev, verbose=verbose)
            for k, v in (tc_overrides or {}).items():
                setattr(tc, k, v)

        trainer = Trainer(model, tc, plan, data_dirs, run_dir, data_weights=data_weights, device=dev,
                          callbacks=callbacks, init_optimizer=init_opt_fn)
        res = trainer.fit(resume=True)
        r = RungResult(rung=name, params=cfg.n_params(), status=res["status"], steps=res["step"],
                       tokens_seen=res["tokens_seen"], best_val_loss=res["best_val_loss"],
                       best_val_bpb=res["best_val_loss"] / math.log(2), run_dir=run_dir,
                       growth_max_logit_diff=growth_diff, seconds=time.time() - t0)
        results.append(r)
        if res["status"] != "completed":
            say(f"[ladder] stopped inside rung {i} ({preset.label}) with status {res['status']!r}; "
                f"re-run the same call to continue")
            return results
        with open(done_path, "w") as fh:
            json.dump({k: v for k, v in asdict(r).items() if k != "skipped"}, fh, indent=2)
        prev_dir, prev_cfg = run_dir, cfg
        if dev.startswith("cuda"):
            del trainer, model
            torch.cuda.empty_cache()
    return results
