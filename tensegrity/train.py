"""Training engine: one resumable, preemption-safe loop for notebooks, scripts and DDP.

What it does that the v1 loops did not
--------------------------------------
* **Correct resume.** Base learning rates are captured from the *fresh* optimizer before any
  state is loaded. (v1 captured them after ``optimizer.load_state_dict``, i.e. the already
  decayed LR, and decayed it a second time.) Model, optimizer, GradScaler, EMA, all RNGs and the
  data sampler are restored, so a resumed run continues the exact batch sequence.
* **Preemption safety.** Checkpoints every N minutes as well as every eval; atomic writes;
  an optional wall-clock limit that stops cleanly with a checkpoint (Colab sessions end).
* **No hidden syncs.** Losses accumulate on-device and are read only when logging.
* **Non-finite protection.** A NaN/Inf gradient skips the step instead of poisoning the weights.
* **Deterministic validation** (fixed windows) reported as loss, bits-per-byte and perplexity,
  for the raw weights and, if enabled, the EMA weights; the better one becomes ``ckpt_best.pt``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from . import manifest as M
from .checkpoint import (capture_rng_state, environment_info, load_checkpoint, restore_rng_state, save_checkpoint,
                         unwrap_model)
from .config import TensegrityConfig
from .continual import hook_info
from .data import ByteDataset
from .evaluation import EvalSuite, evaluate, evaluate_batches  # noqa: F401  (re-exported: tensegrity.train.evaluate*)
from .hw import amp_settings, device_type_of, gpu_info, mfu, pick_device, pick_dtype, set_fast_matmul
from .model import TensegrityLM
from .schedule import RunPlan, plan_change_problems
from .tokenizer import ByteTokenizer

Callback = Callable[[str, Dict[str, Any]], None]


# =============================================================================
# EMA of weights
# =============================================================================

class EMA:
    """Exponential moving average of the model's parameters (kept in fp32).

    The effective decay warms up as ``min(decay, (1 + n) / (10 + n))`` so the average is
    meaningful from the first steps. ``swapped(model)`` temporarily loads the averaged
    weights (bit-exact restore on exit), e.g. for evaluation or export.
    """

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.num_updates = 0
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.shadow: Dict[str, torch.Tensor] = {n: p.detach().clone().float() for n, p in named}
        # cached, aligned lists: the parameters are the same objects for the model's lifetime
        self._params = [p for _, p in named]
        self._shadows = [self.shadow[n] for n, _ in named]

    @torch.no_grad()
    def update(self, model: Optional[nn.Module] = None) -> None:
        """Fold the current parameters into the average (``model`` is accepted for symmetry
        with ``swapped`` but the parameters captured at construction are what is averaged)."""
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        params = [p.detach() for p in self._params]
        try:
            torch._foreach_lerp_(self._shadows, params, 1.0 - d)
        except (AttributeError, RuntimeError, TypeError):
            for s, p in zip(self._shadows, params):
                s.lerp_(p, 1.0 - d)

    @torch.no_grad()
    def _swap(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            s = self.shadow.get(n)
            if s is not None:
                tmp = p.data.clone()
                p.data.copy_(s)
                s.copy_(tmp)

    @contextlib.contextmanager
    def swapped(self, model: nn.Module):
        self._swap(model)
        try:
            yield model
        finally:
            self._swap(model)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "num_updates": self.num_updates,
                "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        for k, v in state["shadow"].items():
            if k in self.shadow:
                self.shadow[k].copy_(v)

    @staticmethod
    @torch.no_grad()
    def apply_state_to_model(state: Dict[str, Any], model: nn.Module) -> None:
        """Load EMA weights from a saved EMA state straight into ``model``."""
        for n, p in model.named_parameters():
            if n in state["shadow"]:
                p.data.copy_(state["shadow"][n].to(p.device, p.dtype))


def default_ema_decay(max_steps: int) -> float:
    """Average over roughly the last 5% of training (capped at 0.9995)."""
    return float(min(0.9995, 1.0 - 1.0 / max(10.0, 0.05 * max_steps)))


# =============================================================================
# Config
# =============================================================================

@dataclass
class TrainConfig:
    # batch
    batch_size: int = 16            # micro-batch, sequences
    grad_accum_steps: int = 1
    block_size: Optional[int] = None  # training sequence length (default: model block_size)
    # optimisation
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    optimizer: str = "adamw"        # "adamw" | "muon"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: Optional[float] = None
    use_8bit_optimizer: bool = False
    # numerics
    dtype: str = "auto"
    compile: bool = False
    # averaging
    ema_decay: float = 0.0          # 0 = off; see default_ema_decay()
    # evaluation / logging
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 20
    eval_batch_size: int = 16
    sample_every_evals: int = 1     # 0 = never
    sample_prompt: str = ""
    sample_len: int = 200
    # persistence
    ckpt_interval_minutes: float = 10.0
    max_minutes: Optional[float] = None
    max_nonfinite: int = 50
    seed: int = 1337
    verbose: bool = True

    # Per-session settings that must NOT stick to a run: re-running with a different wall-clock
    # limit or verbosity has to take effect, instead of replaying whatever the interrupted
    # session used (a persisted ``max_minutes`` would stop every resume immediately).
    EPHEMERAL = ("max_minutes", "verbose")

    # Fields that define the optimisation TRAJECTORY. An exact resume requires them to equal the
    # checkpoint's; everything else (logging / eval / sampling cadence, dtype, compile, ...) may
    # differ between sessions without changing what the run computes.
    TRAJECTORY = ("batch_size", "grad_accum_steps", "block_size", "learning_rate", "min_lr", "optimizer",
                  "weight_decay", "beta1", "beta2", "grad_clip", "muon_lr", "muon_momentum", "muon_weight_decay",
                  "use_8bit_optimizer", "ema_decay", "seed")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def persistent_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.to_dict().items() if k not in self.EPHEMERAL}


# =============================================================================
# Trainer
# =============================================================================

def _cfg_mismatch(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    skip = {"grad_checkpoint", "grad_checkpoint_every"}  # memory knobs: free to differ between runs
    return [f"{k}: checkpoint={a.get(k)!r} vs model={b.get(k)!r}"
            for k in sorted(set(a) | set(b)) if k not in skip and a.get(k) != b.get(k)]


def preflight(model: TensegrityLM, tc: TrainConfig, device: str = "auto", steps: int = 4) -> Dict[str, Any]:
    """
    Run a few REAL optimisation steps (the configured optimizer, dtype, loss scaling; at 10% of the
    peak LR) on random bytes and report whether everything stayed finite -- a cheap way to catch a numerics problem on the
    user's actual GPU before committing hours. The model's weights are restored afterwards; nothing
    is written to disk.

    Returns ``{"ok": bool, "optimizer": str, "dtype": str, "reason": str | None, "loss_first": float, "loss_last": float}``.
    """
    device = pick_device(device)
    dtype = pick_dtype(tc.dtype, device)
    amp = amp_settings(dtype, device)
    snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    was_training, ld = model.training, model.layerdrop_state()
    model.to(device).train()
    model.seed_layerdrop(tc.seed)
    opt = model.configure_optimizer(
        weight_decay=tc.weight_decay, learning_rate=tc.learning_rate, betas=(tc.beta1, tc.beta2),
        device_type=device_type_of(device), use_8bit=False, use_muon=(tc.optimizer == "muon"),
        muon_lr=tc.muon_lr, muon_momentum=tc.muon_momentum, muon_weight_decay=tc.muon_weight_decay)
    for g_ in opt.param_groups:      # numerics check, not a stability test: no warmup here, so use a gentle LR
        g_["lr"] *= 0.1
    scaler = torch.amp.GradScaler("cuda", enabled=amp["use_scaler"])
    T = min(tc.block_size or model.cfg.block_size, 128)
    g = torch.Generator().manual_seed(0)
    out: Dict[str, Any] = {"ok": True, "optimizer": tc.optimizer, "dtype": str(dtype).replace("torch.", ""),
                           "reason": None, "loss_first": float("nan"), "loss_last": float("nan")}
    try:
        for i in range(steps):
            x = torch.randint(0, 256, (2, T), generator=g).to(device)
            with torch.autocast(device_type=amp["autocast_device"], dtype=amp["amp_dtype"], enabled=amp["use_amp"]):
                _, losses, _ = model(x, torch.roll(x, -1, 1))
            loss = losses["total_loss"]
            (scaler.scale(loss) if scaler.is_enabled() else loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            elif torch.isfinite(gnorm):
                opt.step()
            opt.zero_grad(set_to_none=True)
            out["loss_last"] = float(loss.detach())
            if i == 0:
                out["loss_first"] = out["loss_last"]
            if not math.isfinite(out["loss_last"]):
                out.update(ok=False, reason=f"non-finite loss at step {i}")
                break
        else:
            bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
            if bad:
                out.update(ok=False, reason=f"non-finite weights after the steps: {bad[:3]}")
            elif out["loss_last"] > out["loss_first"] * 2.0 + 2.0:
                out.update(ok=False, reason=f"loss diverged ({out['loss_first']:.2f} -> {out['loss_last']:.2f})")
    except Exception as e:  # noqa: BLE001 - any failure of the optimizer path is a failed preflight
        out.update(ok=False, reason=f"{type(e).__name__}: {e}")
    finally:
        model.load_state_dict(snapshot)
        model.set_layerdrop_state(ld)
        model.train(was_training)
        del opt
    return out


def preflight_or_fallback(model: TensegrityLM, tc: TrainConfig, device: str = "auto", verbose: bool = True) -> TrainConfig:
    """Run :func:`preflight`; if the configured optimizer fails (non-finite loss/weights, an exception)
    switch ``tc`` to plain AdamW and re-check. Raises only if AdamW fails too. Mutates and returns ``tc``."""
    r = preflight(model, tc, device)
    say = print if verbose else (lambda *a, **k: None)
    if r["ok"]:
        say(f"[preflight] {r['optimizer']} / {r['dtype']} OK (loss {r['loss_first']:.2f} -> {r['loss_last']:.2f} over 4 test steps)")
        return tc
    if tc.optimizer == "adamw":
        raise RuntimeError(f"preflight failed with AdamW: {r['reason']}")
    say(f"[preflight] WARNING: {tc.optimizer} failed on this device ({r['reason']}); falling back to AdamW.")
    tc.optimizer = "adamw"
    r2 = preflight(model, tc, device)
    if not r2["ok"]:
        raise RuntimeError(f"preflight failed with AdamW as well: {r2['reason']}")
    say(f"[preflight] adamw / {r2['dtype']} OK")
    return tc


def load_run_plan(out_dir: str):
    """Read ``run_plan.json`` written by a previous ``Trainer.fit`` in ``out_dir``.
    Returns ``(RunPlan, TrainConfig, TensegrityConfig)`` or None if the directory has no record.
    Per-session fields (``TrainConfig.EPHEMERAL``: wall-clock limit, verbosity) are not restored --
    set them on the returned config for the session you are about to run."""
    path = os.path.join(out_dir, "run_plan.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    return (RunPlan(**d["plan"]), TrainConfig(**{k: v for k, v in d["train_config"].items() if k in known}),
            TensegrityConfig.from_dict(d["model_config"], strict=True))


class Trainer:
    """
    Parameters
    ----------
    model : TensegrityLM (any device; moved to ``device``)
    tc    : TrainConfig            plan : RunPlan (steps + LR schedule)
    data_dirs / data_weights : one or more prepared corpora (validation always uses data_dirs[0])
    out_dir : checkpoints (``ckpt_latest.pt`` full, ``ckpt_best.pt`` weights-only), ``train_log.jsonl``
    extra_loss : optional ``fn(model) -> scalar tensor`` added to the loss (e.g. an :class:`~tensegrity.continual.EWCState`)
    batch_loss : optional ``fn(model, x, y, logits) -> scalar tensor`` added to the loss; it sees the batch and the
        student's logits (e.g. :class:`~tensegrity.continual.DistillLoss`). Hooks that expose ``fingerprint()`` are
        verified on resume: continuing with a different EWC state or teacher is a policy change.
    callbacks : ``fn(event, payload)`` for events: start, log, eval, sample, checkpoint, end
    eval_suite : an :class:`~tensegrity.evaluation.EvalSuite` of named domains. Without it validation uses
        ``data_dirs[0]`` only. With it every domain is evaluated at each eval, the per-domain numbers are logged,
        and the best checkpoint is chosen by the suite's macro-averaged loss; the suite's fingerprint is recorded and
        checked on resume (best-checkpoint comparisons are only meaningful on the same evaluation set).
    lineage : free-form provenance recorded in every checkpoint (e.g. ``{"init_from": ...}``, ``{"grown_from": ...}``)

    Counters in a checkpoint (all explicit, none inferred):
      * ``step`` == ``completed_updates``: loop iterations finished = the position on the LR schedule.
        Resume continues AT this value; there is no off-by-one between periodic and final checkpoints.
      * ``skipped_steps``: iterations whose update was withheld (non-finite loss/gradients, fp16 overflow);
        ``applied_updates = completed_updates - skipped_steps`` optimizer updates actually applied.
      * ``tokens_seen``: tokens consumed by forward/backward passes (skipped iterations still consume data).
    """

    def __init__(self, model: TensegrityLM, tc: TrainConfig, plan: RunPlan, data_dirs, out_dir: str,
                 data_weights: Optional[Sequence[float]] = None, device: str = "auto",
                 extra_loss: Optional[Callable[[nn.Module], torch.Tensor]] = None,
                 callbacks: Optional[Sequence[Callback]] = None, *,
                 lineage: Optional[Dict[str, Any]] = None,
                 batch_loss: Optional[Callable[[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 eval_suite: Optional[EvalSuite] = None):
        self.model = model
        self._raw = model
        self._log_file = None
        self.cfg: TensegrityConfig = model.cfg
        self.tc, self.plan = tc, plan
        self.data_dirs = [data_dirs] if isinstance(data_dirs, str) else list(data_dirs)
        self.data_weights = data_weights
        self.out_dir = out_dir
        self.device_arg = device
        self.extra_loss = extra_loss
        self.batch_loss = batch_loss
        self.eval_suite = eval_suite
        self._hooks: Dict[str, Any] = {}
        self.callbacks: List[Callback] = list(callbacks or [])
        self.history: List[Dict[str, Any]] = []
        self.best_val_loss = float("inf")
        self.best_source = "raw"
        self.tokens_seen = 0
        self.step = 0
        self.skipped_steps = 0
        self.ema: Optional[EMA] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lineage: Dict[str, Any] = dict(lineage or {})
        self.phases: List[Dict[str, Any]] = []        # adopted policy / data changes, oldest first
        self.base_lrs: List[float] = []
        self._data_identity: Dict[str, Dict[str, Any]] = {}
        self._last_mult = 1.0

    # ------------------------------------------------------------------ utils
    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb(event, payload)

    def _print(self, msg: str) -> None:
        if self.tc.verbose and self.master:
            print(msg, flush=True)

    @property
    def raw_model(self) -> TensegrityLM:
        return self._raw

    def _build_optimizer(self) -> torch.optim.Optimizer:
        tc = self.tc
        if tc.optimizer not in ("adamw", "muon"):
            raise ValueError(f"optimizer must be 'adamw' or 'muon', got {tc.optimizer!r}")
        return self.raw_model.configure_optimizer(
            weight_decay=tc.weight_decay, learning_rate=tc.learning_rate, betas=(tc.beta1, tc.beta2),
            device_type=self.device_type, use_8bit=tc.use_8bit_optimizer, use_muon=(tc.optimizer == "muon"),
            muon_lr=tc.muon_lr, muon_momentum=tc.muon_momentum, muon_weight_decay=tc.muon_weight_decay)

    def _write_log(self, rec: Dict[str, Any]) -> None:
        self.history.append(rec)
        if self._log_file is not None:
            self._log_file.write(json.dumps(rec) + "\n")
            self._log_file.flush()

    # ----------------------------------------------------------- checkpointing
    @staticmethod
    def _optimizer_signature(opt: torch.optim.Optimizer) -> List[Dict[str, Any]]:
        return [{"n_params": len(g["params"]), "numel": int(sum(p.numel() for p in g["params"])),
                 "muon": bool(g.get("use_muon", False))} for g in opt.param_groups]

    def _extra_state(self, step: int) -> Dict[str, Any]:
        return {"format_train": 3, "tokens_seen": self.tokens_seen, "best_source": self.best_source,
                "plan": self.plan.to_dict(), "train_config": self.tc.persistent_dict(),
                "skipped_steps": self.skipped_steps,
                "completed_updates": int(step), "applied_updates": int(step) - self.skipped_steps,
                "lr_state": {"base_lrs": list(self.base_lrs), "last_multiplier": self._last_mult,
                             "schedule": self.plan.schedule, "horizon_steps": self.plan.max_steps},
                "optimizer_signature": self._optimizer_signature(self.optimizer),
                "data": {"identity": self._data_identity, "weights": [float(w) for w in self.ds.weights]},
                "environment": environment_info(), "hooks": self._hooks,
                "eval_suite": self.eval_suite.identity() if self.eval_suite is not None else None,
                "lineage": {**self.lineage, "phases": list(self.phases)}}

    def save_latest(self, step: int) -> str:
        path = os.path.join(self.out_dir, "ckpt_latest.pt")
        save_checkpoint(path, self.raw_model, self.optimizer, self.cfg, step, self.best_val_loss,
                        rng_state=capture_rng_state(self.raw_model, self.ds), scaler=self.scaler,
                        ema_state=self.ema.state_dict() if self.ema else None, extra=self._extra_state(step))
        self._emit("checkpoint", {"path": path, "step": step, "kind": "latest"})
        return path

    def save_best(self, step: int, use_ema: bool) -> str:
        path = os.path.join(self.out_dir, "ckpt_best.pt")
        ctx = self.ema.swapped(self.raw_model) if (use_ema and self.ema) else contextlib.nullcontext()
        with ctx:
            save_checkpoint(path, self.raw_model, None, self.cfg, step, self.best_val_loss,
                            extra={"tokens_seen": self.tokens_seen, "best_source": self.best_source,
                                   "weights_only": True})
        self._emit("checkpoint", {"path": path, "step": step, "kind": "best"})
        return path

    # ------------------------------------------------------------------- resume
    def _policy_diffs(self, ckpt: Dict[str, Any]) -> List[str]:
        """How the training POLICY now differs from the checkpoint's (trajectory fields + the plan)."""
        out: List[str] = []
        saved = ckpt.get("train_config")
        if saved is not None:
            cur = self.tc.persistent_dict()
            out += [f"train_config.{k}: checkpoint={saved[k]!r} vs now={cur[k]!r}"
                    for k in TrainConfig.TRAJECTORY if k in saved and saved[k] != cur[k]]
        if ckpt.get("plan") is not None:
            out += plan_change_problems(ckpt["plan"], self.plan.to_dict(), int(ckpt["step"]))
        if "eval_suite" in ckpt:
            now = self.eval_suite.identity() if self.eval_suite is not None else None
            if ckpt["eval_suite"] != now:
                out.append("eval_suite: the evaluation set differs from the one the checkpoint's best_val_loss was "
                           "measured on (domains, windows or dataset fingerprints changed)")
        if ckpt.get("hooks") is not None:
            for name in ("extra_loss", "batch_loss"):
                if ckpt["hooks"].get(name) != self._hooks.get(name):
                    out.append(f"{name}: checkpoint={ckpt['hooks'].get(name)} vs now={self._hooks.get(name)}")
        return out

    def _data_diffs(self, ckpt: Dict[str, Any]) -> List[str]:
        """How the training DATA now differs from what the checkpoint was trained on."""
        saved = ckpt.get("data")
        if not saved:
            return []                       # written before data identity was recorded: nothing to compare
        out = M.identity_mismatches(saved["identity"], self._data_identity)
        w_old, w_now = saved.get("weights"), [float(w) for w in self.ds.weights]
        if w_old is not None and len(w_old) == len(w_now) and not np.allclose(w_old, w_now):
            out.append(f"mixture weights: checkpoint={[round(w, 4) for w in w_old]} vs now={[round(w, 4) for w in w_now]}")
        return out

    def _resume(self, path: str, on_policy_change: str = "error", on_data_change: str = "error") -> int:
        """Restore EXACTLY what the checkpoint holds, after verifying that this session is the same run.

        Everything is validated before any state is touched, so a refused resume leaves the model and
        optimizer as they were. ``on_policy_change="adopt"`` / ``on_data_change="adopt"`` turn a mismatch
        into a recorded new *phase* of the run (see ``lineage['phases']``) instead of an error: the new
        hyper-parameters / data apply from the resume step on."""
        for name, val in (("on_policy_change", on_policy_change), ("on_data_change", on_data_change)):
            if val not in ("error", "adopt"):
                raise ValueError(f"{name} must be 'error' or 'adopt', got {val!r}")
        ckpt = load_checkpoint(path, self.device)
        bad = _cfg_mismatch(ckpt["config"], self.cfg.to_dict())
        if bad:
            raise ValueError(f"cannot resume from {path}: model config differs from the checkpoint's:\n  "
                             + "\n  ".join(bad))
        if ckpt.get("optimizer_state_dict") is None:
            raise ValueError(f"{path} is a weights-only checkpoint; it has no optimizer state to resume from. "
                             f"Resume from ckpt_latest.pt, or start a new run initialised from these weights.")
        step = int(ckpt["step"]) + (1 if ckpt.get("format_version", 1) == 1 else 0)  # v1 stored the step just completed
        policy, data = self._policy_diffs(ckpt), self._data_diffs(ckpt)
        if policy and on_policy_change != "adopt":
            raise ValueError(
                f"cannot resume {path} exactly: the training policy differs from the checkpoint's:\n  "
                + "\n  ".join(policy)
                + "\nAn exact resume continues the same run. To change the policy from here on, pass "
                  "fit(on_policy_change='adopt') (recorded as a new phase), or start a NEW run from these weights.")
        if data and on_data_change != "adopt":
            raise ValueError(
                f"cannot resume {path} exactly: the training data differs from what the checkpoint was trained on:\n  "
                + "\n  ".join(data)
                + "\nPass fit(on_data_change='adopt') to continue on the new data as a recorded new phase, "
                  "or restore the original dataset.")
        saved_sig = ckpt.get("optimizer_signature")
        if saved_sig is not None and saved_sig != self._optimizer_signature(self.optimizer):
            raise ValueError(f"optimizer state in {path} does not match tc.optimizer={self.tc.optimizer!r} or this model: "
                             f"parameter groups {saved_sig} (checkpoint) vs {self._optimizer_signature(self.optimizer)} (now)")
        env_then, env_now = ckpt.get("environment") or {}, environment_info()
        drift = [f"{k} {env_then[k]} -> {env_now[k]}" for k in ("torch", "numpy") if k in env_then and env_then[k] != env_now[k]]
        if drift:
            warnings.warn(f"resuming under a different software stack ({', '.join(drift)}); results are not guaranteed "
                          f"to be bit-identical to an uninterrupted run across versions.", stacklevel=3)

        fresh_groups = [{k: v for k, v in g.items() if k != "params"} for g in self.optimizer.param_groups]
        self.raw_model.load_state_dict(ckpt["model_state_dict"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError, RuntimeError) as e:
            raise ValueError(f"optimizer state in {path} does not match tc.optimizer={self.tc.optimizer!r}: {e}") from e
        for g, fresh in zip(self.optimizer.param_groups, fresh_groups):
            g.update(fresh)                 # torch restores the checkpoint's hyper-parameters; this session's config rules
        if ckpt.get("scaler_state_dict"):
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.ema is not None and ckpt.get("ema_state_dict"):
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        restore_rng_state(ckpt.get("rng_state"), self.raw_model, self.ds)
        self.best_val_loss = float(ckpt["best_val_loss"])
        self.best_source = ckpt.get("best_source", "raw")
        self.tokens_seen = int(ckpt.get("tokens_seen", 0))
        self.skipped_steps = int(ckpt.get("skipped_steps", 0))
        old_lineage = dict(ckpt.get("lineage") or {})
        self.phases = list(old_lineage.pop("phases", []))
        self.lineage = {**self.lineage, **old_lineage}
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for kind, diffs in (("policy", policy), ("data", data)):
            if diffs:
                self.phases.append({"at_step": step, "kind": kind, "changes": diffs, "time": stamp})
        return step

    # ----------------------------------------------------------------- sampling
    def sample_text(self, use_ema: bool = False) -> str:
        tc = self.tc
        tok = ByteTokenizer()
        ids = [ByteTokenizer.DOC_START] + tok.encode(tc.sample_prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        amp = amp_settings(self.dtype, self.device)
        ctx = self.ema.swapped(self.raw_model) if (use_ema and self.ema) else contextlib.nullcontext()
        devices = [torch.device(self.device)] if self.device_type == "cuda" else []
        with torch.random.fork_rng(devices=devices), ctx:  # sampling must not disturb training RNG
            torch.manual_seed(1234)
            with torch.autocast(device_type=amp["autocast_device"], dtype=amp["amp_dtype"], enabled=amp["use_amp"]):
                out = self.raw_model.generate(idx, tc.sample_len, temperature=0.8, top_k=40, min_p=0.02,
                                              stop_on_eos=True)
        return tok.decode(out[0].tolist())

    # --------------------------------------------------------------------- fit
    def fit(self, resume: bool = True, on_policy_change: str = "error", on_data_change: str = "error") -> Dict[str, Any]:
        """Run (or continue) training.

        ``resume=True`` continues ``out_dir/ckpt_latest.pt`` *exactly* if there is one: same model config,
        training policy (``TrainConfig.TRAJECTORY`` fields and the plan), data fingerprints and mixture
        weights, or it raises and says what differs. ``on_policy_change="adopt"`` /
        ``on_data_change="adopt"`` explicitly accept such differences as a new phase of the run (recorded
        in the checkpoint's ``lineage``); starting a different experiment from the weights is a fork:
        build a new model from them, a new ``out_dir``, and a fresh Trainer."""
        tc, plan = self.tc, self.plan
        # ---- distributed / device setup
        self.is_ddp = int(os.environ.get("RANK", -1)) != -1
        if self.is_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            cuda = torch.cuda.is_available()
            dist.init_process_group(backend="nccl" if cuda else "gloo")
            self.rank, self.world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
            local = int(os.environ.get("LOCAL_RANK", 0))
            self.device = f"cuda:{local}" if cuda else "cpu"
            if cuda:
                torch.cuda.set_device(self.device)
        else:
            self.rank, self.world = 0, 1
            self.device = pick_device(self.device_arg)
        self.master = self.rank == 0
        self.device_type = device_type_of(self.device)
        self.dtype = pick_dtype(tc.dtype, self.device)
        amp = amp_settings(self.dtype, self.device)
        set_fast_matmul()

        torch.manual_seed(tc.seed + self.rank)
        self.raw_model.seed_layerdrop(tc.seed)  # identical on every rank -> identical skip decisions
        os.makedirs(self.out_dir, exist_ok=True)
        self.model.to(self.device)
        T = tc.block_size or self.cfg.block_size
        if T > self.cfg.block_size:
            raise ValueError(f"training block_size {T} exceeds the model's block_size {self.cfg.block_size}")
        if self.master:  # lets a re-run in a fresh session reuse exactly this schedule (see load_run_plan)
            with open(os.path.join(self.out_dir, "run_plan.json"), "w") as fh:
                json.dump({"plan": plan.to_dict(), "train_config": tc.persistent_dict(),
                           "model_config": self.cfg.to_dict()}, fh, indent=2)
        # the data behind a run must not change under it: a replaced/rewritten token file is an error
        self.ds = ByteDataset(self.data_dirs, self.data_weights, seed=tc.seed, rank=self.rank, on_change="error")
        self.ds.verify()                      # token files agree with their manifests (raises DatasetError)
        self._data_identity = self.ds.identity()
        for hook in (self.extra_loss, self.batch_loss):
            if hasattr(hook, "to"):
                hook.to(self.device)           # EWC state / distillation teacher follow the model's device
        self._hooks = {"extra_loss": hook_info(self.extra_loss), "batch_loss": hook_info(self.batch_loss)}
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp["use_scaler"])
        self.optimizer = self._build_optimizer()
        base_lrs = [g["lr"] for g in self.optimizer.param_groups]  # BEFORE any state is loaded (see module doc)
        self.base_lrs = list(base_lrs)
        floor = tc.min_lr / tc.learning_rate if tc.learning_rate else 0.0
        if tc.ema_decay > 0:
            self.ema = EMA(self.raw_model, tc.ema_decay)

        start_step = 0
        latest = os.path.join(self.out_dir, "ckpt_latest.pt")
        if resume and os.path.exists(latest):
            start_step = self._resume(latest, on_policy_change, on_data_change)
            if self.ema is not None and self.ema.num_updates == 0:
                self.ema = EMA(self.raw_model, tc.ema_decay)  # checkpoint had no EMA: start averaging now
            self._print(f"[resume] {latest} -> continuing at step {start_step} "
                        f"({self.tokens_seen / 1e6:.1f}M tokens seen, best_val {self.best_val_loss:.4f})")
        elif resume:
            self._print(f"[resume] no checkpoint at {latest}; starting from scratch")

        compiled = self.model
        if tc.compile:
            try:
                compiled = torch.compile(self.model)
                self._print("[train] torch.compile enabled")
            except Exception as e:  # noqa: BLE001
                self._print(f"[train] torch.compile unavailable ({e}); continuing eagerly")
        if self.is_ddp:
            ids = [torch.device(self.device).index] if self.device_type == "cuda" else None
            compiled = DDP(compiled, device_ids=ids, find_unused_parameters=self.cfg.layerdrop_max > 0)
        run_model = compiled  # what forward/backward go through; self.model stays the plain module

        self._T = T
        eval_batches = (None if self.eval_suite is not None
                        else self.ds.fixed_batches("val", tc.eval_batches, tc.eval_batch_size, T, self.device))
        self._log_file = open(os.path.join(self.out_dir, "train_log.jsonl"), "a") if self.master else None
        info = gpu_info()
        tokens_per_step = tc.batch_size * tc.grad_accum_steps * T * self.world
        if tokens_per_step != plan.tokens_per_step:
            self._print(f"[train] note: plan assumed {plan.tokens_per_step:,} tokens/step, this config does "
                        f"{tokens_per_step:,}; the schedule is defined in steps, tokens are reported as run")
        self._print(f"[train] {self.cfg.describe()}")
        self._print(f"[train] device={self.device} dtype={self.dtype} optimizer={tc.optimizer} "
                    f"steps={plan.max_steps:,} ({plan.total_tokens / 1e6:.0f}M tokens plan) "
                    f"batch={tc.batch_size}x{tc.grad_accum_steps}x{T}")
        self._emit("start", {"step": start_step, "plan": plan.to_dict(), "tc": tc.to_dict()})

        status = "completed"
        t_start = time.time()
        t_log, tok_log = t_start, self.tokens_seen
        t_ckpt = t_start
        n_evals = 0
        consecutive_bad = 0
        step = start_step
        last_val: Dict[str, float] = {}
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            while step < plan.max_steps:
                mult = plan.multiplier(step, floor)
                self._last_mult = mult
                for g, b in zip(self.optimizer.param_groups, base_lrs):
                    g["lr"] = b * mult

                sums: Dict[str, torch.Tensor] = {}
                for micro in range(tc.grad_accum_steps):
                    x, y = self.ds.get_batch("train", tc.batch_size, T, self.device)
                    sync = run_model.no_sync() if (self.is_ddp and micro < tc.grad_accum_steps - 1) \
                        else contextlib.nullcontext()
                    with sync:
                        with torch.autocast(device_type=amp["autocast_device"], dtype=amp["amp_dtype"],
                                            enabled=amp["use_amp"]):
                            logits, losses, _ = run_model(x, y)
                            loss = losses["total_loss"]
                            if self.extra_loss is not None:
                                extra = self.extra_loss(self.raw_model)
                                losses = dict(losses, extra_loss=extra)
                                loss = loss + extra
                            if self.batch_loss is not None:
                                bl = self.batch_loss(self.raw_model, x, y, logits)
                                losses = dict(losses, batch_loss=bl)
                                loss = loss + bl
                        scaled = loss / tc.grad_accum_steps
                        (self.scaler.scale(scaled) if self.scaler.is_enabled() else scaled).backward()
                    for k, v in losses.items():
                        sums[k] = sums.get(k, 0.0) + v.detach().float() / tc.grad_accum_steps

                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                gnorm = (torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), tc.grad_clip)
                         if tc.grad_clip > 0 else None)
                probe = gnorm if gnorm is not None else sums["total_loss"]
                finite = bool(torch.isfinite(probe))
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)   # skips itself on inf/nan gradients
                    self.scaler.update()
                elif finite:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if finite:
                    consecutive_bad = 0
                    if self.ema is not None:
                        self.ema.update(self.raw_model)
                else:
                    consecutive_bad += 1
                    self.skipped_steps += 1
                    # fp16 + GradScaler legitimately skips a few steps while the loss scale calibrates
                    if consecutive_bad > tc.max_nonfinite:
                        raise FloatingPointError(
                            f"{consecutive_bad} consecutive non-finite steps at step {step}; training has diverged. "
                            f"Lower the learning rate or check the data.")
                self.tokens_seen += tokens_per_step
                self.step = step + 1

                # ---- logging
                if self.master and (step % tc.log_interval == 0 or step == plan.max_steps - 1):
                    now = time.time()
                    tps = (self.tokens_seen - tok_log) / (now - t_log) if step > start_step and now > t_log else None
                    rec = {"step": step, "lr": self.optimizer.param_groups[0]["lr"], "lr_mult": mult,
                           "tokens": self.tokens_seen, "elapsed": now - t_start,
                           "grad_norm": float(gnorm) if gnorm is not None else None,
                           "skipped": self.skipped_steps, "tok_per_s": tps,
                           "mfu": mfu(self.cfg, tps, T, info["peak_tflops"]) if tps else None}
                    rec.update({k: float(v) for k, v in sums.items()})
                    t_log, tok_log = now, self.tokens_seen
                    self._write_log(rec)
                    self._emit("log", rec)
                    extra_s = f" conf {rec['confidence_loss']:.3f}" if "confidence_loss" in rec else ""
                    self._print(f"step {step:6d} | lr {rec['lr']:.2e} | loss {rec['total_loss']:.4f} "
                                f"(ce {rec['ce_loss']:.4f} = {rec['ce_loss'] / math.log(2):.3f} bpb{extra_s}) "
                                f"| gnorm {rec['grad_norm'] if rec['grad_norm'] is not None else float('nan'):.2f} "
                                f"| {(tps or 0) / 1e3:.1f}k tok/s | {rec['elapsed'] / 60:.1f} min")

                # ---- evaluation + checkpoint
                done_step = step + 1
                is_last = done_step >= plan.max_steps
                stop_now = False
                if tc.max_minutes is not None and (time.time() - t_start) / 60.0 >= tc.max_minutes:
                    stop_now = True
                if self.is_ddp:
                    flag = torch.tensor([1.0 if stop_now else 0.0], device=self.device)
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                    stop_now = bool(flag.item())
                do_eval = (done_step % tc.eval_interval == 0) or is_last or stop_now
                if do_eval and self.master:
                    last_val = self._eval_and_maybe_save_best(eval_batches, done_step)
                    n_evals += 1
                    if tc.sample_every_evals and n_evals % tc.sample_every_evals == 0:
                        text = self.sample_text(use_ema=last_val.get("source") == "ema")
                        self._emit("sample", {"step": done_step, "text": text})
                        if tc.verbose:
                            print("      sample> " + repr(text[:160]))
                    self.save_latest(done_step)
                    t_ckpt = time.time()
                elif self.master and (time.time() - t_ckpt) / 60.0 >= tc.ckpt_interval_minutes:
                    self.save_latest(done_step)
                    t_ckpt = time.time()
                step = done_step
                if stop_now and not is_last:
                    status = "time_limit"
                    self._print(f"[train] wall-clock limit ({tc.max_minutes} min) reached at step {step}; "
                                f"checkpoint saved. Re-run to continue.")
                    break
        finally:
            if self._log_file is not None:
                self._log_file.close()
            if self.is_ddp:
                dist.destroy_process_group()

        result = {"status": status, "step": step, "best_val_loss": self.best_val_loss,
                  "best_source": self.best_source, "tokens_seen": self.tokens_seen,
                  "elapsed": time.time() - t_start, "last_val": last_val, "skipped_steps": self.skipped_steps,
                  "completed_updates": step, "applied_updates": step - self.skipped_steps,
                  "fingerprints": {k: v["fingerprint"] for k, v in self._data_identity.items()}}
        self._emit("end", result)
        if status == "completed":
            self._print(f"[train] done: step {step}, best val loss {self.best_val_loss:.4f} "
                        f"({self.best_val_loss / math.log(2):.3f} bpb, from {self.best_source} weights)")
        return result

    def _measure(self, eval_batches) -> Dict[str, Any]:
        """Validation metrics of the CURRENT weights: the eval suite (named domains, macro-averaged) if one was
        given, else the fixed windows of ``data_dirs[0]``."""
        if self.eval_suite is not None:
            r = self.eval_suite.evaluate(self.raw_model, self.device, self.dtype, block_size=self._T)
            loss = r.selection_loss
            return {"loss": loss, "bpb": loss / math.log(2.0), "ppl": math.exp(min(loss, 50.0)),
                    "bpb_bytes": r.macro_bpb_bytes,
                    "domains": {k: {"loss": m["loss"], "bpb_bytes": m["bpb_bytes"]} for k, m in r.domains.items()}}
        return evaluate_batches(self.raw_model, eval_batches, self.device, self.dtype)

    def _eval_and_maybe_save_best(self, eval_batches, done_step: int) -> Dict[str, Any]:
        raw = self._measure(eval_batches)
        rec: Dict[str, Any] = {"step": done_step, "val_loss": raw["loss"], "val_bpb": raw["bpb"],
                               "val_ppl": raw["ppl"], "val_bpb_bytes": raw["bpb_bytes"], "tokens": self.tokens_seen}
        if "domains" in raw:
            rec["val_domains"] = raw["domains"]
        best_loss, source = raw["loss"], "raw"
        if self.ema is not None:
            with self.ema.swapped(self.raw_model):
                e = self._measure(eval_batches)
            rec.update(val_loss_ema=e["loss"], val_bpb_ema=e["bpb"], val_bpb_bytes_ema=e["bpb_bytes"])
            if "domains" in e:
                rec["val_domains_ema"] = e["domains"]
            if e["loss"] < best_loss:
                best_loss, source = e["loss"], "ema"
        rec["source"] = source
        self._write_log(rec)
        self._emit("eval", rec)
        msg = f"step {done_step:6d} | val {raw['loss']:.4f} ({raw['bpb_bytes']:.3f} bits/byte)"
        if self.ema is not None:
            msg += f" | ema {rec['val_loss_ema']:.4f} ({rec['val_bpb_bytes_ema']:.3f} bits/byte)"
        if "domains" in raw:
            msg += " | " + ", ".join(f"{k} {m['bpb_bytes']:.3f}" for k, m in raw["domains"].items())
        self._print(msg)
        if best_loss < self.best_val_loss:
            self.best_val_loss, self.best_source = best_loss, source
            self.save_best(done_step, use_ema=(source == "ema"))
            self._print(f"      new best val loss {best_loss:.4f} ({source}) -> ckpt_best.pt")
        return rec
