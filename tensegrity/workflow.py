"""Workflows built on the engine: transactional growth and matched-budget continual-learning comparisons.

These are the *Presenter* layer: they decide what happens in which order and what counts as success,
while the model, the trainer and the evaluation suite stay oblivious to the experiment.

* :func:`grow_transaction` -- growth as a transaction. **Propose** (compatibility) -> **construct** the
  candidate (the parent is never touched) -> **verify** preservation, serialization round trips (config +
  state dict, checkpoint, safetensors) and trainability -> optional **recovery** training (new data +
  replay + distillation from the parent + EWC) -> **retention evaluation** on named domains -> **accept or
  revert**. A rejected candidate is simply not used; the parent is still exactly what it was.
* :func:`compare_continual_strategies` -- new-data-only vs replay vs replay + EWC from the same weights
  under matched budgets, judged on old-domain degradation *and* new-domain gain (not on a total objective).
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .checkpoint import load_model, save_checkpoint, unwrap_model
from .config import TensegrityConfig
from .continual import DistillLoss, EWCState, tensors_fingerprint
from .evaluation import EvalDomain, EvalSuite, SuiteResult
from .export import export_safetensors, load_safetensors
from .growth import (GrowthError, GrowthReport, check_function_preserved, check_growth_trainable, grow_model)
from .hw import pick_dtype
from .model import TensegrityLM
from .schedule import make_plan
from .train import TrainConfig, Trainer


# =============================================================================
# Serialization round trips
# =============================================================================

@torch.no_grad()
def _max_logit_diff(a: TensegrityLM, b: TensegrityLM, cfg: TensegrityConfig, device: str) -> float:
    g = torch.Generator().manual_seed(123)
    x = torch.randint(0, 256, (2, min(48, cfg.block_size)), generator=g).to(device)
    a.eval(), b.eval()
    la, _, ca = a(x)
    lb, _, cb = b(x.to(next(b.parameters()).device))
    d = float((la - lb.to(la.device)).abs().max())
    if ca is not None and cb is not None:
        d = max(d, float((ca - cb.to(ca.device)).abs().max()))
    return d


def verify_roundtrips(model: TensegrityLM, cfg: Optional[TensegrityConfig] = None, atol: float = 1e-6
                      ) -> Dict[str, float]:
    """Rebuild ``model`` three ways and compare outputs: from ``config JSON + state_dict`` (what an
    ordinary reload does), through a training checkpoint, and through a safetensors export. Any function-
    affecting value that lives only in a Python attribute (the norm epsilon of a grown model was one)
    shows up as a difference here. Returns the maximum logit difference per route; raises if any
    exceeds ``atol``."""
    model = unwrap_model(model)
    cfg = cfg or model.cfg
    device = str(next(model.parameters()).device)
    out: Dict[str, float] = {}
    rebuilt = TensegrityLM(TensegrityConfig.from_dict(json.loads(json.dumps(cfg.to_dict())), strict=True)).to(device)
    rebuilt.load_state_dict(model.state_dict())
    out["config+state_dict"] = _max_logit_diff(model, rebuilt, cfg, device)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "c.pt")
        save_checkpoint(path, model, None, cfg, 0, float("inf"))
        out["checkpoint"] = _max_logit_diff(model, load_model(path, device)[0], cfg, device)
        w, c = export_safetensors(model, cfg, td, "m", "float32")
        out["safetensors"] = _max_logit_diff(model, load_safetensors(w, c, device)[0], cfg, device)
    bad = {k: v for k, v in out.items() if v > atol}
    if bad:
        raise GrowthError(f"model does not survive serialization: max |dlogit| {bad} > {atol:g}")
    return out


# =============================================================================
# Growth transaction
# =============================================================================

@dataclass
class GrowthGates:
    """Thresholds a candidate must meet. ``max_retain_regression_bpb``: bits per byte a ``retain`` domain may
    lose, measured after recovery. ``min_adapt_gain_bpb``: bits per byte an ``adapt`` domain must gain
    (0 = must not be worse). Retention is judged on the suite you pass; these are not universal constants."""
    preserve_atol: float = 2e-5
    roundtrip_atol: float = 1e-6
    require_trainable: bool = True
    max_retain_regression_bpb: float = 0.02
    min_adapt_gain_bpb: float = 0.0


@dataclass
class RecoverySpec:
    """Short training of the grown candidate before it is judged. ``distill_weight > 0`` distills the parent
    into the candidate on every batch; ``ewc`` (an :class:`EWCState` estimated on the PARENT) is carried
    through growth automatically. Mix replay by listing the parent's training data with a weight."""
    data_dirs: Sequence[str]
    data_weights: Optional[Sequence[float]] = None
    steps: int = 100
    batch_size: int = 8
    grad_accum_steps: int = 1
    learning_rate: float = 1e-3
    optimizer: str = "adamw"
    distill_weight: float = 1.0
    distill_temperature: float = 1.0
    ewc: Optional[EWCState] = None
    seed: int = 0


@dataclass
class GrowthDecision:
    accepted: bool
    reasons: List[str]                                  # why it was rejected (empty when accepted)
    report: Optional[GrowthReport]
    candidate: Optional[TensegrityLM]
    candidate_cfg: Optional[TensegrityConfig]
    parent_eval: Optional[SuiteResult] = None
    grown_eval: Optional[SuiteResult] = None            # candidate right after growth, before recovery
    final_eval: Optional[SuiteResult] = None            # candidate after recovery (== grown_eval without recovery)
    recovery: Optional[Dict[str, Any]] = None
    parent_unchanged: bool = True

    def raise_if_rejected(self) -> "GrowthDecision":
        if not self.accepted:
            raise GrowthError("growth candidate rejected:\n  " + "\n  ".join(self.reasons))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"accepted": self.accepted, "reasons": self.reasons,
                "report": self.report.to_dict() if self.report else None,
                "parent_eval": self.parent_eval.to_dict() if self.parent_eval else None,
                "grown_eval": self.grown_eval.to_dict() if self.grown_eval else None,
                "final_eval": self.final_eval.to_dict() if self.final_eval else None,
                "recovery": self.recovery, "parent_unchanged": self.parent_unchanged}

    def save(self, directory: str, name: str = "growth_decision") -> str:
        """Write the decision as ``<directory>/<name>[_N].json`` (never overwriting an earlier one); the run
        report lists these files as the run's growth history."""
        os.makedirs(directory, exist_ok=True)
        path, n = os.path.join(directory, f"{name}.json"), 1
        while os.path.exists(path):
            n += 1
            path = os.path.join(directory, f"{name}_{n}.json")
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return path

    def summary(self) -> str:
        head = "ACCEPTED" if self.accepted else "REJECTED"
        lines = [f"growth {head}" + (f": {self.report.summary()}" if self.report else "")]
        lines += [f"  - {r}" for r in self.reasons]
        if self.parent_eval is not None and self.final_eval is not None:
            lines.append("  bits/byte per domain (parent -> candidate): " + ", ".join(
                f"{k} {self.parent_eval.domains[k]['bpb_bytes']:.3f} -> {self.final_eval.domains[k]['bpb_bytes']:.3f}"
                f" [{self.final_eval.roles[k]}]" for k in self.final_eval.domains))
        return "\n".join(lines)


def grow_transaction(parent: TensegrityLM, parent_cfg: Optional[TensegrityConfig] = None, *,
                     target: Optional[TensegrityConfig] = None, suite: Optional[EvalSuite] = None,
                     recovery: Optional[RecoverySpec] = None, gates: Optional[GrowthGates] = None,
                     device: Optional[str] = None, seed: int = 0, run_dir: Optional[str] = None,
                     verbose: bool = True, **grow_kwargs: Any) -> GrowthDecision:
    """Grow ``parent`` into ``target`` (or via ``new_n_layer=`` ... keywords) as a transaction; see the module doc.

    Structural checks (mapping audit, function preservation, serialization, trainability) are hard gates:
    a failure rejects the candidate without spending recovery compute. Retention gates need ``suite`` and
    are judged on the candidate after ``recovery`` (or right after growth if there is none -- then only
    "nothing regressed" is meaningful, since nothing was trained). The decision carries every number it was
    based on. The parent is verified unchanged at the end (``parent_unchanged``)."""
    say = print if verbose else (lambda *a, **k: None)
    gates = gates or GrowthGates()
    parent = unwrap_model(parent)
    parent_cfg = parent_cfg or parent.cfg
    dev = device or str(next(parent.parameters()).device)
    if recovery is not None and suite is None:
        raise ValueError("recovery needs an EvalSuite: without held-out domains there is nothing to judge it by")
    before = tensors_fingerprint(parent.state_dict())

    # ---- 1-2. propose + construct
    try:
        cand, ccfg, report = grow_model(parent, parent_cfg, target=target, device=dev, seed=seed, return_report=True,
                                        **grow_kwargs)
    except (GrowthError, ValueError) as e:
        failed = GrowthDecision(False, [f"construction failed: {e}"], None, None, None)
        say("[grow] " + failed.summary())
        if run_dir is not None:
            failed.save(run_dir)
        return failed
    say(f"[grow] candidate built: {report.summary()}")

    reasons: List[str] = []
    # ---- 3. verify
    try:
        report.checks["preservation_max_logit_diff"] = check_function_preserved(
            parent, cand, parent_cfg, dev, atol=gates.preserve_atol, verbose=False)
    except AssertionError as e:
        reasons.append(f"function not preserved: {e}")
    try:
        report.checks["roundtrip_max_logit_diff"] = verify_roundtrips(cand, ccfg, gates.roundtrip_atol)
    except GrowthError as e:
        reasons.append(str(e))
    try:
        best = check_growth_trainable(cand, parent_cfg, verbose=False)
        report.checks["trainability_min_max_grad"] = min(best.values()) if best else None
    except AssertionError as e:
        if gates.require_trainable:
            reasons.append(f"new capacity cannot learn: {e}")
    unchanged = tensors_fingerprint(parent.state_dict()) == before

    parent_eval = grown_eval = final_eval = None
    rec_info: Optional[Dict[str, Any]] = None
    parent_dev = str(next(parent.parameters()).device)        # the parent is evaluated where it lives
    if not reasons and suite is not None:
        parent_eval = suite.evaluate(parent, parent_dev)
        grown_eval = suite.evaluate(cand, dev)
        final_eval = grown_eval
    # ---- 4. recovery
    if not reasons and recovery is not None:
        rec_info = _recover(parent, cand, ccfg, report, recovery, suite, dev, run_dir, say)
        final_eval = suite.evaluate(cand, dev)
        report.checks["recovery"] = {"steps": rec_info["steps"], "final_train_loss": rec_info["final_train_loss"]}
    # ---- 5. retention + adaptation gates
    if not reasons and suite is not None:
        for name, role in final_eval.roles.items():
            p, f = parent_eval.domains[name]["bpb_bytes"], final_eval.domains[name]["bpb_bytes"]
            if role == "retain" and f - p > gates.max_retain_regression_bpb:
                reasons.append(f"retention: '{name}' regressed {f - p:+.4f} bits/byte ({p:.4f} -> {f:.4f}), "
                               f"limit {gates.max_retain_regression_bpb:g}")
            if role == "adapt" and recovery is not None and p - f < gates.min_adapt_gain_bpb:
                reasons.append(f"adaptation: '{name}' improved only {p - f:+.4f} bits/byte ({p:.4f} -> {f:.4f}), "
                               f"required {gates.min_adapt_gain_bpb:g}")
    unchanged = unchanged and tensors_fingerprint(parent.state_dict()) == before
    if not unchanged:
        reasons.append("the parent model was modified during the transaction")
    decision = GrowthDecision(not reasons, reasons, report, cand, ccfg, parent_eval, grown_eval, final_eval, rec_info,
                              unchanged)
    say("[grow] " + decision.summary())
    if run_dir is not None:
        decision.save(run_dir)
    return decision


def _recover(parent: TensegrityLM, cand: TensegrityLM, ccfg: TensegrityConfig, report: GrowthReport,
             spec: RecoverySpec, suite: EvalSuite, device: str, run_dir: Optional[str], say) -> Dict[str, Any]:
    tokens_per_step = spec.batch_size * spec.grad_accum_steps * ccfg.block_size
    tc = TrainConfig(batch_size=spec.batch_size, grad_accum_steps=spec.grad_accum_steps,
                     learning_rate=spec.learning_rate, min_lr=spec.learning_rate / 10, optimizer=spec.optimizer,
                     ema_decay=0.0, eval_interval=max(1, spec.steps), log_interval=max(1, spec.steps // 5),
                     sample_every_evals=0, seed=spec.seed, verbose=False, ckpt_interval_minutes=1e9)
    plan = make_plan(spec.steps * tokens_per_step, tokens_per_step, "wsd", min_warmup=1)
    teacher = None
    if spec.distill_weight > 0:                          # never move the caller's parent to another device
        same = str(next(parent.parameters()).device) == str(torch.device(device))
        teacher = DistillLoss(parent if same else copy.deepcopy(parent), spec.distill_weight, spec.distill_temperature)
    ewc = spec.ewc.expanded(cand, report) if spec.ewc is not None else None
    cleanup = None
    if run_dir is None:
        cleanup = tempfile.TemporaryDirectory()
        run_dir = cleanup.name
    logs: List[Dict[str, Any]] = []
    try:
        trainer = Trainer(cand, tc, plan, list(spec.data_dirs), os.path.join(run_dir, "recovery"),
                          data_weights=spec.data_weights, device=device, extra_loss=ewc, batch_loss=teacher,
                          eval_suite=suite, callbacks=[lambda e, p: logs.append(p) if e == "log" else None],
                          lineage={"recovery_of": "grow_transaction"})
        res = trainer.fit(resume=False)
    finally:
        if cleanup is not None:
            cleanup.cleanup()
    last = next((r for r in reversed(logs) if "total_loss" in r), {})
    say(f"[grow] recovery: {res['completed_updates']} steps, {res['tokens_seen'] / 1e6:.2f}M tokens")
    return {"steps": res["completed_updates"], "tokens": res["tokens_seen"],
            "final_train_loss": last.get("total_loss"), "distill_weight": spec.distill_weight,
            "ewc": bool(ewc), "data": list(spec.data_dirs)}


# =============================================================================
# Continual-learning comparison
# =============================================================================

ARMS = ("new_only", "replay", "replay_ewc")


@dataclass
class ArmResult:
    arm: str
    seeds: List[int]
    old_bpb: List[float]                # held-out bits/byte on the OLD domain after training, one per seed
    new_bpb: List[float]
    forgetting: List[float]             # old_after - old_before  (bits/byte; positive = got worse)
    new_gain: List[float]               # new_before - new_after  (bits/byte; positive = improved)
    detail: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _ms(v: Sequence[float]) -> Tuple[float, float]:
        m = sum(v) / len(v)
        return m, (math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0)

    @property
    def forgetting_mean_std(self) -> Tuple[float, float]:
        return self._ms(self.forgetting)

    @property
    def new_gain_mean_std(self) -> Tuple[float, float]:
        return self._ms(self.new_gain)


@dataclass
class ContinualReport:
    before: Dict[str, float]            # base model's held-out bits/byte per domain {"old": ., "new": .}
    arms: Dict[str, ArmResult]
    budget: Dict[str, Any]

    def non_dominated(self) -> List[str]:
        """Arms not beaten on BOTH axes (less forgetting AND more new-domain gain) by another arm."""
        out = []
        for a, ra in self.arms.items():
            fa, ga = ra.forgetting_mean_std[0], ra.new_gain_mean_std[0]
            if not any(self.arms[b].forgetting_mean_std[0] <= fa and self.arms[b].new_gain_mean_std[0] >= ga
                       and (self.arms[b].forgetting_mean_std[0] < fa or self.arms[b].new_gain_mean_std[0] > ga)
                       for b in self.arms if b != a):
                out.append(a)
        return out

    def table(self) -> str:
        nd = set(self.non_dominated())
        rows = [f"base model: old {self.before['old']:.4f} bits/byte, new {self.before['new']:.4f} bits/byte  "
                f"(budget: {self.budget['steps']} steps x {self.budget['tokens_per_step']:,} tokens, "
                f"{len(next(iter(self.arms.values())).seeds)} seed(s))",
                f"{'strategy':12s}{'old domain':>12s}{'forgetting':>16s}{'new domain':>12s}{'new-domain gain':>18s}  non-dominated"]
        for a, r in self.arms.items():
            fm, fs = r.forgetting_mean_std
            gm, gs = r.new_gain_mean_std
            rows.append(f"{a:12s}{sum(r.old_bpb) / len(r.old_bpb):12.4f}{fm:+11.4f}±{fs:.3f}"
                        f"{sum(r.new_bpb) / len(r.new_bpb):12.4f}{gm:+13.4f}±{gs:.3f}  {'yes' if a in nd else 'no'}")
        rows.append("forgetting = old-domain bits/byte after minus before (lower is better); "
                    "gain = new-domain bits/byte before minus after (higher is better)")
        return "\n".join(rows)

    def to_dict(self) -> Dict[str, Any]:
        return {"before": self.before, "budget": self.budget, "non_dominated": self.non_dominated(),
                "arms": {a: {"seeds": r.seeds, "old_bpb": r.old_bpb, "new_bpb": r.new_bpb, "forgetting": r.forgetting,
                             "new_gain": r.new_gain, "detail": r.detail} for a, r in self.arms.items()}}


def compare_continual_strategies(
        base: Union[str, TensegrityLM], old_dir: str, new_dir: str, *, steps: int = 100, arms: Sequence[str] = ARMS,
        replay_fraction: float = 0.3, ewc_lambda: float = 100.0, ewc_kind: str = "empirical",
        ewc_sequences: int = 32, batch_size: int = 8, grad_accum_steps: int = 1, learning_rate: float = 1e-3,
        optimizer: str = "adamw", seeds: Sequence[int] = (0,), eval_batches: int = 8, device: str = "cpu",
        run_dir: Optional[str] = None, verbose: bool = True) -> ContinualReport:
    """Compare ways of learning ``new_dir`` after ``old_dir`` from the SAME starting weights, under a matched
    budget: identical optimizer steps, tokens per step, schedule, seeds and evaluation windows. They differ only in
    what each batch is made of and in the loss:

    * ``new_only``   -- every batch from the new corpus;
    * ``replay``     -- ``replay_fraction`` of each batch from the old corpus (so it sees correspondingly fewer
      new-domain tokens at equal total tokens -- that is the trade being measured);
    * ``replay_ewc`` -- replay plus the EWC penalty (``ewc_lambda``, Fisher of ``ewc_kind`` estimated on the old
      corpus' training split).

    ``base`` is a checkpoint path or a model. Both domains are scored on their held-out splits, before and
    after, in bits per byte. Judge strategies by the pair (forgetting, new-domain gain); the report marks the
    ones no other strategy beats on both. One seed is an anecdote: pass several to see the spread."""
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; choose from {ARMS}")
    say = print if verbose else (lambda *a, **k: None)
    model = load_model(base, device)[0] if isinstance(base, str) else unwrap_model(base)
    cfg = model.cfg
    suite = EvalSuite([EvalDomain(n, d, r, n_batches=eval_batches, batch_size=8)
                       for n, d, r in (("old", old_dir, "retain"), ("new", new_dir, "adapt"))])
    model.to(device)
    base_res = suite.evaluate(model, device)
    before = {k: base_res.domains[k]["bpb_bytes"] for k in ("old", "new")}
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    tokens_per_step = batch_size * grad_accum_steps * cfg.block_size
    plan = make_plan(steps * tokens_per_step, tokens_per_step, "wsd", min_warmup=1)
    ewc = EWCState.estimate(model, old_dir, ewc_lambda, kind=ewc_kind, n_sequences=ewc_sequences, device=device) \
        if "replay_ewc" in arms else None
    results: Dict[str, ArmResult] = {a: ArmResult(a, list(seeds), [], [], [], []) for a in arms}
    scratch = None if run_dir else tempfile.TemporaryDirectory(prefix="tg_continual_")
    root = run_dir or scratch.name
    for arm in arms:
        for seed in seeds:
            student = TensegrityLM(cfg).to(device)
            student.load_state_dict(base_state)
            tc = TrainConfig(batch_size=batch_size, grad_accum_steps=grad_accum_steps, learning_rate=learning_rate,
                             min_lr=learning_rate / 10, optimizer=optimizer, ema_decay=0.0, eval_interval=max(1, steps),
                             log_interval=max(1, steps), sample_every_evals=0, seed=seed, verbose=False,
                             ckpt_interval_minutes=1e9)
            dirs, weights = ([new_dir], None) if arm == "new_only" else ([new_dir, old_dir],
                                                                        [1.0 - replay_fraction, replay_fraction])
            out = os.path.join(root, f"{arm}_seed{seed}")
            Trainer(student, tc, plan, dirs, out, data_weights=weights, device=device,
                    extra_loss=ewc if arm == "replay_ewc" else None, eval_suite=suite).fit(resume=False)
            r = suite.evaluate(student, device)
            res = results[arm]
            res.old_bpb.append(r.domains["old"]["bpb_bytes"])
            res.new_bpb.append(r.domains["new"]["bpb_bytes"])
            res.forgetting.append(res.old_bpb[-1] - before["old"])
            res.new_gain.append(before["new"] - res.new_bpb[-1])
            res.detail.append({"seed": seed, "old": r.domains["old"], "new": r.domains["new"]})
            say(f"[continual] {arm:11s} seed {seed}: old {res.old_bpb[-1]:.4f} ({res.forgetting[-1]:+.4f}), "
                f"new {res.new_bpb[-1]:.4f} ({-res.new_gain[-1]:+.4f})")
    if scratch is not None:
        scratch.cleanup()
    report = ContinualReport(before, results, {"steps": steps, "tokens_per_step": tokens_per_step,
                                               "replay_fraction": replay_fraction, "ewc_lambda": ewc_lambda,
                                               "ewc_kind": ewc_kind, "optimizer": optimizer,
                                               "learning_rate": learning_rate, "seeds": list(seeds),
                                               "old_fingerprint": suite.identity()["domains"][0]["fingerprint"],
                                               "new_fingerprint": suite.identity()["domains"][1]["fingerprint"]})
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "continual_report.json"), "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
    say(report.table())
    return report
