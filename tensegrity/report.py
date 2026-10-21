"""Preflight and run reports: what you are about to spend compute on, and what a finished run actually was.

* :func:`preflight_report` -- run before training. One structure (and a printable rendering) answering:
  how big is the model really, what can its attention reach, which exact data will it see (fingerprint,
  split sizes, decoded examples, integrity), is the budget sensible (passes over the data), do the optimizer
  and dtype stay finite on THIS device, is there memory headroom, and does the model survive save and reload.
  ``ok`` is False when anything is a blocker; warnings are advice.
* :func:`write_run_report` -- after (or during) a run: the exact configuration, code and data fingerprints,
  updates and tokens really consumed, hardware and software, held-out results, lineage and growth history,
  and export checks, as ``run_report.json`` plus a readable ``run_report.md``.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import resource
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from . import manifest as M
from .checkpoint import environment_info, load_checkpoint, load_model, unwrap_model
from .config import TensegrityConfig
from .data import ByteDataset, audit_dataset
from .evaluation import EvalSuite, evaluate_batches
from .hw import _fwd_bwd, amp_settings, device_type_of, gpu_info, pick_device, pick_dtype
from .model import TensegrityLM
from .schedule import RunPlan
from .train import TrainConfig, preflight
from .workflow import verify_roundtrips


def package_fingerprint(package_dir: Optional[str] = None) -> str:
    """``sha256:`` over the names and contents of every ``.py`` file of the package: names the exact code a
    run used (the Colab notebook embeds these files verbatim, so it has the same fingerprint)."""
    package_dir = package_dir or os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(package_dir, "*.py"))):
        h.update(os.path.basename(path).encode() + b"\0")
        with open(path, "rb") as fh:
            h.update(fh.read())
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


def _system_memory_gib() -> Optional[float]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def _rss_gib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (rss / 2**30) if sys.platform == "darwin" else (rss / 2**20)


# =============================================================================
# Preflight
# =============================================================================

def preflight_report(model: TensegrityLM, tc: TrainConfig, plan: RunPlan, data_dirs, device: str = "auto",
                     data_weights: Optional[Sequence[float]] = None, check_numerics: bool = True,
                     deep_data: bool = False, n_examples: int = 2) -> Dict[str, Any]:
    """Everything worth knowing before committing hours; see the module doc. Does not modify ``model``
    (the numerics check runs real optimizer steps on a throw-away state and restores the weights)."""
    model = unwrap_model(model)
    cfg = model.cfg
    device = pick_device(device)
    dt = pick_dtype(tc.dtype, device)
    dirs = [data_dirs] if isinstance(data_dirs, str) else list(data_dirs)
    T = tc.block_size or cfg.block_size
    blockers: List[str] = []
    warnings: List[str] = []

    # ---- model + attention layout
    kinds = "".join("G" if cfg.is_global_layer(i) else "l" for i in range(cfg.n_layer))
    rf = cfg.receptive_field()
    mdl = {"description": cfg.describe(), "parameters": model.num_parameters(),
           "parameters_formula": cfg.n_params(), "layout": kinds, "global_layers": [i for i, c in enumerate(kinds) if c == "G"],
           "receptive_field": rf, "block_size": cfg.block_size, "train_block_size": T,
           "head_dim": cfg.head_dim, "ffn_hidden": cfg.ffn_hidden}
    if mdl["parameters"] != mdl["parameters_formula"]:
        blockers.append(f"instantiated parameter count {mdl['parameters']:,} != formula {mdl['parameters_formula']:,}")
    if rf < cfg.block_size:
        warnings.append(f"attention is local-only: a byte can influence at most {rf} later positions of the "
                        f"{cfg.block_size}-byte context (no global layer; global_every={cfg.global_every}, "
                        f"n_layer={cfg.n_layer}). Add a global layer or shorten block_size.")
    if T > cfg.block_size:
        blockers.append(f"training block_size {T} exceeds the model's {cfg.block_size}")

    # ---- data
    data: List[Dict[str, Any]] = []
    for d in dirs:
        try:
            a = audit_dataset(d, n_examples=n_examples, deep=deep_data)
        except Exception as e:  # noqa: BLE001 - a dataset that cannot even be audited is a blocker, whatever the reason
            blockers.append(f"{d}: cannot be audited ({type(e).__name__}: {e})")
            data.append({"dir": d, "error": str(e)})
            continue
        ident = M.data_identity(d)
        rec = {"dir": d, "fingerprint": a["fingerprint"], "kind": a["kind"], "n_train_tokens": ident["n_train_tokens"],
               "n_val_tokens": ident["n_val_tokens"], "problems": a["problems"], "examples": a["examples"],
               "stats": a["stats"], "summary": a["text"]}
        data.append(rec)
        blockers += [f"{d}: {p}" for p in a["problems"]]
        for split, n in (("train", ident["n_train_tokens"]), ("val", ident["n_val_tokens"])):
            if n <= T + 1:
                blockers.append(f"{d}: the {split} split has {n:,} tokens, fewer than one {T + 1}-token window")
        if a["stats"]["train"]["control_fraction"] > 0.05:
            warnings.append(f"{d}: {100 * a['stats']['train']['control_fraction']:.1f}% of the sampled bytes are control "
                            f"characters; is this text?")

    # ---- budget
    tokens_per_step = tc.batch_size * tc.grad_accum_steps * T
    w = list(data_weights) if data_weights is not None else [1.0] * len(dirs)
    train_tokens = sum(x["n_train_tokens"] * (wi / sum(w)) for x, wi in zip(data, w) if "n_train_tokens" in x)
    passes = plan.total_tokens / train_tokens if train_tokens else float("nan")
    budget = {"tokens_per_step": tokens_per_step, "plan_tokens_per_step": plan.tokens_per_step, "steps": plan.max_steps,
              "total_tokens": plan.total_tokens, "passes_over_data": passes,
              "schedule": f"{plan.warmup_steps} warmup / {plan.stable_steps} stable / {plan.decay_steps} decay ({plan.schedule})"}
    if tokens_per_step != plan.tokens_per_step:
        warnings.append(f"the plan assumed {plan.tokens_per_step:,} tokens/step but this configuration does {tokens_per_step:,}")
    if passes > 8:
        warnings.append(f"the plan makes {passes:.1f} passes over the data; the model will memorise it - use more data or fewer tokens")

    # ---- numerics (real optimizer steps on this device)
    numerics: Dict[str, Any] = {"checked": False}
    if check_numerics and not blockers:
        numerics = {"checked": True, **preflight(model, tc, device)}
        if not numerics["ok"]:
            blockers.append(f"numerics: {numerics['reason']}")

    # ---- memory
    memory: Dict[str, Any] = {"device": device}
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    reserve = 3 * param_bytes                                           # optimizer moments + EMA, as autotune assumes
    if device_type_of(device) == "cuda" and not blockers:
        amp = amp_settings(dt, device)
        free, total = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats()
        was = model.training
        model.to(device).train()
        try:
            _fwd_bwd(model, tc.batch_size, T, device, amp)
            peak = torch.cuda.max_memory_allocated()
        except torch.cuda.OutOfMemoryError:
            peak = float("inf")
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was)
        memory.update(total_gib=total / 2**30, free_gib=free / 2**30, fwd_bwd_peak_gib=peak / 2**30,
                      optimizer_and_ema_reserve_gib=reserve / 2**30, headroom_gib=(total - peak - reserve) / 2**30)
        if peak == float("inf") or peak + reserve > total:
            blockers.append(f"a micro-batch of {tc.batch_size} x {T} does not fit in {total / 2**30:.1f} GiB with optimizer state")
        elif (total - peak - reserve) / total < 0.10:
            warnings.append(f"less than 10% memory headroom ({memory['headroom_gib']:.2f} GiB): a longer sample or a "
                            f"fragmented allocator may run out; consider a smaller micro-batch")
    else:
        memory.update(system_gib=_system_memory_gib(), process_rss_peak_gib=_rss_gib(),
                      weights_and_optimizer_gib=(param_bytes + reserve) / 2**30)

    # ---- save + reload
    roundtrip: Dict[str, Any] = {"checked": False}
    if not blockers:
        try:
            roundtrip = {"checked": True, "max_logit_diff": verify_roundtrips(model, cfg)}
        except Exception as e:  # noqa: BLE001
            blockers.append(f"save/reload: {e}")
            roundtrip = {"checked": True, "error": str(e)}

    return {"ok": not blockers, "blockers": blockers, "warnings": warnings, "model": mdl, "data": data, "budget": budget,
            "numerics": numerics, "memory": memory, "roundtrip": roundtrip,
            "dtype": str(dt).replace("torch.", ""), "optimizer": tc.optimizer,
            "code_fingerprint": package_fingerprint(), "environment": environment_info()}


def format_preflight(rep: Dict[str, Any]) -> str:
    m, b = rep["model"], rep["budget"]
    lines = [f"{'PREFLIGHT OK' if rep['ok'] else 'PREFLIGHT FAILED'}  (code {rep['code_fingerprint'][:19]}...)",
             f"model    : {m['description']}",
             f"           layers [{m['layout']}]  (G = global, l = local)  counted {m['parameters']:,} parameters",
             f"budget   : {b['steps']:,} steps x {b['tokens_per_step']:,} bytes = {b['total_tokens'] / 1e6:,.1f}M bytes, "
             f"{b['passes_over_data']:.2f} pass(es) over the data; {b['schedule']}"]
    for d in rep["data"]:
        if "error" in d:
            lines.append(f"data     : {d['dir']}: ERROR {d['error']}")
            continue
        lines.append(f"data     : {d['dir']}")
        lines += ["           " + ln for ln in d["summary"].splitlines()[1:]]
    n = rep["numerics"]
    lines.append("numerics : " + (f"{n['optimizer']} / {n['dtype']} finite over the test steps "
                                   f"(loss {n['loss_first']:.2f} -> {n['loss_last']:.2f})" if n.get("ok") else
                                   ("not checked" if not n.get("checked") else f"FAILED: {n.get('reason')}")))
    mem = rep["memory"]
    lines.append("memory   : " + (f"{mem['fwd_bwd_peak_gib']:.2f} GiB peak fwd+bwd + {mem['optimizer_and_ema_reserve_gib']:.2f} GiB optimizer/EMA "
                                  f"of {mem['total_gib']:.1f} GiB ({mem['headroom_gib']:.2f} GiB headroom)" if "total_gib" in mem else
                                  f"weights+optimizer ~{mem['weights_and_optimizer_gib']:.2f} GiB, process peak {mem['process_rss_peak_gib']:.2f} GiB"
                                  + (f" of {mem['system_gib']:.0f} GiB RAM" if mem.get("system_gib") else "")))
    r = rep["roundtrip"]
    lines.append("save/load: " + ("not checked" if not r.get("checked") else
                                  ("FAILED: " + r["error"] if "error" in r else
                                   "config+state_dict, checkpoint and safetensors reload to identical outputs "
                                   f"(max |dlogit| {max(r['max_logit_diff'].values()):.1e})")))
    lines += [f"WARNING  : {w}" for w in rep["warnings"]] + [f"BLOCKER  : {x}" for x in rep["blockers"]]
    return "\n".join(lines)


# =============================================================================
# Run report
# =============================================================================

def _load_ckpt_light(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:  # noqa: BLE001 - older formats / platforms without mmap: load normally
        return load_checkpoint(path, "cpu")


def write_run_report(run_dir: str, data_dirs=None, eval_suite: Optional[EvalSuite] = None, evaluate: bool = True,
                     export_dir: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write ``run_dir/run_report.json`` and ``run_report.md`` and return the report dict.

    Reads the run's own records (``run_plan.json``, ``ckpt_latest.pt``, ``train_log.jsonl``, growth decisions);
    with ``evaluate`` it also scores ``ckpt_best.pt`` on ``eval_suite`` (or, failing that, the validation split of
    ``data_dirs[0]``). ``export_dir`` adds a check that an exported safetensors file reproduces the best model."""
    from . import __version__
    rep: Dict[str, Any] = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_dir": os.path.abspath(run_dir),
                           "code": {"version": __version__, "fingerprint": package_fingerprint()},
                           "environment": environment_info(), "hardware": gpu_info()}
    plan_path = os.path.join(run_dir, "run_plan.json")
    if os.path.exists(plan_path):
        with open(plan_path) as fh:
            rp = json.load(fh)
        cfg = TensegrityConfig.from_dict(rp["model_config"], strict=True)
        rep.update(plan=rp["plan"], train_config=rp["train_config"],
                   model={"config": rp["model_config"], "parameters": cfg.n_params(), "description": cfg.describe(),
                          "receptive_field": cfg.receptive_field()})
    latest = os.path.join(run_dir, "ckpt_latest.pt")
    if os.path.exists(latest):
        ck = _load_ckpt_light(latest)
        planned = (rep.get("plan") or {}).get("max_steps")
        done = int(ck.get("completed_updates", ck["step"]))
        rep["progress"] = {"completed_updates": done, "applied_updates": int(ck.get("applied_updates", done - ck.get("skipped_steps", 0))),
                           "skipped_steps": int(ck.get("skipped_steps", 0)), "tokens_seen": int(ck.get("tokens_seen", 0)),
                           "planned_updates": planned, "finished": bool(planned and done >= planned),
                           "best_val_loss": float(ck["best_val_loss"]), "best_val_bpb_per_token": float(ck["best_val_loss"]) / math.log(2),
                           "best_source": ck.get("best_source")}
        rep["data"] = ck.get("data")
        rep["lineage"] = ck.get("lineage")
        rep["hooks"] = ck.get("hooks")
        rep["eval_suite"] = ck.get("eval_suite")
        rep["run_environment"] = ck.get("environment")
        rep["lr_state"] = ck.get("lr_state")
        if ck.get("environment") and ck["environment"].get("tensegrity") != __version__:
            rep.setdefault("notes", []).append(f"run trained with tensegrity {ck['environment'].get('tensegrity')}, reported by {__version__}")
    log_path = os.path.join(run_dir, "train_log.jsonl")
    if os.path.exists(log_path):
        recs = [json.loads(x) for x in open(log_path) if x.strip()]
        evals = [r for r in recs if "val_loss" in r]
        steps = [r for r in recs if "elapsed" in r]
        rep["log"] = {"records": len(recs), "evals": len(evals), "last_eval": evals[-1] if evals else None,
                      "wall_seconds_last_session": steps[-1]["elapsed"] if steps else None}
    rep["growth"] = []
    for p in sorted(glob.glob(os.path.join(run_dir, "growth_decision*.json"))):
        with open(p) as fh:
            rep["growth"].append({"file": os.path.basename(p), **json.load(fh)})
    best = os.path.join(run_dir, "ckpt_best.pt")
    if os.path.exists(best):
        rep["best_checkpoint"] = {"sha256": M.sha256_file(best), "bytes": os.path.getsize(best)}
        if evaluate and (eval_suite is not None or data_dirs):
            model, mcfg, _ = load_model(best, "cpu")
            dev = pick_device("auto")
            model.to(dev)
            if eval_suite is not None:
                res = eval_suite.evaluate(model, dev)
                rep["final_evaluation"] = {"suite": eval_suite.identity(), **res.to_dict()}
            else:
                d0 = [data_dirs] if isinstance(data_dirs, str) else list(data_dirs)
                ds = ByteDataset(d0[0])
                batches = ds.fixed_batches("val", 20, 16, mcfg.block_size, dev)
                rep["final_evaluation"] = {"data": M.data_identity(d0[0])["fingerprint"],
                                           **evaluate_batches(model, batches, dev, pick_dtype("auto", dev))}
            if export_dir:
                rep["export"] = _check_export(model, mcfg, export_dir, dev)
    if extra:
        rep["extra"] = extra
    with open(os.path.join(run_dir, "run_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    with open(os.path.join(run_dir, "run_report.md"), "w") as fh:
        fh.write(render_run_report(rep))
    return rep


def _check_export(model: TensegrityLM, cfg: TensegrityConfig, export_dir: str, device: str) -> Dict[str, Any]:
    from .export import load_safetensors
    out: Dict[str, Any] = {"dir": export_dir, "files": {}}
    files = sorted(glob.glob(os.path.join(export_dir, "*.safetensors")))
    if not files:
        return {**out, "error": "no .safetensors file found"}
    g = torch.Generator().manual_seed(5)
    x = torch.randint(0, 256, (1, min(32, cfg.block_size)), generator=g).to(device)
    model.eval()
    with torch.no_grad():
        ref = model(x)[0].float()
    for f in files:
        try:
            m2, _ = load_safetensors(f, None, device)
            with torch.no_grad():
                diff = float((m2.eval()(x)[0].float() - ref).abs().max())
            out["files"][os.path.basename(f)] = {"sha256": M.sha256_file(f), "max_logit_diff_vs_best": diff}
        except Exception as e:  # noqa: BLE001
            out["files"][os.path.basename(f)] = {"error": f"{type(e).__name__}: {e}"}
    return out


def render_run_report(rep: Dict[str, Any]) -> str:
    L = [f"# Run report", "", f"generated {rep['generated_utc']} - `{rep['run_dir']}`", "",
         f"* code: tensegrity {rep['code']['version']} `{rep['code']['fingerprint'][:23]}...`",
         f"* software: torch {rep['environment']['torch']}, numpy {rep['environment']['numpy']}, python {rep['environment']['python']}"]
    hw = rep.get("hardware") or {}
    L.append(f"* hardware: {hw.get('name', 'cpu')}" + (f" ({hw['total_gb']:.0f} GB)" if hw.get("total_gb") else ""))
    if "model" in rep:
        L += ["", "## Model", "", f"{rep['model']['description']}  ", f"{rep['model']['parameters']:,} parameters; attention reaches {rep['model']['receptive_field']} positions"]
    if "progress" in rep:
        p = rep["progress"]
        L += ["", "## Progress", "",
              f"* {'finished' if p['finished'] else 'in progress'}: {p['completed_updates']:,} of {p['planned_updates']:,} planned updates "
              f"({p['applied_updates']:,} applied, {p['skipped_steps']:,} skipped), {p['tokens_seen'] / 1e6:,.2f}M bytes consumed",
              f"* best validation loss {p['best_val_loss']:.4f} nats/token ({p['best_source']} weights)"]
    if rep.get("data"):
        L += ["", "## Data", ""] + [f"* #{k}: `{v['fingerprint']}` ({v['kind']})" for k, v in rep["data"]["identity"].items()]
        L.append(f"* mixture weights {[round(w, 3) for w in rep['data']['weights']]}")
    if rep.get("lineage"):
        L += ["", "## Lineage", "", "```json", json.dumps(rep["lineage"], indent=1, default=str), "```"]
    if rep.get("final_evaluation"):
        fe = rep["final_evaluation"]
        L += ["", "## Held-out evaluation of the best checkpoint", ""]
        if "domains" in fe:
            L += ["| domain | role | bits/byte | nats/token |", "|---|---|---:|---:|"]
            L += [f"| {k} | {fe['roles'][k]} | {m['bpb_bytes']:.4f} | {m['loss']:.4f} |" for k, m in fe["domains"].items()]
            L.append(f"\nmacro mean {fe['macro_bpb_bytes']:.4f} bits/byte; pooled {fe['micro_bpb_bytes']:.4f}")
        else:
            L.append(f"{fe['bpb_bytes']:.4f} bits/byte, {fe['loss']:.4f} nats/token, perplexity {fe['ppl']:.2f} (byte tokens)")
    for g in rep.get("growth", []):
        L += ["", f"## Growth ({g['file']})", "", f"accepted: {g['accepted']}" + ("; " + "; ".join(g["reasons"]) if g["reasons"] else "")]
    if rep.get("export"):
        L += ["", "## Export", ""] + [f"* `{k}`: " + (v.get("error") or f"max |dlogit| vs best {v['max_logit_diff_vs_best']:.1e}") for k, v in rep["export"]["files"].items()]
    return "\n".join(L) + "\n"
