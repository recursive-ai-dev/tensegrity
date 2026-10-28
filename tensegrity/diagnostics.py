"""Offline installation checks, including a small CPU end-to-end exercise.

The dependency report uses the standard library so it can diagnose incomplete
installations without first importing PyTorch.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from . import __version__

_DEPENDENCIES = (("torch", "torch", (2, 3)), ("numpy", "numpy", (1, 23)),
                 ("safetensors", "safetensors", (0, 4)))


def installation_report(smoke: bool = False) -> Dict[str, Any]:
    """Return JSON-compatible readiness information; never download files.

    ``smoke=True`` additionally runs one optimizer update, safe checkpoint and
    safetensors reloads, and exact cached/uncached generation on a tiny CPU model.
    This checks the installed runtime, not model quality or GPU training.
    """
    report: Dict[str, Any] = {"version": __version__, "python": platform.python_version(),
                              "executable": sys.executable, "platform": platform.platform(),
                              "dependencies": {}, "ok": sys.version_info >= (3, 9)}
    for distribution, module, minimum in _DEPENDENCIES:
        entry: Dict[str, Any] = {"minimum": ".".join(map(str, minimum)), "ok": False}
        try:
            entry["version"] = importlib.metadata.version(distribution)
            numbers = re.match(r"^(\d+)\.(\d+)", entry["version"])
            entry["ok"] = (importlib.util.find_spec(module) is not None and numbers is not None
                           and tuple(map(int, numbers.groups())) >= minimum)
            if not entry["ok"]:
                entry["error"] = "missing module or unsupported version"
        except (ImportError, ValueError) as exc:
            entry["error"] = str(exc)
        report["dependencies"][distribution] = entry
        report["ok"] = report["ok"] and entry["ok"]
    if report["ok"]:
        try:
            import torch
            from .hw import gpu_info, pick_device, pick_dtype
            device = pick_device()
            report["device"] = {"selected": device, "dtype": str(pick_dtype("auto", device)),
                                "cuda_runtime": torch.version.cuda, "gpu": gpu_info()}
        except Exception as exc:
            report["runtime_error"] = str(exc)
            report["ok"] = False
    if smoke:
        if report["ok"]:
            try:
                report["smoke"] = _cpu_smoke()
            except Exception as exc:
                report["smoke"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            report["ok"] = report["ok"] and report["smoke"]["ok"]
        else:
            report["smoke"] = {"ok": False, "error": "install supported runtime dependencies first"}
    return report


def _cpu_smoke() -> Dict[str, Any]:
    import torch
    from .checkpoint import load_model, save_checkpoint
    from .config import TensegrityConfig
    from .export import export_safetensors, load_safetensors
    from .model import TensegrityLM
    from .tokenizer import ByteTokenizer

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(2, threads))
        with tempfile.TemporaryDirectory(prefix="tensegrity-doctor-") as tmp, torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(7)
            cfg = TensegrityConfig(n_layer=2, n_embd=32, n_head=2, n_kv_head=1,
                                   block_size=16, sliding_window=8, global_every=2,
                                   ffn_mult=2, ffn_align=8, byte_mix_kernel=4,
                                   layerdrop_max=0, grad_checkpoint=False)
            model = TensegrityLM(cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            ids = torch.randint(0, 256, (2, 16))
            _, losses, _ = model(ids, ids.roll(-1, 1))
            loss = losses["total_loss"]
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("non-finite training gradients")
            optimizer.step()
            model.eval()
            path = str(Path(tmp) / "checkpoint.pt")
            save_checkpoint(path, model, optimizer, cfg, 1, float(loss.detach()))
            loaded, _, _ = load_model(path)
            loaded.eval()
            weights, _ = export_safetensors(model, cfg, tmp)
            exported, _ = load_safetensors(weights)
            exported.eval()
            with torch.no_grad():
                expected = model(ids)[0]
                torch.testing.assert_close(loaded(ids)[0], expected, rtol=0, atol=0)
                torch.testing.assert_close(exported(ids)[0], expected, rtol=0, atol=0)
                prompt = torch.tensor([[ByteTokenizer.DOC_START, 72, 105]])
                cached = loaded.generate(prompt, 18, temperature=0, stop_on_eos=False)
                uncached = loaded.generate(prompt, 18, temperature=0, stop_on_eos=False, use_cache=False)
                if not torch.equal(cached, uncached):
                    raise ValueError("cached and uncached generation differ")
            text = "Hello, café 🌍"
            if ByteTokenizer().decode(ByteTokenizer().encode(text)) != text:
                raise ValueError("UTF-8 round trip failed")
            return {"ok": True, "device": "cpu", "training_loss": float(loss.detach()),
                    "checks": ["optimizer update", "safe checkpoint reload", "safetensors reload",
                               "cached generation across context window", "UTF-8 round trip"]}
    finally:
        torch.set_num_threads(threads)


def format_installation_report(report: Dict[str, Any]) -> str:
    lines = [f"Tensegrity {report['version']} — installation {'OK' if report['ok'] else 'FAILED'}",
             f"Python {report['python']} ({report['executable']})"]
    for name, entry in report["dependencies"].items():
        lines.append(f"{name}: {entry.get('version', 'missing')} (need >= {entry['minimum']}) "
                     f"{'OK' if entry['ok'] else entry.get('error', 'FAILED')}")
    if "device" in report:
        device = report["device"]
        lines.append(f"Device: {device['selected']}; dtype: {device['dtype']}")
    if "runtime_error" in report:
        lines.append(f"Runtime: {report['runtime_error']}")
    if "smoke" in report:
        result = report["smoke"]
        lines.append("CPU smoke: " + (", ".join(result["checks"]) + " OK" if result["ok"] else result["error"]))
    return "\n".join(lines)
