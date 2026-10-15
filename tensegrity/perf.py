"""Measuring performance where it is actually spent: generation latency, attention backends, host syncs.

Nothing here is an estimate: every number is timed (or counted) on the device it is run on. The point is to
profile before substituting components -- e.g. whether the SDPA backend a layer really gets is the fused one,
whether decoding past the context window re-prefills, whether the training step contains device syncs.
"""

from __future__ import annotations

import resource
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import unwrap_model
from .hw import amp_settings, device_type_of, pick_device, pick_dtype
from .layers import SDPA_HAS_ENABLE_GQA, native_gqa_ok


def _sync(device: str) -> None:
    if device_type_of(device) == "cuda":
        torch.cuda.synchronize()


def _peak_memory(device: str) -> Dict[str, Optional[float]]:
    """Peak memory in MiB. CUDA: allocated / reserved since the last reset. CPU: the PROCESS's peak resident set
    (``ru_maxrss``; it never decreases, so read it as 'at most this much by now')."""
    if device_type_of(device) == "cuda":
        return {"allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "process_rss_peak_mib": None}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss      # kilobytes on Linux, bytes on macOS
    return {"allocated_mib": None, "reserved_mib": None,
            "process_rss_peak_mib": rss / 2**20 if sys.platform == "darwin" else rss / 1024.0}


@torch.no_grad()
def benchmark_generation(model: nn.Module, device: str = "auto", new_tokens: int = 64, batch: int = 1,
                         prompt_len: Optional[int] = None, modes: Sequence[str] = ("window", "streaming"),
                         dtype: str = "auto", repeats: int = 2) -> Dict[str, Any]:
    """Time generation separately for its phases, per ``cache_mode``:

    * ``prefill_ms``: encoding a ``prompt_len`` prompt (default half the context);
    * ``decode_tok_s_fits``: tokens/s while prompt + output still fit in ``block_size``;
    * ``decode_tok_s_overflow``: tokens/s when the prompt already fills the context, i.e. EVERY new token is
      generated past the window -- where ``"window"`` mode re-prefills and ``"streaming"`` does not.

    Reports peak memory per phase (see :func:`_peak_memory` for what that means on CPU). Greedy decoding,
    median of ``repeats`` timed runs after one warm-up."""
    model = unwrap_model(model)
    device = pick_device(device)
    dt = pick_dtype(dtype, device)
    amp = amp_settings(dt, device)
    model.to(device).eval()
    bs = model.cfg.block_size
    prompt_len = prompt_len or max(1, bs // 2)
    fit_new = max(1, min(new_tokens, bs - prompt_len))
    g = torch.Generator().manual_seed(0)
    short = torch.randint(0, 256, (batch, prompt_len), generator=g).to(device)
    full = torch.randint(0, 256, (batch, bs), generator=g).to(device)

    def timed(fn) -> float:
        times = []
        for i in range(repeats + 1):                       # first run warms caches / kernels
            _sync(device)
            t0 = time.perf_counter()
            fn()
            _sync(device)
            if i:
                times.append(time.perf_counter() - t0)
        return sorted(times)[len(times) // 2]

    out: Dict[str, Any] = {"device": device, "dtype": str(dt).replace("torch.", ""), "block_size": bs,
                           "prompt_len": prompt_len, "batch": batch, "modes": {}}
    with torch.autocast(device_type=amp["autocast_device"], dtype=amp["amp_dtype"], enabled=amp["use_amp"]):
        for mode in modes:
            if device_type_of(device) == "cuda":
                torch.cuda.reset_peak_memory_stats()
            row: Dict[str, Any] = {}
            row["prefill_ms"] = 1e3 * timed(lambda: model.forward_incremental(short, None, 0))
            row["prefill_tok_s"] = batch * prompt_len / (row["prefill_ms"] / 1e3)
            row["peak_after_prefill"] = _peak_memory(device)
            t = timed(lambda: model.generate(short, fit_new, temperature=0.0, stop_on_eos=False, cache_mode=mode))
            row["decode_tok_s_fits"] = batch * fit_new / t
            t = timed(lambda: model.generate(full, new_tokens, temperature=0.0, stop_on_eos=False, cache_mode=mode))
            row["decode_tok_s_overflow"] = batch * new_tokens / t
            row["peak_after_overflow_decode"] = _peak_memory(device)
            out["modes"][mode] = row
    return out


def format_generation_benchmark(res: Dict[str, Any]) -> str:
    rows = [f"generation on {res['device']} ({res['dtype']}), context {res['block_size']}, prompt {res['prompt_len']}, "
            f"batch {res['batch']}",
            f"{'cache_mode':12s}{'prefill ms':>12s}{'tok/s (fits)':>14s}{'tok/s (past context)':>22s}"]
    for mode, r in res["modes"].items():
        rows.append(f"{mode:12s}{r['prefill_ms']:12.1f}{r['decode_tok_s_fits']:14.1f}{r['decode_tok_s_overflow']:22.1f}")
    return "\n".join(rows)


# =============================================================================
# SDPA backends
# =============================================================================

def sdpa_backend_report(cfg, device: str = "auto", dtype: str = "auto", batch: int = 2, seq_len: Optional[int] = None,
                        iters: int = 3) -> Dict[str, Any]:
    """Which scaled-dot-product-attention backends can run THIS model's attention shapes on THIS device, and
    how fast. Each of flash / memory-efficient / cuDNN / math is forced in turn (``sdpa_kernel``) on the two
    call patterns the model uses -- global causal attention with the KV heads expanded, and the banded local
    pattern with an explicit mask -- forward + backward. A backend that raises is reported with the reason;
    ``default_choice`` is the one PyTorch picks on its own (inferred as the fastest forced backend that works,
    since PyTorch does not expose its dispatch decision). Also reports whether native GQA (``enable_gqa``)
    would be used for this dtype/device."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    device = pick_device(device)
    dt = pick_dtype(dtype, device)
    qdtype = dt if dt in (torch.float16, torch.bfloat16) and device_type_of(device) == "cuda" else torch.float32
    T = seq_len or min(cfg.block_size, 512)
    H, Hk, D = cfg.n_head, cfg.n_kv_head, cfg.head_dim
    q = torch.randn(batch, H, T, D, device=device, dtype=qdtype, requires_grad=True)
    k = torch.randn(batch, H, T, D, device=device, dtype=qdtype, requires_grad=True)
    v = torch.randn(batch, H, T, D, device=device, dtype=qdtype, requires_grad=True)
    W = min(cfg.sliding_window, T)
    i = torch.arange(T, device=device).view(-1, 1)
    j = torch.arange(T, device=device).view(1, -1)
    band = (j <= i) & (j > i - W)
    patterns = {"global_causal": dict(is_causal=True), "local_banded": dict(attn_mask=band)}
    names = [n for n in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH") if hasattr(SDPBackend, n)]
    res: Dict[str, Any] = {"device": device, "dtype": str(qdtype).replace("torch.", ""),
                           "shape": {"batch": batch, "heads": H, "kv_heads": Hk, "seq": T, "head_dim": D},
                           "native_gqa_would_be_used": bool(cfg.gqa_ratio > 1 and native_gqa_ok(q)),
                           "sdpa_has_enable_gqa": SDPA_HAS_ENABLE_GQA, "backends": {}}
    for name in names:
        row: Dict[str, Any] = {}
        for pname, kw in patterns.items():
            try:
                with sdpa_kernel([getattr(SDPBackend, name)]):
                    def step():
                        F.scaled_dot_product_attention(q, k, v, **kw).sum().backward()
                    step()                                   # eligibility + warm-up
                    _sync(device)
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        step()
                    _sync(device)
                row[pname] = {"ok": True, "ms": 1e3 * (time.perf_counter() - t0) / iters}
            except Exception as e:  # noqa: BLE001 - any failure means "this backend cannot run this pattern here"
                row[pname] = {"ok": False, "error": str(e).splitlines()[0][:140]}
        res["backends"][name] = row
    for pname in patterns:
        ok = {n: r[pname]["ms"] for n, r in res["backends"].items() if r[pname]["ok"]}
        res.setdefault("default_choice", {})[pname] = min(ok, key=ok.get) if ok else None
    return res


def format_sdpa_report(res: Dict[str, Any]) -> str:
    rows = [f"SDPA on {res['device']} ({res['dtype']}), shape {res['shape']}; native GQA used: {res['native_gqa_would_be_used']}",
            f"{'backend':22s}{'global causal':>18s}{'local banded':>18s}"]
    for n, r in res["backends"].items():
        cell = lambda p: f"{p['ms']:.2f} ms" if p["ok"] else "unavailable"  # noqa: E731
        rows.append(f"{n:22s}{cell(r['global_causal']):>18s}{cell(r['local_banded']):>18s}")
    rows.append(f"fastest working: {res['default_choice']}")
    return "\n".join(rows)


# =============================================================================
# Training-step profile
# =============================================================================

_SYNC_OPS = ("aten::item", "aten::_local_scalar_dense")


def profile_training_step(model: nn.Module, micro_batch: int = 2, seq_len: Optional[int] = None, device: str = "auto",
                          dtype: str = "auto", top: int = 8) -> Dict[str, Any]:
    """Profile one forward + backward with ``torch.profiler`` and report where time goes and whether the pass
    forces the host to wait on the device: ``host_syncs`` counts ``.item()`` / scalar reads. A clean
    forward/backward has none (the Trainer does exactly one deliberate read per step, for the finite-gradient
    check, outside this pass), so a regression that adds a per-layer ``.item()`` shows up as a number > 0."""
    from torch.profiler import ProfilerActivity, profile
    model = unwrap_model(model)
    device = pick_device(device)
    dt = pick_dtype(dtype, device)
    amp = amp_settings(dt, device)
    T = seq_len or model.cfg.block_size
    model.to(device).train()
    x = torch.randint(0, 256, (micro_batch, T), device=device)
    y = torch.randint(0, 256, (micro_batch, T), device=device)

    def step():
        with torch.autocast(device_type=amp["autocast_device"], dtype=amp["amp_dtype"], enabled=amp["use_amp"]):
            _, losses, _ = model(x, y)
        losses["total_loss"].backward()
        model.zero_grad(set_to_none=True)

    step()                                                   # warm-up
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device_type_of(device) == "cuda" else [])
    with profile(activities=acts) as prof:
        step()
        _sync(device)
    avg = prof.key_averages()
    syncs = sum(e.count for e in avg if e.key in _SYNC_OPS)
    rows: List[Dict[str, Any]] = []
    for e in sorted(avg, key=lambda e: e.self_cpu_time_total, reverse=True)[:top]:
        rows.append({"op": e.key, "calls": e.count, "self_cpu_ms": e.self_cpu_time_total / 1e3})
    return {"device": device, "micro_batch": micro_batch, "seq_len": T, "host_syncs": int(syncs), "top_ops": rows,
            "total_self_cpu_ms": sum(e.self_cpu_time_total for e in avg) / 1e3}
