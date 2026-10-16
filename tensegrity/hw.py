"""Hardware helpers: device / dtype selection, GPU facts, micro-batch autotune, throughput.

The headline fix over the original release: ``torch.cuda.is_bf16_supported()`` returns True on
a Turing T4 (Colab's free GPU) because it counts *emulation*; picking bf16 there is far slower
than fp16. :func:`pick_dtype` decides from the compute capability instead.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Optional

import torch

from .config import TensegrityConfig

# Dense fp16/bf16 tensor-core peak, TFLOPs (vendor figures; used only to display an MFU estimate).
_PEAK_TFLOPS = (
    ("H200", 989.0), ("H100", 989.0), ("B200", 2250.0), ("A100", 312.0), ("L40S", 362.0),
    ("L4", 121.0), ("A10G", 70.0), ("A10", 125.0), ("V100", 125.0), ("T4", 65.0),
    ("RTX 4090", 165.0), ("RTX 4080", 97.5), ("RTX 4070", 58.0), ("RTX 4060", 44.0),
    ("RTX 3090", 71.0), ("RTX 3080", 59.5), ("RTX 3070", 40.6), ("RTX 3060", 25.4),
    ("P100", 18.7),
)


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_type_of(device: str) -> str:
    d = str(device)
    return "cuda" if "cuda" in d else ("mps" if d == "mps" else "cpu")


def native_bf16(device: str = "cuda") -> bool:
    """True only for GPUs with hardware bf16 (compute capability >= 8)."""
    if device_type_of(device) != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(torch.device(device))[0] >= 8


def pick_dtype(requested: str, device: str) -> torch.dtype:
    """'auto' -> bf16 on Ampere+ (native), fp16 on older CUDA GPUs (T4/V100), fp32 elsewhere."""
    dev_type = device_type_of(device)
    if requested != "auto":
        dt = {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16,
              "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}[requested]
        if dt == torch.bfloat16 and dev_type == "cuda" and not native_bf16(device):
            warnings.warn("bfloat16 requested on a GPU without native bf16 (compute capability < 8): "
                          "it will be emulated and much slower than float16.", stacklevel=2)
        return dt
    if dev_type == "cuda":
        return torch.bfloat16 if native_bf16(device) else torch.float16
    return torch.float32


def amp_settings(dtype: torch.dtype, device: str) -> Dict[str, Any]:
    dev_type = device_type_of(device)
    use_amp = dtype in (torch.float16, torch.bfloat16) and dev_type in ("cuda", "cpu")
    return {
        "use_amp": use_amp,
        "amp_dtype": dtype if use_amp else torch.float32,
        "autocast_device": "cpu" if dev_type == "mps" else dev_type,
        "use_scaler": dtype == torch.float16 and dev_type == "cuda",
    }


def gpu_info() -> Dict[str, Any]:
    """Facts about the active GPU (or a CPU stub)."""
    if not torch.cuda.is_available():
        return {"available": False, "name": "cpu", "capability": None, "total_gb": 0.0,
                "native_bf16": False, "peak_tflops": None}
    p = torch.cuda.get_device_properties(0)
    peak = next((v for k, v in _PEAK_TFLOPS if k.lower() in p.name.lower()), None)
    return {"available": True, "name": p.name, "capability": (p.major, p.minor),
            "total_gb": p.total_memory / 2**30, "native_bf16": p.major >= 8, "peak_tflops": peak}


def set_fast_matmul() -> None:
    """Allow TF32 for the few fp32 matmuls that remain under autocast (Ampere+ only)."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision("high")


def mfu(cfg: TensegrityConfig, tokens_per_sec: float, seq_len: int, peak_tflops: Optional[float]) -> Optional[float]:
    """Model FLOPs utilisation estimate in [0, 1] (None if the GPU's peak is unknown)."""
    if not peak_tflops:
        return None
    return cfg.flops_per_token(seq_len, training=True) * tokens_per_sec / (peak_tflops * 1e12)


# =============================================================================
# Autotune / benchmark
# =============================================================================

def _is_oom(e: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
    return (oom_cls and isinstance(e, oom_cls)) or "out of memory" in str(e).lower()


def _fwd_bwd(model: torch.nn.Module, B: int, T: int, device: str, amp: Dict[str, Any]) -> None:
    x = torch.randint(0, 256, (B, T), device=device)
    y = torch.randint(0, 256, (B, T), device=device)
    with torch.autocast(device_type=amp["autocast_device"], dtype=amp["amp_dtype"], enabled=amp["use_amp"]):
        _, losses, _ = model(x, y)
    losses["total_loss"].backward()


def measure_throughput(model: torch.nn.Module, micro_batch: int, seq_len: int, device: str,
                       dtype: torch.dtype, steps: int = 5, warmup: int = 2) -> float:
    """Training tokens/second for fwd+bwd at the given micro-batch (no optimizer step)."""
    amp = amp_settings(dtype, device)
    model.train()
    sync = torch.cuda.synchronize if device_type_of(device) == "cuda" else (lambda: None)
    for _ in range(warmup):
        _fwd_bwd(model, micro_batch, seq_len, device, amp)
        model.zero_grad(set_to_none=True)
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        _fwd_bwd(model, micro_batch, seq_len, device, amp)
        model.zero_grad(set_to_none=True)
    sync()
    return steps * micro_batch * seq_len / (time.perf_counter() - t0)


def autotune_micro_batch(model: torch.nn.Module, seq_len: int, device: str, dtype: torch.dtype,
                         target_batch_tokens: int, max_micro: int = 128, headroom: float = 0.9,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Find the largest power-of-two micro-batch whose fwd+bwd fits in memory, then choose
    ``grad_accum`` so one optimizer step sees >= ``target_batch_tokens`` tokens.

    If even micro-batch 1 does not fit, gradient checkpointing is switched on and the search
    repeats once. On CPU/MPS (no OOM signal) a small fixed micro-batch is used.
    """
    dev_type = device_type_of(device)
    if dev_type != "cuda":
        mb = max(1, min(8, target_batch_tokens // seq_len))
        accum = max(1, round(target_batch_tokens / (mb * seq_len)))
        return {"micro_batch": mb, "grad_accum": accum, "grad_checkpoint": False}

    amp = amp_settings(dtype, device)
    was_training = model.training
    model.train()
    total_mem = torch.cuda.get_device_properties(device).total_memory
    # The probe runs before the optimizer and the weight-EMA exist. AdamW keeps 2 fp32 moments
    # per parameter and the EMA 1 more copy, so reserve 3x the parameter bytes up front
    # (Muon needs 1x, so this is conservative for it).
    reserve = 3 * sum(p.numel() * p.element_size() for p in model.parameters())

    def search() -> Optional[int]:
        best, mb = None, 1
        while mb <= max_micro:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                _fwd_bwd(model, mb, seq_len, device, amp)
                peak = torch.cuda.max_memory_allocated()
                model.zero_grad(set_to_none=True)
                if peak + reserve > headroom * total_mem:
                    break
                best = mb
                if verbose:
                    print(f"[autotune] micro_batch={mb:3d} fits (peak {peak / 2**30:.2f} GiB + "
                          f"{reserve / 2**30:.2f} GiB reserved for optimizer/EMA)")
                mb *= 2
            except Exception as e:  # noqa: BLE001
                model.zero_grad(set_to_none=True)
                if not _is_oom(e):
                    raise
                break
        torch.cuda.empty_cache()
        return best

    ckpt_flag = bool(getattr(model, "grad_checkpoint", False))
    best = search()
    if best is None and not ckpt_flag:
        if verbose:
            print("[autotune] nothing fits without gradient checkpointing; enabling it")
        model.grad_checkpoint = True
        ckpt_flag = True
        best = search()
    model.train(was_training)
    if best is None:
        raise RuntimeError("Even micro_batch=1 does not fit in GPU memory at this sequence length; "
                           "reduce block_size or pick a smaller model.")
    accum = max(1, -(-target_batch_tokens // (best * seq_len)))
    return {"micro_batch": best, "grad_accum": accum, "grad_checkpoint": ckpt_flag}
