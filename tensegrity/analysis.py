"""Model analysis helpers: confidence-head calibration and a per-preset hardware benchmark."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .data import ByteDataset
from .hw import amp_settings, autotune_micro_batch, device_type_of, gpu_info, mfu, measure_throughput, pick_dtype
from .model import TensegrityLM


@torch.no_grad()
def confidence_calibration(model: TensegrityLM, data_dir: str, device: str = "cpu", n_batches: int = 20,
                           batch_size: int = 8, n_bins: int = 10, dtype: str = "auto",
                           seq_len: Optional[int] = None) -> Dict[str, Any]:
    """
    How trustworthy is the confidence head? It predicts P(argmax prediction == target). Tokens are
    bucketed by predicted confidence on the validation split and the empirical accuracy of each
    bucket is measured. A calibrated head has accuracy ~ mean confidence in every bin.

    Returns ``{"bins": [{bin_lo, bin_hi, count, accuracy, mean_confidence}], "ece": float,
    "n_tokens": int, "base_accuracy": float}``; ECE is the count-weighted mean |accuracy - confidence|
    (0 = perfectly calibrated).
    """
    if model.confidence_head is None:
        raise ValueError("model has no confidence head (cfg.use_confidence_head=False)")
    cfg = model.cfg
    amp = amp_settings(pick_dtype(dtype, device), device)
    ds = ByteDataset(data_dir)
    batches = ds.fixed_batches("val", n_batches, batch_size, seq_len or cfg.block_size, device)
    was_training = model.training
    model.eval()
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    count = torch.zeros(n_bins)
    correct_sum = torch.zeros(n_bins)
    conf_sum = torch.zeros(n_bins)
    try:
        for x, y in batches:
            with torch.autocast(device_type=amp["autocast_device"], dtype=amp["amp_dtype"], enabled=amp["use_amp"]):
                logits, _, conf = model(x)
            correct = (logits.argmax(-1) == y).float().reshape(-1).cpu()
            c = conf.float().reshape(-1).clamp(0, 1).cpu()
            idx = torch.bucketize(c, edges[1:-1])
            count += torch.bincount(idx, minlength=n_bins).float()
            correct_sum += torch.bincount(idx, weights=correct, minlength=n_bins).float()
            conf_sum += torch.bincount(idx, weights=c, minlength=n_bins).float()
    finally:
        model.train(was_training)
    total = float(count.sum())
    rows: List[Dict[str, Any]] = []
    ece = 0.0
    for b in range(n_bins):
        n = float(count[b])
        acc = float(correct_sum[b] / n) if n else float("nan")
        mc = float(conf_sum[b] / n) if n else float("nan")
        if n:
            ece += n / total * abs(acc - mc)
        rows.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]), "count": int(n),
                     "accuracy": acc, "mean_confidence": mc})
    return {"bins": rows, "ece": ece, "n_tokens": int(total),
            "base_accuracy": float(correct_sum.sum() / max(total, 1.0))}


def plot_reliability(result: Dict[str, Any]) -> None:
    """Reliability diagram for :func:`confidence_calibration` (no-op without matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    rows = [r for r in result["bins"] if r["count"]]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot([r["mean_confidence"] for r in rows], [r["accuracy"] for r in rows], "o-", label="model")
    ax.set(xlabel="predicted confidence", ylabel="empirical accuracy", xlim=(0, 1), ylim=(0, 1),
           title=f"confidence calibration (ECE {result['ece']:.3f})")
    ax.legend()
    ax.grid(alpha=.3)
    plt.show()


def benchmark_presets(names, device: str = "auto", seq_len: Optional[int] = None, dtype: str = "auto",
                      verbose: bool = True) -> List[Dict[str, Any]]:
    """Measure training throughput of each preset on this machine (autotuned micro-batch)."""
    from .presets import get_preset, get_preset_config
    from .hw import pick_device
    device = pick_device(device)
    dt = pick_dtype(dtype, device)
    info = gpu_info()
    rows = []
    for name in names:
        cfg = get_preset_config(name)
        T = seq_len or cfg.block_size
        model = TensegrityLM(cfg).to(device)
        tune = autotune_micro_batch(model, T, device, dt, get_preset(name).batch_tokens, verbose=False)
        model.grad_checkpoint = tune["grad_checkpoint"]
        tps = measure_throughput(model, tune["micro_batch"], T, device, dt, steps=4, warmup=2)
        budget = get_preset(name).default_tokens
        rows.append({"preset": name, "params": cfg.n_params(), "micro_batch": tune["micro_batch"],
                     "grad_accum": tune["grad_accum"], "grad_checkpoint": tune["grad_checkpoint"],
                     "tokens_per_sec": tps, "mfu": mfu(cfg, tps, T, info["peak_tflops"]),
                     "hours_for_default_budget": budget / tps / 3600.0})
        if verbose:
            print(f"{name:>5s}: {tps / 1e3:8.1f}k tok/s  micro-batch {tune['micro_batch']:3d} x{tune['grad_accum']:<3d}"
                  f"  -> default {budget / 1e6:.0f}M-byte budget in {rows[-1]['hours_for_default_budget']:.1f} h")
        del model
        if device_type_of(device) == "cuda":
            torch.cuda.empty_cache()
    return rows
