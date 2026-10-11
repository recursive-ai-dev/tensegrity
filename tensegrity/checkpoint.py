"""Checkpointing: atomic writes, safe-to-unpickle RNG state, v1 -> v2 compatible loading.

Format v2 differences from v1 (readers of v1 files are fully supported):
  * RNG state is stored as tensors / ints only, so checkpoints load with
    ``torch.load(weights_only=True)`` -- no arbitrary-code-execution surface.
  * ``ckpt_best.pt`` is weights-only by default (a third of the size of a full checkpoint).
  * Extra training state (EMA, dataset sampler, LayerDrop generator, run plan) is optional.
"""

from __future__ import annotations

import os
import pickle
import random
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import TensegrityConfig
from .model import TensegrityLM

CHECKPOINT_FORMAT_VERSION = 2
_REQUIRED = ("model_state_dict", "config", "step", "best_val_loss")


# =============================================================================
# RNG state (safe representation)
# =============================================================================

def capture_rng_state(model: Optional[nn.Module] = None, dataset: Any = None) -> Dict[str, Any]:
    st = random.getstate()
    ns = np.random.get_state()
    state: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": {"version": st[0], "state": list(st[1]), "gauss": st[2]},
        "numpy": {"name": ns[0], "keys": torch.from_numpy(ns[1].astype(np.int64)),
                  "pos": int(ns[2]), "has_gauss": int(ns[3]), "cached": float(ns[4])},
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    raw = getattr(model, "module", model)
    if raw is not None and hasattr(raw, "layerdrop_state"):
        state["layerdrop"] = raw.layerdrop_state()
    if dataset is not None:
        state["dataset"] = dataset.state_dict()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]], model: Optional[nn.Module] = None,
                      dataset: Any = None) -> None:
    """Best-effort restore; accepts both v2 (safe) and v1 (raw tuple / ndarray) layouts."""
    if not state:
        return
    try:
        py = state.get("python", state.get("random"))
        if isinstance(py, dict):
            random.setstate((py["version"], tuple(py["state"]), py["gauss"]))
        elif py is not None:
            random.setstate(py)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"failed to restore python RNG state ({e})")
    if "torch" in state:
        t = state["torch"]
        torch.set_rng_state(t.cpu() if isinstance(t, torch.Tensor) else t)
    if "numpy" in state:
        ns = state["numpy"]
        try:
            if isinstance(ns, dict):
                np.random.set_state((ns["name"], ns["keys"].numpy().astype(np.uint32),
                                     ns["pos"], ns["has_gauss"], ns["cached"]))
            else:
                np.random.set_state(ns)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"failed to restore numpy RNG state ({e})")
    if "cuda" in state and torch.cuda.is_available():
        try:
            cs = [s.cpu() if isinstance(s, torch.Tensor) else s for s in state["cuda"]]
            n = torch.cuda.device_count()
            if len(cs) == n:
                torch.cuda.set_rng_state_all(cs)
            else:  # different GPU count than when saved: restore what exists
                for dev in range(min(len(cs), n)):
                    torch.cuda.set_rng_state(cs[dev], device=dev)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"failed to restore CUDA RNG state ({e})")
    raw = getattr(model, "module", model)
    if "layerdrop" in state and raw is not None and hasattr(raw, "set_layerdrop_state"):
        raw.set_layerdrop_state(state["layerdrop"])
    if dataset is not None and "dataset" in state:
        dataset.load_state_dict(state["dataset"])


# =============================================================================
# save / load
# =============================================================================

def atomic_torch_save(obj: Any, path: str) -> None:
    """torch.save to ``path + '.tmp'``, fsync, then atomic rename: a crash (or a Colab
    disconnect) mid-write can never leave a truncated file that looks like a checkpoint."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer],
                    cfg: TensegrityConfig, step: int, best_val_loss: float,
                    rng_state: Optional[Dict[str, Any]] = None,
                    scaler: Optional[torch.amp.GradScaler] = None,
                    ema_state: Optional[Dict[str, Any]] = None,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    """``optimizer=None`` writes a weights-only checkpoint (still loadable by
    :func:`load_checkpoint` / :func:`load_model`)."""
    raw = getattr(model, "module", model)
    ckpt: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "ema_state_dict": ema_state,
        "config": cfg.to_dict(),
        "step": int(step),
        "best_val_loss": float(best_val_loss),
        "rng_state": rng_state,
    }
    if extra:
        ckpt.update(extra)
    atomic_torch_save(ckpt, path)


def load_checkpoint(path: str, device: str = "cpu", allow_legacy_pickle: bool = True) -> Dict[str, Any]:
    """
    Load a checkpoint dict. v2 files load under ``weights_only=True``. Files written by the
    original release (v1) contain numpy RNG objects and need full unpickling; that is only
    attempted when ``allow_legacy_pickle`` is True, with a warning -- **only load checkpoint
    files you trust** (pickle can execute code).
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, AttributeError, ValueError) as e:
        if not allow_legacy_pickle:
            raise
        warnings.warn(f"{os.path.basename(path)}: not loadable with weights_only=True ({type(e).__name__}); "
                      f"treating it as a legacy (v1) checkpoint and unpickling it. Only do this for files you trust.",
                      stacklevel=2)
        ckpt = torch.load(path, map_location=device, weights_only=False)
    missing = [k for k in _REQUIRED if k not in ckpt]
    if missing:
        raise ValueError(f"{path} is missing checkpoint key(s) {missing}; not a TensegrityLM checkpoint.")
    ver = ckpt.get("format_version")
    if ver not in (1, CHECKPOINT_FORMAT_VERSION):
        warnings.warn(f"{path} has format_version={ver!r}; this code reads 1 and {CHECKPOINT_FORMAT_VERSION}.")
    return ckpt


def load_model(path: str, device: str = "cpu", use_ema: bool = False, verbose: bool = False
               ) -> Tuple[TensegrityLM, TensegrityConfig, Dict[str, Any]]:
    """Build the model described by a checkpoint and load its weights. ``use_ema=True`` loads
    the EMA weights when the checkpoint has them."""
    ckpt = load_checkpoint(path, device)
    cfg = TensegrityConfig.from_dict(ckpt["config"])
    model = TensegrityLM(cfg, verbose=verbose)
    state = ckpt["model_state_dict"]
    if use_ema and ckpt.get("ema_state_dict"):
        from .train import EMA  # local import: train imports this module
        EMA.apply_state_to_model(ckpt["ema_state_dict"], model)
    else:
        model.load_state_dict(state)
    model.to(device)
    return model, cfg, ckpt
