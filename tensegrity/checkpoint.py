"""Checkpointing: atomic writes, safe-to-unpickle RNG state, v1 -> v2 compatible loading.

Format v2 differences from v1 (readers of v1 files are fully supported):
  * RNG state is stored as tensors / ints only, so checkpoints load with
    ``torch.load(weights_only=True)`` -- no arbitrary-code-execution surface.
  * ``ckpt_best.pt`` is weights-only by default (a third of the size of a full checkpoint).
  * Extra training state (EMA, dataset sampler, LayerDrop generator, run plan) is optional.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import random
import tempfile
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import TensegrityConfig
from .model import TensegrityLM

CHECKPOINT_FORMAT_VERSION = 2
_REQUIRED = ("model_state_dict", "config", "step", "best_val_loss")
_WRAPPER_PREFIXES = ("_orig_mod.", "module.")


def unwrap_model(model: nn.Module) -> nn.Module:
    """The plain module under any ``torch.compile`` / DistributedDataParallel / DataParallel wrappers.

    A compiled module's ``state_dict()`` keys carry an ``_orig_mod.`` prefix (DDP's carry ``module.``),
    which a plain model cannot load; serialisation always goes through the unwrapped module."""
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
            model = model.module
        else:
            return model


def normalize_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    """Strip wrapper prefixes (``_orig_mod.`` from torch.compile, ``module.`` from DDP) when EVERY key
    carries them, so a state dict saved from a wrapped model loads into the plain one."""
    changed = True
    while changed and state:
        changed = False
        for pre in _WRAPPER_PREFIXES:
            if all(k.startswith(pre) for k in state):
                state = {k[len(pre):]: v for k, v in state.items()}
                changed = True
    return state


def environment_info() -> Dict[str, str]:
    """Versions that bound how exactly a run can be reproduced (PyTorch documents that results are not
    guaranteed identical across releases or platforms)."""
    import platform
    from . import __version__
    return {"tensegrity": str(__version__), "torch": str(torch.__version__), "numpy": str(np.__version__),
            "python": platform.python_version(), "platform": platform.platform()}


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
    raw = unwrap_model(model) if model is not None else None
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
                np.random.set_state((ns["name"], ns["keys"].cpu().numpy().astype(np.uint32),
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
    raw = unwrap_model(model) if model is not None else None
    if "layerdrop" in state and raw is not None and hasattr(raw, "set_layerdrop_state"):
        raw.set_layerdrop_state(state["layerdrop"])
    if dataset is not None and "dataset" in state:
        dataset.load_state_dict(state["dataset"])


# =============================================================================
# save / load
# =============================================================================

@contextlib.contextmanager
def _atomic_output(path):
    """Stage one file beside its destination and publish only a completed write.

    Unique temporary files allow concurrent writers and avoid following a predictable
    ``.tmp`` symlink. An unsuccessful write leaves the previous destination intact.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        yield temporary
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(obj: Any, path: str) -> None:
    """Write and fsync a unique temporary file, then atomically replace ``path``."""
    with _atomic_output(path) as temporary:
        torch.save(obj, temporary)


def save_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer],
                    cfg: TensegrityConfig, step: int, best_val_loss: float,
                    rng_state: Optional[Dict[str, Any]] = None,
                    scaler: Optional[torch.amp.GradScaler] = None,
                    ema_state: Optional[Dict[str, Any]] = None,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    """``optimizer=None`` writes a weights-only checkpoint (still loadable by
    :func:`load_checkpoint` / :func:`load_model`)."""
    raw = unwrap_model(model)
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
        collisions = set(extra) & set(ckpt)
        if collisions:
            raise ValueError(f"extra cannot replace checkpoint fields: {sorted(collisions)}")
        ckpt.update(extra)
    atomic_torch_save(ckpt, path)


def load_checkpoint(path: str, device: str = "cpu", allow_legacy_pickle: bool = False) -> Dict[str, Any]:
    """
    Load a checkpoint dict. v2 files load under ``weights_only=True``. Files written by the
    original release (v1) contain numpy RNG objects and need full unpickling; that is only
    attempted when ``allow_legacy_pickle=True`` is explicitly passed, with a warning -- **only load checkpoint
    files you trust** (pickle can execute code).
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as e:
        if not allow_legacy_pickle:
            raise ValueError(
                f"{path} could not be loaded safely. Unrestricted pickle loading is disabled. "
                "For a trusted legacy checkpoint only, pass allow_legacy_pickle=True to load_checkpoint/load_model "
                "or migrate it with tensegrity convert-checkpoint --trust-legacy-pickle."
            ) from e
        warnings.warn(f"{os.path.basename(path)}: not loadable with weights_only=True ({type(e).__name__}); "
                      f"treating it as a legacy (v1) checkpoint and unpickling it. Only do this for files you trust.",
                      stacklevel=2)
        ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path} must contain a checkpoint dictionary, got {type(ckpt).__name__}.")
    missing = [k for k in _REQUIRED if k not in ckpt]
    if missing:
        raise ValueError(f"{path} is missing checkpoint key(s) {missing}; not a TensegrityLM checkpoint.")
    if not isinstance(ckpt["config"], dict):
        raise ValueError(f"{path}: config must be a dictionary.")
    state = ckpt["model_state_dict"]
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                              for k, v in state.items()):
        raise ValueError(f"{path}: model_state_dict must map parameter names to tensors.")
    ckpt["model_state_dict"] = normalize_state_dict(state)
    ver = ckpt.get("format_version", 1)
    if ver not in (1, CHECKPOINT_FORMAT_VERSION):
        raise ValueError(f"{path} has unsupported format_version={ver!r}; this code reads 1 and {CHECKPOINT_FORMAT_VERSION}.")
    return ckpt


def load_model(path: str, device: str = "cpu", use_ema: bool = False, verbose: bool = False,
               strict_config: bool = True, allow_legacy_pickle: bool = False
               ) -> Tuple[TensegrityLM, TensegrityConfig, Dict[str, Any]]:
    """Build the model described by a checkpoint and load its weights. ``use_ema=True`` loads
    the EMA weights when the checkpoint has them. A config field this release does not know is an
    error (``strict_config=False`` drops it with a warning instead)."""
    ckpt = load_checkpoint(path, device, allow_legacy_pickle=allow_legacy_pickle)
    cfg = TensegrityConfig.from_dict(ckpt["config"], strict=strict_config)
    model = TensegrityLM(cfg, verbose=verbose)
    state = ckpt["model_state_dict"]
    model.load_state_dict(state)
    if use_ema and ckpt.get("ema_state_dict"):
        from .train import EMA  # local import: train imports this module
        EMA.apply_state_to_model(ckpt["ema_state_dict"], model)
    model.to(device)
    return model, cfg, ckpt
