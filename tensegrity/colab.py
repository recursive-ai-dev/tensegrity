"""Google Colab (and Kaggle / local) helpers. Every function degrades gracefully outside Colab,
so the same notebook runs unchanged on a laptop or in CI."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from typing import Optional


def in_colab() -> bool:
    if "google.colab" in sys.modules:
        return True
    try:  # find_spec imports parent packages and raises if there is no `google` namespace at all
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


def in_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def mount_drive(mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive. Returns True on success, False (with a message) anywhere it is not possible."""
    if not in_colab():
        print("[drive] not running in Colab; skipping Drive mount.")
        return False
    if os.path.isdir(os.path.join(mount_point, "MyDrive")):
        return True
    try:
        from google.colab import drive
        drive.mount(mount_point)
        return os.path.isdir(os.path.join(mount_point, "MyDrive"))
    except Exception as e:  # noqa: BLE001 - user may decline the prompt
        print(f"[drive] could not mount Drive ({type(e).__name__}: {e}); continuing with local disk.")
        return False


def project_dir(use_drive: bool = True, name: str = "tensegrity") -> str:
    """Where checkpoints live: Drive (survives disconnects) when available, else local disk.
    ``TENSEGRITY_PROJECT_DIR`` overrides the location (used by CI)."""
    env = os.environ.get("TENSEGRITY_PROJECT_DIR")
    if env:
        os.makedirs(env, exist_ok=True)
        return env
    if use_drive and mount_drive():
        base = "/content/drive/MyDrive"
    elif in_kaggle():
        base = "/kaggle/working"
    else:
        base = os.getcwd()
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path


def local_scratch(name: str = "tensegrity_data") -> str:
    """Fast local disk for datasets (never Drive: random reads from FUSE are slow).
    ``TENSEGRITY_DATA_DIR`` overrides the location (used by CI)."""
    env = os.environ.get("TENSEGRITY_DATA_DIR")
    if env:
        os.makedirs(env, exist_ok=True)
        return env
    base = "/content" if os.path.isdir("/content") else ("/kaggle/working" if in_kaggle() else os.getcwd())
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path


def get_secret(name: str) -> Optional[str]:
    """Colab Secrets (key icon in the sidebar) or an environment variable."""
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:  # noqa: BLE001 - not in Colab / secret missing / access not granted
        return os.environ.get(name)


def download_to_browser(path: str) -> None:
    if in_colab():
        from google.colab import files
        files.download(path)
    else:
        print(f"[download] not in Colab; file is at {os.path.abspath(path)}")


def free_disk_gb(path: str = ".") -> float:
    return shutil.disk_usage(path).free / 2**30


def total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return float("nan")


def runtime_summary() -> str:
    import torch
    from .hw import gpu_info
    g = gpu_info()
    gpu = (f"{g['name']} ({g['total_gb']:.0f} GB, sm{g['capability'][0]}{g['capability'][1]}, "
           f"{'native bf16' if g['native_bf16'] else 'fp16 (no native bf16)'})") if g["available"] else "none (CPU)"
    where = "Google Colab" if in_colab() else ("Kaggle" if in_kaggle() else "local")
    return (f"{where} | python {platform.python_version()} | torch {torch.__version__} | GPU: {gpu} | "
            f"RAM {total_ram_gb():.0f} GB | free disk {free_disk_gb('/content' if os.path.isdir('/content') else '.'):.0f} GB")


def push_to_hub(folder: str, repo_id: str, token: str, private: bool = True) -> str:
    """Upload an export folder to the Hugging Face Hub (opt-in; needs ``huggingface_hub``)."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"
