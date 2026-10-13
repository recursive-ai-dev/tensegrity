#@title 0.1 · Install the few missing dependencies { display-mode: "form" }
import importlib, subprocess, sys

def _ensure(pkg, module=None):
    """Import `module`; pip-install `pkg` only if it is genuinely missing (Colab ships torch/numpy)."""
    try:
        importlib.import_module(module or pkg)
        return
    except ImportError:
        pass
    for extra in ([], ["--break-system-packages"]):  # second form for externally-managed Pythons
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg, *extra],
                           capture_output=True, text=True)
        if r.returncode == 0:
            importlib.invalidate_caches()
            importlib.import_module(module or pkg)
            return
    raise RuntimeError(f"could not install {pkg!r}:\n{r.stderr[-800:]}\nInstall it manually: pip install {pkg}")

for _pkg, _mod in [("numpy", "numpy"), ("torch", "torch"), ("safetensors", "safetensors"), ("pyarrow", "pyarrow")]:
    _ensure(_pkg, _mod)
print("dependencies OK")
