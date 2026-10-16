"""Export / import: safetensors (+ self-describing config), model card, INT8 dynamic quantisation."""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import torch

from .config import TensegrityConfig
from .model import TensegrityLM

_DTYPES = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16, "fp16": torch.float16,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


def export_safetensors(model: TensegrityLM, cfg: Optional[TensegrityConfig] = None, out_dir: str = "export",
                       name: str = "model", dtype: str = "float32",
                       extra_metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """
    Write ``{name}.safetensors`` and ``{name}.config.json``.

    * The tied ``lm_head.weight`` is not stored (it is the same tensor as ``token_emb.weight``;
      the loader restores the tie), so the file is smaller and safetensors' no-shared-memory
      rule is respected. Files written by the v1 exporter (which stored both) load fine too.
    * The config is also embedded in the safetensors header, so the single file is self-describing.
    * ``dtype``: "float32" (lossless), "float16" or "bfloat16" for a smaller deployment copy.
    """
    from safetensors.torch import save_file
    cfg = cfg or model.cfg
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(set(_DTYPES))}, got {dtype!r}")
    target = _DTYPES[dtype]
    os.makedirs(out_dir, exist_ok=True)
    tensors = {}
    for k, v in model.state_dict().items():
        if k == "lm_head.weight":
            continue
        t = v.detach().cpu().contiguous()
        tensors[k] = t.to(target) if t.is_floating_point() else t
    meta = {"format": "pt", "architecture": "TensegrityLM", "tensegrity_config": json.dumps(cfg.to_dict()),
            "dtype": dtype, "n_params": str(model.num_parameters())}
    if extra_metadata:
        meta.update({k: str(v) for k, v in extra_metadata.items()})
    weights_path = os.path.join(out_dir, f"{name}.safetensors")
    save_file(tensors, weights_path, metadata=meta)
    config_path = os.path.join(out_dir, f"{name}.config.json")
    with open(config_path, "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)
    return weights_path, config_path


def load_safetensors(weights_path: str, config_path: Optional[str] = None, device: str = "cpu"
                     ) -> Tuple[TensegrityLM, TensegrityConfig]:
    """Rebuild a model from a ``.safetensors`` file (config from ``config_path`` or the file header)."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    if config_path is not None:
        with open(config_path) as fh:
            cfg_dict = json.load(fh)
    else:
        with safe_open(weights_path, framework="pt") as f:
            raw = (f.metadata() or {}).get("tensegrity_config")
        if raw is None:
            raise ValueError(f"{weights_path} has no embedded config; pass config_path=")
        cfg_dict = json.loads(raw)
    cfg = TensegrityConfig.from_dict(cfg_dict, strict=True)
    model = TensegrityLM(cfg)
    state = load_file(weights_path)
    state.pop("lm_head.weight", None)  # tied to token_emb.weight
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise ValueError(f"state mismatch loading {weights_path}: missing={missing[:5]} unexpected={unexpected[:5]}")
    assert model.lm_head.weight is model.token_emb.weight, "weight tying lost"
    return model.to(device), cfg


def quantize_int8_dynamic(model: TensegrityLM) -> Optional[torch.nn.Module]:
    """PyTorch dynamic INT8 quantisation of every nn.Linear (CPU inference). Returns None (with a
    warning) when the installed torch no longer ships eager-mode dynamic quantisation.

    The custom architecture has no GGUF/llama.cpp backend, so this is the portable low-memory path."""
    try:
        from torch.ao.quantization import quantize_dynamic
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"dynamic quantisation unavailable in this torch build ({e})")
        return None
    cpu_model = TensegrityLM(model.cfg)
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return quantize_dynamic(cpu_model, {torch.nn.Linear}, dtype=torch.qint8)


def write_model_card(out_dir: str, cfg: TensegrityConfig, name: str = "model",
                     metrics: Optional[Dict[str, Any]] = None, training: Optional[Dict[str, Any]] = None,
                     dataset: str = "unspecified") -> str:
    """Write an honest ``README.md`` model card next to the exported weights."""
    os.makedirs(out_dir, exist_ok=True)
    m = metrics or {}
    t = training or {}
    lines = [
        f"# {name}", "",
        f"A **{cfg.n_params() / 1e6:.1f}M-parameter byte-level language model** (TensegrityLM). "
        "It reads and writes raw UTF-8 bytes (vocabulary 259: 256 bytes + BOS/EOS/PAD); there is no tokenizer file.", "",
        "## Architecture", "",
        f"- layers {cfg.n_layer}, width {cfg.n_embd}, {cfg.n_head} heads / {cfg.n_kv_head} KV heads (head_dim {cfg.head_dim})",
        f"- context {cfg.block_size} bytes; every {cfg.global_every}rd layer global, the rest sliding-window {cfg.sliding_window}",
        f"- SwiGLU FFN width {cfg.ffn_hidden}; RoPE (base {cfg.rope_base:g}); RMSNorm; learnable residual gains",
        f"- Q/K-norm {cfg.use_qk_norm}, logit soft-cap {cfg.logit_soft_cap}, ByteMix kernel {cfg.byte_mix_kernel or 'off'}"
        f"{' (bilinear)' if cfg.byte_mix_bilinear else ''}, "
        f"x0-mix {cfg.x0_mix}, v0-mix {cfg.v0_mix}, loops {cfg.n_loops}, confidence head {cfg.use_confidence_head}", "",
        "## Training", "",
        f"- dataset: {dataset}",
    ]
    for k, v in t.items():
        lines.append(f"- {k}: {v}")
    if m:
        lines += ["", "## Evaluation (held-out split of the training corpus)", ""]
        for k, v in m.items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    lines += [
        "", "## Usage", "", "```python",
        "from tensegrity import load_safetensors, ByteTokenizer", "import torch",
        f"model, cfg = load_safetensors('{name}.safetensors'); model.eval()",
        "tok = ByteTokenizer()",
        "ids = torch.tensor([[tok.DOC_START] + tok.encode('Once upon a time')])",
        "out = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=40)",
        "print(tok.decode(out[0].tolist()))", "```", "",
        "## Limitations", "",
        "- Small research model trained on a narrow corpus: expect fluent-looking but unreliable text; "
        "it has no safety tuning and should not be used for decisions.",
        "- Byte-level modelling needs ~4x more positions than a subword model for the same text.",
        "- Quality claims should be checked on your own data; the architecture combination is not benchmarked "
        "against other models in this repository.", "",
    ]
    path = os.path.join(out_dir, "README.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path
