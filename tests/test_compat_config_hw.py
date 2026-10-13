"""Backward compatibility with v1 checkpoints, config handling, dtype/optimizer correctness."""

import os
import warnings

import pytest
import torch

from conftest import ASSETS, small_cfg
from tensegrity import (ByteTokenizer, TensegrityConfig, TensegrityLM, load_checkpoint, load_model, pick_dtype)
from tensegrity.checkpoint import capture_rng_state, restore_rng_state, save_checkpoint
from tensegrity.hw import amp_settings, native_bf16
from tensegrity.optim import (SingleDeviceMuonWithAuxAdam, _ns_dtype, configure_optimizer, muon_update,
                              zeropower_via_newtonschulz5)


# ---------------------------------------------------------------- v1 checkpoint compatibility
def test_v1_checkpoint_loads_and_reproduces_original_outputs():
    """A checkpoint written by the ORIGINAL release (Muon optimizer, numpy RNG objects) must load
    into the new code as the same network: identical logits, confidence and sampled text."""
    path = os.path.join(ASSETS, "legacy_v1.pt")
    expected = torch.load(os.path.join(ASSETS, "legacy_v1_expected.pt"), weights_only=True)
    with pytest.warns(UserWarning, match="legacy"):
        ckpt = load_checkpoint(path)
    assert ckpt["format_version"] == 1 and ckpt["step"] == 6
    cfg = TensegrityConfig.from_dict(ckpt["config"])
    assert (cfg.byte_mix_kernel, cfg.x0_mix, cfg.confidence_detach) == (0, False, False)  # legacy behaviour
    model = TensegrityLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])          # strict: identical parameter names/shapes
    model.eval()
    with torch.no_grad():
        logits, _, conf = model(expected["x"])
    torch.testing.assert_close(logits, expected["logits"], atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(conf, expected["conf"], atol=2e-5, rtol=1e-5)
    torch.manual_seed(11)
    gen = model.generate(expected["gen_prompt"], 40, temperature=0.8, top_k=30, stop_on_eos=False)
    assert torch.equal(gen, expected["gen"])


def test_v1_optimizer_state_loads_into_new_muon():
    ckpt = load_checkpoint(os.path.join(ASSETS, "legacy_v1.pt"))
    model = TensegrityLM(TensegrityConfig.from_dict(ckpt["config"]))
    opt = model.configure_optimizer(0.1, 3e-3, (0.9, 0.95), "cpu", use_muon=True, muon_lr=0.02)
    opt.load_state_dict(ckpt["optimizer_state_dict"])         # same group layout as v1
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))


def test_load_model_roundtrip_v2_uses_safe_unpickler(tmp_path):
    model = TensegrityLM(small_cfg(byte_mix_kernel=3, x0_mix=True))
    p = str(tmp_path / "c.pt")
    save_checkpoint(p, model, None, model.cfg, 3, 1.5, rng_state=capture_rng_state(model))
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no legacy-pickle warning: it must load under weights_only=True
        m2, cfg2, ck = load_model(p)
    assert cfg2 == model.cfg and ck["step"] == 3 and ck["optimizer_state_dict"] is None
    for (a, pa), (b, pb) in zip(model.state_dict().items(), m2.state_dict().items()):
        assert a == b and torch.equal(pa, pb)


def test_rng_state_roundtrip():
    import random
    import numpy as np
    m = TensegrityLM(small_cfg(layerdrop_max=0.3))
    m.train().seed_layerdrop(5)
    st = capture_rng_state(m)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)), m._layerdrop_plan())
    restore_rng_state(st, m)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)), m._layerdrop_plan())
    assert a == b


def test_load_checkpoint_rejects_garbage(tmp_path):
    p = str(tmp_path / "x.pt")
    torch.save({"hello": 1}, p)
    with pytest.raises(ValueError, match="missing"):
        load_checkpoint(p)


# ------------------------------------------------------------------------------------ config
def test_config_from_dict_old_and_unknown_keys():
    d = small_cfg().to_dict()
    for k in ("norm_eps", "confidence_detach", "byte_mix_kernel", "x0_mix"):
        d.pop(k)                                   # a v1 dict has none of the v2 fields
    cfg = TensegrityConfig.from_dict(d)
    assert (cfg.norm_eps, cfg.confidence_detach, cfg.byte_mix_kernel, cfg.x0_mix) == (1e-6, False, 0, False)
    d["future_field"] = 1
    with pytest.warns(UserWarning, match="future_field"):
        TensegrityConfig.from_dict(d)
    with pytest.raises(TypeError):                 # typos are NOT silently ignored any more
        TensegrityConfig(n_embed=64)


@pytest.mark.parametrize("bad", [dict(n_embd=50), dict(n_head=5, n_kv_head=2), dict(n_embd=36, n_head=4, n_kv_head=2),
                                 dict(n_layer=0), dict(global_every=0), dict(layerdrop_max=1.0),
                                 dict(sliding_window=0), dict(byte_mix_kernel=1), dict(norm_eps=0.0)])
def test_config_validation(bad):
    with pytest.raises((ValueError, ZeroDivisionError)):
        small_cfg(**bad)


def test_config_replace_recomputes_gain_init():
    c = small_cfg(n_layer=4)
    assert c.replace(n_layer=8).gain_init == pytest.approx(1 / (2 * 8) ** 0.5)
    assert c.replace(dropout=0.1).gain_init == c.gain_init


def test_flops_per_token_sane():
    c = small_cfg()
    f = c.flops_per_token(48, training=True)
    n = c.n_params() - c.vocab_size * c.n_embd
    assert 6 * n < f < 6 * (n + c.vocab_size * c.n_embd) * 1.6
    assert c.flops_per_token(48, training=False) == pytest.approx(f / 3)


def test_tokenizer_roundtrip_and_specials():
    t = ByteTokenizer()
    s = "héllo wörld ✓ 日本語"
    assert t.decode(t.encode(s)) == s
    ids = t.encode("ab", add_bos=True, add_eos=True)
    assert ids == [256, 97, 98, 257]
    assert t.decode(ids) == "ab" and t.decode(ids, skip_special_tokens=False) == "<|bos|>ab<|eos|>"
    with pytest.raises(ValueError):
        t.decode([1, 999], strict=True)
    raw = bytes(range(256))
    assert t.decode_bytes(t.encode_bytes(raw)) == raw


# ---------------------------------------------------------------------------- dtype / hardware
class _FakeCuda:
    """Monkeypatch torch.cuda to look like a given GPU generation."""

    def __init__(self, monkeypatch, major, minor=0, name="X"):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (major, minor))
        # the emulation-inclusive check the v1 code used: True even on a T4
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)


@pytest.mark.parametrize("major,expected", [(7, torch.float16), (8, torch.bfloat16), (9, torch.bfloat16)])
def test_pick_dtype_is_capability_aware(monkeypatch, major, expected):
    """Regression: torch.cuda.is_bf16_supported() is True on a T4 (sm75) through emulation, so
    the v1 code trained in slow emulated bf16 on Colab's free GPU."""
    _FakeCuda(monkeypatch, major)
    assert pick_dtype("auto", "cuda") == expected
    assert native_bf16("cuda") == (major >= 8)


def test_pick_dtype_cpu_and_explicit(monkeypatch):
    assert pick_dtype("auto", "cpu") == torch.float32
    assert pick_dtype("float16", "cpu") == torch.float16
    _FakeCuda(monkeypatch, 7)
    with pytest.warns(UserWarning, match="emulated"):
        assert pick_dtype("bfloat16", "cuda") == torch.bfloat16


def test_amp_settings(monkeypatch):
    s = amp_settings(torch.float16, "cuda")
    assert s["use_amp"] and s["use_scaler"] and s["amp_dtype"] == torch.float16
    s = amp_settings(torch.bfloat16, "cuda")
    assert s["use_amp"] and not s["use_scaler"]
    assert not amp_settings(torch.float32, "cpu")["use_amp"]


def test_newton_schulz_dtype_follows_hardware(monkeypatch):
    assert _ns_dtype(torch.device("cpu")) == torch.float32
    _FakeCuda(monkeypatch, 7)
    assert _ns_dtype(torch.device("cuda")) == torch.float16     # not emulated bf16 on a T4
    _FakeCuda(monkeypatch, 8)
    assert _ns_dtype(torch.device("cuda")) == torch.bfloat16


# --------------------------------------------------------------------------------- optimizer
def test_newton_schulz_orthogonalises():
    """Singular values of well-conditioned inputs are pushed to ~[0.5, 1.5] (Muon's documented
    behaviour). A square Gaussian is deliberately NOT used: its smallest singular value is ~1/n and
    five iterations cannot lift it -- that is inherent to Muon, not a defect."""
    torch.manual_seed(0)
    for shape in [(32, 96), (96, 32), (24, 200)]:               # tall/wide Gaussians are well conditioned
        s = torch.linalg.svdvals(zeropower_via_newtonschulz5(torch.randn(*shape)))
        assert 0.4 < float(s.min()) and float(s.max()) < 1.6, (shape, float(s.min()), float(s.max()))
    n = 48                                                      # square, controlled spectrum in [0.2, 1]
    u, _ = torch.linalg.qr(torch.randn(n, n))
    v, _ = torch.linalg.qr(torch.randn(n, n))
    g = u @ torch.diag(torch.linspace(0.2, 1.0, n)) @ v.T
    s = torch.linalg.svdvals(zeropower_via_newtonschulz5(g))
    assert 0.4 < float(s.min()) and float(s.max()) < 1.6
    tiny = torch.randn(16, 48) * 1e-9                           # fp32 norm: tiny grads must not collapse
    out = zeropower_via_newtonschulz5(tiny)
    assert torch.isfinite(out).all() and out.abs().max() > 0.05


def test_muon_update_does_not_mutate_grad():
    g = torch.randn(8, 8)
    g0 = g.clone()
    muon_update(g, torch.zeros(8, 8))
    assert torch.equal(g, g0)


@pytest.mark.parametrize("kw", [{}, {"byte_mix_kernel": 4, "x0_mix": True}])
def test_optimizer_partition_covers_every_param_once(kw):
    m = TensegrityLM(small_cfg(**kw))
    all_ids = {id(p) for p in m.parameters()}
    for use_muon in (False, True):
        opt = configure_optimizer(m, 0.1, 1e-3, (0.9, 0.95), "cpu", use_muon=use_muon)
        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        assert len(ids) == len(set(ids)) and set(ids) == all_ids
    opt = configure_optimizer(m, 0.1, 1e-3, (0.9, 0.95), "cpu", use_muon=True)
    muon = [g for g in opt.param_groups if g["use_muon"]][0]
    muon_ids = {id(p) for p in muon["params"]}
    block_mats = {id(p) for p in m.blocks.parameters() if p.dim() >= 2}
    assert muon_ids == block_mats                           # exactly the hidden matrices
    assert id(m.token_emb.weight) not in muon_ids


def test_decay_rule_excludes_norms_scalars_and_byte_mix():
    m = TensegrityLM(small_cfg(byte_mix_kernel=4, x0_mix=True))
    opt = configure_optimizer(m, 0.1, 1e-3, (0.9, 0.95), "cpu")
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    assert id(m.byte_mix.weight) not in decayed             # 2-D but flagged no-decay
    assert id(m.blocks[0].attn_gain) not in decayed and id(m.blocks[0].x0_gain) not in decayed
    assert id(m.blocks[0].ln1.weight) not in decayed
    assert id(m.blocks[0].attn.q_proj.weight) in decayed and id(m.token_emb.weight) in decayed


def test_muon_optimizer_learns_and_state_roundtrips():
    torch.manual_seed(0)
    m = TensegrityLM(small_cfg(layerdrop_max=0.0))
    opt = configure_optimizer(m, 0.01, 3e-3, (0.9, 0.95), "cpu", use_muon=True, muon_lr=0.02)
    assert isinstance(opt, SingleDeviceMuonWithAuxAdam)
    x = torch.randint(0, 64, (8, 24))
    y = torch.roll(x, -1, dims=1)
    first = None
    for _ in range(40):
        _, l, _ = m(x, y)
        l["total_loss"].backward()
        opt.step()
        opt.zero_grad()
        first = first if first is not None else l["ce_loss"].item()
    assert l["ce_loss"].item() < first - 0.3
    m2 = TensegrityLM(small_cfg())
    o2 = configure_optimizer(m2, 0.01, 3e-3, (0.9, 0.95), "cpu", use_muon=True)
    o2.load_state_dict(opt.state_dict())
