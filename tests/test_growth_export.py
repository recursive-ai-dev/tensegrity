"""Growth (function preservation + trainability), EWC, export/import."""

import json
import os

import pytest
import torch

from conftest import randomized, small_cfg
from tensegrity import (TensegrityLM, check_function_preserved, check_growth_trainable,
                        compute_fisher_diagonal, ewc_penalty, expand_reference_state, export_safetensors, grow_model,
                        load_safetensors, make_ewc_loss, quantize_int8_dynamic, write_model_card)
from tensegrity.growth import derive_target

AXES = {
    "depth": dict(new_n_layer=6),
    "ffn": dict(ffn_growth_mult=1.5),
    "width": dict(new_n_embd=96),
    "context": dict(new_block_size=96),
    "everything": dict(new_n_layer=6, ffn_growth_mult=1.3, new_n_embd=96, new_block_size=96),
}


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("flags", [
    dict(),
    dict(embed_scale=False),
    dict(bias=True, byte_mix_kernel=4, x0_mix=True),
    dict(logit_soft_cap=30.0, use_qk_norm=False, use_confidence_head=False),
])
def test_growth_preserves_function_and_is_trainable(axis, flags):
    cfg = small_cfg(**flags)
    old = randomized(TensegrityLM(cfg), 0.05, seed=1)
    new, ncfg = grow_model(old, cfg, **AXES[axis])
    d = check_function_preserved(old, new, cfg, "cpu", atol=2e-5, verbose=False)
    assert d < 2e-5
    check_growth_trainable(new, cfg, verbose=False)
    assert ncfg.n_params() == new.num_parameters()
    assert (new.num_parameters() > cfg.n_params()) == (axis != "context")      # block_size is parameter-free


def test_grown_model_function_survives_save_and_reload(tmp_path):
    """The norm_eps that makes width growth exact lives in the config, so it survives a round trip."""
    cfg = small_cfg(embed_scale=False)
    old = randomized(TensegrityLM(cfg))
    new, ncfg = grow_model(old, cfg, new_n_embd=96)
    assert ncfg.norm_eps != cfg.norm_eps            # embed_scale off -> eps must be rescaled by D0/D1
    w, c = export_safetensors(new, ncfg, str(tmp_path))
    reloaded, rcfg = load_safetensors(w, c)
    check_function_preserved(old, reloaded, cfg, "cpu", atol=2e-5, verbose=False)


def test_norm_eps_rule_by_embed_scale():
    on = grow_model(randomized(TensegrityLM(small_cfg())), new_n_embd=96)[1]
    off = grow_model(randomized(TensegrityLM(small_cfg(embed_scale=False))), new_n_embd=96)[1]
    assert on.norm_eps == pytest.approx(1e-6)                    # sqrt(D1) embedding factor cancels the dilution
    assert off.norm_eps == pytest.approx(1e-6 * 48 / 96)


def test_check_growth_trainable_detects_the_v1_dead_capacity_bug():
    """Re-create what v1 did on width growth -- zero BOTH sides of every new product -- and make
    sure the verifier flags it. (The v1 code passed its own function-preservation check here.)"""
    cfg = small_cfg()
    old = randomized(TensegrityLM(cfg))
    new, ncfg = grow_model(old, cfg, new_n_embd=96)
    D0, hd = cfg.n_embd, cfg.head_dim
    q0, kv0, h0 = cfg.n_head * hd, cfg.n_kv_head * hd, cfg.ffn_hidden
    with torch.no_grad():
        for b in new.blocks:
            b.attn.q_proj.weight[:, D0:].zero_(); b.attn.q_proj.weight[q0:].zero_()
            b.attn.k_proj.weight[:, D0:].zero_(); b.attn.k_proj.weight[kv0:].zero_()
            b.attn.v_proj.weight[:, D0:].zero_(); b.attn.v_proj.weight[kv0:].zero_()
            b.ffn.w_gate.weight[:, D0:].zero_(); b.ffn.w_gate.weight[h0:].zero_()
            b.ffn.w_up.weight[:, D0:].zero_(); b.ffn.w_up.weight[h0:].zero_()
    check_function_preserved(old, new, cfg, "cpu", atol=2e-5, verbose=False)   # still "preserves the function"
    with pytest.raises(AssertionError, match="can never learn"):
        check_growth_trainable(new, cfg, verbose=False)


def test_new_capacity_actually_moves_under_training():
    """Beyond 'has gradient': after a few optimizer steps the new regions change and the loss drops."""
    torch.manual_seed(0)
    cfg = small_cfg()
    old = randomized(TensegrityLM(cfg))
    new, ncfg = grow_model(old, cfg, new_n_layer=5, new_n_embd=96, ffn_growth_mult=1.4)
    x = torch.randint(0, 64, (8, 24))
    y = torch.roll(x, -1, 1)
    opt = new.configure_optimizer(0.0, 3e-3, (0.9, 0.95), "cpu")
    new.train()
    D0 = cfg.n_embd
    before = new.blocks[0].attn.o_proj.weight[D0:, :].detach().clone()
    gains0 = new.blocks[4].attn_gain.detach().clone()
    first = None
    for _ in range(25):
        _, l, _ = new(x, y)
        l["total_loss"].backward()
        opt.step()
        opt.zero_grad()
        first = first if first is not None else l["ce_loss"].item()
    assert l["ce_loss"].item() < first - 0.3
    assert (new.blocks[0].attn.o_proj.weight[D0:, :] - before).abs().max() > 1e-4       # new channels were written to
    assert (new.blocks[4].attn_gain - gains0).abs() > 1e-4                              # appended block woke up


def test_incompatible_growth_is_rejected_with_reasons():
    cfg = small_cfg()
    old = TensegrityLM(cfg)
    with pytest.raises(ValueError, match="head_dim"):
        grow_model(old, cfg, target=cfg.replace(n_embd=96, n_head=3, n_kv_head=1))   # head_dim 16 -> 32
    with pytest.raises(ValueError, match="cannot shrink"):
        grow_model(old, cfg, target=cfg.replace(n_layer=2))
    with pytest.raises(ValueError, match="changes the function"):
        grow_model(old, cfg, target=cfg.replace(sliding_window=24))
    with pytest.raises(ValueError, match="changes the function"):
        grow_model(old, cfg, target=cfg.replace(byte_mix_kernel=4))
    with pytest.raises(ValueError, match="GQA ratio"):
        grow_model(old, cfg, target=cfg.replace(n_embd=96, n_head=6, n_kv_head=6))
    with pytest.raises(ValueError, match="multiple of head_dim"):
        derive_target(cfg, new_n_embd=100)
    with pytest.raises(ValueError, match="only"):
        derive_target(cfg, new_n_embd=96, new_n_head=3)
    with pytest.raises(ValueError):
        derive_target(cfg, ffn_growth_mult=0.5)


def test_training_hyperparameters_may_change_during_growth():
    cfg = small_cfg()
    old = randomized(TensegrityLM(cfg))
    new, ncfg = grow_model(old, cfg, target=cfg.replace(n_layer=6, dropout=0.1, layerdrop_max=0.2, homeo_weight=0.0,
                                                        grad_checkpoint=True, z_loss_weight=1e-3))
    assert ncfg.dropout == 0.1 and ncfg.layerdrop_max == 0.2
    check_function_preserved(old, new, cfg, "cpu", atol=2e-5, verbose=False)


def test_growth_is_reproducible_and_leaves_old_model_untouched():
    cfg = small_cfg()
    old = randomized(TensegrityLM(cfg))
    snap = {k: v.clone() for k, v in old.state_dict().items()}
    a, _ = grow_model(old, cfg, new_n_embd=96, new_n_layer=6, seed=11)
    b, _ = grow_model(old, cfg, new_n_embd=96, new_n_layer=6, seed=11)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert all(torch.equal(snap[k], v) for k, v in old.state_dict().items())


def test_grown_model_generates_identically_to_parent():
    cfg = small_cfg()
    old = randomized(TensegrityLM(cfg)).eval()
    new, _ = grow_model(old, cfg, new_n_layer=6, new_n_embd=96)
    new.eval()
    p = torch.randint(0, 256, (1, 6))
    a = old.generate(p, 20, temperature=0.0, stop_on_eos=False)
    b = new.generate(p, 20, temperature=0.0, stop_on_eos=False)
    assert torch.equal(a, b)


# ------------------------------------------------------------------------------------- EWC
def test_fisher_penalty_and_expansion(corpus_dir):
    cfg = small_cfg()
    m = randomized(TensegrityLM(cfg))
    fisher, snap = compute_fisher_diagonal(m, corpus_dir, "cpu", n_sequences=6, seq_len=32)
    assert set(fisher) == set(snap) == {n for n, _ in m.named_parameters()}
    assert all((f >= 0).all() for f in fisher.values()) and any(f.sum() > 0 for f in fisher.values())
    assert float(ewc_penalty(m, fisher, snap).detach()) == 0.0               # at the anchor
    with torch.no_grad():
        m.blocks[0].attn.q_proj.weight.add_(0.1)
    pen = float(ewc_penalty(m, fisher, snap).detach())
    manual = float((fisher["blocks.0.attn.q_proj.weight"] * 0.1 ** 2).sum())
    assert pen == pytest.approx(manual, rel=1e-4) and pen > 0
    assert float(make_ewc_loss(fisher, snap, 5.0)(m).detach()) == pytest.approx(5.0 * pen, rel=1e-6)
    # the Fisher uses per-sequence gradients, so it is invariant to how many sequences share a batch
    f2, _ = compute_fisher_diagonal(m, corpus_dir, "cpu", n_sequences=6, seq_len=32)
    assert not any(torch.isnan(f).any() for f in f2.values())
    # expansion onto a grown model: old region keeps its Fisher, new region is unconstrained
    grown, gcfg = grow_model(m, cfg, new_n_embd=96, new_n_layer=6)
    nf, no = expand_reference_state(fisher, snap, grown)
    name = "blocks.0.ffn.w_gate.weight"
    old_shape = fisher[name].shape
    assert torch.equal(nf[name][: old_shape[0], : old_shape[1]], fisher[name])
    assert nf[name][old_shape[0]:].sum() == 0 and nf[name][:, old_shape[1]:].sum() == 0
    assert not any(k.startswith("blocks.4") or k.startswith("blocks.5") for k in nf)     # new blocks: absent
    assert expand_reference_state(None, None, grown) == ({}, {})
    shapes = {n: p.shape for n, p in grown.named_parameters()}
    assert all(nf[n].shape == shapes[n] == no[n].shape for n in nf)           # re-mapped onto the grown shapes


# ----------------------------------------------------------------------------------- export
@pytest.mark.parametrize("flags", [{}, {"byte_mix_kernel": 4, "x0_mix": True, "bias": True}])
def test_safetensors_roundtrip_is_exact(tmp_path, flags):
    cfg = small_cfg(**flags)
    m = randomized(TensegrityLM(cfg)).eval()
    w, c = export_safetensors(m, cfg, str(tmp_path), "m")
    assert os.path.exists(w) and json.load(open(c)) == cfg.to_dict()
    m2, cfg2 = load_safetensors(w, c)
    m3, _ = load_safetensors(w)                       # config recovered from the safetensors header alone
    assert cfg2 == cfg and m2.lm_head.weight is m2.token_emb.weight
    x = torch.randint(0, 256, (2, 20))
    m2.eval(); m3.eval()
    assert torch.equal(m(x)[0], m2(x)[0]) and torch.equal(m(x)[0], m3(x)[0])
    from safetensors import safe_open
    with safe_open(w, framework="pt") as f:
        assert "lm_head.weight" not in f.keys() and "token_emb.weight" in f.keys()


def test_safetensors_fp16_export_is_close_and_smaller(tmp_path):
    m = randomized(TensegrityLM(small_cfg())).eval()
    w32, _ = export_safetensors(m, None, str(tmp_path), "a", "float32")
    w16, _ = export_safetensors(m, None, str(tmp_path), "b", "float16")
    assert os.path.getsize(w16) < 0.6 * os.path.getsize(w32)
    m16, _ = load_safetensors(w16)
    x = torch.randint(0, 256, (2, 20))
    assert (m16.eval()(x)[0] - m(x)[0]).abs().max() < 5e-2
    with pytest.raises(ValueError):
        export_safetensors(m, None, str(tmp_path), "c", "int4")


def test_loads_v1_style_safetensors_with_both_tied_keys(tmp_path):
    from safetensors.torch import save_file
    m = randomized(TensegrityLM(small_cfg())).eval()
    sd = {k: v.detach().clone().contiguous() for k, v in m.state_dict().items()}      # v1 wrote lm_head too
    save_file(sd, str(tmp_path / "v1.safetensors"), metadata={"format": "pt"})
    json.dump(m.cfg.to_dict(), open(tmp_path / "v1.config.json", "w"))
    m2, _ = load_safetensors(str(tmp_path / "v1.safetensors"), str(tmp_path / "v1.config.json"))
    x = torch.randint(0, 256, (1, 12))
    assert torch.equal(m2.eval()(x)[0], m(x)[0])
    with pytest.raises(ValueError, match="no embedded config"):
        load_safetensors(str(tmp_path / "v1.safetensors"))


def test_load_safetensors_detects_missing_tensors(tmp_path):
    from safetensors.torch import save_file
    m = TensegrityLM(small_cfg())
    sd = {k: v.detach().clone().contiguous() for k, v in m.state_dict().items() if k != "lm_head.weight"}
    sd.pop("ln_f.weight")
    save_file(sd, str(tmp_path / "bad.safetensors"))
    json.dump(m.cfg.to_dict(), open(tmp_path / "bad.json", "w"))
    with pytest.raises(ValueError, match="mismatch"):
        load_safetensors(str(tmp_path / "bad.safetensors"), str(tmp_path / "bad.json"))


def test_model_card_and_int8(tmp_path):
    cfg = small_cfg()
    p = write_model_card(str(tmp_path), cfg, "demo", metrics={"val_bpb": 1.2345, "n": 3},
                         training={"steps": 100}, dataset="toy")
    text = open(p).read()
    assert "demo" in text and "val_bpb: 1.2345" in text and "Limitations" in text and "steps: 100" in text
    q = quantize_int8_dynamic(randomized(TensegrityLM(cfg)))
    if q is None:
        pytest.skip("dynamic quantisation not available in this torch build")
    out = q.generate(torch.randint(0, 256, (1, 4)), 8, temperature=0.0, stop_on_eos=False)
    assert out.shape == (1, 12)
