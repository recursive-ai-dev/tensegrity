"""Attention correctness, KV-cache exactness, sampling and LayerDrop behaviour."""

import pytest
import torch
import torch.nn.functional as F

from conftest import randomized, small_cfg
from tensegrity import ByteTokenizer, TensegrityLM
from tensegrity.layers import local_causal_attention, native_gqa_ok


# ----------------------------------------------------------------------------- attention
def _dense_local(q, k, v, window, n_rep):
    """Reference: dense (T, T) masked attention, mask j <= i and j > i - window."""
    T = q.shape[2]
    k, v = k.repeat_interleave(n_rep, 1), v.repeat_interleave(n_rep, 1)
    i = torch.arange(T).view(-1, 1)
    j = torch.arange(T).view(1, -1)
    mask = (j <= i) & (j > i - window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@pytest.mark.parametrize("T,W", [(1, 8), (5, 8), (8, 8), (9, 8), (16, 8), (23, 8), (64, 16), (100, 32)])
@pytest.mark.parametrize("n_rep", [1, 2])
def test_local_attention_equals_dense_masked(T, W, n_rep):
    torch.manual_seed(T * 31 + W)
    B, Hkv, D = 3, 2, 8
    q = torch.randn(B, Hkv * n_rep, T, D)
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)
    got = local_causal_attention(q, k, v, W, n_rep)
    want = _dense_local(q, k, v, W, n_rep)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=2e-6, rtol=1e-5)


def test_local_attention_gradients_match_dense():
    torch.manual_seed(0)
    B, H, T, D, W = 2, 4, 37, 8, 8
    base = [torch.randn(B, H, T, D), torch.randn(B, 2, T, D), torch.randn(B, 2, T, D)]
    a = [t.clone().requires_grad_() for t in base]
    b = [t.clone().requires_grad_() for t in base]
    w = torch.randn(B, H, T, D)
    (local_causal_attention(*a, W, 2) * w).sum().backward()
    (_dense_local(*b, W, 2) * w).sum().backward()
    for x, y in zip(a, b):
        torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=1e-4)


def test_local_attention_window_is_exact():
    """The output at position i depends on keys in (i-W, i] and on nothing else."""
    torch.manual_seed(1)
    B, H, T, D, W = 1, 2, 40, 8, 8
    q, k, v = torch.randn(B, H, T, D), torch.randn(B, H, T, D), torch.randn(B, H, T, D)
    base = local_causal_attention(q, k, v, W, 1)
    i = 25
    k2, v2 = k.clone(), v.clone()
    k2[:, :, : i - W + 1] = torch.randn_like(k2[:, :, : i - W + 1])   # strictly before the window
    v2[:, :, : i - W + 1] = torch.randn_like(v2[:, :, : i - W + 1])
    k2[:, :, i + 1:] = torch.randn_like(k2[:, :, i + 1:])             # strictly in the future
    v2[:, :, i + 1:] = torch.randn_like(v2[:, :, i + 1:])
    out = local_causal_attention(q, k2, v2, W, 1)
    torch.testing.assert_close(out[:, :, i], base[:, :, i], atol=1e-6, rtol=1e-5)
    k3 = k.clone()
    k3[:, :, i] = torch.randn_like(k3[:, :, i])                       # inside the window -> must change
    assert not torch.allclose(local_causal_attention(q, k3, v, W, 1)[:, :, i], base[:, :, i], atol=1e-4)


def test_native_gqa_equals_expanded():
    q = torch.randn(2, 8, 16, 8)
    k, v = torch.randn(2, 2, 16, 8), torch.randn(2, 2, 16, 8)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    b = F.scaled_dot_product_attention(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), is_causal=True)
    torch.testing.assert_close(a, b)


def test_native_gqa_gated_by_hardware():
    """Never route masked/T4-class paths through enable_gqa (it would force the O(T^2) math kernel)."""
    assert native_gqa_ok(torch.zeros(1)) is False  # CPU tensor


def test_model_is_causal_everywhere():
    torch.manual_seed(2)
    m = randomized(TensegrityLM(small_cfg(byte_mix_kernel=4, x0_mix=True))).eval()
    x = torch.randint(0, 256, (2, 40))
    base, _, _ = m(x)
    for t in (0, 7, 19, 33):
        x2 = x.clone()
        x2[:, t + 1:] = torch.randint(0, 256, x2[:, t + 1:].shape)  # change only the future
        out, _, _ = m(x2)
        torch.testing.assert_close(out[:, : t + 1], base[:, : t + 1], atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------------------ KV cache
VARIANTS = [
    {},
    {"byte_mix_kernel": 4},
    {"x0_mix": True},
    {"byte_mix_kernel": 3, "x0_mix": True, "use_qk_norm": False, "bias": True},
    {"logit_soft_cap": 30.0, "embed_scale": False},
]


@pytest.mark.parametrize("variant", VARIANTS)
def test_prefill_matches_forward(variant):
    torch.manual_seed(3)
    m = randomized(TensegrityLM(small_cfg(**variant))).eval()
    x = torch.randint(0, 256, (3, 31))
    a, _, ca = m(x)
    b, _, cb = m.forward_incremental(x)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ca, cb, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("variant", VARIANTS)
def test_cached_decode_matches_full_forward(variant):
    """Token-by-token cached decoding == one full forward, for every position."""
    torch.manual_seed(4)
    cfg = small_cfg(**variant)
    m = randomized(TensegrityLM(cfg)).eval()
    x = torch.randint(0, 256, (2, cfg.block_size))
    full, _, _ = m(x)
    K = cfg.byte_mix_kernel
    n_pre = 5
    logits, past, _ = m.forward_incremental(x[:, :n_pre])
    steps = [logits]
    for t in range(n_pre, cfg.block_size):
        ctx = x[:, max(0, t - K + 1): t] if K else None
        logits, past, _ = m.forward_incremental(x[:, t: t + 1], past, t, ctx)
        steps.append(logits)
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=2e-5, rtol=1e-4)
    # local layers must only keep the trailing window
    for blk, (k, v) in zip(m.blocks, past):
        assert k.shape[2] == (cfg.block_size if blk.attn.is_global else cfg.sliding_window)


@pytest.mark.parametrize("variant", VARIANTS[:3])
@pytest.mark.parametrize("sampling", [dict(temperature=0.0), dict(temperature=0.9, top_k=12),
                                      dict(temperature=0.8, top_p=0.9, min_p=0.05, use_confidence_temp=True)])
def test_generate_cache_equals_no_cache_past_block_size(variant, sampling):
    cfg = small_cfg(**variant)
    m = randomized(TensegrityLM(cfg)).eval()
    prompt = torch.randint(0, 256, (2, 5))
    n_new = cfg.block_size + 30   # forces several window rebuilds
    torch.manual_seed(9)
    a = m.generate(prompt, n_new, stop_on_eos=False, use_cache=True, **sampling)
    torch.manual_seed(9)
    b = m.generate(prompt, n_new, stop_on_eos=False, use_cache=False, **sampling)
    assert a.shape == b.shape == (2, 5 + n_new)
    assert torch.equal(a, b)


def test_rebuild_stride_changes_nothing_before_overflow_and_stays_valid_after():
    cfg = small_cfg()
    m = randomized(TensegrityLM(cfg)).eval()
    prompt = torch.randint(0, 256, (1, 4))
    short = cfg.block_size - 8
    a = m.generate(prompt, short, temperature=0.0, stop_on_eos=False, rebuild_stride=1)
    b = m.generate(prompt, short, temperature=0.0, stop_on_eos=False, rebuild_stride=16)
    assert torch.equal(a, b)  # no overflow -> stride is irrelevant
    c = m.generate(prompt, cfg.block_size * 3, temperature=0.0, stop_on_eos=False, rebuild_stride=16)
    assert c.shape[1] == 4 + cfg.block_size * 3 and int(c.max()) < cfg.vocab_size
    with pytest.raises(ValueError):
        m.generate(prompt, 3, rebuild_stride=0)


def test_generate_restores_training_mode_and_validates_args():
    m = TensegrityLM(small_cfg())
    m.train()
    m.generate(torch.zeros(1, 2, dtype=torch.long), 3)
    assert m.training
    with pytest.raises(ValueError):
        m.generate(torch.zeros(1, 2, dtype=torch.long), 3, temperature=-1)
    with pytest.raises(ValueError):
        m.generate(torch.zeros(1, 2, dtype=torch.long), 3, top_k=0)


def _force_eos_schedule(model, eos_at):
    """Wrap forward_incremental so row r emits EOS from call number eos_at[r] on (None = never)."""
    orig = model.forward_incremental
    calls = {"n": 0}

    def wrapped(idx, past=None, pos=0, ctx=None):
        logits, past, conf = orig(idx, past, pos, ctx)
        logits = logits.clone()
        for r, at in enumerate(eos_at):
            if at is not None and calls["n"] >= at:
                logits[r, -1, ByteTokenizer.EOS] = 1e4
        calls["n"] += 1
        return logits, past, conf
    model.forward_incremental = wrapped


def test_eos_freezes_finished_rows_while_others_continue():
    m = TensegrityLM(small_cfg()).eval()
    _force_eos_schedule(m, [2, 5, None])
    out = m.generate(torch.randint(0, 255, (3, 4)), 20, temperature=0.0, stop_on_eos=True)
    gen = out[:, 4:]
    assert out.shape[1] == 4 + 20                                   # row 2 never finishes -> runs to the limit
    assert gen[0, 2] == ByteTokenizer.EOS and (gen[0, 2:] == ByteTokenizer.EOS).all()   # frozen after EOS
    assert gen[1, 5] == ByteTokenizer.EOS and (gen[1, 5:] == ByteTokenizer.EOS).all()
    assert (gen[1, :5] != ByteTokenizer.EOS).all() and (gen[2] != ByteTokenizer.EOS).all()


def test_generation_stops_when_every_row_finished():
    m = TensegrityLM(small_cfg()).eval()
    _force_eos_schedule(m, [1, 3])
    out = m.generate(torch.randint(0, 255, (2, 4)), 50, temperature=0.0, stop_on_eos=True)
    assert out.shape[1] == 4 + 4                                    # stops right after the last row emits EOS
    assert out[0, 5] == ByteTokenizer.EOS and out[1, 7] == ByteTokenizer.EOS


def test_stream_matches_generate():
    m = randomized(TensegrityLM(small_cfg(use_confidence_head=True))).eval()
    prompt = torch.randint(0, 256, (1, 4))
    torch.manual_seed(5)
    full, conf = m.generate(prompt, 25, temperature=0.9, top_k=20, stop_on_eos=False, return_confidence=True)
    torch.manual_seed(5)
    streamed = list(m.stream(prompt, 25, temperature=0.9, top_k=20, stop_on_eos=False))
    assert [t for t, _ in streamed] == full[0, 4:].tolist()
    assert all(0.0 <= c <= 1.0 for _, c in streamed)
    torch.testing.assert_close(torch.tensor([c for _, c in streamed]), conf[0], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError):
        list(m.stream(torch.zeros(2, 3, dtype=torch.long), 3))
    with pytest.raises(TypeError):
        list(m.stream(prompt, 3, bogus=1))


def test_sampling_filters():
    logits = torch.tensor([[5.0, 4.0, 0.0, -2.0, -9.0]])
    torch.manual_seed(0)
    draws = torch.cat([TensegrityLM._sample_next(logits, 1.0, 2, None, None, None, .5, 1.5) for _ in range(200)])
    assert set(draws.flatten().tolist()) <= {0, 1}                       # top-k=2
    draws = torch.cat([TensegrityLM._sample_next(logits, 1.0, None, 0.5, None, None, .5, 1.5) for _ in range(200)])
    assert set(draws.flatten().tolist()) == {0}                          # nucleus: first token already > 0.5
    draws = torch.cat([TensegrityLM._sample_next(logits, 1.0, None, None, 0.5, None, .5, 1.5) for _ in range(200)])
    assert set(draws.flatten().tolist()) <= {0, 1}                       # min_p: p>=0.5*pmax keeps tokens 0,1 only
    assert TensegrityLM._sample_next(logits, 0.0, None, None, None, torch.tensor([0.1]), .5, 1.5).item() == 0
    nanrow = torch.full((1, 5), float("nan"))
    assert TensegrityLM._sample_next(nanrow, 1.0, None, None, None, None, .5, 1.5).shape == (1, 1)


def test_confidence_temperature_direction():
    """Low confidence -> hotter (flatter), high confidence -> cooler (peakier)."""
    logits = torch.tensor([[2.0, 1.0, 0.0]]).repeat(4000, 1)
    lo = TensegrityLM._sample_next(logits, 1.0, None, None, None, torch.full((4000,), 0.0), 0.5, 1.5)
    hi = TensegrityLM._sample_next(logits, 1.0, None, None, None, torch.full((4000,), 1.0), 0.5, 1.5)
    assert (hi == 0).float().mean() > (lo == 0).float().mean() + 0.1


# ------------------------------------------------------------------------------ training
def test_forward_losses_and_confidence_detach():
    torch.manual_seed(6)
    cfg = small_cfg(z_loss_weight=1e-3, confidence_detach=True)
    m = TensegrityLM(cfg).train()
    x = torch.randint(0, 256, (4, 32))
    y = torch.randint(0, 256, (4, 32))
    _, losses, conf = m(x, y)
    assert conf is None
    assert {"ce_loss", "homeo_loss", "z_loss", "confidence_loss", "total_loss"} <= set(losses)
    expected = (losses["ce_loss"] + cfg.homeo_weight * losses["homeo_loss"] + losses["z_loss"]
                + cfg.confidence_weight * losses["confidence_loss"])
    torch.testing.assert_close(losses["total_loss"], expected)
    # detached head: its loss must not send gradient into the trunk
    m.zero_grad()
    losses["confidence_loss"].backward()
    assert m.blocks[0].attn.q_proj.weight.grad is None or m.blocks[0].attn.q_proj.weight.grad.abs().max() == 0
    assert m.confidence_head.weight.grad.abs().max() > 0
    # undetached (legacy) head does reach the trunk
    m2 = TensegrityLM(small_cfg(confidence_detach=False)).train()
    _, l2, _ = m2(x, y)
    l2["confidence_loss"].backward()
    assert m2.blocks[0].attn.q_proj.weight.grad.abs().max() > 0


def test_ignore_index_and_length_check():
    m = TensegrityLM(small_cfg()).train()
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    y2 = y.clone()
    y2[:, 8:] = -1
    _, a, _ = m(x, y2)
    _, b, _ = m(x[:, :], y)
    assert a["ce_loss"].item() != b["ce_loss"].item()
    with pytest.raises(ValueError):
        m(torch.zeros(1, 49, dtype=torch.long))


def test_layerdrop_deterministic_and_synchronised():
    cfg = small_cfg(layerdrop_max=0.6, n_layer=6)
    a, b = TensegrityLM(cfg).train(), TensegrityLM(cfg).train()
    a.seed_layerdrop(123)
    b.seed_layerdrop(123)
    plans_a = [a._layerdrop_plan() for _ in range(20)]
    plans_b = [b._layerdrop_plan() for _ in range(20)]
    assert plans_a == plans_b                                        # same seed -> same skips on every rank
    assert plans_a[0][0] is False                                    # layer 0 is never dropped
    assert any(any(p) for p in plans_a) and not all(all(p) for p in plans_a)
    state = a.layerdrop_state()
    nxt = a._layerdrop_plan()
    a.set_layerdrop_state(state)
    assert a._layerdrop_plan() == nxt                                # restorable
    a.eval()
    assert a._layerdrop_plan() == [False] * 6                        # never in eval
    # skipped blocks contribute no parameters-in-graph: homeo averaged over active blocks only
    a.train()
    x = torch.randint(0, 256, (2, 16))
    _, losses, _ = a(x, x)
    assert torch.isfinite(losses["total_loss"])


def test_gradient_checkpoint_equals_plain():
    torch.manual_seed(7)
    cfg = small_cfg()
    m1 = randomized(TensegrityLM(cfg)).train()
    m2 = TensegrityLM(cfg.replace(grad_checkpoint=True, grad_checkpoint_every=2)).train()
    m2.load_state_dict(m1.state_dict())
    x = torch.randint(0, 256, (2, 24))
    y = torch.randint(0, 256, (2, 24))
    for m in (m1, m2):
        _, l, _ = m(x, y)
        l["total_loss"].backward()
    for (n, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        torch.testing.assert_close(p1.grad, p2.grad, atol=1e-5, rtol=1e-4, msg=n)


def test_byte_mix_is_identity_at_init_and_uses_history():
    torch.manual_seed(8)
    base = TensegrityLM(small_cfg()).eval()
    mixed = TensegrityLM(small_cfg(byte_mix_kernel=4)).eval()
    mixed.load_state_dict(base.state_dict(), strict=False)
    x = torch.randint(0, 256, (2, 20))
    torch.testing.assert_close(base(x)[0], mixed(x)[0], atol=1e-6, rtol=1e-6)   # zero taps == no-op
    with torch.no_grad():
        mixed.byte_mix.weight.normal_(0, 0.5)
    out = mixed(x)[0]
    x2 = x.clone()
    x2[:, 5] = (x2[:, 5] + 1) % 256
    out2 = mixed(x2)[0]
    assert (out[:, 5:9] - out2[:, 5:9]).abs().max() > 1e-3         # change at t=5 reaches t=6..8 through the taps
    torch.testing.assert_close(out[:, :5], out2[:, :5])             # but never the past


def test_param_counts_match_formula():
    for kw in [{}, {"bias": True}, {"use_qk_norm": False}, {"byte_mix_kernel": 4, "x0_mix": True},
               {"use_confidence_head": False}, {"n_kv_head": 3}]:
        cfg = small_cfg(**kw)
        assert TensegrityLM(cfg).num_parameters() == cfg.n_params(), kw


def test_rope_tables_not_in_state_dict_and_fp32_after_half():
    m = TensegrityLM(small_cfg())
    assert not any("rope" in k for k in m.state_dict())
    m.half()
    cos, sin = m._rope_tables(torch.device("cpu"))
    assert cos.dtype == torch.float32
