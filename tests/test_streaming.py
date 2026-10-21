"""Streaming-cache generation, the long-stream approximation measurement, and the performance instruments."""

import math

import pytest
import torch

from conftest import randomized, small_cfg
from tensegrity import TensegrityLM, long_stream_nll
from tensegrity.layers import precompute_rope
from tensegrity.perf import (benchmark_generation, format_generation_benchmark, format_sdpa_report, profile_training_step,
                             sdpa_backend_report)

BS = 32


def _model(**over):
    kw = dict(block_size=BS, sliding_window=8, global_every=3)
    kw.update(over)
    cfg = small_cfg(**kw)
    torch.manual_seed(0)
    return randomized(TensegrityLM(cfg), 0.05, seed=1).eval()


def _ids(n, seed=0):
    return torch.randint(0, 256, (n,), generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------------- equivalence
@pytest.mark.parametrize("flags", [dict(), dict(byte_mix_kernel=3, x0_mix=True)])
def test_streaming_equals_window_while_everything_fits(flags):
    m = _model(**flags)
    p = _ids(10)[None]
    a = m.generate(p, 20, temperature=0.0, stop_on_eos=False, cache_mode="window")
    b = m.generate(p, 20, temperature=0.0, stop_on_eos=False, cache_mode="streaming")
    c = m.generate(p, 20, temperature=0.0, stop_on_eos=False, use_cache=False)
    assert torch.equal(a, b) and torch.equal(a, c) and a.shape[1] == 30 <= BS


@pytest.mark.parametrize("flags", [
    dict(global_every=3),                                   # the single layer is LOCAL (window 8)
    dict(global_every=1),                                   # the single layer is GLOBAL (bounded to block_size)
    dict(global_every=3, byte_mix_kernel=3, x0_mix=True, bias=True)])      # local + ByteMix: never reaches the window start
def test_a_single_layer_model_streams_exactly_past_the_window(flags):
    """With one layer a cached key depends only on its own token, so dropping old keys is exactly recomputing
    the truncated window: streaming must reproduce the reference at EVERY position, far beyond block_size.
    This pins the absolute-position (RoPE extension), cache-trimming and ByteMix-context logic independently."""
    m = _model(n_layer=1, **flags)
    ids = _ids(4 * BS + 5)
    r = long_stream_nll(m, ids.tolist())
    w, s = torch.tensor(r["window"]["nll_bits"]), torch.tensor(r["streaming"]["nll_bits"])
    assert w.shape == s.shape == (len(ids) - 1,)
    torch.testing.assert_close(w[:BS], s[:BS], atol=1e-5, rtol=0)                   # identical prefix (all fits)
    torch.testing.assert_close(w, s, atol=2e-4, rtol=0)                              # ... and identical beyond it
    p = _ids(BS)[None]
    gw = m.generate(p, 3 * BS, temperature=0.0, stop_on_eos=False, cache_mode="window")
    gs = m.generate(p, 3 * BS, temperature=0.0, stop_on_eos=False, cache_mode="streaming")
    assert torch.equal(gw, gs) and gs.shape[1] == 4 * BS


def test_bytemix_with_a_global_layer_is_the_second_source_of_difference():
    """A re-encoded window starts from PAD left-context (what training windows look like); the streaming cache
    holds keys computed with the real left context. A global layer attends to those first tokens, so even ONE
    layer differs -- and the difference disappears when ByteMix is switched off (zero taps = identity)."""
    m = _model(n_layer=1, global_every=1, byte_mix_kernel=4)
    ids = _ids(3 * BS).tolist()
    r = long_stream_nll(m, ids)
    w, s = torch.tensor(r["window"]["nll_bits"]), torch.tensor(r["streaming"]["nll_bits"])
    torch.testing.assert_close(w[:BS], s[:BS], atol=1e-5, rtol=0)
    assert (w[BS:] - s[BS:]).abs().max() > 1e-3
    with torch.no_grad():
        m.byte_mix.weight.zero_()
    r = long_stream_nll(m, ids)
    torch.testing.assert_close(torch.tensor(r["window"]["nll_bits"]), torch.tensor(r["streaming"]["nll_bits"]), atol=2e-4, rtol=0)


def test_a_deeper_model_streams_approximately_and_the_gap_is_measurable():
    m = _model()                                                                      # 4 layers
    ids = _ids(3 * BS)
    r = long_stream_nll(m, ids.tolist())
    w, s = torch.tensor(r["window"]["nll_bits"]), torch.tensor(r["streaming"]["nll_bits"])
    torch.testing.assert_close(w[:BS], s[:BS], atol=1e-5, rtol=0)                    # within the window: identical
    gap = (w[BS:] - s[BS:]).abs().max()
    assert 1e-4 < gap < 20.0                                                          # beyond it: close, not equal
    assert r["window"]["bpb_within"] == pytest.approx(r["streaming"]["bpb_within"], abs=1e-5)
    assert math.isfinite(r["streaming"]["bpb_beyond"]) and r["window"]["n_scored"] == r["streaming"]["n_scored"]
    only = long_stream_nll(m, ids.tolist(), modes=("streaming",))
    assert list(only) == ["streaming"] and only["streaming"]["nll_bits"] == r["streaming"]["nll_bits"]
    short = long_stream_nll(m, ids[:BS].tolist())                                     # nothing beyond the window
    assert math.isnan(short["window"]["bpb_beyond"]) and short["window"]["bpb"] == pytest.approx(short["streaming"]["bpb"], abs=1e-5)
    with pytest.raises(ValueError, match="mode must be"):
        long_stream_nll(m, ids.tolist(), modes=("exact",))
    with pytest.raises(ValueError, match="at least 3"):
        long_stream_nll(m, [1, 2])


# ------------------------------------------------------------------------------------- the point
def test_streaming_never_reprefills_and_keeps_its_cache_bounded(monkeypatch):
    m = _model()
    calls = []
    real = TensegrityLM.forward_incremental

    def spy(self, idx, past=None, pos_start=0, ctx=None, streaming=False):
        out = real(self, idx, past, pos_start, ctx, streaming)
        calls.append({"prefill": past is None, "T": idx.size(1), "pos": pos_start,
                      "lens": [kv[0].shape[2] for kv in out[1]]})
        return out
    monkeypatch.setattr(TensegrityLM, "forward_incremental", spy)
    p = _ids(BS)[None]
    n_new = 3 * BS
    m.generate(p, n_new, temperature=0.0, stop_on_eos=False, cache_mode="streaming")
    assert sum(c["prefill"] for c in calls) == 1 and len(calls) == n_new               # one prefill, then one step per token
    assert calls[-1]["pos"] == BS + n_new - 2                                          # absolute positions keep advancing
    kinds = [m.cfg.is_global_layer(i) for i in range(m.cfg.n_layer)]
    for c in calls:                                                                    # bounded: global <= block_size, local <= window
        assert all(L <= (BS if g else m.cfg.sliding_window) for L, g in zip(c["lens"], kinds)), c
    calls.clear()
    m.generate(p, n_new, temperature=0.0, stop_on_eos=False, cache_mode="window")
    assert sum(c["prefill"] for c in calls) == n_new                                   # window mode re-prefills past the context
    assert sum(c["prefill"] for c in calls[1:]) == n_new - 1


def test_rope_tables_grow_on_demand_and_stay_accurate_at_large_positions():
    m = _model()
    assert m._rope_tables(torch.device("cpu"))[0].shape[0] == BS                      # normal use: unchanged size
    m.generate(_ids(BS)[None], 4 * BS, temperature=0.0, stop_on_eos=False, cache_mode="streaming")
    cos, sin = m._rope_tables(torch.device("cpu"))
    assert cos.shape[0] >= 5 * BS - 1
    torch.testing.assert_close(cos[:BS], precompute_rope(m.cfg.head_dim, BS, m.cfg.rope_base)[0], atol=0, rtol=0)
    # float64 angles: accurate at position 100_000 where a float32 product is off by ~1e-3 (head_dim 64)
    hd, base, pos = 64, 10000.0, 100_000
    c64 = precompute_rope(hd, pos + 1, base)[0][pos].double()
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    exact = torch.cos(pos * inv)
    naive32 = torch.cos(torch.tensor(float(pos)) * (1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))))
    assert (c64 - exact).abs().max() < 1e-6
    assert (naive32.double() - exact).abs().max() > 5e-4                               # the reason for computing in float64


def test_streaming_api_validation_batches_and_eos():
    m = _model()
    p = _ids(6)[None]
    with pytest.raises(ValueError, match="cache_mode must be"):
        m.generate(p, 3, cache_mode="ring")
    with pytest.raises(ValueError, match="needs use_cache=True"):
        m.generate(p, 3, cache_mode="streaming", use_cache=False)
    toks = [t for t, _ in m.stream(p, 12, temperature=0.0, stop_on_eos=False, cache_mode="streaming")]
    assert toks == m.generate(p, 12, temperature=0.0, stop_on_eos=False, cache_mode="streaming")[0, 6:].tolist()
    batch = torch.stack([_ids(10, 1), _ids(10, 2)])
    out = m.generate(batch, 40, temperature=0.0, stop_on_eos=False, cache_mode="streaming")
    solo = [m.generate(batch[i:i + 1], 40, temperature=0.0, stop_on_eos=False, cache_mode="streaming") for i in range(2)]
    assert torch.equal(out[0:1], solo[0]) and torch.equal(out[1:2], solo[1])           # rows are independent
    # EOS freezes a finished row (as in window mode) and stops early once every row is done
    eos, calls = 257, {"n": 0}
    real = TensegrityLM._sample_next

    def forced(logits, *a, **k):
        calls["n"] += 1
        tok = real(logits, *a, **k)
        if calls["n"] == 3:
            tok[0, 0] = eos                                  # row 0 finishes at step 3 ...
        if calls["n"] == 6:
            tok[1, 0] = eos                                  # ... row 1 at step 6 -> everyone is done
        return tok
    m._sample_next = forced
    g = m.generate(batch, 20, temperature=0.0, cache_mode="streaming")
    assert g.shape == (2, 10 + 6)
    assert (g[0, 10 + 2:] == eos).all() and g[0, 10 + 1] != eos      # frozen on EOS from the step it finished
    assert g[1, 10 + 5] == eos and (g[1, 10:10 + 5] != eos).all()


def test_a_long_prompt_is_truncated_to_the_window_like_window_mode():
    m = _model()
    p = _ids(3 * BS)[None]
    a = m.generate(p, 8, temperature=0.0, stop_on_eos=False, cache_mode="window", rebuild_stride=BS)
    b = m.generate(p, 8, temperature=0.0, stop_on_eos=False, cache_mode="streaming")
    assert torch.equal(a[:, :3 * BS], b[:, :3 * BS]) and b.shape == (1, 3 * BS + 8)
    first_window = m.generate(p[:, -BS:], 1, temperature=0.0, stop_on_eos=False)       # first new byte: same window either way
    assert b[0, 3 * BS] == first_window[0, BS]


# ---------------------------------------------------------------------------------------- perf
def test_generation_benchmark_reports_phases_and_streaming_is_faster_past_the_context():
    cfg = small_cfg(block_size=192, n_embd=64, n_head=4, n_kv_head=2, sliding_window=48, global_every=3, n_layer=4)
    m = TensegrityLM(cfg)
    res = benchmark_generation(m, "cpu", new_tokens=48, repeats=2)
    assert set(res["modes"]) == {"window", "streaming"} and res["block_size"] == 192 and res["prompt_len"] == 96
    for r in res["modes"].values():
        assert r["prefill_ms"] > 0 and r["decode_tok_s_fits"] > 0 and r["decode_tok_s_overflow"] > 0
        assert r["peak_after_prefill"]["process_rss_peak_mib"] > 10 and r["peak_after_prefill"]["allocated_mib"] is None
    w, s = res["modes"]["window"], res["modes"]["streaming"]
    assert s["decode_tok_s_overflow"] > 1.2 * w["decode_tok_s_overflow"]               # no re-prefill vs a re-prefill per token
    assert "tok/s (past context)" in format_generation_benchmark(res)


def test_sdpa_backend_report_names_what_runs_here():
    cfg = small_cfg(block_size=64, n_embd=64, n_head=4, n_kv_head=2, sliding_window=16)
    res = sdpa_backend_report(cfg, "cpu", batch=1, iters=1)
    assert res["backends"]["MATH"]["global_causal"]["ok"] and res["backends"]["MATH"]["local_banded"]["ok"]
    assert res["backends"]["EFFICIENT_ATTENTION"]["global_causal"]["ok"] is False        # CPU has no mem-efficient kernel
    assert "error" in res["backends"]["EFFICIENT_ATTENTION"]["global_causal"]
    assert res["native_gqa_would_be_used"] is False and set(res["default_choice"]) == {"global_causal", "local_banded"}
    assert res["default_choice"]["global_causal"] in res["backends"] and "fastest working" in format_sdpa_report(res)


@pytest.mark.parametrize("layerdrop", [0.0, 0.3])
def test_a_training_pass_contains_no_host_syncs(layerdrop):
    """Regression guard for the v1 problem (a .item() and a collective in EVERY block, every step)."""
    m = TensegrityLM(small_cfg(layerdrop_max=layerdrop))
    r = profile_training_step(m, micro_batch=2, seq_len=24, device="cpu")
    assert r["host_syncs"] == 0 and r["top_ops"] and r["total_self_cpu_ms"] > 0
    assert any("mm" in o["op"] or "linear" in o["op"] for o in r["top_ops"])
    # the instrument itself works: an injected per-layer .item() is counted
    for b in m.blocks:
        orig = b.forward

        def leaky(*a, _orig=orig, **k):
            out = _orig(*a, **k)
            float(out[0].sum())                      # the anti-pattern
            return out
        b.forward = leaky
    assert profile_training_step(m, micro_batch=2, seq_len=24, device="cpu")["host_syncs"] >= len(m.blocks)
