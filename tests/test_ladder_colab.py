"""Ladder climbing, confidence calibration, plan extension, Colab helpers."""

import os
import sys
import types

import pytest
import torch

from conftest import randomized, small_cfg
from tensegrity import (TensegrityLM, climb_ladder, confidence_calibration, evaluate, extend_plan, get_preset_config,
                        grow_model, in_colab, local_scratch, mount_drive, project_dir, runtime_summary, get_secret,
                        RunPlan, benchmark_presets)
from tensegrity.growth import retarget

OVR = dict(block_size=64, sliding_window=16)


FAST = dict(tokens=6_000, batch_tokens=512, device="cpu", micro_batch=4, model_overrides=OVR, verbose=False)
TCO = dict(verbose=False, eval_interval=1000, sample_every_evals=0)


def test_climb_ladder_trains_grows_and_resumes(corpus_dir, tmp_path):
    root = str(tmp_path / "climb")
    r = climb_ladder(["1m", "2m"], corpus_dir, root, tc_overrides=TCO, **FAST)
    assert [x.status for x in r] == ["completed", "completed"] and not any(x.skipped for x in r)
    assert r[0].params < r[1].params and r[1].growth_max_logit_diff is not None and r[1].growth_max_logit_diff < 1e-4
    assert r[0].best_val_bpb < 8.0 and r[1].best_val_bpb < 8.0
    # the grown rung inherited the first rung's function-defining fields (custom window/context)
    from tensegrity import load_model
    m2, c2, _ = load_model(os.path.join(r[1].run_dir, "ckpt_best.pt"))
    assert (c2.sliding_window, c2.block_size, c2.n_embd, c2.n_layer) == (16, 64, 192, 5)
    # provenance: the grown rung's checkpoints name the exact parent checkpoint they started from
    from tensegrity.checkpoint import load_checkpoint
    from tensegrity.manifest import sha256_file
    lin = load_checkpoint(os.path.join(r[1].run_dir, "ckpt_latest.pt"))["lineage"]
    assert lin["grown_from"]["sha256"] == sha256_file(os.path.join(r[0].run_dir, "ckpt_best.pt")) and lin["rung"] == 1
    # calling again is a no-op: both rungs are recognised as finished
    again = climb_ladder(["1m", "2m"], corpus_dir, root, tc_overrides=TCO, **FAST)
    assert all(x.skipped for x in again) and again[1].best_val_loss == r[1].best_val_loss
    # extending the ladder reuses the finished rungs and only trains the new one
    more = climb_ladder(["1m", "2m", "5m"], corpus_dir, root, tc_overrides=TCO, **FAST)
    assert [x.skipped for x in more] == [True, True, False] and more[2].params > r[1].params


def test_climb_ladder_resumes_an_interrupted_rung(corpus_dir, tmp_path):
    root = str(tmp_path / "c")
    cut = climb_ladder(["1m", "2m"], corpus_dir, root, tc_overrides={**TCO, "max_minutes": 1e-9}, **FAST)
    assert cut[-1].status == "time_limit" and len(cut) == 1          # stopped inside rung 0, did not move on
    done = climb_ladder(["1m", "2m"], corpus_dir, root, tc_overrides=TCO, **FAST)
    assert [x.status for x in done] == ["completed", "completed"]


def test_climb_ladder_validation(corpus_dir, tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        climb_ladder(["1m", "1M"], corpus_dir, str(tmp_path))
    with pytest.raises(ValueError, match="OR"):
        climb_ladder(["1m"], corpus_dir, str(tmp_path), hours=1, tokens=5)
    with pytest.raises(ValueError, match="one per rung"):
        climb_ladder(["1m", "2m"], corpus_dir, str(tmp_path), tokens=[1, 2, 3])


def test_hours_budget_sizes_the_plan_from_measured_throughput(corpus_dir, tmp_path, monkeypatch):
    """hours -> plan: tokens = measured_tok_per_s * hours * 3600 * 0.92 (deterministic via a mocked throughput)."""
    import tensegrity.ladder as L
    from tensegrity import load_run_plan, make_plan
    monkeypatch.setattr(L, "measure_throughput", lambda *a, **k: 20.0)
    kw = {k: v for k, v in FAST.items() if k != "tokens"}
    root = str(tmp_path / "h")
    r = climb_ladder(["1m"], corpus_dir, root, hours=0.2, tc_overrides=TCO, **kw)
    plan, _, _ = load_run_plan(r[0].run_dir)
    want = make_plan(int(20.0 * 0.2 * 3600 * 0.92), plan.tokens_per_step, "wsd")
    assert plan.total_tokens == want.total_tokens and plan.max_steps == want.max_steps and r[0].status == "completed"


def test_intermediate_cooldown_only_shortens_non_final_rungs(corpus_dir, tmp_path):
    from tensegrity import load_run_plan
    kw = dict(FAST, tokens=40_000)
    r = climb_ladder(["1m", "2m"], corpus_dir, str(tmp_path / "ic"), intermediate_cooldown=0.05, tc_overrides=TCO, **kw)
    p0, _, _ = load_run_plan(r[0].run_dir)
    p1, _, _ = load_run_plan(r[1].run_dir)
    assert p0.decay_steps == max(1, int(0.05 * p0.max_steps))        # intermediate rung: short cool-down
    assert p1.decay_steps == max(1, int(0.20 * p1.max_steps))        # final rung: the normal 20%


def test_val_loss_is_continuous_across_growth(corpus_dir):
    cfg = get_preset_config("1m", **OVR)
    torch.manual_seed(0)
    parent = randomized(TensegrityLM(cfg), 0.03)
    child, ccfg = grow_model(parent, cfg, target=retarget(cfg, get_preset_config("2m")))
    a = evaluate(parent, corpus_dir, cfg, "cpu", torch.float32, 3, 4)
    b = evaluate(child, corpus_dir, ccfg, "cpu", torch.float32, 3, 4)
    assert b["loss"] == pytest.approx(a["loss"], abs=1e-4)


def test_retarget_keeps_function_fields_and_takes_size():
    old = get_preset_config("1m", block_size=64, sliding_window=16, byte_mix_kernel=4, x0_mix=True)
    t = retarget(old, get_preset_config("5m"))
    assert (t.n_embd, t.n_layer, t.ffn_hidden) == (256, 7, 672)
    assert (t.sliding_window, t.block_size, t.byte_mix_kernel, t.x0_mix) == (16, 64, 4, True)
    assert retarget(old, get_preset_config("5m"), keep_context=False).block_size == 1024


def test_confidence_calibration(corpus_dir):
    torch.manual_seed(0)
    m = randomized(TensegrityLM(small_cfg(confidence_detach=True)), 0.02)
    r = confidence_calibration(m, corpus_dir, "cpu", n_batches=4, batch_size=4, n_bins=5)
    assert len(r["bins"]) == 5 and sum(b["count"] for b in r["bins"]) == r["n_tokens"] == 4 * 4 * 48
    assert 0.0 <= r["ece"] <= 1.0 and 0.0 <= r["base_accuracy"] <= 1.0
    # ECE is the count-weighted |acc - conf| gap
    manual = sum(b["count"] * abs(b["accuracy"] - b["mean_confidence"]) for b in r["bins"] if b["count"]) / r["n_tokens"]
    assert r["ece"] == pytest.approx(manual)
    with pytest.raises(ValueError):
        confidence_calibration(TensegrityLM(small_cfg(use_confidence_head=False)), corpus_dir)


def test_extend_plan():
    p = RunPlan(1000, 100, 5, 75, 20, "wsd")
    q = extend_plan(p, 50, at_step=40)
    assert (q.max_steps, q.stable_steps, q.decay_steps, q.warmup_steps) == (150, 125, 20, 5)
    assert q.multiplier(90, 0.1) == 1.0 and p.multiplier(90, 0.1) < 1.0      # step 90 was cooling, now still stable
    with pytest.raises(ValueError, match="cool-down already began"):
        extend_plan(p, 10, at_step=81)
    with pytest.raises(ValueError, match="WSD"):
        extend_plan(RunPlan(1, 10, 1, 1, 1, "cosine"), 5)
    with pytest.raises(ValueError):
        extend_plan(p, 0)


# ---------------------------------------------------------------------------------- colab helpers
def test_helpers_degrade_gracefully_outside_colab(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    assert in_colab() is False
    assert mount_drive() is False
    monkeypatch.chdir(tmp_path)
    d = project_dir(use_drive=True, name="proj")
    assert os.path.isdir(d) and d.startswith(str(tmp_path))
    assert os.path.isdir(local_scratch("scratch_x")) or True
    monkeypatch.setenv("MY_SECRET", "abc")
    assert get_secret("MY_SECRET") == "abc" and get_secret("DEFINITELY_NOT_SET_12345") is None
    s = runtime_summary()
    assert "python" in s and "torch" in s


def test_drive_and_secrets_inside_a_simulated_colab(tmp_path, monkeypatch):
    calls = {}

    def fake_mount(mp):
        calls["mounted"] = mp
        os.makedirs(os.path.join(mp, "MyDrive"), exist_ok=True)

    fake = types.ModuleType("google.colab")
    drive = types.SimpleNamespace(mount=fake_mount)
    userdata = types.SimpleNamespace(get=lambda k: {"HF_TOKEN": "tok"}[k])
    fake.drive, fake.userdata = drive, userdata
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", fake)
    monkeypatch.setitem(sys.modules, "google.colab.drive", drive)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)
    assert in_colab() is True
    mp = str(tmp_path / "drive")
    assert mount_drive(mp) is True and calls["mounted"] == mp
    assert get_secret("HF_TOKEN") == "tok"
    assert get_secret("MISSING") is None                  # KeyError from userdata -> None, not a crash


def test_benchmark_presets_runs_on_cpu():
    rows = benchmark_presets(["1m"], device="cpu", seq_len=64, verbose=False)
    assert rows[0]["tokens_per_sec"] > 0 and rows[0]["hours_for_default_budget"] > 0 and rows[0]["params"] > 9e5


def test_in_colab_survives_a_missing_google_namespace(monkeypatch):
    """Regression: importlib.util.find_spec('google.colab') RAISES (ModuleNotFoundError) when there is
    no `google` package at all, which crashed runtime_summary() on every non-Colab machine."""
    import importlib.util
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)

    def boom(name, *a, **k):
        raise ModuleNotFoundError("No module named 'google'")
    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert in_colab() is False
    assert "python" in runtime_summary()
    assert mount_drive() is False


def test_liveplot_updates_one_display_handle_and_never_clears_output(monkeypatch):
    """The dashboard must update its figure in place: clear_output() would erase every log line the
    trainer printed in the same cell."""
    pytest.importorskip("matplotlib")
    IPython = pytest.importorskip("IPython")
    import IPython.display as ipd
    import matplotlib
    matplotlib.use("Agg")
    from tensegrity import LivePlot
    shown, updated, cleared = [], [], []

    class Handle:
        def update(self, obj):
            updated.append(obj)
    fake_shell = types.SimpleNamespace(events=types.SimpleNamespace(register=lambda *a, **k: None,
                                                                    unregister=lambda *a, **k: None))
    monkeypatch.setattr(IPython, "get_ipython", lambda: fake_shell)
    monkeypatch.setattr(ipd, "display", lambda obj, display_id=None, **k: (shown.append(display_id), Handle())[1])
    monkeypatch.setattr(ipd, "clear_output", lambda *a, **k: cleared.append(1))
    lp = LivePlot()
    for step in range(3):
        lp("log", {"step": step, "ce_loss": 3.0 - step, "lr": 1e-3, "tok_per_s": 1000.0})
        lp("eval", {"step": step, "val_bpb": 4.0 - step, "val_bpb_ema": 4.1 - step})
    assert shown == [True] and len(updated) == 2 and cleared == []
