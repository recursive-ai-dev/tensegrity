"""Trainer: learning, exact resume, LR schedule after resume, EMA, failure handling, planning."""

import json
import math
import os

import pytest
import torch

from conftest import small_cfg
from tensegrity import (EMA, ByteDataset, RunPlan, TensegrityLM, TrainConfig, Trainer,
                        autotune_micro_batch, default_ema_decay, evaluate, evaluate_batches, extend_plan, get_lr, get_lr_wsd,
                        load_model, load_run_plan, make_plan, measure_throughput, plan_for_time_budget, preflight, preflight_or_fallback)
from tensegrity.checkpoint import load_checkpoint


def _tc(**over) -> TrainConfig:
    kw = dict(batch_size=8, learning_rate=3e-3, min_lr=3e-4, eval_interval=12, eval_batches=3, eval_batch_size=4,
              log_interval=1, sample_every_evals=0, ckpt_interval_minutes=1e9, verbose=False, seed=7)
    kw.update(over)
    return TrainConfig(**kw)


def _plan(max_steps=24, warmup=4, stable=14, decay=6, tokens_per_step=8 * 48) -> RunPlan:
    return RunPlan(tokens_per_step, max_steps, warmup, stable, decay, "wsd")


def _states_equal(a: TensegrityLM, b: TensegrityLM) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_training_learns_and_writes_artifacts(corpus_dir, tmp_path):
    torch.manual_seed(0)
    cfg = small_cfg(n_layer=3, n_embd=64, n_head=4, n_kv_head=2, ffn_mult=2.5)
    model = TensegrityLM(cfg)
    events = []
    out = str(tmp_path / "run")
    tr = Trainer(model, _tc(ema_decay=0.9, eval_interval=20), _plan(60, 6, 40, 14), corpus_dir, out, device="cpu",
                 callbacks=[lambda e, p: events.append((e, p))])
    res = tr.fit()
    assert res["status"] == "completed" and res["step"] == 60
    vals = [p for e, p in events if e == "eval"]
    assert vals[0]["val_bpb"] > vals[-1]["val_bpb"] + 0.5, [v["val_bpb"] for v in vals]
    assert vals[-1]["val_loss_ema"] > 0 and {"start", "log", "eval", "checkpoint", "end"} <= {e for e, _ in events}
    assert sorted(os.listdir(out)) == ["ckpt_best.pt", "ckpt_latest.pt", "run_plan.json", "train_log.jsonl"]
    log = [json.loads(line) for line in open(os.path.join(out, "train_log.jsonl"))]
    assert any("val_loss" in r for r in log) and any("ce_loss" in r for r in log)
    plan, tc, mcfg = load_run_plan(out)
    assert plan == _plan(60, 6, 40, 14) and mcfg == cfg and tc.learning_rate == 3e-3
    best = load_checkpoint(os.path.join(out, "ckpt_best.pt"))              # weights-only, loads under weights_only=True
    assert best["optimizer_state_dict"] is None and best["weights_only"] is True
    assert os.path.getsize(os.path.join(out, "ckpt_best.pt")) < os.path.getsize(os.path.join(out, "ckpt_latest.pt")) / 2


def test_best_checkpoint_reproduces_its_recorded_val_loss(corpus_dir, tmp_path):
    torch.manual_seed(1)
    out = str(tmp_path / "run")
    tc = _tc(ema_decay=0.8, eval_interval=10)
    model = TensegrityLM(small_cfg())
    res = Trainer(model, tc, _plan(30, 4, 20, 6), corpus_dir, out, device="cpu").fit()
    m, cfg, ck = load_model(os.path.join(out, "ckpt_best.pt"))
    ds = ByteDataset(corpus_dir)
    batches = ds.fixed_batches("val", tc.eval_batches, tc.eval_batch_size, cfg.block_size, "cpu")
    again = evaluate_batches(m, batches, "cpu", torch.float32)["loss"]
    assert again == pytest.approx(res["best_val_loss"], rel=1e-5), (res["best_source"], again, res["best_val_loss"])
    assert ck["best_source"] == res["best_source"]


def test_resume_is_bit_exact(corpus_dir, tmp_path):
    """Stop at step 12, resume in a fresh process-like state, finish: the final weights must be
    IDENTICAL to an uninterrupted run (model, optimizer, EMA, dropout/LayerDrop RNG, data order)."""
    cfg = small_cfg(layerdrop_max=0.3, dropout=0.1, byte_mix_kernel=3, x0_mix=True)
    full_plan = _plan(24, 4, 14, 6)
    tc = _tc(ema_decay=0.9)

    torch.manual_seed(5)
    a = TensegrityLM(cfg)
    init_state = {k: v.clone() for k, v in a.state_dict().items()}
    Trainer(a, tc, full_plan, corpus_dir, str(tmp_path / "A"), device="cpu").fit()

    b1 = TensegrityLM(cfg)
    b1.load_state_dict(init_state)
    half = RunPlan(full_plan.tokens_per_step, 12, full_plan.warmup_steps, full_plan.stable_steps, full_plan.decay_steps)
    Trainer(b1, tc, half, corpus_dir, str(tmp_path / "B"), device="cpu").fit()

    torch.manual_seed(999)                          # a *different* global state must not matter after resume
    b2 = TensegrityLM(cfg)
    r = Trainer(b2, tc, full_plan, corpus_dir, str(tmp_path / "B"), device="cpu").fit(resume=True)
    assert r["step"] == 24
    assert _states_equal(a, b2)
    # EMA state is also identical
    ea = load_checkpoint(str(tmp_path / "A" / "ckpt_latest.pt"))["ema_state_dict"]
    eb = load_checkpoint(str(tmp_path / "B" / "ckpt_latest.pt"))["ema_state_dict"]
    assert ea["num_updates"] == eb["num_updates"] == 24
    assert all(torch.equal(ea["shadow"][k], eb["shadow"][k]) for k in ea["shadow"])


def test_lr_after_resume_follows_the_schedule(corpus_dir, tmp_path):
    """Regression for a v1 bug: base LRs were captured AFTER the optimizer state was loaded, so a
    resumed run re-applied the schedule on top of the already-decayed LR (observed 6x too low,
    below the schedule's own floor)."""
    cfg = small_cfg()
    plan = RunPlan(8 * 48, 24, 4, 8, 12, "wsd")           # decay phase starts at step 12
    tc = _tc(eval_interval=100)
    logs = []
    half = RunPlan(plan.tokens_per_step, 16, plan.warmup_steps, plan.stable_steps, plan.decay_steps)
    Trainer(TensegrityLM(cfg), tc, half, corpus_dir, str(tmp_path / "r"), device="cpu").fit()   # ends inside the decay
    Trainer(TensegrityLM(cfg), tc, plan, corpus_dir, str(tmp_path / "r"), device="cpu",
            callbacks=[lambda e, p: logs.append(p) if e == "log" else None]).fit(resume=True)
    floor = tc.min_lr / tc.learning_rate
    seen = [r for r in logs if "lr" in r]
    assert [r["step"] for r in seen][:2] == [16, 17]
    for r in seen:
        want = tc.learning_rate * plan.multiplier(r["step"], floor)
        assert r["lr"] == pytest.approx(want, rel=1e-9), (r["step"], r["lr"], want)
        assert r["lr"] >= tc.min_lr * (1 - 1e-9)           # never below the schedule's own floor


def test_time_limit_stops_cleanly_and_can_continue(corpus_dir, tmp_path):
    out = str(tmp_path / "t")
    cfg = small_cfg()
    r1 = Trainer(TensegrityLM(cfg), _tc(max_minutes=1e-9, eval_interval=100), _plan(10, 2, 5, 3), corpus_dir, out,
                 device="cpu").fit()
    assert r1["status"] == "time_limit" and r1["step"] == 1
    assert os.path.exists(os.path.join(out, "ckpt_latest.pt"))
    r2 = Trainer(TensegrityLM(cfg), _tc(eval_interval=100), _plan(10, 2, 5, 3), corpus_dir, out, device="cpu").fit()
    assert r2["status"] == "completed" and r2["step"] == 10


def test_nonfinite_steps_are_skipped_not_applied(corpus_dir, tmp_path):
    calls = {"n": 0}

    def poison(model):
        calls["n"] += 1
        return model.token_emb.weight.sum() * float("nan") if calls["n"] == 3 else model.token_emb.weight.sum() * 0.0

    model = TensegrityLM(small_cfg())
    tr = Trainer(model, _tc(eval_interval=100), _plan(8, 2, 4, 2), corpus_dir, str(tmp_path / "n"), device="cpu",
                 extra_loss=poison)
    res = tr.fit()
    assert res["skipped_steps"] == 1 and res["status"] == "completed"
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # a model that diverges for good raises instead of silently looping
    tr2 = Trainer(TensegrityLM(small_cfg()), _tc(max_nonfinite=3), _plan(20, 2, 15, 3), corpus_dir,
                  str(tmp_path / "d"), device="cpu", extra_loss=lambda m: m.token_emb.weight.sum() * float("nan"))
    with pytest.raises(FloatingPointError, match="diverged"):
        tr2.fit()


def test_resume_guards(corpus_dir, tmp_path):
    out = str(tmp_path / "g")
    cfg = small_cfg()
    Trainer(TensegrityLM(cfg), _tc(eval_interval=100), _plan(4, 2, 1, 1), corpus_dir, out, device="cpu").fit()
    other = small_cfg(n_embd=24, n_head=2, n_kv_head=1)
    with pytest.raises(ValueError, match="config differs"):
        Trainer(TensegrityLM(other), _tc(), _plan(8, 2, 4, 2), corpus_dir, out, device="cpu").fit()
    with pytest.raises(ValueError, match="optimizer"):
        Trainer(TensegrityLM(cfg), _tc(optimizer="muon"), _plan(8, 2, 4, 2), corpus_dir, out, device="cpu").fit()
    # a weights-only checkpoint (ckpt_best) is not resumable: clear message, not a KeyError
    os.replace(os.path.join(out, "ckpt_best.pt"), os.path.join(out, "ckpt_latest.pt"))
    with pytest.raises(ValueError, match="weights-only"):
        Trainer(TensegrityLM(cfg), _tc(), _plan(8, 2, 4, 2), corpus_dir, out, device="cpu").fit()


def test_grad_checkpointing_changes_nothing_but_memory(corpus_dir, tmp_path):
    outs = []
    for flag in (False, True):
        torch.manual_seed(3)
        m = TensegrityLM(small_cfg(grad_checkpoint=flag))
        Trainer(m, _tc(eval_interval=100), _plan(6, 2, 3, 1), corpus_dir, str(tmp_path / str(flag)), device="cpu").fit()
        outs.append(m)
    for pa, pb in zip(outs[0].parameters(), outs[1].parameters()):
        torch.testing.assert_close(pa, pb, atol=2e-5, rtol=1e-4)


def test_muon_training_runs_and_resumes(corpus_dir, tmp_path):
    out = str(tmp_path / "m")
    tc = _tc(optimizer="muon", learning_rate=2e-3, min_lr=2e-4, eval_interval=6)
    short = RunPlan(8 * 48, 4, 2, 3, 1, "wsd")             # stopped at step 4, before its cool-down (which starts at 5)
    Trainer(TensegrityLM(small_cfg()), tc, short, corpus_dir, out, device="cpu").fit()
    r = Trainer(TensegrityLM(small_cfg()), tc, extend_plan(short, 8), corpus_dir, out, device="cpu").fit()
    assert r["step"] == 12


# -------------------------------------------------------------------------------------- EMA
def test_ema_math_swap_and_state():
    torch.manual_seed(0)
    m = TensegrityLM(small_cfg())
    ema = EMA(m, 0.9)
    ref = {n: p.detach().clone() for n, p in m.named_parameters()}
    for step in range(1, 6):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        ema.update(m)
        d = min(0.9, (1 + step) / (10 + step))
        for n, p in m.named_parameters():
            ref[n] = ref[n] * d + p.detach() * (1 - d)
    for n, p in m.named_parameters():
        torch.testing.assert_close(ema.shadow[n], ref[n], atol=1e-6, rtol=1e-5)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    with ema.swapped(m):
        for n, p in m.named_parameters():
            torch.testing.assert_close(p, ref[n], atol=1e-6, rtol=1e-5)
    assert all(torch.equal(p, before[n]) for n, p in m.named_parameters())      # bit-exact restore
    e2 = EMA(m, 0.5)
    e2.load_state_dict(ema.state_dict())
    assert e2.decay == 0.9 and e2.num_updates == 5
    m2 = TensegrityLM(small_cfg())
    EMA.apply_state_to_model(ema.state_dict(), m2)
    for n, p in m2.named_parameters():
        torch.testing.assert_close(p, ref[n], atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError):
        EMA(m, 1.0)


def test_default_ema_decay():
    assert default_ema_decay(100) == pytest.approx(1 - 1 / 10)
    assert default_ema_decay(10 ** 6) == 0.9995


# --------------------------------------------------------------------------- planning/schedule
def test_schedules():
    assert get_lr(0, 10, 100, 1.0, 0.1) == pytest.approx(0.1) and get_lr(9, 10, 100, 1.0, 0.1) == pytest.approx(1.0)
    assert get_lr(10, 10, 100, 1.0, 0.1) == pytest.approx(1.0) and get_lr(100, 10, 100, 1.0, 0.1) == 0.1
    assert get_lr(55, 10, 100, 1.0, 0.1) == pytest.approx(0.55)
    lrs = [get_lr(s, 10, 100, 1.0, 0.1) for s in range(10, 100)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    assert get_lr_wsd(5, 10, 20, 10, 1.0, 0.1) == pytest.approx(0.6)
    assert get_lr_wsd(15, 10, 20, 10, 1.0, 0.1) == 1.0 == get_lr_wsd(29, 10, 20, 10, 1.0, 0.1)
    assert get_lr_wsd(35, 10, 20, 10, 1.0, 0.1) == pytest.approx(0.55) and get_lr_wsd(40, 10, 20, 10, 1.0, 0.1) == 0.1
    with pytest.raises(AssertionError):
        get_lr_wsd(0, 0, 1, 1, 1.0, 0.1)


def test_make_plan_and_time_budget():
    p = make_plan(10_000_000, 10_000, "wsd")
    assert p.max_steps == 1000 and p.warmup_steps + p.stable_steps + p.decay_steps == 1000
    assert p.decay_steps == 200 and p.total_tokens == 10_000_000
    assert p.multiplier(0, 0.1) == pytest.approx(1 / p.warmup_steps) and p.multiplier(p.max_steps, 0.1) == 0.1
    c = make_plan(10_000_000, 10_000, "cosine")
    assert c.multiplier(c.warmup_steps, 0.1) == pytest.approx(1.0) and c.multiplier(c.max_steps, 0.1) == 0.1
    q = plan_for_time_budget(tokens_per_sec=1000, hours=1, tokens_per_step=1000, safety=1.0)
    assert q.max_steps == 3600
    capped = plan_for_time_budget(1e6, 100, 1000, max_tokens=5_000_000)
    assert capped.total_tokens <= 5_000_000
    with pytest.raises(ValueError):
        make_plan(100, 0)
    with pytest.raises(ValueError):
        make_plan(100, 10, "triangle")
    with pytest.raises(ValueError):
        plan_for_time_budget(0, 1, 10)


def test_autotune_and_throughput_cpu_path():
    m = TensegrityLM(small_cfg())
    t = autotune_micro_batch(m, 48, "cpu", torch.float32, target_batch_tokens=48 * 16)
    assert t["micro_batch"] * t["grad_accum"] * 48 >= 48 * 16 - 48 * t["micro_batch"] and t["grad_checkpoint"] is False
    assert measure_throughput(m, 2, 24, "cpu", torch.float32, steps=2, warmup=1) > 0


def test_evaluate_helper_and_extra_loss_hook(corpus_dir, tmp_path):
    m = TensegrityLM(small_cfg())
    r = evaluate(m, corpus_dir, m.cfg, "cpu", n_batches=3, batch_size=4)
    assert r["bpb"] == pytest.approx(r["loss"] / math.log(2)) and r["ppl"] == pytest.approx(math.exp(r["loss"]))
    assert r["loss"] > 4.0            # untrained: near ln(259) = 5.56
    seen = []
    Trainer(m, _tc(eval_interval=100), _plan(3, 1, 1, 1), corpus_dir, str(tmp_path / "x"), device="cpu",
            extra_loss=lambda mod: (seen.append(1), mod.token_emb.weight.sum() * 0.0)[1]).fit()
    assert len(seen) == 3


def test_session_limits_are_not_persisted_with_the_run(corpus_dir, tmp_path):
    out = str(tmp_path / "e")
    Trainer(TensegrityLM(small_cfg()), _tc(max_minutes=1e-9, verbose=False, eval_interval=100), _plan(6, 2, 3, 1),
            corpus_dir, out, device="cpu").fit()
    saved = json.load(open(os.path.join(out, "run_plan.json")))["train_config"]
    assert "max_minutes" not in saved and "verbose" not in saved and saved["learning_rate"] == 3e-3
    plan, tc, _ = load_run_plan(out)
    assert tc.max_minutes is None and tc.learning_rate == 3e-3          # a resume is not throttled by the last session


@pytest.mark.parametrize("opt", ["adamw", "muon"])
def test_preflight_passes_restores_weights_and_catches_bad_numerics(opt):
    torch.manual_seed(0)
    m = TensegrityLM(small_cfg(layerdrop_max=0.2))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    ld = m.layerdrop_state().clone()
    r = preflight(m, _tc(optimizer=opt), "cpu", steps=3)
    assert r["ok"] and r["reason"] is None and r["optimizer"] == opt and r["loss_first"] > 4
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())      # weights restored
    assert torch.equal(m.layerdrop_state(), ld)                                    # RNG state untouched
    # a numerics failure is reported, not raised
    broken = TensegrityLM(small_cfg())
    with torch.no_grad():
        broken.token_emb.weight.fill_(float("nan"))
    rb = preflight(broken, _tc(), "cpu", steps=2)
    assert rb["ok"] is False and "non-finite" in rb["reason"]


def test_preflight_or_fallback_switches_to_adamw_and_raises_when_both_fail(monkeypatch, capsys):
    import tensegrity.optim as O

    def exploding_step(self, closure=None):
        raise RuntimeError("muon exploded on this device")
    monkeypatch.setattr(O.SingleDeviceMuonWithAuxAdam, "step", exploding_step)
    m = TensegrityLM(small_cfg())
    tc = _tc(optimizer="muon")
    out = preflight_or_fallback(m, tc, "cpu", verbose=True)
    assert out is tc and tc.optimizer == "adamw"
    assert "falling back to AdamW" in capsys.readouterr().out
    assert preflight_or_fallback(m, _tc(optimizer="adamw"), "cpu", verbose=False).optimizer == "adamw"   # healthy: untouched
    broken = TensegrityLM(small_cfg())
    with torch.no_grad():
        broken.token_emb.weight.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="AdamW"):
        preflight_or_fallback(broken, _tc(optimizer="adamw"), "cpu", verbose=False)
