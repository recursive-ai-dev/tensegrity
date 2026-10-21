"""Exact resume is a precisely defined operation: same policy, same data, same counters, same state."""

import json
import os
import warnings

import numpy as np
import pytest
import torch

from conftest import make_text, small_cfg
from tensegrity import (RunPlan, TensegrityConfig, TensegrityLM, TrainConfig, Trainer, data_identity, extend_plan,
                        load_model, prepare_data)
from tensegrity.checkpoint import (atomic_torch_save, capture_rng_state, load_checkpoint, normalize_state_dict,
                                   save_checkpoint, unwrap_model)
from tensegrity.manifest import DatasetError
from tensegrity.schedule import plan_change_problems

def _compiled(model):
    """torch.compile wraps lazily; only its availability can differ between torch/python builds."""
    try:
        return torch.compile(model)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"torch.compile unavailable in this environment: {e}")


TPS = 8 * 48


def _tc(**over) -> TrainConfig:
    kw = dict(batch_size=8, learning_rate=3e-3, min_lr=3e-4, eval_interval=100, eval_batches=2, eval_batch_size=4,
              log_interval=1, sample_every_evals=0, ckpt_interval_minutes=1e9, verbose=False, seed=7)
    kw.update(over)
    return TrainConfig(**kw)


FULL = RunPlan(TPS, 12, 2, 8, 2, "wsd")
HALF = RunPlan(TPS, 6, 2, 8, 2, "wsd")           # the same LR curve, stopped at step 6 (cool-down would start at 10)


def _fit(model, tc, plan, data, out, **fit_kw):
    return Trainer(model, tc, plan, data, str(out), device="cpu").fit(**fit_kw)


def _weights(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


# --------------------------------------------------------------------------------------- policy
def test_resume_refuses_a_changed_policy_and_touches_nothing(corpus_dir, tmp_path):
    cfg, out = small_cfg(), tmp_path / "r"
    _fit(TensegrityLM(cfg), _tc(), HALF, corpus_dir, out)
    fresh = TensegrityLM(cfg)
    before = _weights(fresh)
    with pytest.raises(ValueError) as e:
        _fit(fresh, _tc(learning_rate=1e-3, weight_decay=0.0), FULL, corpus_dir, out)
    msg = str(e.value)
    assert "learning_rate: checkpoint=0.003 vs now=0.001" in msg and "weight_decay" in msg and "on_policy_change" in msg
    assert _same(before, _weights(fresh))                      # a refused resume loads nothing
    # non-trajectory knobs may differ freely between sessions
    _fit(TensegrityLM(cfg), _tc(eval_interval=3, log_interval=5, eval_batches=1, sample_every_evals=2), FULL,
         corpus_dir, out)
    assert load_checkpoint(str(out / "ckpt_latest.pt"))["step"] == 12


def test_adopting_a_policy_change_is_explicit_consistent_and_recorded(corpus_dir, tmp_path):
    cfg, out = small_cfg(), tmp_path / "r"
    _fit(TensegrityLM(cfg), _tc(weight_decay=0.1), HALF, corpus_dir, out)
    tc2 = _tc(learning_rate=1e-3, min_lr=1e-4, weight_decay=0.0)
    tr = Trainer(TensegrityLM(cfg), tc2, FULL, corpus_dir, str(out), device="cpu")
    tr.fit(on_policy_change="adopt")
    # v-review probe: torch restores the CHECKPOINT's weight decay (0.1) into the param groups; the session's config must win
    assert all(g["weight_decay"] == 0.0 for g in tr.optimizer.param_groups)
    ck = load_checkpoint(str(out / "ckpt_latest.pt"))
    (phase,) = ck["lineage"]["phases"]
    assert phase["kind"] == "policy" and phase["at_step"] == 6 and any("learning_rate" in c for c in phase["changes"])
    assert ck["train_config"]["learning_rate"] == 1e-3 and ck["lr_state"]["base_lrs"] == [1e-3, 1e-3]
    log = [json.loads(line) for line in open(out / "train_log.jsonl")]
    lr_by_step = {r["step"]: r["lr"] for r in log if "lr" in r}
    assert lr_by_step[5] == pytest.approx(3e-3) and lr_by_step[6] == pytest.approx(1e-3)    # new base LR from step 6 on
    # afterwards the run is simply a run with the new policy: a plain resume is exact again
    r = _fit(TensegrityLM(cfg), tc2, FULL, corpus_dir, out)
    assert r["step"] == 12
    with pytest.raises(ValueError, match="on_policy_change"):
        _fit(TensegrityLM(cfg), tc2, FULL, corpus_dir, out, on_policy_change="maybe")


def test_legal_and_illegal_plan_changes():
    old = FULL.to_dict()
    assert plan_change_problems(old, old, 5) == []
    assert plan_change_problems(old, HALF.to_dict(), 6) == []                                  # loop bound only
    assert plan_change_problems(HALF.to_dict(), FULL.to_dict(), 6) == []
    ext = extend_plan(HALF, 4)
    assert plan_change_problems(HALF.to_dict(), ext.to_dict(), 6) == []                        # extend_plan, cool-down not begun
    assert "cool-down already began" in plan_change_problems(FULL.to_dict(), extend_plan(FULL, 4, at_step=0).to_dict(), 12)[0]
    bad = RunPlan(TPS, 12, 2, 8, 2, "wsd").to_dict() | {"decay_steps": 3}
    assert any("decay_steps" in p for p in plan_change_problems(old, bad, 5))
    bad = old | {"stable_steps": 9}                                                            # curve changed, horizon did not
    assert any("extend_plan" in p for p in plan_change_problems(old, bad, 5))
    assert any("warmup_steps" in p for p in plan_change_problems(old, old | {"warmup_steps": 3}, 5))
    cos = RunPlan(TPS, 12, 2, 8, 2, "cosine").to_dict()
    assert any("cosine" in p for p in plan_change_problems(cos, cos | {"max_steps": 16}, 5))
    assert plan_change_problems(cos, cos, 5) == []


# ------------------------------------------------------------------------------------------- data
def test_resume_requires_the_same_data(corpus_dir, corpus_dir2, tmp_path):
    cfg, out = small_cfg(), tmp_path / "r"
    _fit(TensegrityLM(cfg), _tc(), HALF, corpus_dir, out)
    with pytest.raises(ValueError, match="training data differs") as e:
        _fit(TensegrityLM(cfg), _tc(), FULL, corpus_dir2, out)
    assert "fingerprint" in str(e.value)
    # the same CONTENT prepared again elsewhere is the same data (content-addressed fingerprint)
    again = tmp_path / "again"
    src = tmp_path / "s.txt"
    src.write_text(make_text(), encoding="utf-8")
    prepare_data(str(src), str(again), val_fraction=0.1, doc_sep="<|endoftext|>", verbose=False)
    assert data_identity(str(again))["fingerprint"] == data_identity(corpus_dir)["fingerprint"]
    assert _fit(TensegrityLM(cfg), _tc(), FULL, str(again), out)["step"] == 12
    # adopting new data is explicit and recorded as a phase
    out2 = tmp_path / "r2"
    _fit(TensegrityLM(cfg), _tc(), HALF, corpus_dir, out2)
    _fit(TensegrityLM(cfg), _tc(), FULL, corpus_dir2, out2, on_data_change="adopt")
    phases = load_checkpoint(str(out2 / "ckpt_latest.pt"))["lineage"]["phases"]
    assert [p["kind"] for p in phases] == ["data"] and phases[0]["at_step"] == 6


def test_resume_requires_the_same_replay_mixture(corpus_dir, corpus_dir2, tmp_path):
    cfg, out = small_cfg(), tmp_path / "r"
    both = [corpus_dir, corpus_dir2]
    Trainer(TensegrityLM(cfg), _tc(), HALF, both, str(out), data_weights=[1, 1], device="cpu").fit()
    with pytest.raises(ValueError, match="mixture weights"):
        Trainer(TensegrityLM(cfg), _tc(), FULL, both, str(out), data_weights=[3, 1], device="cpu").fit()
    with pytest.raises(ValueError, match="the run used 2 data directories, this session passes 1"):
        Trainer(TensegrityLM(cfg), _tc(), FULL, [corpus_dir], str(out), device="cpu").fit()
    Trainer(TensegrityLM(cfg), _tc(), FULL, both, str(out), data_weights=[2, 2], device="cpu").fit()   # normalised: same mix


def test_data_must_be_consistent_and_must_not_change_during_a_run(corpus_dir, tmp_path):
    d = tmp_path / "d"
    prepare_data(str(_write(tmp_path / "s.txt")), str(d), val_fraction=0.1, doc_sep="<|endoftext|>", verbose=False)
    with open(d / "val.bin", "ab") as fh:                                  # size no longer matches the manifest
        fh.write(b"\x00\x00")
    with pytest.raises(DatasetError, match="inconsistent"):
        _fit(TensegrityLM(small_cfg()), _tc(), HALF, str(d), tmp_path / "r1")
    d2 = tmp_path / "d2"
    prepare_data(str(_write(tmp_path / "s2.txt")), str(d2), val_fraction=0.1, doc_sep="<|endoftext|>", verbose=False)
    swapped = {"done": False}

    def swap_data_mid_run(event, payload):
        if event == "log" and payload["step"] == 3 and not swapped["done"]:
            swapped["done"] = True
            data = np.fromfile(d2 / "train.bin", dtype=np.uint16)
            (data + 1).astype(np.uint16).tofile(d2 / "new.bin")
            os.replace(d2 / "new.bin", d2 / "train.bin")                   # replaced on disk, same name and size

    with pytest.raises(RuntimeError, match="must not change during a run"):
        Trainer(TensegrityLM(small_cfg()), _tc(), HALF, str(d2), str(tmp_path / "r2"), device="cpu",
                callbacks=[swap_data_mid_run]).fit()


def _write(path):
    path.write_text(make_text(120, seed=3), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------- trajectories
def test_evaluation_cadence_cannot_change_the_training_trajectory(corpus_dir, tmp_path):
    """Validation and sampling use their own RNGs: asking for more evaluations must not move a single weight."""
    cfg = small_cfg(layerdrop_max=0.3, dropout=0.1)
    torch.manual_seed(1)
    init = _weights(TensegrityLM(cfg))
    finals = []
    for i, kw in enumerate((dict(eval_interval=100), dict(eval_interval=2, eval_batches=3, sample_every_evals=1,
                                                         sample_len=8))):
        m = TensegrityLM(cfg)
        m.load_state_dict(init)
        _fit(m, _tc(ema_decay=0.9, **kw), FULL, corpus_dir, tmp_path / str(i))
        finals.append(_weights(m))
    assert _same(*finals)


def test_interrupted_run_matches_uninterrupted_in_every_respect(corpus_dir, tmp_path):
    """Weights, optimizer moments, EMA, LR trace, sampler state and counters -- not just the final weights."""
    cfg = small_cfg(layerdrop_max=0.3, dropout=0.1, byte_mix_kernel=3)
    tc = _tc(ema_decay=0.9, eval_interval=4)
    torch.manual_seed(5)
    init = _weights(TensegrityLM(cfg))
    a, b1, b2 = TensegrityLM(cfg), TensegrityLM(cfg), TensegrityLM(cfg)
    for m in (a, b1):
        m.load_state_dict(init)
    _fit(a, tc, FULL, corpus_dir, tmp_path / "A")
    _fit(b1, tc, HALF, corpus_dir, tmp_path / "B")
    torch.manual_seed(12345)                                              # a different global state must not matter
    _fit(b2, tc, FULL, corpus_dir, tmp_path / "B")
    assert _same(_weights(a), _weights(b2))
    ca, cb = (load_checkpoint(str(tmp_path / n / "ckpt_latest.pt")) for n in "AB")
    sa, sb = ca["optimizer_state_dict"]["state"], cb["optimizer_state_dict"]["state"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k][f], sb[k][f]) for k in sa for f in sa[k])
    for key in ("tokens_seen", "step", "completed_updates", "applied_updates", "skipped_steps"):
        assert ca[key] == cb[key], key
    assert ca["rng_state"]["dataset"] == cb["rng_state"]["dataset"]       # the next batch would be the same batch
    assert all(torch.equal(ca["ema_state_dict"]["shadow"][k], cb["ema_state_dict"]["shadow"][k])
               for k in ca["ema_state_dict"]["shadow"])
    lrs = [{r["step"]: r["lr"] for r in map(json.loads, open(tmp_path / n / "train_log.jsonl")) if "lr" in r} for n in "AB"]
    assert lrs[0] == lrs[1] and len(lrs[0]) == 12


def test_checkpoint_counters_provenance_and_environment(corpus_dir, tmp_path):
    calls = {"n": 0}

    def poison(model):
        calls["n"] += 1
        return model.token_emb.weight.sum() * (float("nan") if calls["n"] == 3 else 0.0)

    out = tmp_path / "r"
    tr = Trainer(TensegrityLM(small_cfg()), _tc(), FULL, corpus_dir, str(out), device="cpu", extra_loss=poison,
                 lineage={"init_from": "somewhere.pt"})
    res = tr.fit()
    ck = load_checkpoint(str(out / "ckpt_latest.pt"))
    assert ck["step"] == ck["completed_updates"] == res["completed_updates"] == 12
    assert ck["skipped_steps"] == 1 and ck["applied_updates"] == 11 == res["applied_updates"]
    assert ck["tokens_seen"] == 12 * TPS                                   # a skipped update still consumed its data
    assert ck["data"]["identity"]["0"]["fingerprint"] == data_identity(corpus_dir)["fingerprint"]
    assert res["fingerprints"] == {"0": data_identity(corpus_dir)["fingerprint"]}
    assert ck["lr_state"]["base_lrs"] == [3e-3, 3e-3] and ck["lr_state"]["horizon_steps"] == 12
    n_numel = sum(p.numel() for p in tr.raw_model.parameters())
    assert sum(g["numel"] for g in ck["optimizer_signature"]) == n_numel
    assert ck["environment"]["torch"] == torch.__version__ and ck["lineage"] == {"init_from": "somewhere.pt", "phases": []}


def test_resume_under_a_different_software_stack_warns(corpus_dir, tmp_path):
    out = tmp_path / "r"
    _fit(TensegrityLM(small_cfg()), _tc(), HALF, corpus_dir, out)
    path = str(out / "ckpt_latest.pt")
    ck = load_checkpoint(path)
    ck["environment"]["torch"] = "0.0.1"
    atomic_torch_save(ck, path)
    with pytest.warns(UserWarning, match=r"software stack \(torch 0\.0\.1"):
        _fit(TensegrityLM(small_cfg()), _tc(), FULL, corpus_dir, out)


# --------------------------------------------------------------- serialisation of wrapped models
def test_checkpoints_from_compiled_or_ddp_wrapped_models_are_clean(tmp_path):
    m = TensegrityLM(small_cfg(layerdrop_max=0.2))
    compiled = _compiled(m)
    assert next(iter(compiled.state_dict())).startswith("_orig_mod.")      # the pitfall: wrapper keys
    assert unwrap_model(compiled) is m and unwrap_model(m) is m
    p = str(tmp_path / "c.pt")
    save_checkpoint(p, compiled, None, m.cfg, 0, 1.0, rng_state=capture_rng_state(compiled))
    assert not any(k.startswith("_orig_mod.") for k in load_checkpoint(p)["model_state_dict"])
    loaded, _, ck = load_model(p)
    assert _same(_weights(loaded), _weights(m)) and "layerdrop" in ck["rng_state"]
    # a state dict written by other tooling with wrapper prefixes still loads
    sd = {"module._orig_mod." + k: v for k, v in m.state_dict().items()}
    assert set(normalize_state_dict(sd)) == set(m.state_dict())
    ck2 = load_checkpoint(p)
    ck2["model_state_dict"] = sd
    atomic_torch_save(ck2, p)
    assert _same(_weights(load_model(p)[0]), _weights(m))


def test_unknown_config_fields_are_errors_when_loading(tmp_path):
    with pytest.raises(ValueError, match="unknown TensegrityConfig field.*n_layr"):
        TensegrityConfig.from_dict({"n_layr": 2}, strict=True)
    with pytest.warns(UserWarning, match="ignoring unknown"):
        assert TensegrityConfig.from_dict({"n_layr": 2, "n_layer": 3}).n_layer == 3     # lenient mode still available
    m = TensegrityLM(small_cfg())
    p = str(tmp_path / "c.pt")
    save_checkpoint(p, m, None, m.cfg, 0, 1.0)
    ck = load_checkpoint(p)
    ck["config"]["field_from_the_future"] = 1
    atomic_torch_save(ck, p)
    with pytest.raises(ValueError, match="field_from_the_future"):
        load_model(p)
    with pytest.warns(UserWarning, match="ignoring unknown"):
        load_model(p, strict_config=False)


# ------------------------------------------------------------------------ compatibility with 2.0
def test_checkpoints_written_by_release_2_0_still_resume(corpus_dir, tmp_path):
    """The new provenance (data identity, hooks, eval suite, optimizer signature, environment, lineage,
    lr_state) is additive: a checkpoint without it resumes exactly as before -- with the checks that
    need it skipped, never with a crash and never by pretending to have verified what it cannot."""
    out = tmp_path / "r"
    _fit(TensegrityLM(small_cfg()), _tc(ema_decay=0.9), HALF, corpus_dir, out)
    path = str(out / "ckpt_latest.pt")
    ck = load_checkpoint(path)
    for key in ("data", "hooks", "eval_suite", "optimizer_signature", "environment", "lineage", "lr_state",
                "completed_updates", "applied_updates"):
        ck.pop(key)
    ck["format_train"] = 2
    atomic_torch_save(ck, path)
    old_style = _fit(TensegrityLM(small_cfg()), _tc(ema_decay=0.9), FULL, corpus_dir, out)
    assert old_style["step"] == 12
    torch.manual_seed(0)
    init = _weights(TensegrityLM(small_cfg()))
    a, b = TensegrityLM(small_cfg()), TensegrityLM(small_cfg())
    a.load_state_dict(init), b.load_state_dict(init)
    _fit(a, _tc(ema_decay=0.9), FULL, corpus_dir, tmp_path / "A")
    _fit(b, _tc(ema_decay=0.9), HALF, corpus_dir, tmp_path / "B")
    p2 = str(tmp_path / "B" / "ckpt_latest.pt")
    ck = load_checkpoint(p2)
    for key in ("data", "hooks", "eval_suite", "optimizer_signature", "environment", "lineage", "lr_state"):
        ck.pop(key)
    atomic_torch_save(ck, p2)
    c = TensegrityLM(small_cfg())
    _fit(c, _tc(ema_decay=0.9), FULL, corpus_dir, tmp_path / "B")
    assert _same(_weights(a), _weights(c))                              # still the same trajectory as an uninterrupted run
    final = load_checkpoint(p2)
    assert final["data"]["identity"] and final["lineage"]["phases"] == []   # and it upgrades itself to the new format
