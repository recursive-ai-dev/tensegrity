"""Growth as a transaction and continual learning as a measured comparison."""

import json
import os

import pytest
import torch

import tensegrity.workflow as W
from conftest import make_text, randomized, small_cfg
from tensegrity import (EWCState, EvalDomain, EvalSuite, GrowthError, GrowthGates, RecoverySpec, RunPlan, TensegrityLM,
                        TrainConfig, Trainer, compare_continual_strategies, grow_model, grow_transaction, prepare_data,
                        save_checkpoint, verify_roundtrips)
from tensegrity.checkpoint import load_checkpoint
from tensegrity.continual import tensors_fingerprint


def _tc(**over):
    kw = dict(batch_size=8, learning_rate=4e-3, min_lr=4e-4, eval_interval=1000, eval_batches=2, eval_batch_size=4,
              log_interval=1000, sample_every_evals=0, ckpt_interval_minutes=1e9, verbose=False, seed=7)
    kw.update(over)
    return TrainConfig(**kw)


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    d = tmp_path_factory.mktemp("wf")
    out = {}
    for name, seed, tweak in (("old", 1, lambda s: s), ("new", 2, lambda s: s.upper())):
        (d / f"{name}.txt").write_text(tweak(make_text(250, seed=seed)), encoding="utf-8")
        prepare_data(str(d / f"{name}.txt"), str(d / name), val_fraction=0.1, doc_sep="<|endoftext|>", verbose=False)
        out[name] = str(d / name)
    return out


@pytest.fixture(scope="module")
def parent(corpora, tmp_path_factory):
    torch.manual_seed(0)
    m = TensegrityLM(small_cfg())
    Trainer(m, _tc(), RunPlan(8 * 48, 40, 4, 28, 8, "wsd"), corpora["old"], str(tmp_path_factory.mktemp("p")),
            device="cpu").fit()
    return m


def _fresh_parent(parent):
    m = TensegrityLM(parent.cfg)
    m.load_state_dict(parent.state_dict())
    return m


# ------------------------------------------------------------------------------------- structure
def test_a_healthy_growth_is_accepted_with_every_check_recorded(parent):
    snap = tensors_fingerprint(parent.state_dict())
    d = grow_transaction(parent, new_n_embd=96, new_n_layer=6, ffn_growth_mult=1.3, verbose=False)
    assert d.accepted and d.reasons == [] and d.parent_unchanged and d.candidate is not None
    c = d.report.checks
    assert c["preservation_max_logit_diff"] < 2e-5 and c["trainability_min_max_grad"] > 0
    assert set(c["roundtrip_max_logit_diff"]) == {"config+state_dict", "checkpoint", "safetensors"}
    assert all(v < 1e-6 for v in c["roundtrip_max_logit_diff"].values())
    assert d.report.axes == ["depth", "ffn", "width"] and d.candidate_cfg.n_layer == 6
    json.dumps(d.to_dict())
    d.raise_if_rejected()
    assert "ACCEPTED" in d.summary()
    assert tensors_fingerprint(parent.state_dict()) == snap


def test_construction_failures_are_rejections_not_crashes(parent):
    d = grow_transaction(parent, target=parent.cfg.replace(n_layer=2), verbose=False)
    assert not d.accepted and d.candidate is None and "construction failed" in d.reasons[0] and "cannot shrink" in d.reasons[0]
    with pytest.raises(GrowthError, match="rejected"):
        d.raise_if_rejected()
    assert "REJECTED" in d.summary()


def test_the_transaction_rejects_a_grower_that_breaks_the_function(parent, monkeypatch):
    real = W.grow_model

    def sabotaged(*a, **k):
        m, c, r = real(*a, **k)
        with torch.no_grad():
            m.blocks[0].ffn.w_down.weight[:48, :48] += 0.05            # a carried weight silently altered
        return m, c, r
    monkeypatch.setattr(W, "grow_model", sabotaged)
    d = grow_transaction(parent, new_n_layer=5, verbose=False)
    assert not d.accepted and any("function not preserved" in r for r in d.reasons)
    assert d.candidate is not None and d.parent_unchanged                   # inspectable; the parent is untouched


def test_the_transaction_rejects_state_that_lives_only_in_a_python_attribute(parent, monkeypatch):
    """The v1 grower changed RMSNorm eps as an attribute: invisible to state_dict and config, so a reload drifted."""
    real = W.grow_model

    def sabotaged(*a, **k):
        m, c, r = real(*a, **k)
        m.blocks[1].ln1.eps = 3e-3                                           # not in cfg.norm_eps, not in the state dict
        return m, c, r
    monkeypatch.setattr(W, "grow_model", sabotaged)
    d = grow_transaction(parent, new_n_layer=5, verbose=False)
    assert not d.accepted and any("does not survive serialization" in r for r in d.reasons)
    with pytest.raises(GrowthError, match="serialization"):
        verify_roundtrips(d.candidate)
    ok = _fresh_parent(parent)
    assert max(verify_roundtrips(ok).values()) == 0.0


def test_the_transaction_rejects_dead_capacity(parent, monkeypatch):
    real = W.grow_model

    def v1_style(*a, **k):
        m, c, r = real(*a, **k)
        cfg = parent.cfg
        D0, hd = cfg.n_embd, cfg.head_dim
        with torch.no_grad():
            for b in m.blocks:
                b.ffn.w_gate.weight[:, D0:].zero_(); b.ffn.w_gate.weight[cfg.ffn_hidden:].zero_()
                b.ffn.w_up.weight[:, D0:].zero_(); b.ffn.w_up.weight[cfg.ffn_hidden:].zero_()
                b.attn.q_proj.weight[:, D0:].zero_(); b.attn.q_proj.weight[cfg.n_head * hd:].zero_()
                b.attn.k_proj.weight[:, D0:].zero_(); b.attn.k_proj.weight[cfg.n_kv_head * hd:].zero_()
                b.attn.v_proj.weight[:, D0:].zero_(); b.attn.v_proj.weight[cfg.n_kv_head * hd:].zero_()
        return m, c, r
    monkeypatch.setattr(W, "grow_model", v1_style)
    d = grow_transaction(parent, new_n_embd=96, verbose=False)
    assert not d.accepted and any("cannot learn" in r for r in d.reasons)
    assert grow_transaction(parent, new_n_embd=96, gates=GrowthGates(require_trainable=False), verbose=False).accepted


# ------------------------------------------------------------------------------- recovery + gates
def test_recovery_adapts_to_the_new_domain_and_the_gates_decide(parent, corpora, tmp_path):
    suite = EvalSuite([EvalDomain("old", corpora["old"], "retain", n_batches=4, batch_size=4),
                       EvalDomain("new", corpora["new"], "adapt", n_batches=4, batch_size=4)])
    p = _fresh_parent(parent)
    snap = tensors_fingerprint(p.state_dict())
    ewc = EWCState.estimate(p, corpora["old"], 20.0, n_sequences=4, seq_len=32)
    rec = RecoverySpec([corpora["new"], corpora["old"]], [0.7, 0.3], steps=40, learning_rate=4e-3, distill_weight=2.0, ewc=ewc)
    gates = GrowthGates(max_retain_regression_bpb=5.0, min_adapt_gain_bpb=0.1)       # generous on retention, real on adaptation
    d = grow_transaction(p, suite=suite, recovery=rec, gates=gates, new_n_layer=6, new_n_embd=96, run_dir=str(tmp_path / "g"),
                         verbose=False)
    assert d.accepted, d.summary()
    assert d.parent_eval is not None and d.grown_eval is not None and d.final_eval is not None
    for k in ("old", "new"):                                                         # right after growth: identical to the parent
        assert d.grown_eval.domains[k]["bpb_bytes"] == pytest.approx(d.parent_eval.domains[k]["bpb_bytes"], abs=1e-3)
    assert d.parent_eval.domains["new"]["bpb_bytes"] - d.final_eval.domains["new"]["bpb_bytes"] >= 0.1
    assert d.recovery["steps"] == 40 and d.recovery["ewc"] is True and d.recovery["distill_weight"] == 2.0
    assert d.parent_unchanged and tensors_fingerprint(p.state_dict()) == snap
    ck = load_checkpoint(str(tmp_path / "g" / "recovery" / "ckpt_latest.pt"))        # the recovery run is a recorded run
    assert ck["hooks"]["batch_loss"]["type"] == "DistillLoss" and ck["hooks"]["extra_loss"]["type"] == "EWCState"
    assert ck["lineage"]["recovery_of"] == "grow_transaction"
    # same candidate, stricter gates -> the SAME numbers lead to rejection, with the reason spelled out
    d2 = grow_transaction(_fresh_parent(parent), suite=suite, recovery=rec, new_n_layer=6, new_n_embd=96, verbose=False,
                          gates=GrowthGates(max_retain_regression_bpb=-5.0, min_adapt_gain_bpb=100.0))
    assert not d2.accepted
    assert any(r.startswith("retention: 'old'") for r in d2.reasons) and any(r.startswith("adaptation: 'new'") for r in d2.reasons)
    assert d2.final_eval.domains["new"]["bpb_bytes"] == pytest.approx(d.final_eval.domains["new"]["bpb_bytes"], rel=1e-6)
    with pytest.raises(ValueError, match="needs an EvalSuite"):
        grow_transaction(_fresh_parent(parent), recovery=rec, new_n_layer=5, verbose=False)


def test_without_recovery_only_nothing_regressed_is_judged(parent, corpora):
    suite = EvalSuite.from_dirs({"old": corpora["old"], "new": corpora["new"]}, roles={"new": "adapt"}, n_batches=2, batch_size=4)
    d = grow_transaction(_fresh_parent(parent), suite=suite, new_n_layer=5, verbose=False,
                         gates=GrowthGates(min_adapt_gain_bpb=1.0))                  # unreachable, but nothing was trained
    assert d.accepted and d.recovery is None
    for k in ("old", "new"):
        assert d.final_eval.domains[k]["bpb_bytes"] == pytest.approx(d.parent_eval.domains[k]["bpb_bytes"], abs=1e-3)


# ------------------------------------------------------------------------------ continual comparison
def test_continual_strategies_are_compared_under_a_matched_budget(parent, corpora, tmp_path):
    kw = dict(steps=40, batch_size=8, learning_rate=4e-3, replay_fraction=0.4, ewc_lambda=50.0, ewc_sequences=4,
              seeds=(0, 1), eval_batches=4, verbose=False)
    rep = compare_continual_strategies(_fresh_parent(parent), corpora["old"], corpora["new"], run_dir=str(tmp_path / "cl"), **kw)
    assert list(rep.arms) == ["new_only", "replay", "replay_ewc"]
    # identical budget in every arm: same steps, same tokens consumed
    runs = [load_checkpoint(str(tmp_path / "cl" / f"{a}_seed{s}" / "ckpt_latest.pt")) for a in rep.arms for s in (0, 1)]
    assert {(c["completed_updates"], c["tokens_seen"]) for c in runs} == {(40, 40 * 8 * 48)}
    na, rp = rep.arms["new_only"], rep.arms["replay"]
    assert all(len(x) == 2 for x in (na.forgetting, na.new_gain, rp.old_bpb))
    assert rep.before["old"] > 0 and rep.before["new"] > rep.before["old"]            # the new domain is harder at the start
    for a in rep.arms.values():                                                        # the definitions hold
        assert a.forgetting == pytest.approx([o - rep.before["old"] for o in a.old_bpb])
        assert a.new_gain == pytest.approx([rep.before["new"] - n for n in a.new_bpb])
    # replay of the old corpus protects it: less forgetting than training on the new corpus alone
    assert rp.forgetting_mean_std[0] < na.forgetting_mean_std[0]
    # ... and costs some new-domain progress at equal total tokens (it sees fewer new-domain tokens)
    assert na.new_gain_mean_std[0] > 0 and rp.new_gain_mean_std[0] > 0
    nd = rep.non_dominated()
    assert nd and set(nd) <= set(rep.arms)
    table = rep.table()
    assert "forgetting" in table and "non-dominated" in table and all(a in table for a in rep.arms)
    saved = json.load(open(tmp_path / "cl" / "continual_report.json"))
    assert saved["non_dominated"] == nd and saved["budget"]["steps"] == 40 and set(saved["arms"]) == set(rep.arms)
    again = compare_continual_strategies(_fresh_parent(parent), corpora["old"], corpora["new"], **kw)
    assert json.dumps(again.to_dict()["arms"], sort_keys=True) == json.dumps(rep.to_dict()["arms"], sort_keys=True)   # reproducible


def test_continual_comparison_accepts_a_checkpoint_and_validates_arms(parent, corpora, tmp_path):
    path = str(tmp_path / "base.pt")
    save_checkpoint(path, parent, None, parent.cfg, 0, 1.0)
    rep = compare_continual_strategies(path, corpora["old"], corpora["new"], steps=6, arms=("new_only", "replay"),
                                       batch_size=4, eval_batches=2, verbose=False)
    assert list(rep.arms) == ["new_only", "replay"] and "replay_ewc" not in rep.table()
    with pytest.raises(ValueError, match="unknown arm"):
        compare_continual_strategies(path, corpora["old"], corpora["new"], arms=("everything",), verbose=False)
