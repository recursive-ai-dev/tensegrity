"""Growth is audited, not trusted: every old state entry is verified in the grown model; reports are
recorded; EWC reference state follows the transformation; Fisher kinds are what they claim to be."""

import json

import pytest
import torch
import torch.nn as nn

from conftest import randomized, small_cfg
from tensegrity import (DistillLoss, EWCState, GrowthError, GrowthReport, TensegrityLM, TrainConfig, Trainer, RunPlan,
                        compute_fisher_diagonal, ewc_penalty, expand_reference_state, grow_model, load_model,
                        verify_growth_mapping)
from tensegrity.growth import _rescale_map


def _compiled(model):
    """torch.compile wraps lazily; only its availability can differ between torch/python builds."""
    try:
        return torch.compile(model)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"torch.compile unavailable in this environment: {e}")


def _state(m):
    return {k: v.clone() for k, v in m.state_dict().items()}


# --------------------------------------------------------------------------------- the audit
@pytest.mark.parametrize("kw", [dict(new_n_layer=6), dict(new_n_embd=96), dict(ffn_growth_mult=1.5),
                                dict(new_n_embd=96, new_n_layer=6, ffn_growth_mult=1.3, new_block_size=96)])
@pytest.mark.parametrize("flags", [dict(), dict(embed_scale=False, bias=True, byte_mix_kernel=3, x0_mix=True)])
def test_every_old_state_entry_is_found_in_the_grown_model(kw, flags):
    cfg = small_cfg(**flags)
    old = randomized(TensegrityLM(cfg), 0.05, seed=3)
    new, ncfg, rep = grow_model(old, cfg, return_report=True, **kw)
    n_old = len([k for k in old.state_dict() if k != "lm_head.weight"])
    assert rep.carried_entries == n_old and rep.max_carry_deviation == 0.0 or rep.rescale
    assert all(k.split(".")[1].isdigit() and int(k.split(".")[1]) >= cfg.n_layer for k in rep.new_only)
    assert rep.new_params == new.num_parameters() == ncfg.n_params() and rep.old_params == old.num_parameters()
    assert set(rep.rescale) == set(_rescale_map(cfg, ncfg))
    json.dumps(rep.to_dict())                                            # storable next to a checkpoint
    assert rep.summary().startswith("growth [") and (("width" in rep.axes) == ("new_n_embd" in kw))


def test_the_report_names_the_rescaling_that_width_growth_applies():
    for embed_scale in (True, False):
        cfg = small_cfg(embed_scale=embed_scale)
        _, _, rep = grow_model(randomized(TensegrityLM(cfg)), cfg, new_n_embd=96, return_report=True)
        assert rep.rescale["ln_f.weight"] == pytest.approx((48 / 96) ** 0.5)
        assert rep.rescale["blocks.0.ln1.weight"] == pytest.approx((48 / 96) ** 0.5)
        assert ("blocks.0.attn_gain" in rep.rescale) == embed_scale
        if embed_scale:
            assert rep.rescale["blocks.3.ffn_gain"] == pytest.approx((96 / 48) ** 0.5)
    _, _, depth = grow_model(randomized(TensegrityLM(small_cfg())), new_n_layer=6, return_report=True)
    assert depth.rescale == {} and depth.axes == ["depth"] and len(depth.new_only) > 0


def test_the_verifier_catches_lost_altered_and_unaccounted_state():
    cfg = small_cfg()
    old = randomized(TensegrityLM(cfg), 0.05, seed=4)
    new, _, rep = grow_model(old, cfg, new_n_embd=96, return_report=True)
    verify_growth_mapping(old, new, rep.rescale)                          # the honest transplant passes
    # (1) the v1 grower forgot the Q/K-norm scales: simulate a transplant that did
    broken = TensegrityLM(new.cfg)
    broken.load_state_dict(new.state_dict())
    with torch.no_grad():
        broken.blocks[1].attn.q_norm.weight.fill_(1.0)
    with pytest.raises(GrowthError, match=r"blocks\.1\.attn\.q_norm\.weight: carried values differ"):
        verify_growth_mapping(old, broken, rep.rescale)
    # (2) a carried value altered in the corner of a bigger tensor
    broken.load_state_dict(new.state_dict())
    with torch.no_grad():
        broken.blocks[0].ffn.w_up.weight[0, 0] += 1e-3
    with pytest.raises(GrowthError, match=r"w_up\.weight: carried values differ"):
        verify_growth_mapping(old, broken, rep.rescale)
    # (3) forgetting the declared rescale is also a violation (norm weights are x sqrt(D0/D1))
    with pytest.raises(GrowthError, match="ln1.weight"):
        verify_growth_mapping(old, new, {})
    # (4) state the old model has and the grown one lacks -> the operation fails, it does not drop it silently
    extended = randomized(TensegrityLM(cfg))
    extended.blocks[0].register_parameter("future_gain", nn.Parameter(torch.full((), 2.0)))
    with pytest.raises(GrowthError, match=r"blocks\.0\.future_gain: present in the old model, absent"):
        grow_model(extended, cfg, new_n_layer=5)
    # (5) a carried tensor may not shrink; tying must survive
    small = TensegrityLM(cfg)
    with pytest.raises(GrowthError, match="may only grow"):
        verify_growth_mapping(new, small, {})
    untied = TensegrityLM(new.cfg)
    untied.load_state_dict(new.state_dict())
    untied.lm_head.weight = nn.Parameter(untied.lm_head.weight.detach().clone())
    with pytest.raises(GrowthError, match="no longer tied"):
        verify_growth_mapping(old, untied, rep.rescale)


def test_growth_never_modifies_or_moves_the_parent():
    cfg = small_cfg()
    old = randomized(TensegrityLM(cfg))
    before, dev = _state(old), next(old.parameters()).device
    grow_model(old, cfg, new_n_embd=96, new_n_layer=6, device="cpu")
    assert next(old.parameters()).device == dev
    assert all(torch.equal(before[k], v) for k, v in old.state_dict().items())
    # a torch.compile-wrapped parent is accepted (state_dict keys carry the wrapper prefix otherwise)
    grown, _ = grow_model(_compiled(old), cfg, new_n_layer=5)
    assert not any(k.startswith("_orig_mod") for k in grown.state_dict())


# ------------------------------------------------------------------------------------ EWC
@pytest.mark.parametrize("embed_scale", [True, False])
def test_ewc_penalty_is_zero_at_the_instant_of_growth_and_matches_in_function_space(corpus_dir, embed_scale):
    cfg = small_cfg(embed_scale=embed_scale)
    old = randomized(TensegrityLM(cfg), 0.05, seed=2)
    fisher, anchor = compute_fisher_diagonal(old, corpus_dir, "cpu", n_sequences=6, seq_len=32)
    new, _, rep = grow_model(old, cfg, new_n_embd=96, new_n_layer=6, return_report=True)
    # without the rescale map the rescaled norm weights are pulled back: a spurious penalty on an unchanged function
    naive = expand_reference_state(fisher, anchor, new)
    assert float(ewc_penalty(new, *naive)) > 0
    f2, p2 = expand_reference_state(fisher, anchor, new, rep)
    assert float(ewc_penalty(new, f2, p2)) == pytest.approx(0.0, abs=1e-12)
    # theta' = a theta  =>  F' = F / a^2: the SAME perturbation of the function costs the SAME penalty
    name, a = "blocks.0.ln1.weight", rep.rescale["blocks.0.ln1.weight"]
    delta = 0.07
    with torch.no_grad():
        old.blocks[0].ln1.weight.add_(delta)
        new.blocks[0].ln1.weight[:48].add_(a * delta)
    p_old = float((fisher[name] * delta ** 2).sum())
    p_new = float((f2[name][:48] * (a * delta) ** 2).sum())
    assert p_new == pytest.approx(p_old, rel=1e-5) and p_old > 0
    assert torch.equal(f2[name][48:], torch.zeros(48))                  # new channels: unconstrained
    # the report object itself is accepted wherever the mapping is
    f3, _ = expand_reference_state(fisher, anchor, new, rep.rescale)
    assert torch.equal(f3[name], f2[name])


def test_fisher_kinds_are_labelled_and_differ(corpus_dir):
    m = randomized(TensegrityLM(small_cfg()), 0.02, seed=1)
    emp, snap = compute_fisher_diagonal(m, corpus_dir, "cpu", n_sequences=12, seq_len=32, kind="empirical")
    smp, _ = compute_fisher_diagonal(m, corpus_dir, "cpu", n_sequences=12, seq_len=32, kind="sampled")
    smp2, _ = compute_fisher_diagonal(m, corpus_dir, "cpu", n_sequences=12, seq_len=32, kind="sampled")
    assert emp.keys() == smp.keys() == snap.keys()
    assert all(torch.equal(smp[k], smp2[k]) for k in smp)                # sampling is seeded -> reproducible
    assert any(not torch.allclose(emp[k], smp[k]) for k in emp)
    assert all((f >= 0).all() and torch.isfinite(f).all() for f in smp.values())
    with pytest.raises(ValueError, match="kind must be"):
        compute_fisher_diagonal(m, corpus_dir, "cpu", n_sequences=2, kind="exact")


def test_sampled_labels_have_zero_mean_score_empirical_labels_do_not(corpus_dir):
    """The defining difference: under labels drawn from the model's own distribution the gradient of
    the log-likelihood averages to zero (E[score] = 0); under the data's labels it does not."""
    torch.manual_seed(0)
    m = TensegrityLM(small_cfg()).eval()
    from tensegrity import ByteDataset
    batches = ByteDataset(corpus_dir).fixed_batches("train", 96, 1, 32, "cpu", seed=3)
    gen = torch.Generator().manual_seed(5)
    params = [p for p in m.parameters()]

    def mean_over_rms(sampled: bool) -> float:
        total = [torch.zeros_like(p) for p in params]
        sq = 0.0
        for x, y in batches:
            m.zero_grad()
            logits, _, _ = m(x)
            flat = logits.reshape(-1, logits.size(-1))
            tgt = (torch.multinomial(torch.softmax(flat.detach(), -1), 1, generator=gen).view(-1) if sampled
                   else y.view(-1))
            torch.nn.functional.cross_entropy(flat, tgt).backward()
            for t, p in zip(total, params):
                if p.grad is not None:                       # e.g. the confidence head: unused without targets
                    t += p.grad
            sq += sum(float(p.grad.pow(2).sum()) for p in params if p.grad is not None)
        mean_sq = sum(float((t / len(batches)).pow(2).sum()) for t in total)
        return (mean_sq / (sq / len(batches))) ** 0.5

    assert mean_over_rms(sampled=True) < 0.5 * mean_over_rms(sampled=False)


def test_ewc_state_estimate_save_load_fingerprint_and_growth(corpus_dir, tmp_path):
    cfg = small_cfg()
    m = randomized(TensegrityLM(cfg), 0.05, seed=2)
    with pytest.raises(TypeError, match="needs a model object"):
        EWCState.estimate("runs/x/ckpt_best.pt", corpus_dir, 10.0)                  # the notebook's v1 crash, now explained
    st = EWCState.estimate(m, corpus_dir, 25.0, kind="sampled", n_sequences=6, seq_len=32)
    assert st.meta["kind"] == "sampled" and "sequence" in st.meta["unit"] and st.meta["data_fingerprint"].startswith("sha256:")
    assert float(st(m)) == 0.0                                           # at the anchor
    with torch.no_grad():
        m.blocks[0].attn.q_proj.weight.add_(0.1)
    assert float(st(m)) == pytest.approx(25.0 * float(st.penalty(m))) and float(st(m)) > 0
    p = str(tmp_path / "ewc.pt")
    st.save(p)
    back = EWCState.load(p)                                              # loads under weights_only=True
    assert back.fingerprint() == st.fingerprint() and back.lam == 25.0 and back.meta == st.meta
    assert EWCState(st.fisher, st.params, 26.0, st.meta).fingerprint() != st.fingerprint()
    other = EWCState({k: v * 2 for k, v in st.fisher.items()}, st.params, 25.0, st.meta)
    assert other.fingerprint() != st.fingerprint()
    with pytest.raises(ValueError):
        EWCState(st.fisher, {}, 1.0)
    # carried through growth: still zero penalty on the unchanged function
    old = randomized(TensegrityLM(cfg), 0.05, seed=2)
    st2 = EWCState.estimate(old, corpus_dir, 25.0, n_sequences=6, seq_len=32)
    new, _, rep = grow_model(old, cfg, new_n_embd=96, return_report=True)
    grown_state = st2.expanded(new, rep)
    assert float(grown_state(new)) == pytest.approx(0.0, abs=1e-12) and grown_state.lam == 25.0
    assert grown_state.meta["carried_through_growth_from"] == st2.fingerprint()


# ------------------------------------------------------------------------------ distillation
def test_distill_loss_math_masking_and_gradients():
    torch.manual_seed(0)
    cfg = small_cfg()
    teacher = randomized(TensegrityLM(cfg), 0.05, seed=1)
    student = TensegrityLM(cfg)
    x = torch.randint(0, 256, (3, 24))
    y = torch.roll(x, -1, 1)
    loss_fn = DistillLoss(teacher, weight=2.0, temperature=1.5)
    logits, _, _ = student(x)
    got = loss_fn(student, x, y, logits)
    t_log, _, _ = teacher(x)
    t = torch.softmax(t_log / 1.5, -1)
    want = 2.0 * 1.5 ** 2 * (t * (t.log() - torch.log_softmax(logits / 1.5, -1))).sum(-1).mean()
    assert float(got) == pytest.approx(float(want), rel=1e-5)
    got.backward()
    assert all(p.grad is None for p in teacher.parameters()) and any(p.grad is not None for p in student.parameters())
    y_masked = y.clone()
    y_masked[:, 12:] = -1                                                # half the positions have no target
    half = loss_fn(student, x, y_masked, logits)
    manual = 2.0 * 1.5 ** 2 * (t * (t.log() - torch.log_softmax(logits / 1.5, -1))).sum(-1)[:, :12].mean()
    assert float(half) == pytest.approx(float(manual), rel=1e-5)
    same = DistillLoss(teacher)(teacher, x, y, teacher(x)[0])
    assert float(same) == pytest.approx(0.0, abs=1e-6)                   # student == teacher -> nothing to distill
    with pytest.raises(ValueError):
        DistillLoss(teacher, temperature=0.0)
    long_x = torch.randint(0, 256, (1, cfg.block_size))
    short_teacher = DistillLoss(TensegrityLM(small_cfg(block_size=16, sliding_window=8)))
    with pytest.raises(ValueError, match="shorter than the training sequences"):
        short_teacher(student, long_x, long_x, student(long_x)[0])
    assert DistillLoss(teacher).fingerprint() == DistillLoss(teacher).fingerprint() != DistillLoss(student).fingerprint()


def _tc(**over):
    kw = dict(batch_size=8, learning_rate=3e-3, min_lr=3e-4, eval_interval=100, eval_batches=2, eval_batch_size=4,
              log_interval=1, sample_every_evals=0, ckpt_interval_minutes=1e9, verbose=False, seed=7)
    kw.update(over)
    return TrainConfig(**kw)


def test_trainer_hooks_log_their_terms_and_are_verified_on_resume(corpus_dir, corpus_dir2, tmp_path):
    cfg = small_cfg()
    parent = randomized(TensegrityLM(cfg), 0.05, seed=2)
    ewc = EWCState.estimate(parent, corpus_dir, 5.0, n_sequences=4, seq_len=32)
    half = RunPlan(8 * 48, 6, 2, 8, 2, "wsd")
    full = RunPlan(8 * 48, 12, 2, 8, 2, "wsd")
    out = str(tmp_path / "r")
    logs = []
    student = TensegrityLM(cfg)
    student.load_state_dict(parent.state_dict())
    Trainer(student, _tc(), half, [corpus_dir2, corpus_dir], out, data_weights=[0.7, 0.3], device="cpu", extra_loss=ewc,
            batch_loss=DistillLoss(parent, 1.0), callbacks=[lambda e, p: logs.append(p) if e == "log" else None]).fit()
    assert all("extra_loss" in r and "batch_loss" in r for r in logs) and logs[0]["batch_loss"] == pytest.approx(0.0, abs=1e-6)
    kw = dict(data_weights=[0.7, 0.3], device="cpu")
    # the same constraint resumes; a different EWC strength or a different teacher is a policy change
    other_ewc = EWCState(ewc.fisher, ewc.params, 6.0, ewc.meta)
    with pytest.raises(ValueError, match="extra_loss"):
        Trainer(TensegrityLM(cfg), _tc(), full, [corpus_dir2, corpus_dir], out, extra_loss=other_ewc,
                batch_loss=DistillLoss(parent, 1.0), **kw).fit()
    with pytest.raises(ValueError, match="batch_loss"):
        Trainer(TensegrityLM(cfg), _tc(), full, [corpus_dir2, corpus_dir], out, extra_loss=ewc,
                batch_loss=DistillLoss(TensegrityLM(cfg), 1.0), **kw).fit()
    with pytest.raises(ValueError, match="batch_loss"):
        Trainer(TensegrityLM(cfg), _tc(), full, [corpus_dir2, corpus_dir], out, extra_loss=ewc, **kw).fit()
    r = Trainer(TensegrityLM(cfg), _tc(), full, [corpus_dir2, corpus_dir], out, extra_loss=ewc,
                batch_loss=DistillLoss(parent, 1.0), **kw).fit()
    assert r["step"] == 12


def test_distillation_keeps_a_grown_candidate_closer_to_its_parent(corpus_dir, corpus_dir2, tmp_path):
    """Recovery after growth: adapting the candidate to a new corpus drifts the function away from the
    parent's; the distillation term measurably reduces that drift. (An approximate recovery procedure --
    this does not claim the drift is harmless.)"""
    cfg = small_cfg()
    torch.manual_seed(0)
    parent = TensegrityLM(cfg)
    Trainer(parent, _tc(learning_rate=4e-3), RunPlan(8 * 48, 40, 4, 28, 8, "wsd"), corpus_dir, str(tmp_path / "p"),
            device="cpu").fit()
    parent.eval()
    drifts = {}
    for weight in (0.0, 20.0):
        cand, ccfg = grow_model(parent, cfg, new_n_layer=6, new_n_embd=96)
        Trainer(cand, _tc(learning_rate=4e-3, seed=3), RunPlan(8 * 48, 30, 3, 20, 7, "wsd"), corpus_dir2,
                str(tmp_path / f"c{weight}"), device="cpu",
                batch_loss=DistillLoss(parent, weight) if weight else None).fit()
        cand.eval()
        x = torch.randint(0, 256, (8, 40), generator=torch.Generator().manual_seed(9))
        with torch.no_grad():
            tp = torch.log_softmax(parent(x)[0], -1)
            cp = torch.log_softmax(cand(x)[0], -1)
            drifts[weight] = float((tp.exp() * (tp - cp)).sum(-1).mean())
    assert drifts[20.0] < 0.8 * drifts[0.0], drifts
