"""Evaluation answers the questions the architecture raises: exact bits per byte, named domains with a
defined aggregate, what the attention layout can reach, what a model actually retrieves, and how much
each sublayer moves the residual stream."""

import json
import math
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import make_text, randomized, small_cfg
from tensegrity import (ByteDataset, ByteTokenizer, EvalDomain, EvalSuite, RunPlan, TensegrityConfig, TensegrityLM,
                        TrainConfig, Trainer, context_probe, evaluate, evaluate_batches, format_probe, get_preset_config,
                        prepare_data, residual_report, LADDER)
from tensegrity.checkpoint import load_checkpoint
from tensegrity.manifest import DatasetError

LN2 = math.log(2)
EOS = ByteTokenizer.EOS


# ------------------------------------------------------------------------------ exact accounting
def _reference(model, batches):
    """Independent re-derivation: per-position NLL from the logits, split by target class."""
    nll_b = nll_s = 0.0
    n_b = n_s = 0
    for x, y in batches:
        with torch.no_grad():
            logits, _, _ = model.eval()(x)
        ce = -torch.log_softmax(logits.double(), -1).gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        for c, t in zip(ce.flatten().tolist(), y.flatten().tolist()):
            if 0 <= t < 256:
                nll_b += c; n_b += 1
            elif t >= 256:
                nll_s += c; n_s += 1
    return nll_b, n_b, nll_s, n_s


def test_bits_per_byte_separates_bytes_from_document_markers():
    torch.manual_seed(0)
    model = randomized(TensegrityLM(small_cfg()), 0.05)
    x = torch.randint(0, 256, (4, 32))
    y = torch.randint(0, 256, (4, 32))
    y[:, 7] = EOS                                         # one document boundary per row
    y[0, :4] = -1                                         # ignored positions count as nothing
    r = evaluate_batches(model, [(x, y)], "cpu", torch.float32)
    nb, n_b, ns, n_s = _reference(model, [(x, y)])
    assert (r["n_bytes"], r["n_special"], r["n_tokens"]) == (n_b, n_s, n_b + n_s)
    assert r["n_special"] == 4 and r["n_bytes"] == 4 * 32 - 4 - 4 and r["n_tokens"] == 4 * 32 - 4
    assert r["bpb_bytes"] == pytest.approx(nb / n_b / LN2, rel=1e-5)
    assert r["bits_per_special"] == pytest.approx(ns / n_s / LN2, rel=1e-5)
    assert r["loss"] == pytest.approx((nb + ns) / (n_b + n_s), rel=1e-5)
    assert r["bpb"] == pytest.approx(r["loss"] / LN2) and r["ppl"] == pytest.approx(math.exp(r["loss"]))
    assert r["bpb_bytes"] != pytest.approx(r["bpb"], rel=1e-4)       # the legacy number mixes the two populations
    # the same loss the model reports for the same targets (the number checkpoints are selected on is unchanged)
    with torch.no_grad():
        _, losses, _ = model.eval()(x, y)
    assert r["loss"] == pytest.approx(float(losses["ce_loss"]), rel=1e-5)


def test_without_special_targets_both_bpb_definitions_agree(corpus_dir):
    model = TensegrityLM(small_cfg())
    ds = ByteDataset(corpus_dir)
    x = torch.randint(0, 256, (2, 24))
    r = evaluate_batches(model, [(x, torch.roll(x, -1, 1))], "cpu", torch.float32)
    assert r["n_special"] == 0 and math.isnan(r["bits_per_special"]) and r["bpb"] == pytest.approx(r["bpb_bytes"])
    only_special = evaluate_batches(model, [(x, torch.full_like(x, EOS))], "cpu", torch.float32)
    assert math.isnan(only_special["bpb_bytes"]) and only_special["n_bytes"] == 0
    full = evaluate(model, corpus_dir, model.cfg, "cpu", n_batches=3, batch_size=4)
    assert full["n_special"] > 0 and {"bpb_bytes", "bits_per_special", "nll_bits_bytes"} <= set(full)   # real data has EOS markers


# ------------------------------------------------------------------------------- receptive field
def test_receptive_field_matches_the_reviewed_example_and_the_architecture():
    four_local = TensegrityConfig(n_layer=4, n_embd=64, n_head=4, n_kv_head=2, block_size=1024, sliding_window=124,
                                  global_every=8)
    assert four_local.n_global_layers == 0 and four_local.receptive_field() == 1 + 4 * (124 - 1) == 493
    assert "LOCAL-ONLY: sees at most 493 of 1024" in four_local.describe() and "reach=493" in four_local.describe()
    assert four_local.replace(global_every=4).receptive_field() == 1024                 # one global layer sees everything
    assert four_local.replace(byte_mix_kernel=4).receptive_field() == 496                # ByteMix adds K-1 at the input
    assert four_local.replace(block_size=256).receptive_field() == 256                   # never beyond the context
    assert "LOCAL-ONLY" not in four_local.replace(global_every=4).describe()
    for p in LADDER:                                                                     # every shipped rung can see its whole context
        c = get_preset_config(p.name)
        assert c.n_global_layers >= 1 and c.receptive_field() == c.block_size, p.name
    for name in ("base15m", "balanced135m", "highintel400m"):
        c = get_preset_config(name)
        assert c.receptive_field() == c.block_size, name


# --------------------------------------------------------------------------------------- suites
@pytest.fixture(scope="module")
def two_domains(tmp_path_factory):
    d = tmp_path_factory.mktemp("domains")
    out = {}
    for name, seed, tweak in (("stories", 1, lambda s: s), ("shouting", 2, lambda s: s.upper())):
        src = d / f"{name}.txt"
        src.write_text(tweak(make_text(300, seed=seed)), encoding="utf-8")
        prepare_data(str(src), str(d / name), val_fraction=0.1, doc_sep="<|endoftext|>", verbose=False)
        out[name] = str(d / name)
    return out


def test_suite_reports_each_domain_and_defines_its_aggregate(two_domains):
    torch.manual_seed(0)
    model = randomized(TensegrityLM(small_cfg()), 0.05)
    suite = EvalSuite([EvalDomain("stories", two_domains["stories"], "retain", weight=3.0, n_batches=4, batch_size=4),
                       EvalDomain("shouting", two_domains["shouting"], "adapt", weight=1.0, n_batches=2, batch_size=4)])
    r = suite.evaluate(model, "cpu", torch.float32)
    T = model.cfg.block_size
    for d in suite.domains:                                  # each domain equals evaluate_batches on its own fixed windows
        ds = ByteDataset(d.data_dir)
        want = evaluate_batches(model, ds.fixed_batches("val", d.n_batches, d.batch_size, T, "cpu"), "cpu", torch.float32)
        assert r.domains[d.name]["bpb_bytes"] == pytest.approx(want["bpb_bytes"], rel=1e-6)
        assert r.domains[d.name]["loss"] == pytest.approx(want["loss"], rel=1e-6)
    a, b = r.domains["stories"], r.domains["shouting"]
    assert a["bpb_bytes"] != pytest.approx(b["bpb_bytes"], abs=1e-3)
    assert r.macro_loss == pytest.approx((3 * a["loss"] + b["loss"]) / 4)                  # weighted mean of domain means
    assert r.macro_bpb_bytes == pytest.approx((3 * a["bpb_bytes"] + b["bpb_bytes"]) / 4)
    assert r.micro_bpb_bytes == pytest.approx((a["nll_bits_bytes"] + b["nll_bits_bytes"]) / (a["n_bytes"] + b["n_bytes"]))
    assert r.selection_loss == r.macro_loss and r.roles == {"stories": "retain", "shouting": "adapt"}
    assert "macro mean" in r.table() and "pooled" in r.table() and "adapt" in r.table()
    json.dumps(r.to_dict())
    again = suite.evaluate(model, "cpu", torch.float32)
    assert json.dumps(again.to_dict(), sort_keys=True) == json.dumps(r.to_dict(), sort_keys=True)   # fixed windows -> repeatable
    other = EvalSuite.from_dirs(two_domains, roles={"shouting": "adapt"}, n_batches=4, batch_size=4)
    assert other.domains[1].role == "adapt" and other.fingerprint() != suite.fingerprint()
    delta = r.delta(r)
    assert delta == {"stories": 0.0, "shouting": 0.0}


def test_suite_validation_and_identity(two_domains, tmp_path):
    ok = EvalDomain("a", two_domains["stories"])
    with pytest.raises(ValueError, match="at least one"):
        EvalSuite([])
    with pytest.raises(ValueError, match="unique"):
        EvalSuite([ok, EvalDomain("a", two_domains["shouting"])])
    with pytest.raises(ValueError, match="role must be"):
        EvalSuite([EvalDomain("a", two_domains["stories"], role="forget")])
    with pytest.raises(ValueError, match="positive"):
        EvalSuite([EvalDomain("a", two_domains["stories"], weight=0)])
    with pytest.raises(ValueError, match="split"):
        EvalSuite([EvalDomain("a", two_domains["stories"], split="test")])
    base = EvalSuite([ok]).fingerprint()
    assert EvalSuite([ok]).fingerprint() == base and EvalSuite([ok], seed=1).fingerprint() != base
    assert EvalSuite([EvalDomain("a", two_domains["stories"], n_batches=3)]).fingerprint() != base
    bad = tmp_path / "bad"
    prepare_data(str(_corpus(tmp_path)), str(bad), doc_sep="<|endoftext|>", verbose=False)
    with open(bad / "val.bin", "ab") as fh:
        fh.write(b"\0\0")
    with pytest.raises(DatasetError):
        EvalSuite([EvalDomain("bad", str(bad))])


def _corpus(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text(make_text(120, seed=4), encoding="utf-8")
    return p


def _tc(**over):
    kw = dict(batch_size=8, learning_rate=3e-3, min_lr=3e-4, eval_interval=6, eval_batches=2, eval_batch_size=4,
              log_interval=1, sample_every_evals=0, ckpt_interval_minutes=1e9, verbose=False, seed=7)
    kw.update(over)
    return TrainConfig(**kw)


def test_trainer_validates_on_the_suite_and_checks_it_on_resume(two_domains, tmp_path):
    cfg = small_cfg()
    suite = EvalSuite.from_dirs(two_domains, roles={"shouting": "adapt"}, n_batches=2, batch_size=4)
    half, full = RunPlan(8 * 48, 6, 2, 8, 2, "wsd"), RunPlan(8 * 48, 12, 2, 8, 2, "wsd")
    events, out = [], str(tmp_path / "r")
    res = Trainer(TensegrityLM(cfg), _tc(ema_decay=0.9), half, two_domains["stories"], out, device="cpu",
                  eval_suite=suite, callbacks=[lambda e, p: events.append(p) if e == "eval" else None]).fit()
    ev = events[-1]
    assert set(ev["val_domains"]) == set(ev["val_domains_ema"]) == {"stories", "shouting"}
    assert {"val_bpb_bytes", "val_bpb_bytes_ema"} <= set(ev)
    macro = np.mean([m["loss"] for m in ev["val_domains"].values()])                        # equal weights
    assert ev["val_loss"] == pytest.approx(macro) and res["best_val_loss"] <= ev["val_loss"] + 1e-9
    ck = load_checkpoint(os.path.join(out, "ckpt_latest.pt"))
    assert ck["eval_suite"]["domains"][0]["name"] == "stories" and ck["eval_suite"]["domains"][0]["fingerprint"].startswith("sha256:")
    # a different evaluation set would make best_val_loss incomparable -> refuse; the same one resumes
    other = EvalSuite([EvalDomain("stories", two_domains["stories"], n_batches=2, batch_size=4)])
    with pytest.raises(ValueError, match="eval_suite"):
        Trainer(TensegrityLM(cfg), _tc(ema_decay=0.9), full, two_domains["stories"], out, device="cpu", eval_suite=other).fit()
    with pytest.raises(ValueError, match="eval_suite"):
        Trainer(TensegrityLM(cfg), _tc(ema_decay=0.9), full, two_domains["stories"], out, device="cpu").fit()
    assert Trainer(TensegrityLM(cfg), _tc(ema_decay=0.9), full, two_domains["stories"], out, device="cpu",
                   eval_suite=suite).fit()["step"] == 12


# -------------------------------------------------------------------------------- context probe
class InductionOracle(nn.Module):
    """A 'model' that does exactly one thing: continue the longest earlier match of the current suffix, but only
    if that match lies within ``reach`` positions. It stands in for a trained retriever with a known range."""

    def __init__(self, cfg, reach):
        super().__init__()
        self.cfg, self.reach = cfg, reach
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, seq):
        B, T = seq.shape
        logits = torch.zeros(B, T, 259)
        for b in range(B):
            s = seq[b].tolist()
            for j in range(T):
                best_len, best_next = 0, None
                for i in range(max(0, j - self.reach), j):               # candidate match ends at i (< j)
                    n = 0
                    while n <= i and s[i - n] == s[j - n] and n < 16:
                        n += 1
                    if n >= best_len and n > 0:
                        best_len, best_next = n, s[i + 1]
                if best_next is not None:
                    logits[b, j, best_next] = 30.0
        return logits, None, None


def test_context_probe_measures_retrieval_and_flags_what_the_layout_cannot_reach():
    cfg = TensegrityConfig(n_layer=2, n_embd=32, n_head=2, n_kv_head=1, block_size=96, sliding_window=16, global_every=99)
    rf = cfg.receptive_field()
    assert rf == 1 + 2 * 15 == 31
    rows = context_probe(InductionOracle(cfg, reach=rf), distances=(0, 8, 24, 40), key_len=12, n_trials=48, seed=1)
    by_d = {r["distance"]: r for r in rows}
    assert [by_d[d]["reachable"] for d in (0, 8, 24, 40)] == [True, True, False, False]   # d + key_len <= 31
    for d in (0, 8):                                                                       # inside the range: it retrieves
        assert by_d[d]["accuracy"] > 0.8 and by_d[d]["gain_bits"] > 2.0, by_d[d]
    for d in (24, 40):                                                                     # beyond it: back to chance
        assert by_d[d]["accuracy"] < 0.2 and abs(by_d[d]["gain_bits"]) < 1.5, by_d[d]
    assert all(by_d[d]["control_accuracy"] < 0.25 for d in by_d)                           # without a first copy nothing helps
    assert "does not fit" not in format_probe(rows) and format_probe(rows).count("\n") == 4
    far = context_probe(InductionOracle(cfg, reach=rf), distances=(80,), key_len=12, n_trials=4)
    assert far == [{"distance": 80, "fits": False, "reachable": False}] and "does not fit" in format_probe(far)


def test_context_probe_on_an_untrained_model_is_at_chance_and_restores_mode():
    torch.manual_seed(0)
    m = TensegrityLM(small_cfg(block_size=96, sliding_window=24)).train()
    rows = context_probe(m, distances=(0, 16), key_len=8, n_trials=32)
    assert m.training and all(r["accuracy"] < 0.2 and abs(r["gain_bits"]) < 1.0 for r in rows)
    assert all(math.isfinite(r["nll_bits"]) for r in rows)


# ------------------------------------------------------------------------- residual diagnostics
def test_residual_report_follows_the_real_forward_pass_and_scales_with_the_gain():
    torch.manual_seed(0)
    m = randomized(TensegrityLM(small_cfg(x0_mix=True, byte_mix_kernel=3)), 0.05).eval()
    for b in m.blocks:
        b.x0_gain.data.fill_(0.1)
    idx = torch.randint(0, 256, (2, 24))
    rows = residual_report(m, idx)
    assert [r["kind"] for r in rows] == ["local", "local", "global", "local"] and len(rows) == 4
    # re-assemble the logits from the final stream the report walked: must equal the model's own forward
    # (so the report cannot silently drift from Block.forward)
    x0 = m._embed(idx)
    x = x0
    cos, sin = (t[:24] for t in m._rope_tables(idx.device))
    with torch.no_grad():
        for b in m.blocks:
            x, _ = b(x, cos, sin, x0)
        logits = m.lm_head(m.ln_f(x))
    torch.testing.assert_close(logits, m(idx)[0], atol=1e-6, rtol=1e-5)
    assert rows[-1]["stream_rms"] == pytest.approx(float(torch.sqrt(x.pow(2).mean(-1)).mean()), rel=1e-5)
    # layer 0 sees the same stream whatever its gain is, so rho is exactly proportional to the gain
    g0 = float(m.blocks[0].attn_gain.detach())
    m.blocks[0].attn_gain.data.mul_(2.0)
    assert residual_report(m, idx)[0]["attn_rho"] == pytest.approx(2.0 * rows[0]["attn_rho"], rel=1e-5)
    assert residual_report(m, idx)[0]["attn_branch_rms"] == pytest.approx(rows[0]["attn_branch_rms"], rel=1e-5)
    m.blocks[0].attn_gain.data.fill_(0.0)
    assert residual_report(m, idx)[0]["attn_rho"] == 0.0
    assert g0 != 0.0 and all(r["ffn_rho"] >= 0 for r in rows)
