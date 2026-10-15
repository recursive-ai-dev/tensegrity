"""Preflight and run reports: decisive before a run, reproducible after it."""

import json
import os
import shutil

import pytest
import torch

from conftest import make_text, small_cfg
from tensegrity import (EvalSuite, GrowthGates, RunPlan, TensegrityLM, TrainConfig, Trainer, export_safetensors,
                        format_preflight, grow_transaction, package_fingerprint, prepare_data, preflight_report,
                        render_run_report, write_run_report)
from tensegrity.checkpoint import load_checkpoint


def _tc(**over):
    kw = dict(batch_size=8, learning_rate=3e-3, min_lr=3e-4, eval_interval=1000, eval_batches=2, eval_batch_size=4,
              log_interval=1, sample_every_evals=0, ckpt_interval_minutes=1e9, verbose=False, seed=7)
    kw.update(over)
    return TrainConfig(**kw)


PLAN = RunPlan(8 * 48, 12, 2, 8, 2, "wsd")


# ------------------------------------------------------------------------------------ preflight
def test_preflight_answers_the_questions_before_compute_is_spent(corpus_dir):
    m = TensegrityLM(small_cfg())
    before = {k: v.clone() for k, v in m.state_dict().items()}
    rep = preflight_report(m, _tc(), PLAN, corpus_dir, "cpu")
    assert rep["ok"] and rep["blockers"] == [] and rep["warnings"] == []
    assert rep["model"]["layout"] == "llGl" and rep["model"]["global_layers"] == [2] and rep["model"]["receptive_field"] == 48
    assert rep["model"]["parameters"] == rep["model"]["parameters_formula"] == m.cfg.n_params()
    d = rep["data"][0]
    assert d["fingerprint"].startswith("sha256:") and d["problems"] == [] and d["n_train_tokens"] > 5000
    assert d["examples"]["train"] and d["examples"]["val"]
    assert rep["budget"]["total_tokens"] == PLAN.total_tokens and rep["budget"]["passes_over_data"] < 1
    assert rep["numerics"]["ok"] and rep["numerics"]["checked"] and rep["roundtrip"]["checked"]
    assert set(rep["roundtrip"]["max_logit_diff"]) == {"config+state_dict", "checkpoint", "safetensors"}
    assert rep["memory"]["weights_and_optimizer_gib"] > 0 and rep["memory"]["process_rss_peak_gib"] > 0
    assert rep["code_fingerprint"] == package_fingerprint() and rep["environment"]["torch"] == torch.__version__
    text = format_preflight(rep)
    for needle in ("PREFLIGHT OK", "layers [llGl]", d["fingerprint"], "finite over the test steps", "reload to identical outputs"):
        assert needle in text, needle
    json.dumps(rep, default=str)
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())          # the model is left exactly as it was


def test_preflight_blocks_on_real_problems(corpus_dir, tmp_path):
    # a dataset that no longer matches its manifest
    bad = tmp_path / "bad"
    shutil.copytree(corpus_dir, bad)
    with open(bad / "val.bin", "ab") as fh:
        fh.write(b"\0\0")
    rep = preflight_report(TensegrityLM(small_cfg()), _tc(), PLAN, str(bad), "cpu")
    assert not rep["ok"] and any("manifest says" in b for b in rep["blockers"]) and "PREFLIGHT FAILED" in format_preflight(rep)
    assert rep["numerics"] == {"checked": False} and rep["roundtrip"] == {"checked": False}      # no compute on bad data
    # a model whose numerics are broken
    broken = TensegrityLM(small_cfg())
    with torch.no_grad():
        broken.token_emb.weight.fill_(float("nan"))
    rep = preflight_report(broken, _tc(), PLAN, corpus_dir, "cpu")
    assert not rep["ok"] and any(b.startswith("numerics:") for b in rep["blockers"])
    # a corpus too small for one training window
    tiny = tmp_path / "tiny.txt"
    tiny.write_text("short text. " * 6)
    prepare_data(str(tiny), str(tmp_path / "t"), val_fraction=0.3, verbose=False)
    rep = preflight_report(TensegrityLM(small_cfg(block_size=64, sliding_window=16)), _tc(), PLAN, str(tmp_path / "t"), "cpu")
    assert not rep["ok"] and any("fewer than one" in b for b in rep["blockers"])
    # training longer sequences than the model has context for
    rep = preflight_report(TensegrityLM(small_cfg()), _tc(block_size=96), PLAN, corpus_dir, "cpu")
    assert any("exceeds the model" in b for b in rep["blockers"])
    # a dataset path that does not exist is a blocker, not a crash
    rep = preflight_report(TensegrityLM(small_cfg()), _tc(), PLAN, str(tmp_path / "nowhere"), "cpu")
    assert not rep["ok"] and "cannot be audited" in rep["blockers"][0]


def test_preflight_warns_about_local_only_attention_and_overtraining(corpus_dir):
    cfg = small_cfg(global_every=99, sliding_window=12, n_layer=2)             # reach = 1 + 2 * 11 = 23 of 48
    rep = preflight_report(TensegrityLM(cfg), _tc(), PLAN, corpus_dir, "cpu")
    assert rep["ok"] and rep["model"]["receptive_field"] == 23
    assert any("local-only" in w and "23" in w for w in rep["warnings"]) and "WARNING" in format_preflight(rep)
    many = RunPlan(8 * 48, 5000, 20, 4000, 980, "wsd")                        # ~190 passes over a 100k-token corpus
    rep = preflight_report(TensegrityLM(small_cfg()), _tc(), many, corpus_dir, "cpu", check_numerics=False)
    assert any("memorise" in w for w in rep["warnings"]) and rep["numerics"] == {"checked": False}
    rep = preflight_report(TensegrityLM(small_cfg()), _tc(batch_size=4), PLAN, corpus_dir, "cpu", check_numerics=False)
    assert any("assumed" in w for w in rep["warnings"])


# --------------------------------------------------------------------------------------- reports
def test_package_fingerprint_names_the_exact_code(tmp_path):
    pkg = os.path.dirname(os.path.abspath(__import__("tensegrity").__file__))
    copy = tmp_path / "tensegrity"
    shutil.copytree(pkg, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert package_fingerprint(str(copy)) == package_fingerprint()
    with open(copy / "schedule.py", "a") as fh:
        fh.write("\n# changed\n")
    assert package_fingerprint(str(copy)) != package_fingerprint()


def test_run_report_records_what_the_run_actually_was(corpus_dir, tmp_path):
    out = str(tmp_path / "run")
    suite = EvalSuite.from_dirs({"stories": corpus_dir}, n_batches=2, batch_size=4)
    Trainer(TensegrityLM(small_cfg()), _tc(), PLAN, corpus_dir, out, device="cpu", eval_suite=suite,
            lineage={"init_from": "x.pt"}).fit()
    ck = load_checkpoint(out + "/ckpt_latest.pt")
    export_safetensors(__import__("tensegrity").load_model(out + "/ckpt_best.pt")[0], None, str(tmp_path / "exp"), "m")
    rep = write_run_report(out, eval_suite=suite, export_dir=str(tmp_path / "exp"))
    p = rep["progress"]
    assert p["finished"] and p["completed_updates"] == ck["completed_updates"] == 12 and p["planned_updates"] == 12
    assert p["tokens_seen"] == ck["tokens_seen"] == 12 * 8 * 48 and p["applied_updates"] == 12
    assert rep["data"]["identity"]["0"]["fingerprint"].startswith("sha256:") and rep["lineage"]["init_from"] == "x.pt"
    assert rep["code"]["fingerprint"] == package_fingerprint() and rep["environment"]["torch"] == torch.__version__
    assert rep["model"]["receptive_field"] == 48 and rep["train_config"]["learning_rate"] == 3e-3
    fe = rep["final_evaluation"]
    assert set(fe["domains"]) == {"stories"} and fe["suite"]["domains"][0]["fingerprint"] == rep["data"]["identity"]["0"]["fingerprint"]
    assert rep["best_checkpoint"]["sha256"] and rep["log"]["evals"] >= 1
    (f,) = rep["export"]["files"].values()
    assert f["max_logit_diff_vs_best"] == 0.0 and f["sha256"]
    on_disk = json.load(open(out + "/run_report.json"))
    assert on_disk["progress"] == rep["progress"]
    md = open(out + "/run_report.md").read()
    for needle in ("# Run report", "finished: 12 of 12", "bits/byte", rep["code"]["fingerprint"][:23], "stories"):
        assert needle in md, needle
    assert render_run_report(on_disk) == md
    # without a suite the validation split of the first data dir is used; without any data nothing is evaluated
    rep2 = write_run_report(out, data_dirs=corpus_dir)
    assert rep2["final_evaluation"]["bpb_bytes"] > 0 and "domains" not in rep2["final_evaluation"]
    assert "final_evaluation" not in write_run_report(out, evaluate=False)
    # a damaged export is reported, not hidden
    shutil.rmtree(tmp_path / "exp")
    os.makedirs(tmp_path / "exp")
    assert "error" in write_run_report(out, data_dirs=corpus_dir, export_dir=str(tmp_path / "exp"))["export"]


def test_run_report_of_a_run_in_progress_and_of_growth_history(corpus_dir, tmp_path):
    out = str(tmp_path / "run")
    Trainer(TensegrityLM(small_cfg()), _tc(max_minutes=1e-9), PLAN, corpus_dir, out, device="cpu").fit()
    rep = write_run_report(out, evaluate=False)
    assert rep["progress"]["finished"] is False and rep["progress"]["completed_updates"] == 1
    assert "in progress" in open(out + "/run_report.md").read()
    d = grow_transaction(TensegrityLM(small_cfg()), new_n_layer=5, verbose=False, run_dir=out)
    d2 = grow_transaction(TensegrityLM(small_cfg()), target=small_cfg().replace(n_layer=2), verbose=False, run_dir=out)
    assert d.accepted and not d2.accepted
    assert sorted(f for f in os.listdir(out) if f.startswith("growth_decision")) == ["growth_decision.json", "growth_decision_2.json"]
    rep = write_run_report(out, evaluate=False)
    assert [g["accepted"] for g in rep["growth"]] == [True, False] and "construction failed" in rep["growth"][1]["reasons"][0]
    md = open(out + "/run_report.md").read()
    assert "Growth (growth_decision.json)" in md and "accepted: False; construction failed" in md
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "progress" not in write_run_report(str(empty), evaluate=False)             # nothing to report on is not an error
