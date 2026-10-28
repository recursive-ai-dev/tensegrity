"""Installed CLI ergonomics and offline readiness checks."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tensegrity import load_checkpoint, load_model, save_checkpoint, TensegrityLM
from tensegrity.cli import build_parser, main
from tensegrity.diagnostics import installation_report
from conftest import ASSETS, small_cfg

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("flag", ["--version", "--help"])
def test_cli_bootstraps_without_site_packages(flag):
    result = subprocess.run([sys.executable, "-S", "-m", "tensegrity", flag], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Tensegrity" in result.stdout


def test_doctor_diagnoses_missing_dependencies_without_import_crash():
    result = subprocess.run([sys.executable, "-S", "-m", "tensegrity", "doctor", "--json", "--smoke"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert not report["ok"] and not report["dependencies"]["torch"]["ok"]
    assert not report["smoke"]["ok"] and "Traceback" not in result.stderr


def test_lazy_public_api_preserves_names_without_importing_torch():
    code = ("import sys, tensegrity; assert 'torch' not in sys.modules; "
            "assert 'Trainer' in dir(tensegrity); "
            "assert tensegrity.ByteTokenizer().encode('é') == [195,169]; "
            "assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_doctor_smoke_preserves_rng_and_thread_settings(capsys):
    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    main(["doctor", "--smoke", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["smoke"]["ok"]
    assert len(report["smoke"]["checks"]) == 5
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.get_num_threads() == threads


@pytest.mark.parametrize("command,option,value", [
    ("generate", "--temperature", "nan"), ("generate", "--temperature", "-1"),
    ("generate", "--top_k", "0"), ("generate", "--top_p", "0"),
    ("generate", "--min_p", "1.1"), ("generate", "--max_new_tokens", "-1"),
    ("train", "--tokens", "inf"), ("train", "--micro_batch", "0"),
    ("train", "--x0_mix", "typo"), ("train", "--hours", "-1"),
])
def test_cli_rejects_invalid_knobs_before_loading(command, option, value):
    required = ["--ckpt", "missing"] if command == "generate" else ["--data_dir", "missing", "--out_dir", "unused"]
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, *required, option, value])
    assert exc.value.code == 2


def test_cli_missing_file_is_actionable_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--ckpt", "/nonexistent/tensegrity.pt", "--device", "cpu"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "tensegrity:" in err and "Traceback" not in err


def test_cli_legacy_conversion_is_explicit_and_loads_safely(tmp_path, capsys):
    source = os.path.join(ASSETS, "legacy_v1.pt")
    dest = tmp_path / "converted.pt"
    with pytest.raises(SystemExit) as exc:
        main(["convert-checkpoint", "--ckpt", source, "--out", str(dest)])
    assert exc.value.code == 1 and not dest.exists()
    main(["convert-checkpoint", "--ckpt", source, "--out", str(dest), "--trust-legacy-pickle"])
    checkpoint = load_checkpoint(dest)
    assert checkpoint["format_version"] == 2 and checkpoint["optimizer_state_dict"] is None
    original, _, _ = load_model(source, allow_legacy_pickle=True)
    converted, _, _ = load_model(dest)
    for key, value in original.state_dict().items():
        assert torch.equal(value, converted.state_dict()[key])
    content = dest.read_bytes()
    with pytest.raises(SystemExit):
        main(["convert-checkpoint", "--ckpt", source, "--out", str(dest), "--trust-legacy-pickle"])
    assert content == dest.read_bytes()


def test_cli_stream_decodes_utf8_across_byte_boundaries(monkeypatch, capsys):
    class Fake:
        def eval(self):
            return self
        def stream(self, *args, **kwargs):
            for token in ' café 🌍'.encode('utf-8'):
                yield token, None
            yield 257, None
    monkeypatch.setattr('tensegrity.checkpoint.load_model', lambda *a, **k: (Fake(), None, {}))
    main(["generate", "--ckpt", "unused", "--prompt", "Hello", "--stream", "--device", "cpu"])
    assert capsys.readouterr().out == "Hello café 🌍\n"


def test_cli_stream_matches_buffered_generation(tmp_path, capsys):
    model = TensegrityLM(small_cfg())
    ckpt = tmp_path / "weights.pt"
    save_checkpoint(ckpt, model, None, model.cfg, 0, 1)
    args = ["generate", "--ckpt", str(ckpt), "--prompt", "Bonjour é", "--max_new_tokens", "32",
            "--temperature", "0", "--no_stop_on_eos", "--device", "cpu"]
    main(args)
    buffered = capsys.readouterr().out
    main(args + ["--stream"])
    assert capsys.readouterr().out == buffered


def test_cli_json_eval_and_audit_are_valid(tmp_path, corpus_dir, capsys):
    model = TensegrityLM(small_cfg())
    ckpt = tmp_path / "weights.pt"
    save_checkpoint(ckpt, model, None, model.cfg, 0, 1)
    main(["eval", "--ckpt", str(ckpt), "--data_dir", corpus_dir, "--eval_batches", "1", "--device", "cpu", "--json"])
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["bpb_bytes"] > 0 and metrics["n_bytes"] > 0
    main(["audit", "--data_dir", corpus_dir, "--deep", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["problems"] == [] and report["fingerprint"].startswith("sha256:")


def test_cli_seed_covers_model_initialization(tmp_path, corpus_dir):
    states = []
    for index in range(2):
        # Different surrounding randomness must not change a seeded fresh run.
        torch.manual_seed(100 + index)
        run = tmp_path / str(index)
        main(["train", "--preset", "1m", "--data_dir", corpus_dir, "--out_dir", str(run),
              "--micro_batch", "2", "--tokens", "512", "--block_size", "32",
              "--sliding_window", "16", "--optimizer", "adamw", "--device", "cpu", "--seed", "42"])
        states.append(load_checkpoint(run / "ckpt_latest.pt")["model_state_dict"])
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
