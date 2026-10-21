"""The CLI exposes the same engine the notebook uses: preflight, report, probe, bench, continual, streaming."""

import json
import os

import pytest
import torch

from conftest import make_text, small_cfg
from tensegrity import TensegrityLM, prepare_data, save_checkpoint
from tensegrity.cli import main


@pytest.fixture(scope="module")
def two(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    out = {}
    for name, seed, tw in (("old", 1, lambda s: s), ("new", 2, lambda s: s.upper())):
        (d / f"{name}.txt").write_text(tw(make_text(200, seed=seed)), encoding="utf-8")
        prepare_data(str(d / f"{name}.txt"), str(d / name), doc_sep="<|endoftext|>", verbose=False)
        out[name] = str(d / name)
    ck = str(d / "m.pt")
    torch.manual_seed(0)
    m = TensegrityLM(small_cfg(block_size=64, sliding_window=16))
    save_checkpoint(ck, m, None, m.cfg, 0, 1.0)
    return out, ck, d


def test_cli_preflight_report_and_exit_code(two, capsys, tmp_path):
    import shutil
    dirs, ck, d = two
    args = ["preflight", "--preset", "1m", "--micro_batch", "2", "--tokens", "1e5", "--device", "cpu", "--optimizer", "adamw"]
    main(args + ["--data_dir", dirs["old"]])
    out = capsys.readouterr().out
    for needle in ("PREFLIGHT OK", "model    :", "layers [llGll]", "fingerprint : sha256:", "train example:", "finite over the test steps"):
        assert needle in out, needle
    bad = tmp_path / "bad"
    shutil.copytree(dirs["old"], bad)
    with open(bad / "val.bin", "ab") as fh:
        fh.write(b"\0\0")
    with pytest.raises(SystemExit) as e:
        main(args + ["--data_dir", str(bad)])
    assert e.value.code == 1 and "PREFLIGHT FAILED" in capsys.readouterr().out


def test_cli_probe_bench_and_streaming_generate(two, capsys):
    dirs, ck, d = two
    main(["probe", "--ckpt", ck, "--distances", "0,8,200", "--trials", "8", "--device", "cpu"])
    out = capsys.readouterr().out
    assert "reach=" in out and "does not fit" in out and "accuracy" in out
    main(["bench", "--ckpt", ck, "--new_tokens", "16", "--device", "cpu"])
    out = capsys.readouterr().out
    assert "streaming" in out and "tok/s (past context)" in out and "SDPA on cpu" in out
    main(["generate", "--ckpt", ck, "--prompt", "Once", "--max_new_tokens", "80", "--cache_mode", "streaming",
          "--temperature", "0", "--no_stop_on_eos", "--device", "cpu"])
    assert capsys.readouterr().out.startswith("Once")
    with pytest.raises(SystemExit):
        main(["generate", "--ckpt", ck, "--start_token", "none", "--device", "cpu"])           # nothing to condition on


def test_cli_continual_and_report(two, capsys, tmp_path):
    dirs, ck, d = two
    out_dir = str(tmp_path / "cl")
    main(["continual", "--ckpt", ck, "--old_dir", dirs["old"], "--new_dir", dirs["new"], "--steps", "8", "--batch_size", "4",
          "--seeds", "0", "--out_dir", out_dir, "--device", "cpu"])
    assert "non-dominated" in capsys.readouterr().out and os.path.exists(os.path.join(out_dir, "continual_report.json"))
    run = str(tmp_path / "run")
    main(["train", "--preset", "1m", "--data_dir", dirs["old"], "--out_dir", run, "--tokens", "20000", "--micro_batch", "2",
          "--block_size", "64", "--sliding_window", "16", "--device", "cpu", "--optimizer", "adamw"])
    capsys.readouterr()
    main(["report", "--run_dir", run, "--data_dir", dirs["old"]])
    assert "run_report.md" in capsys.readouterr().out
    rep = json.load(open(os.path.join(run, "run_report.json")))
    assert rep["progress"]["finished"] and rep["final_evaluation"]["bpb_bytes"] > 0
