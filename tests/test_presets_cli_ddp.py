"""Preset ladder invariants, the full growth walk, CLI, and 2-process DDP."""

import os
import socket
import subprocess
import sys

import pytest
import torch
import torch.multiprocessing as mp

from conftest import small_cfg
from tensegrity import (LADDER, LADDER_NAMES, MODEL_PRESETS, TensegrityLM, check_function_preserved,
                        check_growth_trainable, describe_presets, get_preset, get_preset_config, grow_model,
                        next_preset, train_setup)
from tensegrity.cli import build_parser, main as cli_main


# ----------------------------------------------------------------------------- the ladder
TARGETS = {"1m": 1e6, "2m": 2e6, "5m": 5e6, "10m": 10e6, "50m": 50e6, "115m": 115e6}


@pytest.mark.parametrize("name", LADDER_NAMES)
def test_preset_param_counts_match_their_names(name):
    cfg = get_preset_config(name)
    assert abs(cfg.n_params() - TARGETS[name]) / TARGETS[name] < 0.02, cfg.describe()
    assert cfg.ffn_hidden == get_preset(name).ffn_hidden
    assert cfg.head_dim == 32 and cfg.gqa_ratio == 2
    assert (cfg.byte_mix_kernel, cfg.embed_scale, cfg.x0_mix) == (4, False, False)   # ablation-chosen ladder defaults


@pytest.mark.parametrize("name", LADDER_NAMES[:4])
def test_preset_analytic_count_equals_instantiated(name):
    cfg = get_preset_config(name)
    assert TensegrityLM(cfg).num_parameters() == cfg.n_params()


def test_ladder_is_monotone_and_growth_compatible():
    cfgs = [get_preset_config(p.name) for p in LADDER]
    for a, b in zip(cfgs, cfgs[1:]):
        for f in ("n_embd", "n_layer", "n_head", "n_kv_head", "ffn_hidden", "block_size"):
            assert getattr(b, f) >= getattr(a, f), (f, a.describe(), b.describe())
        assert (a.head_dim, a.gqa_ratio, a.sliding_window, a.global_every, a.vocab_size) == \
               (b.head_dim, b.gqa_ratio, b.sliding_window, b.global_every, b.vocab_size)
    assert [get_preset(n).label for n in LADDER_NAMES] == ["1M", "2M", "5M", "10M", "50M", "115M"]
    assert next_preset("5M").name == "10m" and next_preset("115m") is None
    assert get_preset("5M") is get_preset("5m")
    with pytest.raises(ValueError):
        get_preset("7m")
    with pytest.raises(ValueError):
        get_preset_config("nope")


@pytest.mark.parametrize("src,dst", list(zip(LADDER_NAMES, LADDER_NAMES[1:]))[:4])
def test_every_adjacent_pair_grows_exactly_and_trainably(src, dst):
    """Walk the ladder 1M -> 2M -> 5M -> 10M -> 50M with real presets (the 50M -> 115M pair is
    exercised in test_top_rung_growth)."""
    torch.manual_seed(0)
    c0, c1 = get_preset_config(src, block_size=64, sliding_window=16), get_preset_config(dst, block_size=64, sliding_window=16)
    old = TensegrityLM(c0)
    with torch.no_grad():
        for p in old.parameters():
            p.add_(torch.randn_like(p) * 0.03)
    new, ncfg = grow_model(old, c0, target=c1)
    assert ncfg.n_embd == c1.n_embd and ncfg.n_layer == c1.n_layer and ncfg.ffn_hidden == c1.ffn_hidden
    check_function_preserved(old, new, c0, "cpu", n_tokens=32, atol=5e-5)
    check_growth_trainable(new, c0)


@pytest.mark.slow
def test_top_rung_growth():
    c0 = get_preset_config("50m", block_size=64, sliding_window=16)
    c1 = get_preset_config("115m", block_size=64, sliding_window=16)
    torch.manual_seed(0)
    old = TensegrityLM(c0)
    new, _ = grow_model(old, c0, target=c1)
    check_function_preserved(old, new, c0, "cpu", n_tokens=24, atol=1e-4)
    check_growth_trainable(new, c0)


def test_legacy_presets_unchanged():
    """Parameter counts of the original release's named presets (measured from the original code)."""
    assert get_preset_config("base15m").n_params() == 15_867_401
    assert get_preset_config("balanced135m").n_params() == 133_200_409
    assert get_preset_config("highintel400m").n_params() == 403_107_361
    assert set(MODEL_PRESETS) >= {"base15m", "balanced135m", "highintel400m", *LADDER_NAMES}
    c = get_preset_config("base15m", dropout=0.1)
    assert c.dropout == 0.1 and c.n_layer == 4 and c.byte_mix_kernel == 0 and c.confidence_detach is False


def test_describe_and_train_setup():
    table = describe_presets()
    for label in ("1M", "5M", "115M"):
        assert label in table
    tc, plan = train_setup("5m", micro_batch=16, grad_accum=4)
    assert plan.tokens_per_step == 16 * 4 * 1024 and plan.total_tokens <= get_preset("5m").default_tokens
    assert tc.learning_rate == get_preset("5m").learning_rate and tc.min_lr == tc.learning_rate / 10
    assert 0 < tc.ema_decay < 1 and tc.optimizer == "muon" and tc.muon_weight_decay == 0.0
    tc2, _ = train_setup("1m", 8, 1, total_tokens=1_000_000, ema=False, optimizer="adamw", eval_interval=7)
    assert tc2.ema_decay == 0 and tc2.optimizer == "adamw" and tc2.eval_interval == 7
    with pytest.raises(TypeError):
        train_setup("1m", 8, 1, bogus=1)


# --------------------------------------------------------------------------------------- CLI
def test_cli_end_to_end(tmp_path, capsys):
    src = tmp_path / "t.txt"
    from conftest import make_text
    src.write_text(make_text(200), encoding="utf-8")
    data, run, run2, export = (str(tmp_path / n) for n in ("data", "run", "run2", "export"))
    cli_main(["presets"])
    cli_main(["prepare", "--data_path", str(src), "--out_dir", data, "--doc_sep", "<|endoftext|>"])
    cli_main(["train", "--preset", "1m", "--data_dir", data, "--out_dir", run, "--micro_batch", "4", "--tokens", "40000",
              "--block_size", "64", "--sliding_window", "16", "--device", "cpu"])
    assert os.path.exists(os.path.join(run, "ckpt_best.pt"))
    cli_main(["train", "--preset", "1m", "--data_dir", data, "--out_dir", run, "--device", "cpu"])   # resume: completed -> no-op
    cli_main(["eval", "--ckpt", os.path.join(run, "ckpt_best.pt"), "--data_dir", data, "--eval_batches", "2", "--device", "cpu"])
    out = capsys.readouterr().out
    assert "bits/byte" in out and "val_loss" in out
    cli_main(["generate", "--ckpt", os.path.join(run, "ckpt_best.pt"), "--prompt", "The", "--max_new_tokens", "12",
              "--seed", "0", "--device", "cpu"])
    cli_main(["export", "--ckpt", os.path.join(run, "ckpt_best.pt"), "--out_dir", export, "--dtype", "float16"])
    assert os.path.exists(os.path.join(export, "tensegrity.safetensors")) and os.path.exists(os.path.join(export, "README.md"))


def test_cli_grow_then_continue(tmp_path):
    from conftest import make_text
    src = tmp_path / "t.txt"
    src.write_text(make_text(200), encoding="utf-8")
    data, run, grown_dir = str(tmp_path / "d"), str(tmp_path / "r"), str(tmp_path / "g")
    cli_main(["prepare", "--data_path", str(src), "--out_dir", data, "--doc_sep", "<|endoftext|>"])
    common = ["--data_dir", data, "--micro_batch", "4", "--tokens", "30000", "--block_size", "64",
              "--sliding_window", "16", "--device", "cpu"]
    cli_main(["train", "--preset", "1m", "--out_dir", run, *common])
    os.makedirs(grown_dir)
    cli_main(["grow", "--ckpt", os.path.join(run, "ckpt_best.pt"), "--to", "2m", "--out", os.path.join(grown_dir, "w.pt"),
              "--device", "cpu"])
    cont = ["--data_dir", data, "--micro_batch", "4", "--tokens", "30000", "--device", "cpu"]
    with pytest.raises(SystemExit, match="fixes the architecture"):
        cli_main(["train", "--preset", "2m", "--out_dir", grown_dir + "/bad", "--init_weights",
                  os.path.join(grown_dir, "w.pt"), "--block_size", "128", *cont])
    cli_main(["train", "--preset", "2m", "--out_dir", grown_dir + "/run", "--init_weights", os.path.join(grown_dir, "w.pt"), *cont])
    assert os.path.exists(os.path.join(grown_dir, "run", "ckpt_best.pt"))
    from tensegrity import load_model
    m, c, _ = load_model(os.path.join(grown_dir, "run", "ckpt_best.pt"))
    assert (c.n_embd, c.n_layer, c.sliding_window, c.block_size) == (192, 5, 16, 64)   # grown size, kept pattern/context


def test_cli_parser_rejects_bad_input():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["grow", "--ckpt", "x", "--to", "7m", "--out", "y"])
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_module_entrypoint_runs():
    r = subprocess.run([sys.executable, "-m", "tensegrity", "presets"], capture_output=True, text=True,
                       cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    assert r.returncode == 0 and "115M" in r.stdout


# --------------------------------------------------------------------------------------- DDP
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_worker(rank, world, port, data_dir, out_dir, q):
    import traceback
    try:
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                          MASTER_PORT=str(port))
        torch.set_num_threads(1)
        from tensegrity import RunPlan, TrainConfig, Trainer
        cfg = small_cfg(layerdrop_max=0.5, n_layer=4)         # LayerDrop + DDP is the risky combination
        torch.manual_seed(0)
        model = TensegrityLM(cfg)
        tc = TrainConfig(batch_size=4, learning_rate=3e-3, min_lr=3e-4, eval_interval=100, log_interval=100,
                         sample_every_evals=0, verbose=False, seed=3, ema_decay=0.9)
        plan = RunPlan(4 * 48 * world, 8, 2, 4, 2, "wsd")
        res = Trainer(model, tc, plan, data_dir, out_dir, device="cpu").fit(resume=False)
        path = os.path.join(out_dir + f"_rank{rank}.pt")
        torch.save(model.state_dict(), path)      # not via the queue: fd-passing tensors dies with the worker
        q.put((rank, res["step"], path, None))
    except BaseException:  # noqa: BLE001 - report to the parent instead of dying silently
        q.put((rank, -1, None, traceback.format_exc()))


@pytest.mark.slow
def test_ddp_two_processes_stay_in_lockstep(corpus_dir, tmp_path):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_ddp_worker, args=(r, 2, port, corpus_dir, str(tmp_path / "ddp"), q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, step, sd, err = q.get(timeout=300)
        assert err is None, f"rank {rank} crashed:\n{err}"
        results[rank] = (step, torch.load(sd, weights_only=True))
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert results[0][0] == results[1][0] == 8
    for (k, a), b in zip(results[0][1].items(), results[1][1].values()):
        assert torch.allclose(a, b, atol=1e-6), k           # gradients were all-reduced: replicas identical
