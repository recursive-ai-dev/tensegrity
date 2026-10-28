"""Regression coverage for concurrent preparation and malformed runtime inputs."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
import torch

from conftest import small_cfg
from tensegrity import ByteDataset, TensegrityLM, prepare_data
from tensegrity import ingest, manifest


def _source(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_concurrent_preparations_never_share_staging(tmp_path, monkeypatch):
    sources = [_source(tmp_path / f"{i}.txt", str(i) * 1000) for i in range(2)]
    destination = str(tmp_path / "dataset")
    barrier = Barrier(2)
    publish = ingest._publish

    def synchronized_publish(*args):
        barrier.wait(timeout=10)
        return publish(*args)

    monkeypatch.setattr(ingest, "_publish", synchronized_publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare_data, source, destination, verbose=False) for source in sources]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except (FileExistsError, manifest.DatasetError) as exc:
                outcomes.append(exc)
    assert sum(isinstance(outcome, dict) for outcome in outcomes) == 1
    assert manifest.verify_dataset(destination, deep=True) == []
    assert not list(tmp_path.glob("dataset.staging-*"))
    assert not (tmp_path / "dataset.prepare-lock").exists()


def test_repeated_replacements_preserve_every_previous_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.time, "time", lambda: 1000.0)
    destination = str(tmp_path / "dataset")
    fingerprints = []
    for i in range(3):
        source = _source(tmp_path / f"{i}.txt", f"source {i}" * 200)
        fingerprints.append(prepare_data(source, destination, overwrite=True, verbose=False)["fingerprint"])
    backups = list(tmp_path.glob("dataset.replaced-*"))
    assert len(backups) == 2
    assert {manifest.read_meta(str(p))["fingerprint"] for p in backups} == set(fingerprints[:2])
    assert all(manifest.verify_dataset(str(p), deep=True) == [] for p in backups)


def test_matching_manifest_does_not_hide_corrupt_data_on_reprepare(tmp_path):
    source = _source(tmp_path / "source.txt", "hello world " * 200)
    destination = str(tmp_path / "dataset")
    prepare_data(source, destination, verbose=False)
    with open(tmp_path / "dataset" / "train.bin", "r+b") as fh:
        fh.write(b"\x00\x00")
    with pytest.raises(FileExistsError, match="immutable"):
        prepare_data(source, destination, verbose=False)
    prepare_data(source, destination, overwrite=True, verbose=False)
    assert manifest.verify_dataset(destination, deep=True) == []


def test_legacy_identity_tracks_changes_in_tail_between_one_and_two_mib(tmp_path):
    values = np.zeros((3 << 18), dtype=np.uint16)  # 1.5 MiB
    values.tofile(tmp_path / "train.bin")
    first = manifest.quick_fingerprint(str(tmp_path))
    values[-1] = 17
    values.tofile(tmp_path / "train.bin")
    assert manifest.quick_fingerprint(str(tmp_path)) != first


@pytest.mark.parametrize("payload", ["{broken", "[]", '{"manifest_version":1}',
                                      '{"n_train_tokens": "many"}'])
def test_invalid_metadata_is_reported_as_a_consistency_problem(tmp_path, payload):
    for name in manifest.SPLIT_FILES:
        np.arange(20, dtype=np.uint16).tofile(tmp_path / name)
    (tmp_path / "meta.json").write_text(payload)
    assert manifest.verify_dataset(str(tmp_path))
    with pytest.raises(manifest.DatasetError):
        ByteDataset(tmp_path).verify()


def test_manifest_cannot_request_files_outside_the_dataset(tmp_path):
    source = _source(tmp_path / "source.txt", "hello world " * 200)
    destination = str(tmp_path / "dataset")
    meta = prepare_data(source, destination, verbose=False)
    meta["files"]["../private"] = meta["files"].pop("train.bin")
    (tmp_path / "dataset" / "meta.json").write_text(json.dumps(meta))
    assert "exactly train.bin and val.bin" in manifest.verify_dataset(destination, deep=True)[0]


@pytest.mark.parametrize("weights", [[float("nan"), 1], [float("inf"), 1], [[1], [2]], 1])
def test_nonfinite_or_malformed_mixture_weights_fail_immediately(tmp_path, weights):
    with pytest.raises(ValueError, match="weights"):
        ByteDataset([str(tmp_path / "a"), str(tmp_path / "b")], weights=weights)


def test_large_finite_mixture_weights_are_normalized_without_overflow(tmp_path):
    dataset = ByteDataset([str(tmp_path / "a"), str(tmp_path / "b")], weights=[1e308, 1e308])
    np.testing.assert_array_equal(dataset.weights, [0.5, 0.5])


def test_a_single_exact_size_window_is_valid_for_training_and_evaluation(tmp_path):
    for split in ("train", "val"):
        np.arange(17, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    ds = ByteDataset(tmp_path)
    x, y = ds.get_batch("train", 2, 16, "cpu")
    assert torch.equal(x[0], torch.arange(16))
    assert torch.equal(y[0], torch.arange(1, 17))
    fixed = ds.fixed_batches("val", 1, 2, 16, "cpu")
    assert torch.equal(fixed[0][0], x) and torch.equal(fixed[0][1], y)


@pytest.mark.parametrize("field,value", [
    ("n_head", 0), ("n_kv_head", 0), ("block_size", 0), ("n_layer", 1.5), ("ffn_align", True),
    ("dropout", -0.1), ("dropout", 1.0), ("norm_eps", float("nan")),
    ("gain_init", float("inf")), ("rope_base", 0), ("ffn_mult", -1),
    ("logit_soft_cap", 0), ("homeo_weight", -1), ("byte_mix_kernel", 2.5),
])
def test_invalid_architecture_fails_before_model_allocation(field, value):
    with pytest.raises(ValueError, match=field):
        small_cfg(**{field: value})


def test_recursive_local_blocks_have_the_reported_longer_reach():
    torch.manual_seed(9)
    cfg = small_cfg(n_layer=2, n_loops=3, global_every=99, sliding_window=3, block_size=32,
                    layerdrop_max=0, use_confidence_head=False)
    model = TensegrityLM(cfg).eval()
    a = torch.arange(8).unsqueeze(0)
    b = a.clone()
    b[0, 0] = 19
    with torch.no_grad():
        original = model(a)[0][0, -1]
        changed = model(b)[0][0, -1]
    # Two layers on one pass reach only five positions; looping carries information
    # from byte zero to byte seven, which the architectural report must include.
    assert (original - changed).abs().max() > 1e-8
    assert cfg.receptive_field() >= 8
    assert cfg.replace(n_loops=1).receptive_field() < 8


@pytest.mark.parametrize("field,value", [
    ("batch_size", 0), ("grad_accum_steps", 0), ("log_interval", 0),
    ("eval_interval", 0), ("eval_batches", 0), ("block_size", -1),
    ("learning_rate", float("nan")), ("beta2", 1.0), ("ema_decay", float("inf")),
])
def test_invalid_training_config_fails_before_a_run(field, value):
    from tensegrity import TrainConfig
    with pytest.raises(ValueError, match=field):
        TrainConfig(**{field: value})


def test_nonfinite_gradients_skip_updates_when_clipping_is_disabled(corpus_dir, tmp_path):
    from tensegrity import RunPlan, TrainConfig, Trainer
    torch.manual_seed(7)
    model = TensegrityLM(small_cfg())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    handle = next(model.parameters()).register_hook(lambda grad: torch.full_like(grad, float("nan")))
    tc = TrainConfig(batch_size=2, grad_clip=0, eval_batches=1, eval_batch_size=1,
                     sample_every_evals=0, verbose=False)
    trainer = Trainer(model, tc, RunPlan(96, 2, 1, 0, 1, "wsd"), corpus_dir,
                      str(tmp_path / "run"), device="cpu")
    try:
        trainer.fit()
    finally:
        handle.remove()
    assert trainer.skipped_steps == 2
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
