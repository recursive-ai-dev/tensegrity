"""Trust boundaries and interrupted writes for model artifacts."""

import concurrent.futures
import os

import pytest
import torch

from conftest import ASSETS, small_cfg
from tensegrity import TensegrityLM, load_checkpoint, load_model
from tensegrity.checkpoint import atomic_torch_save, environment_info, save_checkpoint


class _UntrustedPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')",)


def test_untrusted_checkpoint_cannot_trigger_pickle_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "untrusted.pt"
    torch.save(_UntrustedPayload(marker), path)
    with pytest.raises(ValueError, match="Unrestricted pickle loading is disabled"):
        load_checkpoint(path)
    with pytest.raises(ValueError, match="Unrestricted pickle loading is disabled"):
        load_model(path)
    assert not marker.exists()


def test_legacy_checkpoint_requires_explicit_trust():
    path = os.path.join(ASSETS, "legacy_v1.pt")
    with pytest.raises(ValueError, match="allow_legacy_pickle=True"):
        load_checkpoint(path)
    with pytest.warns(UserWarning, match="legacy"):
        model, cfg, checkpoint = load_model(path, allow_legacy_pickle=True)
    assert cfg == model.cfg and checkpoint["format_version"] == 1


def test_environment_info_is_safe_serializable(tmp_path):
    versions = environment_info()
    assert all(type(value) is str for value in versions.values())
    path = tmp_path / "environment.pt"
    atomic_torch_save(versions, path)
    assert torch.load(path, weights_only=True) == versions


@pytest.mark.parametrize("value", [None, [], 4, {"config": []}])
def test_invalid_checkpoint_structure_is_rejected(tmp_path, value):
    path = tmp_path / "invalid.pt"
    torch.save(value, path)
    with pytest.raises(ValueError, match="dictionary|missing"):
        load_checkpoint(path)


def test_unsupported_format_and_model_state_rejected(tmp_path):
    model = TensegrityLM(small_cfg())
    path = tmp_path / "valid.pt"
    save_checkpoint(path, model, None, model.cfg, 0, 1.0)
    checkpoint = load_checkpoint(path)
    checkpoint["format_version"] = 999
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="unsupported format_version"):
        load_checkpoint(path)
    checkpoint["format_version"] = 2
    checkpoint["model_state_dict"] = {"broken": "not a tensor"}
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="model_state_dict"):
        load_checkpoint(path)


def test_checkpoint_extra_cannot_replace_reserved_fields(tmp_path):
    model = TensegrityLM(small_cfg())
    with pytest.raises(ValueError, match="cannot replace"):
        save_checkpoint(tmp_path / "invalid.pt", model, None, model.cfg, 0, 1.0,
                        extra={"model_state_dict": {}})


def test_failed_atomic_save_preserves_existing_checkpoint_and_cleans_staging(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"complete": 1}, path)

    def fail(obj, destination):
        with open(destination, "wb") as stream:
            stream.write(b"incomplete")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        atomic_torch_save({"complete": 2}, path)
    assert torch.load(path, weights_only=True) == {"complete": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_concurrent_atomic_saves_publish_one_complete_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.pt"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda value: atomic_torch_save({"value": value}, path), range(16)))
    assert torch.load(path, weights_only=True)["value"] in range(16)
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_ema_load_retains_checkpoint_weights_for_frozen_parameters(tmp_path):
    model = TensegrityLM(small_cfg())
    named = dict(model.named_parameters())
    name, parameter = next(iter(named.items()))
    ema = {"shadow": {name: torch.full_like(parameter, 0.5)}}
    path = tmp_path / "partial_ema.pt"
    save_checkpoint(path, model, None, model.cfg, 0, 1.0, ema_state=ema)
    loaded, _, _ = load_model(path, use_ema=True)
    assert torch.equal(dict(loaded.named_parameters())[name], ema["shadow"][name])
    for other_name, other_parameter in list(named.items())[1:]:
        assert torch.equal(dict(loaded.named_parameters())[other_name], other_parameter)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA RNG state loaded on device")
def test_numpy_rng_restoration_after_loading_checkpoint_on_cuda(tmp_path):
    import numpy as np
    from tensegrity.checkpoint import capture_rng_state, restore_rng_state
    path = tmp_path / "rng.pt"
    state = capture_rng_state()
    atomic_torch_save(state, path)
    expected = np.random.rand(8)
    loaded = torch.load(path, map_location="cuda", weights_only=True)
    restore_rng_state(loaded)
    assert np.array_equal(np.random.rand(8), expected)
