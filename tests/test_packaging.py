"""Exercise the real Debian archive builder without changing the host system."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import io
import json

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11) or shutil.which("dpkg-deb") is None,
    reason="Debian builder requires Python 3.11+ and dpkg-deb",
)


def run(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, check=True, **kwargs)


def test_source_archive_timestamp_without_git(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("build_deb", ROOT / "tools/build_deb.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    monkeypatch.setattr(builder.shutil, "which", lambda name: None)

    def unexpected_subprocess(*args, **kwargs):
        pytest.fail("source_epoch must not invoke git when it is unavailable")

    monkeypatch.setattr(builder.subprocess, "run", unexpected_subprocess)
    assert builder.source_epoch(tmp_path) == 946684800


@pytest.fixture(scope="module")
def deb(tmp_path_factory):
    output = tmp_path_factory.mktemp("deb")
    env = dict(os.environ, SOURCE_DATE_EPOCH="1700000000")
    command = (sys.executable, str(ROOT / "tools/build_deb.py"), "--out-dir")
    run(*command, str(output / "first"), env=env)
    run(*command, str(output / "second"), env=env)
    first = next((output / "first").glob("*.deb"))
    second = next((output / "second").glob("*.deb"))
    extracted = output / "extracted"
    run("dpkg-deb", "--extract", str(first), str(extracted))
    return first, second, extracted


def test_deb_is_reproducible_and_has_valid_manifest(deb):
    first, second, extracted = deb
    assert first.read_bytes() == second.read_bytes()
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    assert first.with_suffix(".deb.sha256").read_text().strip() == f"{digest}  {first.name}"
    control = run("dpkg-deb", "--field", str(first), text=True).stdout
    assert "Architecture: all" in control
    assert "python3-torch (>= 2.3)" in control
    assert "python3-safetensors (>= 0.4)" in control
    assert "python3-numpy (>= 1.23)" in control
    checksums = run("dpkg-deb", "--ctrl-tarfile", str(first)).stdout
    with tarfile.open(fileobj=io.BytesIO(checksums)) as archive:
        manifest = archive.extractfile("./md5sums").read().decode()
    for line in manifest.splitlines():
        checksum, relative = line.split("  ", 1)
        assert hashlib.md5((extracted / relative).read_bytes(), usedforsecurity=False).hexdigest() == checksum
    data = run("dpkg-deb", "--fsys-tarfile", str(first)).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            assert (member.uid, member.gid) == (0, 0)
            assert member.mtime == 1700000000
            assert "__pycache__" not in member.name
            assert not member.name.endswith((".pyc", ".pt", ".bin"))
    launcher = extracted / "usr/bin/tensegrity"
    assert launcher.stat().st_mode & 0o111 == 0o111
    assert launcher.read_text().startswith("#!/usr/bin/python3\n")
    manual = gzip.decompress((extracted / "usr/share/man/man1/tensegrity.1.gz").read_bytes())
    assert b"train, grow and run" in manual


def test_extracted_cli_imports_packaged_modules(deb, tmp_path):
    if any(importlib.util.find_spec(module) is None for module in ("torch", "numpy", "safetensors")):
        pytest.skip("install runtime dependencies to smoke-test the extracted package")
    _, _, extracted = deb
    modules = extracted / "usr/lib/python3/dist-packages"
    env = dict(os.environ, PYTHONPATH=str(modules), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    launcher = extracted / "usr/bin/tensegrity"
    result = run(sys.executable, str(launcher), "--help", cwd=tmp_path, env=env, text=True)
    assert "presets" in result.stdout
    result = run(sys.executable, str(launcher), "presets", cwd=tmp_path, env=env, text=True)
    assert "115" in result.stdout
    result = run(sys.executable, str(launcher), "doctor", "--smoke", "--json",
                 cwd=tmp_path, env=env, text=True)
    diagnostics = json.loads(result.stdout)
    assert diagnostics["ok"]
    result = run(sys.executable, "-c", "import tensegrity; print(tensegrity.__file__)",
                 cwd=tmp_path, env=env, text=True)
    assert str(modules) in result.stdout


def test_deb_installs_upgrades_and_removes_in_isolated_root(deb, tmp_path):
    if not shutil.which("fakeroot") or not shutil.which("dpkg"):
        pytest.skip("isolated lifecycle check requires fakeroot and dpkg")
    first, _, _ = deb
    target = tmp_path / "root"
    target.mkdir()
    base = ("fakeroot", "dpkg", f"--root={target}", f"--log={tmp_path / 'dpkg.log'}",
            "--force-not-root", "--force-depends", "--force-script-chrootless")
    # Dependencies are deliberately ignored in this empty filesystem. Runtime
    # behavior is checked by test_extracted_cli_imports_packaged_modules.
    run(*base, "--install", str(first))
    modules = target / "usr/lib/python3/dist-packages/tensegrity"
    assert (target / "usr/bin/tensegrity").is_file()
    assert (modules / "__init__.py").is_file()
    if shutil.which("py3compile"):
        assert list(modules.rglob("*.pyc"))
    run(*base, "--install", str(first))
    run(*base, "--remove", "tensegrity-lm")
    assert not (target / "usr/bin/tensegrity").exists()
    assert not modules.exists()
