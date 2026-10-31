#!/usr/bin/env python3
"""Build a native, reproducible Debian package without root, pip, or build isolation.

Build requirements: Python >= 3.11 and dpkg-deb. Runtime dependencies are resolved
by apt on the destination machine; no dependencies are vendored in this package.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import format_datetime
import gzip
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "tensegrity-lm"
DEFAULT_MAINTAINER = "Tensegrity contributors <noreply@localhost>"


def source_epoch(root: Path) -> int:
    """Use an explicit timestamp, the source commit, or a fixed archive fallback."""
    if "SOURCE_DATE_EPOCH" in os.environ:
        value = os.environ["SOURCE_DATE_EPOCH"]
    elif shutil.which("git"):
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct"], cwd=root,
            capture_output=True, text=True, check=False,
        )
        value = result.stdout.strip() if result.returncode == 0 else "946684800"
    else:
        value = "946684800"
    try:
        epoch = int(value)
        if epoch < 0:
            raise ValueError
        datetime.fromtimestamp(epoch, timezone.utc)
    except (ValueError, OverflowError, OSError):
        raise ValueError("SOURCE_DATE_EPOCH must be a nonnegative Unix timestamp") from None
    return epoch


def write(path: Path, content: str | bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    path.chmod(mode)


def build(output_dir: Path, revision: str, maintainer: str) -> Path:
    if not shutil.which("dpkg-deb"):
        raise ValueError("dpkg-deb is required (install the dpkg package)")
    if not re.fullmatch(r"[0-9][a-zA-Z0-9+.~]*", revision):
        raise ValueError("revision must start with a digit and contain only letters, digits, +, ., or ~")
    if not re.fullmatch(r"[^\r\n<>]+ <[^\s<>@]+@[^\s<>@]+>", maintainer):
        raise ValueError("maintainer must be 'Name <email>' without line breaks")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", version):
        raise ValueError("project.version must be a numeric release version for Debian packaging")
    python_requirement = project["requires-python"]
    match = re.fullmatch(r">=\s*([0-9]+\.[0-9]+)", python_requirement)
    if not match:
        raise ValueError("requires-python must have a single >= major.minor lower bound")
    if project["scripts"].get("tensegrity") != "tensegrity.cli:main":
        raise ValueError("update the Debian launcher for the changed tensegrity entry point")
    deb_version = f"{version}-{revision}"
    epoch = source_epoch(ROOT)
    date = format_datetime(datetime.fromtimestamp(epoch, timezone.utc))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    artifact = output_dir / f"{PACKAGE}_{deb_version}_all.deb"
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch), LC_ALL="C", TZ="UTC")

    with tempfile.TemporaryDirectory(prefix="tensegrity-deb-") as temp:
        staging = Path(temp) / "root"
        modules = staging / "usr/lib/python3/dist-packages"
        for source in sorted((ROOT / "tensegrity").rglob("*.py")):
            if "__pycache__" not in source.parts:
                write(modules / "tensegrity" / source.relative_to(ROOT / "tensegrity"), source.read_bytes())
        metadata = (
            "Metadata-Version: 2.1\n"
            f"Name: {project['name']}\nVersion: {version}\n"
            f"Summary: {project['description']}\nRequires-Python: {python_requirement}\n"
        )
        for dependency in project.get("dependencies", []):
            metadata += f"Requires-Dist: {dependency}\n"
        for extra, dependencies in project.get("optional-dependencies", {}).items():
            metadata += f"Provides-Extra: {extra}\n"
            for dependency in dependencies:
                metadata += f"Requires-Dist: {dependency}; extra == '{extra}'\n"
        info = modules / f"tensegrity_lm-{version}.dist-info"
        write(info / "METADATA", metadata + "\n")
        write(info / "INSTALLER", "deb\n")
        write(info / "entry_points.txt", "[console_scripts]\ntensegrity = tensegrity.cli:main\n")
        write(staging / "usr/bin/tensegrity", (
            "#!/usr/bin/python3\n"
            "from tensegrity.cli import main\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n"
        ), 0o755)
        docs = staging / f"usr/share/doc/{PACKAGE}"
        for source, name in ((ROOT / "README.md", "README.md.gz"),
                             (ROOT / "packaging/README.md", "README.Debian.gz")):
            write(docs / name, gzip.compress(source.read_bytes(), mtime=0))
        write(docs / "copyright", (ROOT / "packaging/debian/copyright").read_bytes())
        if (ROOT / "LICENSE").exists():
            write(docs / "LICENSE", (ROOT / "LICENSE").read_bytes())
        changelog = (
            f"{PACKAGE} ({deb_version}) unstable; urgency=medium\n\n"
            "  * Local build of the Tensegrity byte-level language model toolkit.\n\n"
            f" -- {maintainer}  {date}\n"
        )
        write(docs / "changelog.Debian.gz", gzip.compress(changelog.encode(), mtime=0))
        write(staging / "usr/share/man/man1/tensegrity.1.gz",
              gzip.compress((ROOT / "packaging/debian/tensegrity.1").read_bytes(), mtime=0))
        data_files = sorted(path for path in staging.rglob("*") if path.is_file())
        installed_size = (sum(path.stat().st_size for path in data_files) + 1023) // 1024
        control = (ROOT / "packaging/debian/control").read_text(encoding="utf-8").format(
            version=deb_version, maintainer=maintainer, installed_size=installed_size,
            python_minimum=match.group(1),
        )
        write(staging / "DEBIAN/control", control)
        write(staging / "DEBIAN/md5sums", "".join(
            f"{hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()}  {path.relative_to(staging)}\n"
            for path in data_files
        ))
        for name in ("postinst", "prerm"):
            write(staging / "DEBIAN" / name, (ROOT / "packaging/debian" / name).read_bytes(), 0o755)
        for path in sorted(staging.rglob("*")):
            if path.is_dir():
                path.chmod(0o755)
            os.utime(path, (epoch, epoch))
        staging.chmod(0o755)
        os.utime(staging, (epoch, epoch))
        pending = Path(temp) / artifact.name
        subprocess.run([
            "dpkg-deb", "--root-owner-group", "--uniform-compression", "-Zxz", "-z6",
            "--threads-max=1", "--build", str(staging), str(pending),
        ], check=True, env=env)
        # Stage on the destination filesystem before atomically replacing a prior build.
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".deb-", delete=False) as stream:
            temporary_output = Path(stream.name)
        try:
            shutil.copyfile(pending, temporary_output)
            temporary_output.chmod(0o644)
            os.replace(temporary_output, artifact)
        finally:
            temporary_output.unlink(missing_ok=True)
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    write(artifact.with_suffix(".deb.sha256"), f"{checksum}  {artifact.name}\n")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--revision", default="1", help="Debian revision (default: 1)")
    parser.add_argument("--maintainer", default=DEFAULT_MAINTAINER,
                        help="package contact as 'Name <email>' (default: local placeholder)")
    args = parser.parse_args()
    try:
        artifact = build(args.out_dir, args.revision, args.maintainer)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"build_deb: {exc}\n")
    print(artifact)
    print(f"SHA256: {artifact.with_suffix('.deb.sha256')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
