# Debian package

Build the package without root from a checkout using Python 3.11+ and `dpkg-deb`
(provided by `dpkg`):

```bash
python3 tools/build_deb.py
dpkg-deb --info dist/tensegrity-lm_*.deb
```

The builder reads the release version from `pyproject.toml` and writes an
architecture-independent `.deb` plus its SHA-256 file to `dist/`. No Python
dependencies are required to build. `--out-dir`, `--revision` and `--maintainer
'Name <email>'` customize the output. The default maintainer contact is an
explicit local placeholder; set a real contact when distributing a release.

The same sources, builder version, revision and `SOURCE_DATE_EPOCH` produce the
same package bytes. The timestamp defaults to the most recent Git commit, or
2000-01-01 when building from an archive without Git metadata. Compression is
single-threaded, ownership is normalized to root, and file modes and timestamps
are normalized. A build replaces an existing artifact with the same name only
after the new package has completed successfully.

## Install

Target Debian 13 (trixie) or a newer Debian/Ubuntu release whose repositories
provide `python3-torch >= 2.3`, `python3-numpy >= 1.23` and
`python3-safetensors >= 0.4`. These dependencies are available on Debian 13 and
Ubuntu 26.04. Earlier distributions may not satisfy the dependency versions;
APT reports that rather than installing an incompatible runtime.

```bash
sudo apt install ./dist/tensegrity-lm_2.2.0-1_all.deb
tensegrity --help
tensegrity --version
tensegrity doctor --smoke
tensegrity presets
man tensegrity
```

APT resolves dependencies from the distribution repositories. The package
installs the `tensegrity` command, Python modules under
`/usr/lib/python3/dist-packages`, a manual and documentation. Installation does
not use pip, access external package indexes, download models, or configure
background services. Python bytecode is compiled by the distribution tools at
installation and removed on upgrade/removal.

The package uses distribution PyTorch and its supported hardware backends.
For a separate CUDA-enabled PyTorch environment, use the project's Python
installation workflow in an isolated environment instead. Parquet ingestion
requires optional `python3-pyarrow`; plotting requires optional
`python3-matplotlib`. Hugging Face integrations require their corresponding
optional Python packages.

Remove the package with `sudo apt remove tensegrity-lm`. User-created corpora,
checkpoints and run directories remain in the paths selected when running it.

## Check without installing on the host

```bash
python3 -m pytest -q tests/test_packaging.py
dpkg-deb --contents dist/tensegrity-lm_2.2.0-1_all.deb
```

The tests build twice and verify reproducibility, control metadata, archive
ownership, checksums, executable permissions and the extracted command,
including `doctor --smoke --json`. When `fakeroot` is available they also install,
reinstall and remove the package in a temporary dpkg root. That lifecycle check
ignores dependencies only within the empty test root; it does not alter the host
package database. Maintainer scripts respect `DPKG_ROOT` for these installs. The
command smoke test uses the current Python interpreter with the extracted
modules and runs outside the checkout so it cannot accidentally import the
working tree. It requires the runtime dependencies to be installed for that
interpreter.

The repository did not include an upstream license at the time packaging was
added. The package records this fact instead of inventing a license. Add a
license grant before redistributing; a root `LICENSE` file is included verbatim
by the builder.

Dependency reference: [Debian 13 PyTorch](https://packages.debian.org/trixie/python3-torch)
and [Debian 13 safetensors](https://packages.debian.org/trixie/python3-safetensors).
