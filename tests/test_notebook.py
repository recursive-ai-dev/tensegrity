"""The Colab notebook is a generated artefact: it must match the package exactly, be valid,
and run top-to-bottom headlessly (CPU, offline, SMOKE_TEST) including every optional section."""

import json
import os
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NB = os.path.join(ROOT, "notebooks", "Tensegrity_Colab.ipynb")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_notebook  # noqa: E402


def _load():
    with open(NB, encoding="utf-8") as fh:
        return json.load(fh)


def test_notebook_matches_generator_output():
    assert open(NB, encoding="utf-8").read() == build_notebook.render(build_notebook.build()), \
        "notebook is stale: run `python tools/build_notebook.py`"


def test_embedded_sources_are_exactly_the_package_files():
    seen = {}
    for c in _load()["cells"]:
        src = "".join(c["source"])
        m = re.match(r"%%writefile tensegrity_src/tensegrity/(\w+)\.py\n", src)
        if m:
            seen[m.group(1)] = src[m.end():]
    assert set(seen) == set(build_notebook.MODULE_ORDER)
    pkg = os.path.join(ROOT, "tensegrity")
    on_disk = {f[:-3] for f in os.listdir(pkg) if f.endswith(".py")}
    assert set(seen) == on_disk, f"package modules not embedded in the notebook: {on_disk ^ set(seen)}"
    for name, text in seen.items():
        assert text.rstrip("\n") == open(os.path.join(pkg, f"{name}.py"), encoding="utf-8").read().rstrip("\n"), name


def test_notebook_is_valid_and_all_cells_compile():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(NB, as_version=4)
    nbformat.validate(nb)
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in nb.cells if c.cell_type == "code")
    ids = [c["id"] for c in nb.cells]
    assert len(ids) == len(set(ids))
    for c in nb.cells:
        if c.cell_type == "code":
            body = re.sub(r"^%%writefile .*\n", "", c.source)
            compile(body, c["id"], "exec")
    meta = nb.metadata
    assert meta["accelerator"] == "GPU" and meta["colab"]["collapsed_sections"]


def test_every_library_name_the_notebook_uses_exists():
    """Static guard: each `tg.<name>` referenced by a notebook cell must be exported by the package
    (a missing export is only otherwise discovered at the very end of a long Colab run)."""
    import tensegrity
    names = set()
    for c in _load()["cells"]:
        if c["cell_type"] == "code" and not "".join(c["source"]).startswith("%%writefile"):
            names |= set(re.findall(r"\btg\.(\w+)", "".join(c["source"])))
    assert len(names) > 40
    missing = sorted(n for n in names if not hasattr(tensegrity, n))
    assert not missing, f"notebook uses names the package does not export: {missing}"


def test_notebook_form_fields_and_presets_are_documented():
    text = "\n".join("".join(c["source"]) for c in _load()["cells"])
    for size in ("1M", "2M", "5M", "10M", "50M", "115M"):
        assert f'"{size}"' in text and f"**{size}**" in text
    assert "SMOKE_TEST" in text and "Open In Colab" in text and "colab-badge.svg" in text


def _flip(src: str, name: str, value: str) -> str:
    return re.sub(rf"^{name} = .*?(\s+#@param.*)?$", lambda m: f"{name} = {value}" + (m.group(1) or ""), src, flags=re.M)


@pytest.mark.slow
def test_notebook_executes_end_to_end_offline(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    from conftest import make_text
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "raw" / "tinyshakespeare.txt").write_text(make_text(120, seed=2, sep="\n\n"), encoding="utf-8")  # pre-seeded cache: no network
    cont = tmp_path / "new_corpus.txt"
    cont.write_text(make_text(60, seed=9, sep="\n").upper(), encoding="utf-8")

    nb = nbformat.read(NB, as_version=4)
    flips = {"SMOKE_TEST": "True", "USE_GOOGLE_DRIVE": "False", "RUN_THE_LADDER": "True", "LADDER_RUNGS": '"1M,2M"',
             "HOURS_PER_RUNG": '"0.004,0.004"', "GROW_NOW": "True", "RUN_CONTINUAL": "True",
             "NEW_DATASET": '"custom"', "NEW_CUSTOM_PATH": f'"{cont}"', "CONTINUAL_MEGABYTES_OF_TRAINING": "0.05",
             "RUN_BENCHMARK": "True", "BENCHMARK_SIZES": '"1M,2M"', "ALSO_INT8_CPU_COPY": "True"}
    hits = set()
    for c in nb.cells:
        if c.cell_type == "code" and not c.source.startswith("%%writefile"):
            for k, v in flips.items():
                new = _flip(c.source, k, v)
                if new != c.source:
                    hits.add(k)
                    c.source = new
    assert hits == set(flips), set(flips) - hits
    env = dict(os.environ, TENSEGRITY_DATA_DIR=str(data_dir), TENSEGRITY_PROJECT_DIR=str(tmp_path / "proj"),
               PYTHONWARNINGS="ignore")
    old_env = dict(os.environ)
    os.environ.update(env)
    try:
        client = nbclient.NotebookClient(nb, timeout=1500, kernel_name="python3",
                                         resources={"metadata": {"path": str(tmp_path)}})
        client.execute()
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    out = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.outputs
                    if o.get("output_type") == "stream")
    if os.environ.get("TENSEGRITY_NB_OUT"):          # debugging aid: keep the executed notebook's text output
        open(os.environ["TENSEGRITY_NB_OUT"], "w").write(out)
    import tensegrity
    for needle in ("dependencies OK", f"Tensegrity {tensegrity.__version__}", "fingerprint : sha256:", "integrity   : OK",
                   "data ready:", "training plan", "PREFLIGHT OK", "reload to identical outputs", "Finished. Best validation",
                   "validation: ", "bits/byte", "copy/retrieval probe", "growth ACCEPTED", "function preserved", "trainability:",
                   "validation unchanged at", "base model: old", "non-dominated", "forgetting", "tok/s", "wrote run report"):
        assert needle in out, f"missing in notebook output: {needle!r}"
    assert (tmp_path / "proj" / "runs").is_dir() and (tmp_path / "proj" / "ladders").is_dir()
    assert not any(o.get("output_type") == "error" for c in nb.cells if c.cell_type == "code" for o in c.outputs)
