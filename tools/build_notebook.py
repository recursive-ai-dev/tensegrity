#!/usr/bin/env python3
"""Build notebooks/Tensegrity_Colab.ipynb from tools/nb/* and the tensegrity/ package.

The notebook is a GENERATED, self-contained artefact: its library cells are byte-for-byte the files
in tensegrity/, so a user can upload the .ipynb alone to Colab and everything works, while the
repository stays the single source of truth.

    python tools/build_notebook.py            # (re)write the notebook
    python tools/build_notebook.py --check    # exit 1 if the committed notebook is stale (used by CI)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "notebooks", "Tensegrity_Colab.ipynb")
MODULE_ORDER = ["config", "tokenizer", "layers", "model", "optim", "manifest", "ingest", "data", "checkpoint", "hw",
                "schedule", "evaluation", "growth", "continual", "train", "presets", "ladder", "workflow", "analysis",
                "export", "perf", "viz", "colab", "report", "cli", "__init__", "__main__"]


def _cell_id(kind: str, key: str) -> str:
    return hashlib.sha1(f"{kind}:{key}".encode()).hexdigest()[:12]


def _lines(src: str):
    parts = src.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def md(key: str, src: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id("md", key), "metadata": {}, "source": _lines(src.rstrip("\n"))}


def code(key: str, src: str, form: bool = False) -> dict:
    meta = {"cellView": "form"} if form else {}
    return {"cell_type": "code", "id": _cell_id("code", key), "metadata": meta, "execution_count": None,
            "outputs": [], "source": _lines(src.rstrip("\n"))}


def preset_table() -> str:
    from tensegrity.presets import LADDER, get_preset_config
    rows = ["| size | parameters | layers × width | heads (q/kv) | FFN | context | default byte budget | what it is for |",
            "|---|---:|---|---|---:|---:|---:|---|"]
    for p in LADDER:
        c = get_preset_config(p.name)
        rows.append(f"| **{p.label}** | {c.n_params() / 1e6:.2f}M | {c.n_layer} × {c.n_embd} | {c.n_head}/{c.n_kv_head} | "
                    f"{c.ffn_hidden} | {c.block_size} | {p.default_tokens / 1e6:,.0f}M | {p.blurb} |")
    return "\n".join(rows)


def build() -> dict:
    cells = []
    collapsed = []
    for path in sorted(glob.glob(os.path.join(ROOT, "tools", "nb", "*"))):
        name = os.path.basename(path)
        body = open(path, encoding="utf-8").read()
        if name.startswith("03_source_header"):
            n_lines = sum(open(os.path.join(ROOT, "tensegrity", f"{m}.py"), encoding="utf-8").read().count("\n")
                          for m in MODULE_ORDER)
            hdr = md(name, body.replace("{{SOURCE_LINES}}", f"{round(n_lines, -2):,}"))
            cells.append(hdr)
            collapsed.append(hdr["id"])
            # the package sources, one %%writefile cell per module
            cells.append(code("mkdir", 'import os\nos.makedirs("tensegrity_src/tensegrity", exist_ok=True)\n'
                                       'print("writing the tensegrity package to ./tensegrity_src/tensegrity/ ...")'))
            for mod in MODULE_ORDER:
                src = open(os.path.join(ROOT, "tensegrity", f"{mod}.py"), encoding="utf-8").read()
                cells.append(code(f"src-{mod}", f"%%writefile tensegrity_src/tensegrity/{mod}.py\n{src}"))
            continue
        if name.endswith(".md"):
            cells.append(md(name, body.replace("{{PRESET_TABLE}}", preset_table())))
        elif name.endswith(".py"):
            cells.append(code(name, body, form=body.lstrip().startswith("#@title")))
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "Tensegrity_Colab.ipynb", "provenance": [], "gpuType": "T4",
                      "collapsed_sections": collapsed, "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def render(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the notebook on disk is out of date")
    args = ap.parse_args()
    text = render(build())
    if args.check:
        have = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if have != text:
            sys.exit("notebooks/Tensegrity_Colab.ipynb is stale: run `python tools/build_notebook.py` and commit it.")
        print("notebook is up to date")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {OUT} ({len(text) / 1e3:.0f} kB, {len(json.loads(text)['cells'])} cells)")


if __name__ == "__main__":
    main()
