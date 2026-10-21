"""Dataset identity: content hashes, manifests, fingerprints and verification.

A prepared dataset directory holds ``train.bin`` / ``val.bin`` (flat ``uint16`` token streams) and
``meta.json``. Since manifest version 1, ``meta.json`` is a *manifest*: besides the counts it records
the SHA-256 of both token files, the tokenizer and split policy that produced them, the provenance of
every source file, how many records were rejected and why, and a single ``fingerprint`` that names
the exact token content. A training run records the fingerprints it trained on, so "which data made
this model?" has a checkable answer and "resume on different data" is detectable.

Directories prepared by older releases have no fingerprint; :func:`data_identity` then derives a
cheap *quick* identity (file sizes plus the hashes of the first and last MiB of each file) that still
changes whenever a file is replaced by different data of the same name.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List

MANIFEST_VERSION = 1
_BLOCK = 1 << 22
_QUICK_EDGE = 1 << 20
SPLIT_FILES = ("train.bin", "val.bin")


class DatasetError(ValueError):
    """A prepared dataset is missing, inconsistent with its manifest, or not what a run expects."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, block: int = _BLOCK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(block)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def file_record(path: str) -> Dict[str, Any]:
    """``{"path", "bytes", "sha256"}`` of a file -- how lineage names the checkpoint a run started from."""
    return {"path": os.path.abspath(path), "bytes": os.path.getsize(path), "sha256": sha256_file(path)}


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes (sorted keys, no whitespace, no NaN) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def compute_fingerprint(manifest: Dict[str, Any]) -> str:
    """``sha256:<hex>`` over what the *model sees*: the tokenizer, the split policy and the content
    hashes and counts of the token files. Paths, timestamps and source provenance are excluded, so
    preparing the same data twice gives the same fingerprint."""
    split = dict(manifest["split"])
    split.pop("reason", None)            # human-readable commentary, not identity
    core = {"manifest_version": manifest.get("manifest_version", MANIFEST_VERSION),
            "tokenizer": manifest["tokenizer"], "split": split,
            "files": {k: {"sha256": v["sha256"], "bytes": v["bytes"]} for k, v in sorted(manifest["files"].items())},
            "counts": {k: manifest["counts"][k] for k in ("n_train_tokens", "n_val_tokens", "n_train_documents",
                                                          "n_val_documents")}}
    return "sha256:" + sha256_bytes(canonical_json(core))


def read_meta(data_dir: str) -> Dict[str, Any]:
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        raise DatasetError(f"{data_dir} has no meta.json; it was not produced by prepare_data()")
    with open(path) as fh:
        return json.load(fh)


def has_manifest(meta: Dict[str, Any]) -> bool:
    return "fingerprint" in meta and "files" in meta


def _quick_hash(path: str) -> str:
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(_QUICK_EDGE))
        if size > 2 * _QUICK_EDGE:
            fh.seek(size - _QUICK_EDGE)
            h.update(fh.read(_QUICK_EDGE))
    return h.hexdigest()


def quick_fingerprint(data_dir: str) -> str:
    """Identity for directories without a manifest: sizes + the first/last MiB of each token file."""
    parts = []
    for name in SPLIT_FILES:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            parts.append(f"{name}:{_quick_hash(p)}")
    if not parts:
        raise DatasetError(f"{data_dir} has neither train.bin nor val.bin")
    return "quick:" + sha256_bytes("|".join(parts).encode())


def verify_dataset(data_dir: str, deep: bool = False) -> List[str]:
    """Problems found with a prepared dataset (empty list = consistent).

    Shallow (default, O(1)): token files exist, sizes are whole ``uint16`` counts and match the
    manifest, and the manifest's fingerprint matches its own contents. ``deep=True`` additionally
    re-hashes both files against the manifest's SHA-256 (reads the whole dataset once)."""
    problems: List[str] = []
    for name in SPLIT_FILES:
        p = os.path.join(data_dir, name)
        if not os.path.exists(p):
            problems.append(f"{name} is missing")
        elif os.path.getsize(p) % 2:
            problems.append(f"{name} has an odd byte size ({os.path.getsize(p)}); not a uint16 stream")
    if problems:
        return problems
    try:
        meta = read_meta(data_dir)
    except DatasetError:
        return problems          # legacy / hand-made directory: nothing to compare against
    if not has_manifest(meta):
        for key, name in (("n_train_tokens", "train.bin"), ("n_val_tokens", "val.bin")):
            if key in meta and meta[key] * 2 != os.path.getsize(os.path.join(data_dir, name)):
                problems.append(f"{name} holds {os.path.getsize(os.path.join(data_dir, name)) // 2:,} tokens but "
                                f"meta.json says {meta[key]:,}")
        return problems
    for name, rec in meta["files"].items():
        p = os.path.join(data_dir, name)
        if os.path.getsize(p) != rec["bytes"]:
            problems.append(f"{name}: {os.path.getsize(p):,} bytes on disk, manifest says {rec['bytes']:,}")
        elif deep and sha256_file(p) != rec["sha256"]:
            problems.append(f"{name}: content hash differs from the manifest (file was modified after prepare)")
    if compute_fingerprint(meta) != meta["fingerprint"]:      # O(1): the manifest must agree with itself
        problems.append("manifest fingerprint does not match its own contents (meta.json was edited)")
    return problems


def data_identity(data_dir: str) -> Dict[str, Any]:
    """``{"fingerprint", "kind", "n_train_tokens", "n_val_tokens"}`` for a prepared directory.
    ``kind`` is ``"manifest"`` (hash recorded at prepare time) or ``"quick"`` (derived, legacy dirs)."""
    meta: Dict[str, Any] = {}
    try:
        meta = read_meta(data_dir)
    except DatasetError:
        pass
    if has_manifest(meta):
        fp, kind = meta["fingerprint"], "manifest"
    else:
        fp, kind = quick_fingerprint(data_dir), "quick"
    sizes = {n: os.path.getsize(os.path.join(data_dir, n)) // 2 for n in SPLIT_FILES
             if os.path.exists(os.path.join(data_dir, n))}
    return {"fingerprint": fp, "kind": kind, "n_train_tokens": sizes.get("train.bin", 0),
            "n_val_tokens": sizes.get("val.bin", 0)}


def identity_mismatches(saved: Dict[str, Dict[str, Any]], current: Dict[str, Dict[str, Any]]) -> List[str]:
    """Human-readable differences between two ``{dir_key: identity}`` mappings (keyed by position)."""
    out: List[str] = []
    if len(saved) != len(current):
        out.append(f"the run used {len(saved)} data director{'y' if len(saved) == 1 else 'ies'}, "
                   f"this session passes {len(current)}")
        return out
    for (sk, sv), (ck, cv) in zip(saved.items(), current.items()):
        if sv["fingerprint"] != cv["fingerprint"]:
            out.append(f"data #{sk}: fingerprint {sv['fingerprint'][:23]}… (checkpoint) vs {cv['fingerprint'][:23]}… "
                       f"(now) for {ck}")
    return out


def render_manifest_summary(meta: Dict[str, Any]) -> str:
    """Short multi-line description for logs and the notebook's data audit."""
    if not has_manifest(meta):
        return (f"legacy dataset (no manifest): {meta.get('n_train_tokens', '?'):,} train / "
                f"{meta.get('n_val_tokens', '?'):,} val tokens" if isinstance(meta.get("n_train_tokens"), int)
                else "legacy dataset (no manifest)")
    c, s = meta["counts"], meta["split"]
    rej = c.get("n_rejected", 0)
    docs = (f"({c['n_train_documents']:,} / {c['n_val_documents']:,} documents)" if s["policy"] == "document" else
            f"({c['n_train_documents'] + c['n_val_documents']:,} document(s) in total; the boundary cuts the token stream, not a document)")
    lines = [f"fingerprint : {meta['fingerprint']}",
             f"tokens      : {c['n_train_tokens']:,} train / {c['n_val_tokens']:,} val {docs}",
             f"split       : {s['policy']} (val_fraction {s['val_fraction']})" + (f" - {s['reason']}" if s.get("reason") else ""),
             f"sources     : {len(meta.get('sources', []))} file(s), {c.get('source_bytes', 0):,} source bytes",
             f"rejected    : {rej:,} record(s)" + (f" {c.get('rejected_reasons')}" if rej else "")]
    if c.get("n_duplicates_dropped"):
        lines.append(f"duplicates  : {c['n_duplicates_dropped']:,} exact duplicate document(s) dropped")
    return "\n".join(lines)
