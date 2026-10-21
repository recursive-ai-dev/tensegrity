"""Data access: deterministic/resumable batching over prepared datasets, dataset downloads, audits.

Ingestion (``prepare_data`` and friends) lives in :mod:`tensegrity.ingest` and is re-exported here, so
``tensegrity.data.prepare_data`` keeps working. On-disk format (unchanged from v1): ``train.bin`` /
``val.bin`` are flat ``uint16`` token streams (bytes 0-255, EOS = 257 after every document) plus a
``meta.json`` (a verifiable manifest since manifest version 1, see :mod:`tensegrity.manifest`).
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import manifest as M
from .ingest import (SUPPORTED_EXTS, _StreamEncoder, _partial_suffix_len, list_source_files,  # noqa: F401
                     prepare_data)
from .tokenizer import ByteTokenizer


def load_meta(data_dir: str) -> Dict[str, Any]:
    with open(os.path.join(data_dir, "meta.json")) as fh:
        return json.load(fh)


# =============================================================================
# Batching
# =============================================================================

class ByteDataset:
    """Random-window sampler over one or more prepared directories.

    * Owns a private ``numpy.random.Generator``: batch sampling never perturbs (or is
      perturbed by) global RNG use elsewhere, and its state is checkpointable
      (``state_dict`` / ``load_state_dict``) so a resumed run continues the exact sequence.
    * With several directories, each row of a batch is drawn from directory ``i`` with
      probability ``weights[i]`` (replay / continual-learning mixes).
    * ``fixed_batches`` gives a deterministic evaluation set, so validation numbers are
      comparable from step to step (the v1 loop drew fresh random batches every eval).
    * Memory maps are tied to the file they opened (inode, size, mtime). If a token file is replaced
      or rewritten while the dataset object lives, the next access notices: ``on_change="remap"``
      (default) warns and maps the new file, ``on_change="error"`` raises -- the Trainer uses the
      latter, because the data behind a run must not change under it.
    """

    def __init__(self, data_dirs, weights: Optional[Sequence[float]] = None, seed: int = 1337, rank: int = 0,
                 on_change: str = "remap"):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.data_dirs = list(data_dirs)
        if not self.data_dirs:
            raise ValueError("data_dirs is empty")
        if on_change not in ("remap", "error"):
            raise ValueError(f"on_change must be 'remap' or 'error', got {on_change!r}")
        w = np.asarray(weights if weights is not None else [1.0] * len(self.data_dirs), dtype=np.float64)
        if len(w) != len(self.data_dirs) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative, sum > 0, one per data dir")
        self.weights = w / w.sum()
        self.rng = np.random.default_rng([seed, rank])
        self.on_change = on_change
        self.n_remaps = 0
        self._arrs: Dict[Tuple[str, str], np.memmap] = {}
        self._sigs: Dict[Tuple[str, str], Tuple[int, int, int]] = {}

    # ------------------------------------------------------------------ storage
    @staticmethod
    def _signature(path: str) -> Tuple[int, int, int]:
        st = os.stat(path)
        return st.st_ino, st.st_size, st.st_mtime_ns

    def array(self, data_dir: str, split: str) -> np.memmap:
        key = (data_dir, split)
        path = os.path.join(data_dir, f"{split}.bin")
        try:
            sig = self._signature(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"{path} not found; run prepare_data first") from None
        arr = self._arrs.get(key)
        if arr is not None and self._sigs[key] == sig:
            return arr
        if arr is not None:
            msg = (f"{path} changed on disk while it was open (was inode/size/mtime {self._sigs[key]}, "
                   f"now {sig})")
            if self.on_change == "error":
                raise RuntimeError(msg + ". Prepared datasets must not change during a run; prepare the new data "
                                         "into a new directory instead.")
            warnings.warn(msg + "; mapping the new file.", stacklevel=2)
            self.n_remaps += 1
            self._arrs.pop(key, None)
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        self._arrs[key], self._sigs[key] = arr, sig
        return arr

    def close(self) -> None:
        """Drop every memory map (they are re-opened lazily on the next access)."""
        self._arrs.clear()
        self._sigs.clear()

    def n_tokens(self, split: str, dir_index: int = 0) -> int:
        return len(self.array(self.data_dirs[dir_index], split))

    def identity(self) -> Dict[str, Dict[str, Any]]:
        """``{"0": {"dir", "fingerprint", "kind", ...}, ...}`` -- what a checkpoint records about the data."""
        return {str(i): {"dir": d, **M.data_identity(d)} for i, d in enumerate(self.data_dirs)}

    def verify(self, deep: bool = False) -> None:
        """Raise :class:`~tensegrity.manifest.DatasetError` if any directory disagrees with its manifest."""
        for d in self.data_dirs:
            problems = M.verify_dataset(d, deep=deep)
            if problems:
                raise M.DatasetError(f"dataset {d} is inconsistent: " + "; ".join(problems))

    # ----------------------------------------------------------------- sampling
    def _check(self, arr: np.ndarray, path: str, split: str, block_size: int) -> None:
        if len(arr) <= block_size + 1:
            raise ValueError(f"{path}/{split}.bin has only {len(arr)} tokens, need > block_size+1 "
                             f"({block_size + 1}). Use a larger corpus or a smaller block_size.")

    def _gather(self, arr: np.ndarray, starts: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = starts[:, None] + np.arange(block_size + 1)[None, :]
        xy = arr[idx].astype(np.int64)
        return xy[:, :-1], xy[:, 1:]

    def _to_device(self, x: np.ndarray, y: np.ndarray, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        xt = torch.from_numpy(np.ascontiguousarray(x))
        yt = torch.from_numpy(np.ascontiguousarray(y))
        if str(device).startswith("cuda"):
            return xt.pin_memory().to(device, non_blocking=True), yt.pin_memory().to(device, non_blocking=True)
        return xt.to(device), yt.to(device)

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str):
        x = np.empty((batch_size, block_size), dtype=np.int64)
        y = np.empty((batch_size, block_size), dtype=np.int64)
        if len(self.data_dirs) == 1:
            choices = np.zeros(batch_size, dtype=np.int64)
        else:
            choices = self.rng.choice(len(self.data_dirs), size=batch_size, p=self.weights)
        for d_idx, d in enumerate(self.data_dirs):
            rows = np.nonzero(choices == d_idx)[0]
            if rows.size == 0:
                continue
            arr = self.array(d, split)
            self._check(arr, d, split, block_size)
            # a window starting at s needs tokens s .. s+block_size, so s <= len - block_size - 1;
            # `integers` excludes its upper bound, so len - block_size is exactly right.
            starts = self.rng.integers(0, len(arr) - block_size, size=rows.size)
            x[rows], y[rows] = self._gather(arr, starts, block_size)
        return self._to_device(x, y, device)

    def fixed_batches(self, split: str, n_batches: int, batch_size: int, block_size: int,
                      device: str, dir_index: int = 0, seed: int = 0):
        """Deterministic evaluation windows from ``data_dirs[dir_index]`` (same every call)."""
        arr = self.array(self.data_dirs[dir_index], split)
        self._check(arr, self.data_dirs[dir_index], split, block_size)
        rng = np.random.default_rng([seed, 7919])
        out = []
        for _ in range(n_batches):
            starts = rng.integers(0, len(arr) - block_size, size=batch_size)
            x, y = self._gather(arr, starts, block_size)
            out.append(self._to_device(x, y, device))
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {"bit_generator": self.rng.bit_generator.state}

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if state and "bit_generator" in state:
            self.rng.bit_generator.state = state["bit_generator"]


# =============================================================================
# Audit
# =============================================================================

def _decode_head(path: str, n_examples: int, example_bytes: int, read_tokens: int = 1 << 20) -> List[str]:
    toks = np.fromfile(path, dtype=np.uint16, count=read_tokens)
    out, start = [], 0
    for e in np.flatnonzero(toks == ByteTokenizer.EOS).tolist():
        doc = toks[start:e]
        start = e + 1
        if doc.size:
            out.append(bytes(doc[doc < 256].astype(np.uint8)[:example_bytes]).decode("utf-8", errors="replace"))
        if len(out) >= n_examples:
            break
    if not out and toks.size:
        out.append(bytes(toks[toks < 256].astype(np.uint8)[:example_bytes]).decode("utf-8", errors="replace"))
    return out


def audit_dataset(data_dir: str, n_examples: int = 2, example_bytes: int = 200, deep: bool = False) -> Dict[str, Any]:
    """Everything worth looking at before spending compute on a dataset: the manifest summary,
    consistency problems (``deep=True`` re-hashes the files), decoded example documents from both
    splits, and byte statistics (a corpus of mostly control bytes is not text). Returns a dict whose
    ``"text"`` entry is ready to print."""
    problems = M.verify_dataset(data_dir, deep=deep)
    meta = M.read_meta(data_dir) if os.path.exists(os.path.join(data_dir, "meta.json")) else {}
    ident = M.data_identity(data_dir)
    stats: Dict[str, Any] = {}
    for split in ("train", "val"):
        arr = np.fromfile(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, count=1 << 22)
        b = arr[arr < 256]
        stats[split] = {
            "sampled_tokens": int(arr.size), "eos_rate": float(np.mean(arr == ByteTokenizer.EOS)) if arr.size else 0.0,
            "printable_fraction": float(np.mean(((b >= 32) & (b < 127)) | (b == 10) | (b == 9) | (b >= 128))) if b.size else 0.0,
            "control_fraction": float(np.mean((b < 32) & (b != 10) & (b != 9) & (b != 13))) if b.size else 0.0}
    examples = {s: _decode_head(os.path.join(data_dir, f"{s}.bin"), n_examples, example_bytes) for s in ("train", "val")}
    lines = [f"dataset {data_dir}", M.render_manifest_summary(meta) if meta else f"identity    : {ident['fingerprint']}"]
    if ident["kind"] == "quick":
        lines.append("note        : no manifest (prepared by an older release); identity is a quick size+edges hash")
    lines.append("integrity   : " + ("OK" + (" (content re-hashed)" if deep else " (sizes match the manifest)")
                                      if not problems else "PROBLEMS: " + "; ".join(problems)))
    for s in ("train", "val"):
        st = stats[s]
        lines.append(f"{s:5s} bytes  : {100 * st['printable_fraction']:.1f}% text-like, {100 * st['control_fraction']:.2f}% "
                     f"control, 1 EOS per {1 / st['eos_rate']:.0f} tokens" if st["eos_rate"] else
                     f"{s:5s} bytes  : {100 * st['printable_fraction']:.1f}% text-like, no EOS markers in the sample")
        for ex in examples[s]:
            lines.append(f"  {s} example: {ex!r}")
    return {"fingerprint": ident["fingerprint"], "kind": ident["kind"], "problems": problems, "stats": stats,
            "examples": examples, "meta": meta, "text": "\n".join(lines)}


# =============================================================================
# Downloads (resumable, retrying, hash-recorded, with a clean cut at a document boundary)
# =============================================================================

_UA = {"User-Agent": "tensegrity-lm/2.1 (+https://github.com/recursive-ai-dev/tensegrity)"}


def _truncate_at_boundary(path: str, boundary: bytes, window: int = 8 << 20) -> None:
    """Cut ``path`` right after the LAST ``boundary`` in its final ``window`` bytes so a
    byte-limited slice never ends mid-document."""
    size = os.path.getsize(path)
    start = max(0, size - window)
    with open(path, "r+b") as fh:
        fh.seek(start)
        tail = fh.read()
        k = tail.rfind(boundary)
        if k >= 0:
            fh.truncate(start + k + len(boundary))


def _sidecar(dest: str) -> str:
    return dest + ".meta.json"


def read_download_record(dest: str) -> Optional[Dict[str, Any]]:
    """The provenance record written next to a downloaded file (url, size, sha256), or None."""
    try:
        with open(_sidecar(dest)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def download_file(url: str, dest: str, max_bytes: Optional[int] = None, retries: int = 5,
                  timeout: float = 30.0, chunk: int = 1 << 20, boundary: Optional[bytes] = None,
                  verbose: bool = True, expected_sha256: Optional[str] = None) -> str:
    """
    Download ``url`` to ``dest`` (streaming, never holds the file in memory).

    * Resumes a partial ``dest + '.part'`` with an HTTP Range request (restarts cleanly if
      the server ignores Range).
    * ``max_bytes``: stop after that many bytes (only that range is requested); with
      ``boundary`` the slice is then trimmed back to the last complete document.
    * Retries network errors with exponential backoff; raises after ``retries`` failures.
    * The finished file's SHA-256 is recorded in ``dest + '.meta.json'``; ``expected_sha256`` makes a
      mismatch an error (the partial file is discarded). A cached ``dest`` whose size disagrees with its
      record is treated as corrupt and fetched again.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        rec = read_download_record(dest)
        ok = rec is None or rec.get("bytes") == os.path.getsize(dest)
        if ok and expected_sha256 is not None:
            ok = (rec or {}).get("sha256", M.sha256_file(dest)) == expected_sha256
        if ok:
            if verbose:
                print(f"[download] using cached {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
            return dest
        if verbose:
            print(f"[download] cached {dest} does not match its record; downloading again")
        os.replace(dest, dest + f".corrupt-{int(time.time())}")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    part = dest + ".part"
    last_err: Optional[BaseException] = None
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if max_bytes is not None and have >= max_bytes:
            break
        headers = dict(_UA)
        if have or max_bytes is not None:
            end = "" if max_bytes is None else str(max_bytes - 1)
            headers["Range"] = f"bytes={have}-{end}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                total = resp.headers.get("Content-Length")
                total = int(total) + (have if status == 206 else 0) if total else None
                mode = "ab" if (status == 206 and have) else "wb"
                if mode == "wb":
                    have = 0
                got, t_last = have, time.time()
                with open(part, mode) as out:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        if max_bytes is not None and got + len(buf) > max_bytes:
                            buf = buf[: max_bytes - got]
                        out.write(buf)
                        got += len(buf)
                        if verbose and time.time() - t_last > 2.0:
                            pct = f" ({100 * got / total:5.1f}%)" if total else ""
                            print(f"\r[download] {got / 1e6:9.1f} MB{pct}", end="")
                            t_last = time.time()
                        if max_bytes is not None and got >= max_bytes:
                            break
            if total is not None and max_bytes is None and got < total:
                raise ConnectionError(f"short read: {got} of {total} bytes")
            last_err = None
            break
        except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError, OSError) as e:
            last_err = e
            wait = min(60, 2 ** attempt)
            if verbose:
                print(f"\n[download] attempt {attempt + 1}/{retries} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    if last_err is not None:
        raise RuntimeError(f"download of {url} failed after {retries} attempts: {last_err}") from last_err
    if verbose:
        print()
    if boundary:
        _truncate_at_boundary(part, boundary)
    digest = M.sha256_file(part)
    if expected_sha256 is not None and digest != expected_sha256:
        os.remove(part)
        raise RuntimeError(f"{url}: SHA-256 {digest} does not match the expected {expected_sha256}; "
                           f"the partial download was discarded")
    size = os.path.getsize(part)
    os.replace(part, dest)
    with open(_sidecar(dest), "w") as fh:
        json.dump({"url": url, "bytes": size, "sha256": digest, "max_bytes": max_bytes,
                   "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh, indent=2)
    if verbose:
        print(f"[download] saved {dest} ({size / 1e6:.1f} MB, sha256 {digest[:16]}...)")
    return dest


# name -> spec.  `doc_sep` is the literal separator inside the text file.
DATASETS: Dict[str, Dict[str, Any]] = {
    "shakespeare": dict(
        url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        filename="tinyshakespeare.txt", doc_sep=None, val_fraction=0.05, approx_mb=1.1,
        description="Tiny Shakespeare (1.1 MB). Smoke tests / quick demos."),
    "tinystories-valid": dict(
        url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
        filename="TinyStories-valid.txt", doc_sep="<|endoftext|>", val_fraction=0.05, approx_mb=19,
        description="TinyStories validation split (19 MB). Good for the 1M-5M models."),
    "tinystories": dict(
        url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
        filename="TinyStories-train.txt", doc_sep="<|endoftext|>", val_fraction=0.01, approx_mb=1920,
        description="TinyStories train split (1.9 GB; use max_mb to take a slice)."),
}


def fetch_dataset(name: str, root: str, max_mb: Optional[float] = None, verbose: bool = True) -> Dict[str, Any]:
    """Download (cached) a named dataset into ``root`` and return its spec plus ``path`` and ``sha256``."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    spec = dict(DATASETS[name])
    max_bytes = int(max_mb * 1e6) if max_mb else None
    suffix = f".{int(max_mb)}mb" if max_bytes else ""
    dest = os.path.join(root, spec["filename"] + suffix)
    boundary = spec["doc_sep"].encode() if (max_bytes and spec["doc_sep"]) else None
    spec["path"] = download_file(spec["url"], dest, max_bytes=max_bytes, boundary=boundary, verbose=verbose,
                                 expected_sha256=spec.get("sha256"))
    spec["sha256"] = (read_download_record(dest) or {}).get("sha256")
    return spec


def fetch_hf_streaming(repo: str, config: Optional[str], split: str, text_field: str, out_path: str,
                       max_mb: float, doc_sep: str = "<|endoftext|>", verbose: bool = True,
                       revision: Optional[str] = None) -> str:
    """Stream documents from a Hugging Face dataset (needs ``pip install datasets``) into one
    text file separated by ``doc_sep`` until ``max_mb`` megabytes were written.

    ``revision`` pins a commit/tag/branch of the dataset repository. Without it the *current* head is
    read, which is not reproducible; a warning says so and the sidecar record notes ``revision: null``."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError("this dataset needs `pip install datasets`") from e
    if revision is None:
        warnings.warn(f"fetch_hf_streaming({repo!r}): no revision pinned; the data may change between runs. "
                      f"Pass revision='<commit sha>' to make this download reproducible.", stacklevel=2)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ds = load_dataset(repo, config, split=split, streaming=True, revision=revision)
    budget, written, n_docs, t_last = int(max_mb * 1e6), 0, 0, time.time()
    part = out_path + ".part"
    with open(part, "wb") as fh:
        for row in ds:
            text = (row.get(text_field) or "").replace(doc_sep, "")
            if not text:
                continue
            blob = text.encode("utf-8", errors="replace") + doc_sep.encode()
            fh.write(blob)
            written += len(blob)
            n_docs += 1
            if verbose and time.time() - t_last > 2.0:
                print(f"\r[stream] {written / 1e6:8.1f} / {max_mb:.0f} MB ({n_docs:,} docs)", end="")
                t_last = time.time()
            if written >= budget:
                break
    digest = M.sha256_file(part)
    os.replace(part, out_path)
    with open(_sidecar(out_path), "w") as fh:
        json.dump({"source": f"huggingface:{repo}", "config": config, "split": split, "revision": revision,
                   "text_field": text_field, "bytes": written, "documents": n_docs, "sha256": digest,
                   "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh, indent=2)
    if verbose:
        print(f"\n[stream] saved {out_path} ({written / 1e6:.1f} MB, {n_docs:,} docs, sha256 {digest[:16]}...)")
    return out_path
