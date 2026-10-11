"""Data pipeline: streaming ``prepare``, deterministic/resumable batching, dataset downloads.

On-disk format (unchanged from v1): ``train.bin`` / ``val.bin`` are flat ``uint16`` token
streams (bytes 0-255, EOS = 257 between documents) plus ``meta.json``. Existing prepared
directories keep working.
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .tokenizer import ByteTokenizer

SUPPORTED_EXTS = {".txt", ".md", ".parquet", ".jsonl"}
_TEXT_KEYS = ("text", "content", "body", "story")
_EOS = ByteTokenizer.EOS


# =============================================================================
# Streaming encoder: raw bytes -> uint16 tokens, optional document separator -> EOS
# =============================================================================

def _partial_suffix_len(tail: bytes, sep: bytes) -> int:
    """Length of the longest proper prefix of ``sep`` that ``tail`` ends with."""
    for k in range(min(len(sep) - 1, len(tail)), 0, -1):
        if tail.endswith(sep[:k]):
            return k
    return 0


class _StreamEncoder:
    """Feeds byte chunks in, emits uint16 token arrays. A literal ``sep`` (e.g. TinyStories'
    ``<|endoftext|>``) becomes ONE real EOS token instead of 13 ordinary bytes the model
    would have to learn to spell. A separator split across chunk boundaries is handled."""

    def __init__(self, sink, sep: Optional[bytes] = None):
        self.sink = sink
        self.sep = sep or None
        self._rx = re.compile(re.escape(self.sep)) if self.sep else None
        self.carry = b""
        self.n_tokens = 0
        self.n_docs = 0

    def _emit(self, arr: np.ndarray) -> None:
        if arr.size:
            self.n_tokens += int(arr.size)
            self.sink(arr)

    def _encode(self, data: bytes) -> None:
        if not data:
            return
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint16)
        if self._rx is None:
            self._emit(arr)
            return
        pos = np.fromiter((m.start() for m in self._rx.finditer(data)), dtype=np.int64)
        if pos.size:
            arr[pos] = _EOS
            if len(self.sep) > 1:
                keep = np.ones(arr.size, dtype=bool)
                for off in range(1, len(self.sep)):
                    keep[pos + off] = False
                arr = arr[keep]
            self.n_docs += int(pos.size)
        self._emit(arr)

    def feed(self, raw: bytes) -> None:
        data = self.carry + raw
        self.carry = b""
        if self._rx is not None:
            last_end = 0
            for m in self._rx.finditer(data):
                last_end = m.end()
            hold = _partial_suffix_len(data[last_end:], self.sep)
            if hold:
                self.carry, data = data[-hold:], data[:-hold]
        self._encode(data)

    def end_document(self) -> None:
        """Flush any held partial separator as ordinary bytes, then emit one EOS."""
        if self.carry:
            tail, self.carry = self.carry, b""
            self._encode(tail)
        self._emit(np.array([_EOS], dtype=np.uint16))
        self.n_docs += 1


# =============================================================================
# File readers
# =============================================================================

def _iter_jsonl_docs(path: str) -> Iterator[bytes]:
    bad = total = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            val = None
            if isinstance(obj, dict):
                for k in _TEXT_KEYS:
                    if obj.get(k):
                        val = obj[k]
                        break
                if val is None:
                    val = "\n".join(str(v) for v in obj.values() if isinstance(v, (str, int, float)))
            elif isinstance(obj, str):
                val = obj
            if val:
                yield str(val).encode("utf-8", errors="replace")
    if total and bad / total > 0.01:
        print(f"[prepare] warning: {bad}/{total} unparseable lines in {path}")


def _pick_text_column(names: Sequence[str], string_cols: Sequence[str], requested: str, path: str) -> str:
    if requested != "auto":
        if requested not in names:
            raise ValueError(f"{path}: column {requested!r} not found; available: {list(names)}")
        return requested
    for k in _TEXT_KEYS:
        if k in string_cols:
            return k
    if string_cols:
        return string_cols[0]
    raise ValueError(f"{path}: no string column to use as text; columns: {list(names)}")


def _iter_parquet_docs(path: str, text_column: str) -> Iterator[bytes]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError("reading .parquet needs `pip install pyarrow`") from e
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    string_cols = [f.name for f in schema if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
    col = _pick_text_column(schema.names, string_cols, text_column, path)
    for batch in pf.iter_batches(columns=[col], batch_size=4096):
        for s in batch.column(0).to_pylist():
            if s:
                yield str(s).encode("utf-8", errors="replace")


def _iter_text_chunks(path: str, chunk_bytes: int) -> Iterator[bytes]:
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                return
            yield chunk


def list_source_files(data_path: str) -> List[str]:
    if os.path.isdir(data_path):
        allf = sorted(f for f in glob.glob(os.path.join(data_path, "**", "*"), recursive=True) if os.path.isfile(f))
        files = [f for f in allf if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
        return files or allf  # fall back to everything if nothing has a known extension
    return [data_path]


# =============================================================================
# prepare
# =============================================================================

def prepare_data(data_path: str, out_dir: str, val_fraction: float = 0.1, seed: int = 1337,
                 shuffle_files: bool = False, doc_sep: Optional[str] = None,
                 text_column: str = "auto", max_bytes: Optional[int] = None,
                 chunk_bytes: int = 64 << 20, verbose: bool = True) -> Dict[str, Any]:
    """
    Stream text / markdown / parquet / jsonl file(s) into ``out_dir/{train,val}.bin``.

    * Memory use is O(chunk_bytes), not O(corpus): a multi-GB file is processed in chunks.
    * Plain-text files: one EOS after each file, plus one at every occurrence of ``doc_sep``
      (a literal string such as ``"<|endoftext|>"``). Parquet / jsonl: one EOS per row/record.
    * ``max_bytes``: stop after this many *source* bytes (cheap way to take a slice).
    * The last ``val_fraction`` of the token stream becomes the validation set.

    Returns the metadata dict that is also written to ``meta.json``.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    files = list_source_files(data_path)
    if not files:
        raise ValueError(f"No files found at {data_path}")
    if shuffle_files:
        files = list(files)
        random.Random(seed).shuffle(files)
    os.makedirs(out_dir, exist_ok=True)

    sep = doc_sep.encode("utf-8") if doc_sep else None
    tmp_path = os.path.join(out_dir, "stream.tmp")
    n_src = 0
    t0 = time.time()
    with open(tmp_path, "wb") as sink_f:
        enc = _StreamEncoder(lambda a: a.tofile(sink_f), sep)
        budget = max_bytes
        for f in files:
            if budget is not None and n_src >= budget:
                break
            ext = os.path.splitext(f)[1].lower()
            if ext == ".parquet" or ext == ".jsonl":
                docs = _iter_parquet_docs(f, text_column) if ext == ".parquet" else _iter_jsonl_docs(f)
                for doc in docs:
                    if budget is not None and n_src >= budget:
                        break
                    enc.feed(doc)
                    enc.end_document()
                    n_src += len(doc)
            else:
                for chunk in _iter_text_chunks(f, chunk_bytes):
                    if budget is not None and n_src + len(chunk) > budget:
                        chunk = chunk[: budget - n_src]
                    enc.feed(chunk)
                    n_src += len(chunk)
                    if verbose:
                        print(f"\r[prepare] {n_src / 1e6:8.1f} MB read, {enc.n_tokens / 1e6:8.1f}M tokens", end="")
                    if budget is not None and n_src >= budget:
                        break
                enc.end_document()
        n = enc.n_tokens
        n_docs = enc.n_docs
    if verbose:
        print()
    if n == 0:
        os.remove(tmp_path)
        raise ValueError(f"All files at {data_path} are empty; nothing to tokenize.")

    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    if n_train <= 0:
        os.remove(tmp_path)
        raise ValueError("Dataset too small after train/val split; provide more data.")

    # split in place: copy the tail to val.bin, truncate the stream to become train.bin
    val_path = os.path.join(out_dir, "val.bin")
    train_path = os.path.join(out_dir, "train.bin")
    stream = np.memmap(tmp_path, dtype=np.uint16, mode="r", shape=(n,))
    with open(val_path, "wb") as vf:
        for s in range(n_train, n, 1 << 24):
            np.asarray(stream[s: min(n, s + (1 << 24))]).tofile(vf)
    del stream
    os.truncate(tmp_path, n_train * 2)
    os.replace(tmp_path, train_path)

    meta = {
        "vocab_size": ByteTokenizer.VOCAB_SIZE,
        "dtype": "uint16",
        "n_train_tokens": int(n_train),
        "n_val_tokens": int(n_val),
        "n_documents": int(n_docs),
        "source_bytes": int(n_src),
        "source_files": files,
        "val_fraction": val_fraction,
        "shuffle_files": shuffle_files,
        "seed": seed if shuffle_files else None,
        "doc_sep": doc_sep,
        "max_bytes": max_bytes,
        "prepare_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    if verbose:
        print(f"[prepare] {len(files)} file(s) -> {n_train:,} train / {n_val:,} val tokens "
              f"({n_docs:,} documents) in {meta['prepare_seconds']}s -> {out_dir}")
    return meta


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
    """

    def __init__(self, data_dirs, weights: Optional[Sequence[float]] = None, seed: int = 1337, rank: int = 0):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.data_dirs = list(data_dirs)
        if not self.data_dirs:
            raise ValueError("data_dirs is empty")
        w = np.asarray(weights if weights is not None else [1.0] * len(self.data_dirs), dtype=np.float64)
        if len(w) != len(self.data_dirs) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative, sum > 0, one per data dir")
        self.weights = w / w.sum()
        self.rng = np.random.default_rng([seed, rank])
        self._arrs: Dict[Tuple[str, str], np.memmap] = {}

    def array(self, data_dir: str, split: str) -> np.memmap:
        key = (data_dir, split)
        arr = self._arrs.get(key)
        if arr is None:
            path = os.path.join(data_dir, f"{split}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found; run prepare_data first")
            arr = np.memmap(path, dtype=np.uint16, mode="r")
            self._arrs[key] = arr
        return arr

    def n_tokens(self, split: str, dir_index: int = 0) -> int:
        return len(self.array(self.data_dirs[dir_index], split))

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
# Downloads (resumable, retrying, with a clean cut at a document boundary)
# =============================================================================

_UA = {"User-Agent": "tensegrity-lm/2.0 (+https://github.com/recursive-ai-dev/tensegrity)"}


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


def download_file(url: str, dest: str, max_bytes: Optional[int] = None, retries: int = 5,
                  timeout: float = 30.0, chunk: int = 1 << 20, boundary: Optional[bytes] = None,
                  verbose: bool = True) -> str:
    """
    Download ``url`` to ``dest`` (streaming, never holds the file in memory).

    * Resumes a partial ``dest + '.part'`` with an HTTP Range request (restarts cleanly if
      the server ignores Range).
    * ``max_bytes``: stop after that many bytes (only that range is requested); with
      ``boundary`` the slice is then trimmed back to the last complete document.
    * Retries network errors with exponential backoff; raises after ``retries`` failures.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        if verbose:
            print(f"[download] using cached {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return dest
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
    os.replace(part, dest)
    if verbose:
        print(f"[download] saved {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
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
    """Download (cached) a named dataset into ``root`` and return its spec plus ``path``."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    spec = dict(DATASETS[name])
    max_bytes = int(max_mb * 1e6) if max_mb else None
    suffix = f".{int(max_mb)}mb" if max_bytes else ""
    dest = os.path.join(root, spec["filename"] + suffix)
    boundary = spec["doc_sep"].encode() if (max_bytes and spec["doc_sep"]) else None
    spec["path"] = download_file(spec["url"], dest, max_bytes=max_bytes, boundary=boundary, verbose=verbose)
    return spec


def fetch_hf_streaming(repo: str, config: Optional[str], split: str, text_field: str, out_path: str,
                       max_mb: float, doc_sep: str = "<|endoftext|>", verbose: bool = True) -> str:
    """Stream documents from a Hugging Face dataset (needs ``pip install datasets``) into one
    text file separated by ``doc_sep`` until ``max_mb`` megabytes were written."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError("this dataset needs `pip install datasets`") from e
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ds = load_dataset(repo, config, split=split, streaming=True)
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
    os.replace(part, out_path)
    if verbose:
        print(f"\n[stream] saved {out_path} ({written / 1e6:.1f} MB, {n_docs:,} docs)")
    return out_path
