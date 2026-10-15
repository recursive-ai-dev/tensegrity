"""Corpus ingestion: validated documents -> deterministic document-level split -> packed tokens.

The contract (the reason this module exists):

* **A reader returns validated documents in its declared format or raises an actionable error.**
  It never falls back to training on arbitrary file bytes: unknown extensions are an error (unless
  ``format=`` says how to read them), a ``.jsonl`` / ``.parquet`` record without a usable text field
  is *rejected and counted* (never glued together from whatever columns it has), text files must be
  valid UTF-8, and if more than ``max_reject_fraction`` of a file's records are rejected the whole
  preparation fails instead of quietly training on the remainder.
* **Documents are split before tokens are packed.** Every document (a record, a ``doc_sep``-delimited
  segment, or a whole file) is assigned to train or validation by a hash of its content (or of a
  ``group_field`` such as a shared context/id), so the same document always lands in the same split,
  exact duplicates can never straddle it, and a document is never cut in two by the boundary. A
  contiguous ``tail`` split is still available (and chosen, loudly, for corpora with too few documents
  to split by document, e.g. one 1 MB text file).
* **Every prepared directory is a verifiable artefact.** ``meta.json`` is a manifest (see
  :mod:`tensegrity.manifest`) with content hashes, counts, rejected-record accounting and a
  fingerprint. Preparing is staged in a temporary directory and published file by file, so a crash
  never leaves a half-written dataset, and an existing *different* dataset is never overwritten
  without ``overwrite=True`` (the old files are moved aside, not deleted).

On-disk format is unchanged from v1/v2: ``train.bin`` / ``val.bin`` are flat ``uint16`` streams
(bytes 0-255, EOS = 257 after every document). Existing prepared directories keep working.
"""

from __future__ import annotations

import codecs
import glob
import hashlib
import json
import os
import random
import re
import shutil
import string
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import manifest as M
from .tokenizer import ByteTokenizer

SUPPORTED_EXTS = {".txt": "text", ".md": "text", ".jsonl": "jsonl", ".parquet": "parquet"}
FORMATS = ("text", "jsonl", "parquet")
_TEXT_KEYS = ("text", "content", "body", "story")
_EOS = ByteTokenizer.EOS
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_U64 = float(2 ** 64)

Record = Tuple[bytes, Optional[bytes]]       # (document bytes, optional split-group key)


# =============================================================================
# Streaming encoder: raw bytes -> uint16 tokens, optional document separator -> EOS
# =============================================================================

def _partial_suffix_len(tail: bytes, sep: bytes) -> int:
    """Length of the longest proper prefix of ``sep`` that ``tail`` ends with."""
    for k in range(min(len(sep) - 1, len(tail)), 0, -1):
        if tail.endswith(sep[:k]):
            return k
    return 0


def _hash64(*parts: bytes) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p)
    return int.from_bytes(h.digest(), "little")


_INDEX_DTYPE = np.dtype([("end", "<u8"), ("split", "<u8"), ("content", "<u8")])


class _DocIndex:
    """Append-only (end offset, split hash, content hash) record per document, spilled to disk so a
    corpus with tens of millions of documents needs O(chunk) memory."""

    def __init__(self, path: str, flush_at: int = 1 << 16):
        self.path = path
        self.n = 0
        self._fh = open(path, "wb")
        self._buf: List[Tuple[int, int, int]] = []
        self._flush_at = flush_at

    def add(self, end: int, split_hash: int, content_hash: int) -> None:
        self._buf.append((end, split_hash, content_hash))
        self.n += 1
        if len(self._buf) >= self._flush_at:
            self._flush()

    def _flush(self) -> None:
        if self._buf:
            np.array(self._buf, dtype=_INDEX_DTYPE).tofile(self._fh)
            self._buf = []

    def close(self) -> None:
        if not self._fh.closed:
            self._flush()
            self._fh.close()

    def chunks(self, size: int = 1 << 20) -> Iterator[np.ndarray]:
        with open(self.path, "rb") as fh:
            while True:
                a = np.fromfile(fh, dtype=_INDEX_DTYPE, count=size)
                if a.size == 0:
                    return
                yield a


class _StreamEncoder:
    """Feeds byte chunks in, emits uint16 token arrays. A literal ``sep`` (e.g. TinyStories'
    ``<|endoftext|>``) becomes ONE real EOS token instead of 13 ordinary bytes the model would have
    to learn to spell. A separator split across chunk boundaries is handled.

    With an ``index`` it also records, for every document (every EOS), where it ends and two 64-bit
    hashes: of its content and of its split key (the content, unless a record supplied a group key
    through :meth:`begin_record`). ``salt`` makes the split seed-dependent."""

    def __init__(self, sink, sep: Optional[bytes] = None, index: Optional[_DocIndex] = None, salt: bytes = b""):
        self.sink = sink
        self.sep = sep or None
        self._rx = re.compile(re.escape(self.sep)) if self.sep else None
        self.carry = b""
        self.n_tokens = 0
        self.n_docs = 0
        self.index = index
        self._salt = salt
        self._hasher = None
        self._group: Optional[int] = None

    # ---- document bookkeeping
    def begin_record(self, group_key: Optional[bytes]) -> None:
        """Documents emitted until the next ``end_document`` split by ``group_key`` instead of content."""
        self._group = None if group_key is None else _hash64(self._salt, b"\x00group\x00", group_key)

    def _track(self, arr: np.ndarray) -> None:
        eos = np.flatnonzero(arr == _EOS)
        prev = 0
        for e in eos.tolist():
            if self._hasher is None:
                self._hasher = hashlib.blake2b(self._salt, digest_size=8)
            if e > prev:
                self._hasher.update(arr[prev:e].tobytes())
            content = int.from_bytes(self._hasher.digest(), "little")
            self._hasher = None
            self.index.add(self.n_tokens + e + 1, content if self._group is None else self._group, content)
            prev = e + 1
        if prev < arr.size:
            if self._hasher is None:
                self._hasher = hashlib.blake2b(self._salt, digest_size=8)
            self._hasher.update(arr[prev:].tobytes())

    def _emit(self, arr: np.ndarray) -> None:
        if arr.size:
            if self.index is not None:
                self._track(arr)
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
        self._group = None


# =============================================================================
# Record extraction (jsonl / parquet): explicit schema, rejects are counted
# =============================================================================

class _Rejects:
    """Counts records that were examined and dropped, by reason, with a few examples."""

    def __init__(self, max_examples: int = 5):
        self.counts: Dict[str, int] = {}
        self.examples: List[Dict[str, str]] = []
        self.n_seen = 0
        self._max = max_examples

    def seen(self) -> None:
        self.n_seen += 1

    def add(self, source: str, reason: str, example: str = "") -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if len(self.examples) < self._max:
            self.examples.append({"source": os.path.basename(source), "reason": reason, "example": example[:120]})

    @property
    def n_rejected(self) -> int:
        return sum(self.counts.values())

    def snapshot(self) -> Tuple[int, int]:
        return self.n_seen, self.n_rejected

    def check_since(self, snap: Tuple[int, int], source: str, max_fraction: float) -> None:
        seen, rejected = self.n_seen - snap[0], self.n_rejected - snap[1]
        if seen and rejected / seen > max_fraction:
            raise ValueError(
                f"{source}: {rejected:,} of {seen:,} records ({100 * rejected / seen:.1f}%) were rejected, more than "
                f"max_reject_fraction={max_fraction:g}. Reasons: {dict(self.counts)}. Examples: {self.examples}. "
                f"Fix the source, name the right field (text_field=/record_template=), or raise max_reject_fraction.")


class _RecordSpec:
    """How to turn one JSON object / table row into a document: a text field, or a template that
    stitches several fields into ONE record (keeping question/context/answer together)."""

    def __init__(self, text_field: Optional[str] = None, template: Optional[str] = None,
                 group_field: Optional[str] = None):
        if template is not None and text_field not in (None, "auto"):
            raise ValueError("pass either text_field/text_column or record_template, not both")
        self.text_field = text_field or "auto"
        self.template = template
        self.group_field = group_field
        self.template_fields = _template_fields(template) if template is not None else []

    def with_text_field(self, name: str) -> "_RecordSpec":
        return _RecordSpec(name, self.template, self.group_field)

    @staticmethod
    def _scalar(v: Any) -> Optional[str]:
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, (str, int, float)):
            return str(v)
        return None

    def extract(self, obj: Any) -> Tuple[Optional[Record], Optional[Tuple[str, str]]]:
        """-> (record, None) or (None, (reason, detail))."""
        group: Optional[bytes] = None
        if self.group_field is not None:
            if not isinstance(obj, dict) or obj.get(self.group_field) is None:
                return None, ("missing_group_field", self.group_field)
            g = self._scalar(obj[self.group_field])
            if g is None:
                return None, ("non_scalar_group_field", self.group_field)
            group = g.encode("utf-8", errors="replace")
        if self.template is not None:
            if not isinstance(obj, dict):
                return None, ("not_an_object", type(obj).__name__)
            values = {}
            for f in self.template_fields:
                if obj.get(f) is None:
                    return None, (f"missing_field:{f}", f)
                s = self._scalar(obj[f])
                if s is None:
                    return None, (f"non_scalar_field:{f}", f)
                values[f] = s
            text = self.template.format_map(values)
        elif isinstance(obj, str):
            if self.group_field is not None:
                return None, ("missing_group_field", self.group_field)
            text = obj
        elif isinstance(obj, dict):
            key = self.text_field
            if key == "auto":
                key = next((k for k in _TEXT_KEYS if k in obj), None)
                if key is None:
                    return None, ("no_text_field", f"keys {sorted(obj)[:8]}")
            elif key not in obj:
                return None, ("no_text_field", f"keys {sorted(obj)[:8]}")
            text = obj[key]
            if text is None:
                return None, ("null_text", key)
            if not isinstance(text, str):
                return None, ("non_string_text", f"{key}: {type(text).__name__}")
        else:
            return None, ("not_an_object", type(obj).__name__)
        if not text:
            return None, ("empty_text", "")
        try:
            return (text.encode("utf-8"), group), None
        except UnicodeEncodeError:
            return None, ("unencodable_text", text[:40])


def _template_fields(template: str) -> List[str]:
    names: List[str] = []
    for _, field, _, _ in string.Formatter().parse(template):
        if field is None:
            continue
        if not _IDENT.match(field):
            raise ValueError(f"record_template field {field!r} must be a plain column name like {{question}} "
                             f"(no indexing, attributes or empty braces)")
        names.append(field)
    if not names:
        raise ValueError("record_template has no {fields}")
    return list(dict.fromkeys(names))


def _iter_jsonl_records(path: str, spec: _RecordSpec, rej: _Rejects) -> Iterator[Record]:
    with open(path, "rb") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            rej.seen()
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                rej.add(path, "invalid_utf8", f"line {lineno}")
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                rej.add(path, "invalid_json", f"line {lineno}: {line[:80]}")
                continue
            rec, why = spec.extract(obj)
            if rec is None:
                rej.add(path, why[0], f"line {lineno}: {why[1]}")
                continue
            yield rec


def _pick_text_column(names: Sequence[str], string_cols: Sequence[str], requested: str, path: str) -> str:
    if requested != "auto":
        if requested not in names:
            raise ValueError(f"{path}: column {requested!r} not found; available: {list(names)}")
        if requested not in string_cols:
            raise ValueError(f"{path}: column {requested!r} is not a string column; string columns: {list(string_cols)}")
        return requested
    for k in _TEXT_KEYS:
        if k in string_cols:
            return k
    if len(string_cols) == 1:
        return string_cols[0]
    if not string_cols:
        raise ValueError(f"{path}: no string column to use as text; columns: {list(names)}")
    raise ValueError(f"{path}: several string columns {list(string_cols)} and none is named like one of "
                     f"{list(_TEXT_KEYS)}; say which one is the text with text_field=... "
                     f"(or stitch several into one record with record_template=...)")


def _iter_parquet_records(path: str, spec: _RecordSpec, rej: _Rejects) -> Iterator[Record]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError("reading .parquet needs `pip install pyarrow`") from e
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    names = list(schema.names)
    string_cols = [f.name for f in schema if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
    if spec.template is not None:
        missing = [f for f in spec.template_fields if f not in names]
        if missing:
            raise ValueError(f"{path}: record_template uses column(s) {missing} that do not exist; available: {names}")
        cols = list(spec.template_fields)
    else:
        col = _pick_text_column(names, string_cols, spec.text_field, path)
        spec = spec.with_text_field(col)
        cols = [col]
    if spec.group_field is not None:
        if spec.group_field not in names:
            raise ValueError(f"{path}: group_field {spec.group_field!r} not found; available: {names}")
        if spec.group_field not in cols:
            cols.append(spec.group_field)
    row_no = 0
    for batch in pf.iter_batches(columns=cols, batch_size=4096):
        data = batch.to_pydict()
        for r in range(batch.num_rows):
            row_no += 1
            rej.seen()
            rec, why = spec.extract({c: data[c][r] for c in cols})
            if rec is None:
                rej.add(path, why[0], f"row {row_no}: {why[1]}")
                continue
            yield rec


def _iter_text_chunks(path: str, chunk_bytes: int) -> Iterator[bytes]:
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                return
            yield chunk


def _utf8_cut(chunk: bytes, limit: int) -> int:
    """Largest ``cut <= limit`` such that ``chunk[:cut]`` does not end inside a multi-byte character."""
    cut = min(limit, len(chunk))
    while 0 < cut < len(chunk) and (chunk[cut] & 0xC0) == 0x80:
        cut -= 1
    return cut


# =============================================================================
# Source discovery
# =============================================================================

def list_source_files(data_path: str, format: Optional[str] = None) -> List[str]:
    """Files to ingest, in a stable order. Only known extensions (``.txt .md .jsonl .parquet``) are
    taken from a directory; a single explicit file must have one too unless ``format`` says how to
    read it. There is deliberately no "read everything if nothing matches" fallback."""
    if format is not None and format not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {format!r}")
    if os.path.isdir(data_path):
        allf = sorted(f for f in glob.glob(os.path.join(data_path, "**", "*"), recursive=True)
                      if os.path.isfile(f) and not os.path.basename(f).startswith("."))
        files = allf if format else [f for f in allf if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
        if not files:
            found = sorted({os.path.splitext(f)[1].lower() or "(no extension)" for f in allf})
            raise ValueError(f"no ingestible files under {data_path}: supported extensions are "
                             f"{sorted(SUPPORTED_EXTS)}; found {found or 'no files'}. "
                             f"Pass format='text'|'jsonl'|'parquet' to read other files explicitly.")
        return files
    if os.path.isfile(data_path):
        if format is None and os.path.splitext(data_path)[1].lower() not in SUPPORTED_EXTS:
            raise ValueError(f"{data_path}: unrecognised extension {os.path.splitext(data_path)[1]!r} "
                             f"(supported: {sorted(SUPPORTED_EXTS)}). Pass format='text'|'jsonl'|'parquet' "
                             f"to say how it should be read.")
        return [data_path]
    raise FileNotFoundError(f"{data_path} does not exist")


def _format_of(path: str, forced: Optional[str]) -> str:
    return forced or SUPPORTED_EXTS[os.path.splitext(path)[1].lower()]


# =============================================================================
# Output sinks and the split
# =============================================================================

class _HashingSink:
    """Writes uint16 arrays to a file while hashing them and counting tokens / EOS markers."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "wb")
        self._sha = hashlib.sha256()
        self.n_tokens = 0
        self.n_docs = 0

    def write(self, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr, dtype=np.uint16)
        mv = memoryview(arr).cast("B")
        self._fh.write(mv)
        self._sha.update(mv)
        self.n_tokens += int(arr.size)
        self.n_docs += int(np.count_nonzero(arr == _EOS))

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    def digest(self) -> str:
        return self._sha.hexdigest()


def _copy_range(stream: np.ndarray, a: int, b: int, sink: _HashingSink, block: int = 1 << 24) -> None:
    for s in range(a, b, block):
        sink.write(stream[s: min(b, s + block)])


def _doc_bounds(chunk: np.ndarray, prev_end: int) -> Tuple[np.ndarray, np.ndarray]:
    ends = chunk["end"].astype(np.int64)
    starts = np.empty_like(ends)
    starts[0] = prev_end
    starts[1:] = ends[:-1]
    return starts, ends


def _keep_mask(index: _DocIndex, dedup: bool) -> Tuple[Optional[np.ndarray], int]:
    """Boolean keep-mask (first occurrence of every distinct content hash) or None."""
    if not dedup or index.n == 0:
        return None, 0
    content = np.concatenate([c["content"] for c in index.chunks()])
    _, first = np.unique(content, return_index=True)
    keep = np.zeros(index.n, dtype=bool)
    keep[first] = True
    return keep, int(index.n - first.size)


def _document_split_stats(index: _DocIndex, keep: Optional[np.ndarray], val_fraction: float) -> Dict[str, int]:
    prev, off = 0, 0
    st = dict(train_docs=0, val_docs=0, train_tokens=0, val_tokens=0)
    for chunk in index.chunks():
        starts, ends = _doc_bounds(chunk, prev)
        prev = int(ends[-1])
        lens = ends - starts
        is_val = (chunk["split"].astype(np.float64) / _U64) < val_fraction
        k = np.ones(len(chunk), dtype=bool) if keep is None else keep[off: off + len(chunk)]
        off += len(chunk)
        st["val_docs"] += int(np.count_nonzero(is_val & k))
        st["train_docs"] += int(np.count_nonzero(~is_val & k))
        st["val_tokens"] += int(lens[is_val & k].sum())
        st["train_tokens"] += int(lens[~is_val & k].sum())
    return st


def _write_document_split(stream: np.ndarray, index: _DocIndex, keep: Optional[np.ndarray], val_fraction: float,
                          train: _HashingSink, val: _HashingSink) -> None:
    prev, off = 0, 0
    for chunk in index.chunks():
        starts, ends = _doc_bounds(chunk, prev)
        prev = int(ends[-1])
        assign = ((chunk["split"].astype(np.float64) / _U64) < val_fraction).astype(np.int8)
        if keep is not None:
            assign = np.where(keep[off: off + len(chunk)], assign, np.int8(-1))
        off += len(chunk)
        change = np.flatnonzero(np.diff(assign)) + 1
        run_lo = np.concatenate(([0], change))
        run_hi = np.concatenate((change, [len(assign)]))
        for lo, hi in zip(run_lo.tolist(), run_hi.tolist()):
            a = int(assign[lo])
            if a >= 0:
                _copy_range(stream, int(starts[lo]), int(ends[hi - 1]), val if a else train)


# =============================================================================
# prepare
# =============================================================================

def _publish(staging: str, out_dir: str, files: Sequence[str], fingerprint: str, overwrite: bool, say) -> str:
    """Move the staged files into ``out_dir``. Returns "published" | "unchanged" | "replaced"."""
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in ("meta.json", "train.bin", "val.bin") if os.path.exists(os.path.join(out_dir, f))]
    status = "published"
    if existing:
        old_fp = None
        try:
            old_fp = M.read_meta(out_dir).get("fingerprint")
        except (M.DatasetError, ValueError):
            pass
        if old_fp == fingerprint:
            shutil.rmtree(staging, ignore_errors=True)
            say(f"[prepare] {out_dir} already holds this exact dataset ({fingerprint[:23]}...); left untouched")
            return "unchanged"
        if not overwrite:
            shutil.rmtree(staging, ignore_errors=True)
            raise FileExistsError(
                f"{out_dir} already contains a different prepared dataset ({old_fp or 'no fingerprint'} vs the new "
                f"{fingerprint}). Prepared datasets are immutable: write to a new directory, or pass "
                f"overwrite=True to move the old files aside to '{out_dir}.replaced-<time>' and replace them.")
        aside = f"{out_dir}.replaced-{int(time.time())}"
        os.makedirs(aside, exist_ok=True)
        for f in existing:                       # meta.json first: a crash mid-way leaves "not a dataset", not a mismatch
            os.replace(os.path.join(out_dir, f), os.path.join(aside, f))
        say(f"[prepare] previous dataset moved to {aside}")
        status = "replaced"
    for f in files:                              # meta.json last
        os.replace(os.path.join(staging, f), os.path.join(out_dir, f))
    shutil.rmtree(staging, ignore_errors=True)
    return status


def prepare_data(data_path: str, out_dir: str, val_fraction: float = 0.1, seed: int = 1337,
                 shuffle_files: bool = False, doc_sep: Optional[str] = None,
                 text_column: str = "auto", max_bytes: Optional[int] = None,
                 chunk_bytes: int = 64 << 20, verbose: bool = True, *,
                 text_field: Optional[str] = None, record_template: Optional[str] = None,
                 group_field: Optional[str] = None, format: Optional[str] = None,
                 split: str = "auto", min_documents: int = 50, dedup: bool = False,
                 validate_utf8: bool = True, max_reject_fraction: float = 0.01,
                 overwrite: bool = False, hash_sources: bool = True) -> Dict[str, Any]:
    """
    Turn text / markdown / jsonl / parquet file(s) into ``out_dir/{train,val}.bin`` plus a manifest.

    Documents
      * ``.txt`` / ``.md``: one document per file, or one per ``doc_sep`` occurrence (a literal such as
        ``"<|endoftext|>"``, mapped to the real EOS token). Must be valid UTF-8 (``validate_utf8``).
      * ``.jsonl`` / ``.parquet``: one document per record. The text is ``text_field`` (alias
        ``text_column``; ``"auto"`` = the first of text/content/body/story), **or** ``record_template``
        such as ``"Q: {question}\\nContext: {context}\\nA: {answer}"`` which stitches several fields
        into one record. Records that lack the field are rejected and counted; more than
        ``max_reject_fraction`` of a file's records being rejected is an error.
      * ``format`` forces a reader for files whose extension is not recognised.

    Splitting (``split``)
      * ``"document"``: a document goes to validation iff the hash of its content (or of its
        ``group_field`` value, e.g. a shared context id) falls below ``val_fraction``. Deterministic,
        content-addressed, seeded by ``seed``; duplicates cannot straddle the split; ``dedup=True``
        additionally drops exact duplicate documents (keeping the first).
      * ``"tail"``: the last ``val_fraction`` of the token stream (the v1 behaviour; may cut one
        document in two).
      * ``"auto"`` (default): ``document`` when there are at least ``min_documents`` documents and both
        sides come out non-empty, otherwise ``tail``; the choice and the reason are recorded.

    ``max_bytes`` stops after that many source bytes. The result is staged and then published
    atomically; an existing different dataset in ``out_dir`` is only replaced with ``overwrite=True``
    (and is moved aside, not deleted). Returns the manifest (also written as ``meta.json``).
    """
    from . import __version__

    say = print if verbose else (lambda *a, **k: None)
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if split not in ("auto", "document", "tail"):
        raise ValueError(f"split must be 'auto', 'document' or 'tail', got {split!r}")
    if dedup and split == "tail":
        raise ValueError("dedup needs a document-level split (split='document' or 'auto')")
    if not 0.0 <= max_reject_fraction <= 1.0:
        raise ValueError("max_reject_fraction must be in [0, 1]")
    if text_field is not None and text_column not in ("auto", None, text_field):
        raise ValueError("pass text_field or text_column, not two different values")
    spec = _RecordSpec(text_field if text_field is not None else text_column, record_template, group_field)

    files = list_source_files(data_path, format)
    skipped: List[str] = []
    if os.path.isdir(data_path) and format is None:
        skipped = sorted(f for f in glob.glob(os.path.join(data_path, "**", "*"), recursive=True)
                         if os.path.isfile(f) and f not in files and not os.path.basename(f).startswith("."))
    if group_field is not None and all(_format_of(f, format) == "text" for f in files):
        raise ValueError("group_field only applies to .jsonl / .parquet records")
    if shuffle_files:
        files = list(files)
        random.Random(seed).shuffle(files)

    out_dir = os.path.abspath(out_dir)
    staging = out_dir + ".staging"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)

    sep = doc_sep.encode("utf-8") if doc_sep else None
    salt = f"tensegrity-split/{seed}".encode()
    stream_path = os.path.join(staging, "stream.tmp")
    index = _DocIndex(os.path.join(staging, "docs.idx"))
    rej = _Rejects()
    sources: List[Dict[str, Any]] = []
    n_src = 0
    t0 = time.time()
    try:
        with open(stream_path, "wb") as sink_f:
            enc = _StreamEncoder(lambda a: a.tofile(sink_f), sep, index, salt)
            budget = max_bytes
            for f in files:
                if budget is not None and n_src >= budget:
                    break
                fmt = _format_of(f, format)
                src_rec: Dict[str, Any] = {"path": os.path.abspath(f), "name": os.path.basename(f), "format": fmt}
                snap = rej.snapshot()
                n_before = n_src
                if fmt in ("jsonl", "parquet"):
                    recs = (_iter_jsonl_records if fmt == "jsonl" else _iter_parquet_records)(f, spec, rej)
                    n_rec = 0
                    for doc, group in recs:
                        if budget is not None and n_src >= budget:
                            break
                        enc.begin_record(group)
                        enc.feed(doc)
                        enc.end_document()
                        n_src += len(doc)
                        n_rec += 1
                    rej.check_since(snap, f, max_reject_fraction)
                    src_rec.update(records=n_rec, bytes_consumed=n_src - n_before, size=os.path.getsize(f),
                                   sha256=M.sha256_file(f) if hash_sources else None)
                else:
                    if group_field is not None:
                        raise ValueError(f"{f}: group_field cannot be applied to plain text files")
                    if record_template is not None:
                        raise ValueError(f"{f}: record_template applies to .jsonl/.parquet records, not text files; "
                                         f"prepare them separately")
                    dec = codecs.getincrementaldecoder("utf-8")("strict") if validate_utf8 else None
                    sha = hashlib.sha256() if hash_sources else None
                    offset, truncated = 0, False
                    for chunk in _iter_text_chunks(f, chunk_bytes):
                        if budget is not None and n_src + len(chunk) > budget:
                            chunk = chunk[: _utf8_cut(chunk, budget - n_src)]
                            truncated = True
                        if dec is not None:
                            try:
                                dec.decode(chunk)
                            except UnicodeDecodeError as e:
                                raise ValueError(
                                    f"{f}: not valid UTF-8 (byte {offset + e.start:,}: {e.reason}). Fix or re-encode "
                                    f"the file, or pass validate_utf8=False to train on its raw bytes anyway.") from e
                        if sha is not None:
                            sha.update(chunk)
                        enc.feed(chunk)
                        n_src += len(chunk)
                        offset += len(chunk)
                        if verbose:
                            print(f"\r[prepare] {n_src / 1e6:8.1f} MB read, {enc.n_tokens / 1e6:8.1f}M tokens", end="")
                        if truncated or (budget is not None and n_src >= budget):
                            break
                    if dec is not None and not truncated:
                        try:
                            dec.decode(b"", final=True)
                        except UnicodeDecodeError as e:
                            raise ValueError(f"{f}: ends in the middle of a UTF-8 character") from e
                    enc.end_document()
                    src_rec.update(bytes_consumed=offset, size=os.path.getsize(f),
                                   sha256=sha.hexdigest() if sha is not None else None,
                                   complete=offset == os.path.getsize(f))
                sources.append(src_rec)
        n = enc.n_tokens
        index.close()
        if verbose:
            print()
        if n == 0:
            raise ValueError(f"All files at {data_path} are empty (or entirely rejected); nothing to tokenize.")

        stream = np.memmap(stream_path, dtype=np.uint16, mode="r", shape=(n,))
        keep, n_dup = _keep_mask(index, dedup)

        # ---- decide the split policy
        reason = ""
        policy = "tail" if split == "tail" else "document"
        stats: Optional[Dict[str, int]] = None
        if policy == "document":
            stats = _document_split_stats(index, keep, val_fraction)
            why_not = ""
            if index.n - n_dup < max(2, min_documents if split == "auto" else 2):
                why_not = (f"only {index.n - n_dup:,} document(s) (< {min_documents if split == 'auto' else 2}); "
                           f"a document-level split needs many documents - set doc_sep, or use jsonl/parquet records")
            elif stats["train_docs"] == 0 or stats["val_docs"] == 0:
                why_not = (f"the hash split left one side empty (train {stats['train_docs']:,} docs, "
                           f"val {stats['val_docs']:,} docs)")
            if why_not:
                if split == "document":
                    raise ValueError(f"split='document' is not possible: {why_not}. Use split='tail' (or 'auto').")
                policy, reason = "tail", why_not + "; using a contiguous tail split"
                if dedup:
                    raise ValueError(f"dedup needs a document-level split but {why_not}")
                stats = None
        if verbose and reason:
            print(f"[prepare] note: {reason}")

        # ---- write the two splits (hashing as we go)
        train = _HashingSink(os.path.join(staging, "train.bin"))
        val = _HashingSink(os.path.join(staging, "val.bin"))
        if policy == "document":
            _write_document_split(stream, index, keep, val_fraction, train, val)
        else:
            n_val = max(1, int(n * val_fraction))
            n_train = n - n_val
            if n_train <= 0:
                raise ValueError("Dataset too small after train/val split; provide more data.")
            _copy_range(stream, 0, n_train, train)
            _copy_range(stream, n_train, n, val)
        train.close()
        val.close()
        del stream
        for tmp in (stream_path, index.path):
            os.remove(tmp)
        if train.n_tokens == 0 or val.n_tokens == 0:
            raise ValueError("Dataset too small after train/val split; provide more data.")

        counts = {
            "n_train_tokens": train.n_tokens, "n_val_tokens": val.n_tokens,
            "n_train_documents": train.n_docs, "n_val_documents": val.n_docs,
            "n_rejected": rej.n_rejected, "rejected_reasons": dict(rej.counts), "rejected_examples": rej.examples,
            "n_duplicates_dropped": n_dup, "source_bytes": int(n_src),
        }
        meta: Dict[str, Any] = {
            # ---- fields every release has written
            "vocab_size": ByteTokenizer.VOCAB_SIZE, "dtype": "uint16",
            "n_train_tokens": train.n_tokens, "n_val_tokens": val.n_tokens,
            "n_documents": train.n_docs + val.n_docs, "source_bytes": int(n_src),
            "source_files": [s["path"] for s in sources], "val_fraction": val_fraction,
            "shuffle_files": shuffle_files, "seed": seed if (shuffle_files or policy == "document") else None,
            "doc_sep": doc_sep, "max_bytes": max_bytes, "prepare_seconds": round(time.time() - t0, 1),
            # ---- manifest
            "manifest_version": M.MANIFEST_VERSION,
            "tokenizer": {"name": "ByteTokenizer", "vocab_size": ByteTokenizer.VOCAB_SIZE, "eos": ByteTokenizer.EOS,
                          "bos": ByteTokenizer.BOS, "pad": ByteTokenizer.PAD,
                          "document_format": "bytes+EOS (no BOS in the data)"},
            "split": {"policy": policy, "requested": split, "val_fraction": val_fraction,
                      "salt": salt.decode() if policy == "document" else None, "group_field": group_field,
                      "min_documents": min_documents, "dedup": dedup, "reason": reason},
            "counts": counts,
            "files": {"train.bin": {"sha256": train.digest(), "bytes": train.n_tokens * 2},
                      "val.bin": {"sha256": val.digest(), "bytes": val.n_tokens * 2}},
            "sources": sources, "skipped_files": skipped,
            "preprocessing": {"text_field": spec.text_field, "record_template": record_template, "format": format,
                              "validate_utf8": validate_utf8, "max_reject_fraction": max_reject_fraction,
                              "chunk_bytes": chunk_bytes},
            "code_version": __version__,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta["fingerprint"] = M.compute_fingerprint(meta)
        with open(os.path.join(staging, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        status = _publish(staging, out_dir, ("train.bin", "val.bin", "meta.json"), meta["fingerprint"], overwrite, say)
        if status == "unchanged":
            meta = M.read_meta(out_dir)
    except BaseException:
        index.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if verbose and status != "unchanged":
        print(f"[prepare] {len(files)} file(s) -> {meta['n_train_tokens']:,} train / {meta['n_val_tokens']:,} val tokens "
              f"({meta['n_documents']:,} documents, {policy} split, {rej.n_rejected:,} rejected) in "
              f"{meta['prepare_seconds']}s -> {out_dir}\n[prepare] fingerprint {meta['fingerprint']}")
    return meta
