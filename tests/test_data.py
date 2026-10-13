"""Data pipeline: streaming prepare, document separators, batching, downloads."""

import json
import os
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import pytest

from conftest import make_text
from tensegrity import ByteDataset, ByteTokenizer, download_file, prepare_data
from tensegrity.data import _StreamEncoder, _partial_suffix_len

EOS = ByteTokenizer.EOS


def _tokens(out_dir, split):
    return np.fromfile(os.path.join(out_dir, f"{split}.bin"), dtype=np.uint16)


def _all(out_dir):
    return np.concatenate([_tokens(out_dir, "train"), _tokens(out_dir, "val")])


def _reference(text: bytes, sep: bytes):
    """What the encoder must produce: sep -> single EOS, plus one EOS at end of file."""
    parts = text.split(sep)
    toks = []
    for k, part in enumerate(parts):
        toks.extend(part)
        toks.append(EOS)                       # each split point -> EOS; the last one is the end-of-file EOS
    return np.array(toks, dtype=np.uint16)


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7, 16, 1 << 20])
@pytest.mark.parametrize("sep", ["<|endoftext|>", "aa", "x"])
def test_separator_handled_across_every_chunk_boundary(tmp_path, chunk, sep):
    """Streaming must equal one-shot processing for ANY chunk size, including separators that
    straddle chunk boundaries and self-overlapping separators ('aa')."""
    body = ("one aaa two" + sep + "three xx aa" + sep + sep + "tail aaaa") if sep != "<|endoftext|>" else \
        ("one two" + sep + "three " + sep + sep + "tail <|endof")
    src = tmp_path / "s.txt"
    src.write_bytes(body.encode())
    out = tmp_path / "out"
    prepare_data(str(src), str(out), val_fraction=0.2, doc_sep=sep, chunk_bytes=chunk, verbose=False)
    got = _all(str(out))
    want = _reference(body.encode(), sep.encode())
    assert np.array_equal(got, want), (got.tolist(), want.tolist())


def test_partial_suffix_len():
    assert _partial_suffix_len(b"abc<|end", b"<|endoftext|>") == 5
    assert _partial_suffix_len(b"abc", b"<|endoftext|>") == 0
    assert _partial_suffix_len(b"xa", b"aa") == 1


def test_doc_sep_gives_real_eos_tokens_not_spelled_bytes(tmp_path):
    src = tmp_path / "s.txt"
    src.write_text(make_text(50), encoding="utf-8")
    out = tmp_path / "o"
    meta = prepare_data(str(src), str(out), doc_sep="<|endoftext|>", verbose=False)
    toks = _all(str(out))
    assert int((toks == EOS).sum()) == meta["n_documents"] == 51   # 50 separators + end-of-file EOS
    text_bytes = bytes(toks[toks < 256].astype(np.uint8))
    assert b"endoftext" not in text_bytes and b"<|" not in text_bytes   # never spelled out as 13 bytes


def test_legacy_behaviour_without_doc_sep_is_one_eos_per_file(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.txt").write_bytes(b"hello")
    (d / "b.md").write_bytes(b"world!")
    (d / "ignored.bin").write_bytes(b"zzzz")      # unsupported extension is skipped when supported ones exist
    out = tmp_path / "o"
    prepare_data(str(d), str(out), val_fraction=0.3, verbose=False)
    want = list(b"hello") + [EOS] + list(b"world!") + [EOS]
    assert _all(str(out)).tolist() == want
    meta = json.load(open(out / "meta.json"))
    assert meta["n_train_tokens"] + meta["n_val_tokens"] == len(want)
    assert meta["n_val_tokens"] == max(1, int(len(want) * 0.3))


def test_prepare_jsonl_and_parquet_one_eos_per_record(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    j = tmp_path / "x.jsonl"
    j.write_text("\n".join([json.dumps({"text": "alpha"}), "not json", json.dumps({"content": "beta"}),
                            json.dumps("gamma"), json.dumps({"other": 1, "name": "delta"}), ""]))
    out = tmp_path / "oj"
    prepare_data(str(j), str(out), val_fraction=0.1, verbose=False)
    # records: {"text"}, an unparseable line (skipped), {"content"}, a bare JSON string, and a
    # dict with no text key (its scalar values are joined with newlines)
    want = (list(b"alpha") + [EOS] + list(b"beta") + [EOS] + list(b"gamma") + [EOS]
            + list(b"1\ndelta") + [EOS])
    assert _all(str(out)).tolist() == want
    table = pa.table({"id": [1, 2, 3], "story": ["one", None, "three"], "junk": ["x", "y", "z"]})
    pq.write_table(table, tmp_path / "t.parquet")
    out2 = tmp_path / "op"
    prepare_data(str(tmp_path / "t.parquet"), str(out2), val_fraction=0.1, text_column="story", verbose=False)
    assert _all(str(out2)).tolist() == list(b"one") + [EOS] + list(b"three") + [EOS]
    out3 = tmp_path / "op2"                            # 'auto' prefers a known text column name
    prepare_data(str(tmp_path / "t.parquet"), str(out3), val_fraction=0.1, verbose=False)
    assert _all(str(out3)).tolist() == list(b"one") + [EOS] + list(b"three") + [EOS]
    with pytest.raises(ValueError, match="not found"):
        prepare_data(str(tmp_path / "t.parquet"), str(tmp_path / "bad"), text_column="nope", verbose=False)


def test_max_bytes_and_error_paths(tmp_path):
    src = tmp_path / "s.txt"
    src.write_bytes(b"x" * 1000)
    meta = prepare_data(str(src), str(tmp_path / "o"), max_bytes=300, val_fraction=0.1, verbose=False)
    assert meta["source_bytes"] == 300 and meta["n_train_tokens"] + meta["n_val_tokens"] == 301
    empty = tmp_path / "e.txt"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        prepare_data(str(empty), str(tmp_path / "oe"), verbose=False)
    with pytest.raises(ValueError):
        prepare_data(str(src), str(tmp_path / "ov"), val_fraction=1.5, verbose=False)


def test_stream_encoder_unit():
    chunks = []
    enc = _StreamEncoder(chunks.append, b"##")
    for piece in (b"ab#", b"#cd", b"#", b"#e"):
        enc.feed(piece)
    enc.end_document()
    got = np.concatenate(chunks).tolist()
    assert got == list(b"ab") + [EOS] + list(b"cd") + [EOS] + list(b"e") + [EOS]


# ------------------------------------------------------------------------------- batching
def test_batches_are_shifted_windows_and_deterministic(corpus_dir):
    ds1, ds2 = ByteDataset(corpus_dir, seed=3), ByteDataset(corpus_dir, seed=3)
    x1, y1 = ds1.get_batch("train", 6, 32, "cpu")
    x2, y2 = ds2.get_batch("train", 6, 32, "cpu")
    assert (x1 == x2).all() and (y1 == y2).all()
    assert (x1[:, 1:] == y1[:, :-1]).all() and x1.dtype.is_floating_point is False
    assert ByteDataset(corpus_dir, seed=4).get_batch("train", 6, 32, "cpu")[0].ne(x1).any()
    arr = ds1.array(corpus_dir, "train")
    for row in range(6):                       # every row really is a contiguous window of the corpus
        w = x1[row].numpy()
        starts = np.nonzero(arr[: len(arr) - 40] == w[0])[0]
        assert any((arr[s: s + 32] == w).all() for s in starts)


def test_dataset_state_resumes_exact_sequence(corpus_dir):
    a, b = ByteDataset(corpus_dir, seed=1), ByteDataset(corpus_dir, seed=1)
    for _ in range(3):
        a.get_batch("train", 4, 16, "cpu")
    b.load_state_dict(a.state_dict())
    assert (a.get_batch("train", 4, 16, "cpu")[0] == b.get_batch("train", 4, 16, "cpu")[0]).all()


def test_mixture_weights_respected(corpus_dir, corpus_dir2):
    ds = ByteDataset([corpus_dir, corpus_dir2], weights=[1, 0], seed=0)
    x, _ = ds.get_batch("train", 64, 16, "cpu")
    a2 = ds.array(corpus_dir, "train")
    assert all(any((a2[s: s + 16] == r.numpy()).all() for s in np.nonzero(a2[:-20] == r[0].item())[0]) for r in x[:8])
    ds = ByteDataset([corpus_dir, corpus_dir2], weights=[1, 1], seed=0)
    assert len(ds.weights) == 2 and abs(ds.weights.sum() - 1) < 1e-12
    with pytest.raises(ValueError):
        ByteDataset([corpus_dir], weights=[1, 2])
    with pytest.raises(ValueError):
        ByteDataset([corpus_dir, corpus_dir2], weights=[0, 0])


def test_fixed_batches_deterministic_and_independent_of_sampling_rng(corpus_dir):
    ds = ByteDataset(corpus_dir, seed=1)
    a = ds.fixed_batches("val", 3, 4, 24, "cpu")
    ds.get_batch("train", 8, 24, "cpu")                 # consume the sampler's RNG
    b = ds.fixed_batches("val", 3, 4, 24, "cpu")
    assert all((xa == xb).all() and (ya == yb).all() for (xa, ya), (xb, yb) in zip(a, b))


def test_too_small_corpus_error(tmp_path):
    src = tmp_path / "s.txt"
    src.write_bytes(b"tiny")
    prepare_data(str(src), str(tmp_path / "o"), val_fraction=0.4, verbose=False)
    with pytest.raises(ValueError, match="need > block_size"):
        ByteDataset(str(tmp_path / "o")).get_batch("train", 2, 64, "cpu")


# ------------------------------------------------------------------------------ downloads
class _RangeHandler(SimpleHTTPRequestHandler):
    """Static file server that honours Range requests (python's built-in one does not)."""
    fail_first = {"n": 0}

    def log_message(self, *a):
        pass

    def do_GET(self):
        if _RangeHandler.fail_first["n"] > 0:
            _RangeHandler.fail_first["n"] -= 1
            self.send_error(503)
            return
        path = self.translate_path(self.path)
        data = open(path, "rb").read()
        rng = self.headers.get("Range")
        if rng:
            start, end = rng.replace("bytes=", "").split("-")
            start = int(start)
            end = int(end) if end else len(data) - 1
            body = data[start: end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def http_dir(tmp_path):
    payload = ("story one. <|endoftext|>" * 5000).encode()
    (tmp_path / "f.txt").write_bytes(payload)
    handler = partial(_RangeHandler, directory=str(tmp_path))
    srv = HTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/f.txt", payload, tmp_path
    srv.shutdown()


def test_download_full_slice_retry_and_resume(http_dir):
    url, payload, tmp = http_dir
    dest = str(tmp / "dl" / "full.txt")
    download_file(url, dest, verbose=False)
    assert open(dest, "rb").read() == payload
    # cached: no second request needed
    _RangeHandler.fail_first["n"] = 99
    download_file(url, dest, verbose=False)
    _RangeHandler.fail_first["n"] = 0
    # byte-limited slice, trimmed back to the last complete document
    sl = str(tmp / "dl" / "slice.txt")
    download_file(url, sl, max_bytes=10_000, boundary=b"<|endoftext|>", verbose=False)
    got = open(sl, "rb").read()
    assert 9_000 < len(got) <= 10_000 and got.endswith(b"<|endoftext|>") and payload.startswith(got)
    # transient server errors are retried
    _RangeHandler.fail_first["n"] = 2
    ok = str(tmp / "dl" / "retry.txt")
    import tensegrity.data as D
    real_sleep, D.time.sleep = D.time.sleep, lambda s: None
    try:
        download_file(url, ok, retries=5, verbose=False)
    finally:
        D.time.sleep = real_sleep
    assert open(ok, "rb").read() == payload
    # resume from a partial .part file
    part_dest = str(tmp / "dl" / "resume.txt")
    open(part_dest + ".part", "wb").write(payload[:5000])
    download_file(url, part_dest, verbose=False)
    assert open(part_dest, "rb").read() == payload
    # permanent failure surfaces a clear error
    _RangeHandler.fail_first["n"] = 99
    D.time.sleep = lambda s: None
    try:
        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            download_file(url, str(tmp / "dl" / "never.txt"), retries=2, verbose=False)
    finally:
        D.time.sleep = real_sleep
        _RangeHandler.fail_first["n"] = 0
