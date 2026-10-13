"""Trustworthy data: strict readers, document-level splits, manifests, immutability, change detection."""

import json
import os
import random
import warnings
from functools import partial
from http.server import HTTPServer
import threading

import numpy as np
import pytest

from tensegrity import (ByteDataset, ByteTokenizer, DatasetError, audit_dataset, data_identity, download_file,
                        prepare_data, read_download_record, verify_dataset)
from tensegrity.manifest import compute_fingerprint, read_meta

from test_data import _RangeHandler

EOS = ByteTokenizer.EOS
SEP = "<|endoftext|>"
_WORDS = "the a one little big red old happy sad dog cat bird girl boy tree house river sun moon".split()


def make_docs(n, seed=0):
    rnd = random.Random(seed)
    return [f"Doc {i:05d} " + " ".join(rnd.choice(_WORDS) for _ in range(rnd.randint(6, 30))) + "." for i in range(n)]


def write_corpus(path, docs, sep=SEP):
    path.write_text(sep.join(docs) + sep, encoding="utf-8")
    return str(path)


def split_docs(out_dir, split):
    """The documents (as bytes) of a prepared split; asserts the file consists of whole documents."""
    toks = np.fromfile(os.path.join(out_dir, f"{split}.bin"), dtype=np.uint16)
    assert toks.size and toks[-1] == EOS, f"{split}.bin must end on a document boundary"
    docs, start = [], 0
    for e in np.flatnonzero(toks == EOS).tolist():
        docs.append(bytes(toks[start:e].astype(np.uint8)))
        start = e + 1
    return docs


# --------------------------------------------------------------------------- document-level split
def test_document_split_keeps_every_document_whole_and_disjoint(tmp_path):
    docs = make_docs(2000)
    src = write_corpus(tmp_path / "c.txt", docs)
    meta = prepare_data(src, str(tmp_path / "o"), val_fraction=0.1, doc_sep=SEP, verbose=False)
    assert meta["split"]["policy"] == "document" and meta["split"]["requested"] == "auto"
    tr, va = split_docs(str(tmp_path / "o"), "train"), split_docs(str(tmp_path / "o"), "val")
    want = {d.encode() for d in docs}
    assert set(tr).isdisjoint(va)                                   # no document on both sides
    assert set(tr) | set(va) == want | {b""}                        # the trailing separator makes one empty doc
    assert 150 < len(va) < 250                                      # ~10% of 2000 documents
    assert meta["counts"]["n_train_documents"] == len(tr) and meta["counts"]["n_val_documents"] == len(va)
    assert meta["n_documents"] == len(tr) + len(va)


def test_split_is_deterministic_content_addressed_and_seeded(tmp_path):
    docs = make_docs(600)
    a = prepare_data(write_corpus(tmp_path / "a.txt", docs), str(tmp_path / "A"), doc_sep=SEP, verbose=False)
    b = prepare_data(write_corpus(tmp_path / "b.txt", docs), str(tmp_path / "B"), doc_sep=SEP, verbose=False)
    assert a["fingerprint"] == b["fingerprint"]                     # same data + policy -> same identity
    assert open(tmp_path / "A" / "train.bin", "rb").read() == open(tmp_path / "B" / "train.bin", "rb").read()
    shuffled = list(docs)
    random.Random(5).shuffle(shuffled)                              # the SAME documents in another order ...
    prepare_data(write_corpus(tmp_path / "s.txt", shuffled), str(tmp_path / "S"), doc_sep=SEP, verbose=False)
    assert set(split_docs(str(tmp_path / "S"), "val")) == set(split_docs(str(tmp_path / "A"), "val"))   # ... same val set
    c = prepare_data(write_corpus(tmp_path / "c.txt", docs), str(tmp_path / "C"), doc_sep=SEP, seed=7, verbose=False)
    assert c["fingerprint"] != a["fingerprint"]                     # another seed -> another split
    assert set(split_docs(str(tmp_path / "C"), "val")) != set(split_docs(str(tmp_path / "A"), "val"))


def test_duplicates_cannot_straddle_the_split_and_dedup_drops_them(tmp_path):
    base = make_docs(400)
    docs = base + base[:100] + base[:100]                            # 100 documents appear three times
    src = write_corpus(tmp_path / "d.txt", docs)
    prepare_data(src, str(tmp_path / "o"), val_fraction=0.2, doc_sep=SEP, verbose=False)
    tr, va = split_docs(str(tmp_path / "o"), "train"), split_docs(str(tmp_path / "o"), "val")
    assert set(tr).isdisjoint(va)                                    # a copy of a val doc is never in train
    meta = prepare_data(src, str(tmp_path / "dd"), val_fraction=0.2, doc_sep=SEP, dedup=True, verbose=False)
    tr2, va2 = split_docs(str(tmp_path / "dd"), "train"), split_docs(str(tmp_path / "dd"), "val")
    assert len(tr2) + len(va2) == 400 + 1 and len(set(tr2 + va2)) == len(tr2 + va2)      # +1: the empty doc
    assert meta["counts"]["n_duplicates_dropped"] == 200
    with pytest.raises(ValueError, match="dedup"):
        prepare_data(src, str(tmp_path / "x"), doc_sep=SEP, dedup=True, split="tail", verbose=False)


def test_group_field_keeps_related_records_together(tmp_path):
    rows = []
    for c in range(80):                                              # 80 contexts x 5 questions each
        for q in range(5):
            rows.append({"ctx_id": f"ctx{c}", "context": f"Passage {c}. " * 4, "question": f"Q{c}.{q}?", "answer": f"A{c}.{q}"})
    p = tmp_path / "qa.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    tmpl = "Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    prepare_data(str(p), str(tmp_path / "o"), val_fraction=0.25, record_template=tmpl, group_field="ctx_id",
                 verbose=False)
    for split in ("train", "val"):
        seen = {d.decode().split("Passage ")[1].split(".")[0] for d in split_docs(str(tmp_path / "o"), split)}
        if split == "train":
            train_ctx = seen
        else:
            assert train_ctx.isdisjoint(seen) and seen                # no passage appears on both sides
    # without the group key the same questions DO straddle the split
    prepare_data(str(p), str(tmp_path / "o2"), val_fraction=0.25, record_template=tmpl, verbose=False)
    t2 = {d.decode().split("Passage ")[1].split(".")[0] for d in split_docs(str(tmp_path / "o2"), "train")}
    v2 = {d.decode().split("Passage ")[1].split(".")[0] for d in split_docs(str(tmp_path / "o2"), "val")}
    assert t2 & v2


def test_tail_split_and_auto_fallback_are_explicit_and_recorded(tmp_path):
    one = tmp_path / "one.txt"
    one.write_text("hello world. " * 400)                            # ONE document: nothing to split by document
    m = prepare_data(str(one), str(tmp_path / "a"), val_fraction=0.1, verbose=False)
    assert m["split"]["policy"] == "tail" and m["split"]["requested"] == "auto" and "document" in m["split"]["reason"]
    with pytest.raises(ValueError, match="split='document' is not possible"):
        prepare_data(str(one), str(tmp_path / "b"), val_fraction=0.1, split="document", verbose=False)
    m2 = prepare_data(str(one), str(tmp_path / "c"), val_fraction=0.1, split="tail", verbose=False)
    assert m2["split"]["policy"] == "tail" and m2["split"]["reason"] == ""
    toks = np.concatenate([np.fromfile(tmp_path / "c" / f"{s}.bin", dtype=np.uint16) for s in ("train", "val")])
    assert bytes(toks[:-1].astype(np.uint8)) == one.read_bytes() and toks[-1] == EOS     # contiguous, original order
    with pytest.raises(ValueError, match="split must be"):
        prepare_data(str(one), str(tmp_path / "d"), split="random", verbose=False)


# ----------------------------------------------------------------------------------- strict readers
def test_unknown_extensions_are_never_silently_ingested(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "blob.dat").write_bytes(os.urandom(2000))
    (d / ".hidden.txt").write_text("secret")
    with pytest.raises(ValueError, match="no ingestible files"):
        prepare_data(str(d), str(tmp_path / "o"), verbose=False)
    with pytest.raises(ValueError, match="unrecognised extension"):
        prepare_data(str(d / "blob.dat"), str(tmp_path / "o"), verbose=False)
    with pytest.raises(FileNotFoundError):
        prepare_data(str(tmp_path / "missing"), str(tmp_path / "o"), verbose=False)
    (d / "notes.dat").write_text("plain text in an oddly named file. " * 20)
    m = prepare_data(str(d / "notes.dat"), str(tmp_path / "ok"), format="text", verbose=False)   # explicit opt-in
    assert m["sources"][0]["format"] == "text"
    (d / "a.txt").write_text("real text. " * 50)
    m = prepare_data(str(d), str(tmp_path / "mixed"), verbose=False)                           # skipped files are listed
    assert [os.path.basename(s["path"]) for s in m["sources"]] == ["a.txt"]
    assert sorted(os.path.basename(f) for f in m["skipped_files"]) == ["blob.dat", "notes.dat"]


def test_text_must_be_utf8_and_budget_cuts_never_split_a_character(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"fine text " * 5 + b"\xff\xfe broken " + b"more " * 5)
    with pytest.raises(ValueError, match="not valid UTF-8 \\(byte 50"):
        prepare_data(str(bad), str(tmp_path / "o"), verbose=False)
    assert not os.path.exists(tmp_path / "o") and not os.path.exists(tmp_path / "o.staging")
    prepare_data(str(bad), str(tmp_path / "raw"), validate_utf8=False, verbose=False)         # explicit opt-out
    euro = tmp_path / "euro.txt"
    euro.write_text("€" * 400, encoding="utf-8")                                              # 3 bytes per character
    m = prepare_data(str(euro), str(tmp_path / "e"), max_bytes=100, verbose=False, split="tail")
    toks = np.concatenate([np.fromfile(tmp_path / "e" / f"{s}.bin", dtype=np.uint16) for s in ("train", "val")])
    assert m["source_bytes"] == 99                                                           # 33 whole characters, not 100 bytes
    bytes(toks[toks < 256].astype(np.uint8)).decode("utf-8")                                  # strict decode succeeds
    trunc = tmp_path / "trunc.txt"
    trunc.write_bytes("ok ".encode() * 20 + "€".encode()[:2])
    with pytest.raises(ValueError, match="middle of a UTF-8 character"):
        prepare_data(str(trunc), str(tmp_path / "t"), verbose=False)


def test_structured_records_stay_one_document_and_rejects_are_accounted(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    rows = [{"question": f"What is {i}?", "context": f"ctx {i}", "answer": str(i), "source": "unit"} for i in range(200)]
    rows[3]["answer"] = None                       # missing field -> rejected
    rows[4]["context"] = ["a", "b"]                 # non-scalar -> rejected
    j = tmp_path / "qa.jsonl"
    j.write_text("\n".join(json.dumps(r) for r in rows))
    tmpl = "Q: {question}\nC: {context}\nA: {answer}"
    m = prepare_data(str(j), str(tmp_path / "o"), record_template=tmpl, split="tail", verbose=False, max_reject_fraction=0.05)
    assert m["counts"]["rejected_reasons"] == {"missing_field:answer": 1, "non_scalar_field:context": 1}
    toks = np.concatenate([np.fromfile(tmp_path / "o" / f"{s}.bin", dtype=np.uint16) for s in ("train", "val")])
    text = bytes(np.where(toks == EOS, 0, toks).astype(np.uint8)).decode().split("\x00")[:-1]
    assert len(text) == 198 and text[0] == "Q: What is 0?\nC: ctx 0\nA: 0"                    # fields stay in ONE record
    assert not any("What is 3?" in t for t in text)
    with pytest.raises(ValueError, match="plain column name"):
        prepare_data(str(j), str(tmp_path / "x"), record_template="{question.__class__}", verbose=False)
    with pytest.raises(ValueError, match="either text_field"):
        prepare_data(str(j), str(tmp_path / "x"), record_template=tmpl, text_field="question", verbose=False)
    # parquet: ambiguity is an error, not a silent first-column guess
    pq.write_table(pa.table({"a": ["x"] * 80, "b": ["y"] * 80}), tmp_path / "amb.parquet")
    with pytest.raises(ValueError, match="several string columns"):
        prepare_data(str(tmp_path / "amb.parquet"), str(tmp_path / "x"), verbose=False)
    with pytest.raises(ValueError, match="not a string column"):
        pq.write_table(pa.table({"n": list(range(80)), "t": ["x"] * 80}), tmp_path / "n.parquet")
        prepare_data(str(tmp_path / "n.parquet"), str(tmp_path / "x"), text_field="n", verbose=False)
    m = prepare_data(str(tmp_path / "amb.parquet"), str(tmp_path / "pt"), record_template="{a}|{b}", verbose=False)
    assert m["sources"][0]["records"] == 80


# --------------------------------------------------------------------------- manifest + immutability
def test_manifest_hashes_fingerprint_and_verification(tmp_path):
    src = write_corpus(tmp_path / "c.txt", make_docs(300))
    out = str(tmp_path / "o")
    m = prepare_data(src, out, doc_sep=SEP, verbose=False)
    import hashlib
    for name in ("train.bin", "val.bin"):
        assert m["files"][name]["sha256"] == hashlib.sha256(open(os.path.join(out, name), "rb").read()).hexdigest()
        assert m["files"][name]["bytes"] == os.path.getsize(os.path.join(out, name))
    assert m["sources"][0]["sha256"] == hashlib.sha256(open(src, "rb").read()).hexdigest()
    assert m["tokenizer"]["eos"] == EOS and m["code_version"] and m["created_utc"].endswith("Z")
    assert m["fingerprint"] == compute_fingerprint(m) == read_meta(out)["fingerprint"]
    assert verify_dataset(out) == [] and verify_dataset(out, deep=True) == []
    ident = data_identity(out)
    assert ident["kind"] == "manifest" and ident["fingerprint"] == m["fingerprint"]
    # in-place corruption of the same size is caught by the deep check; a size change by the shallow one
    with open(os.path.join(out, "train.bin"), "r+b") as fh:
        fh.seek(40)
        fh.write(b"\x07\x07")
    assert verify_dataset(out) == [] and "content hash differs" in verify_dataset(out, deep=True)[0]
    with open(os.path.join(out, "val.bin"), "ab") as fh:
        fh.write(b"\x00\x00")
    assert "manifest says" in verify_dataset(out)[0]
    with open(os.path.join(out, "meta.json")) as fh:
        meta = json.load(fh)
    meta["counts"]["n_val_tokens"] += 1
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    os.truncate(os.path.join(out, "val.bin"), m["files"]["val.bin"]["bytes"])
    assert any("fingerprint does not match" in p for p in verify_dataset(out, deep=False) + verify_dataset(out, deep=True))
    with pytest.raises(DatasetError):
        ByteDataset(out).verify()


def test_legacy_directories_get_a_quick_identity_that_tracks_replacement(tmp_path):
    d = tmp_path / "legacy"
    d.mkdir()
    np.arange(5000, dtype=np.uint16).tofile(d / "train.bin")
    np.arange(500, dtype=np.uint16).tofile(d / "val.bin")
    json.dump({"n_train_tokens": 5000, "n_val_tokens": 500}, open(d / "meta.json", "w"))
    a = data_identity(str(d))
    assert a["kind"] == "quick" and a["fingerprint"].startswith("quick:") and verify_dataset(str(d)) == []
    assert data_identity(str(d))["fingerprint"] == a["fingerprint"]
    (np.arange(5000, dtype=np.uint16) + 1).tofile(d / "train.bin")                   # same size, different data
    assert data_identity(str(d))["fingerprint"] != a["fingerprint"]
    with pytest.raises(DatasetError):
        data_identity(str(tmp_path / "nothing"))


def test_prepared_datasets_are_immutable_atomic_and_never_deleted(tmp_path):
    src_a = write_corpus(tmp_path / "a.txt", make_docs(200, seed=1))
    src_b = write_corpus(tmp_path / "b.txt", make_docs(200, seed=2))
    out = str(tmp_path / "o")
    a = prepare_data(src_a, out, doc_sep=SEP, verbose=False)
    before = os.stat(os.path.join(out, "train.bin")).st_mtime_ns
    again = prepare_data(src_a, out, doc_sep=SEP, verbose=False)                      # identical -> untouched
    assert again["fingerprint"] == a["fingerprint"] and os.stat(os.path.join(out, "train.bin")).st_mtime_ns == before
    with pytest.raises(FileExistsError, match="immutable"):
        prepare_data(src_b, out, doc_sep=SEP, verbose=False)
    assert read_meta(out)["fingerprint"] == a["fingerprint"] and not os.path.exists(out + ".staging")
    b = prepare_data(src_b, out, doc_sep=SEP, verbose=False, overwrite=True)
    assert b["fingerprint"] != a["fingerprint"] and read_meta(out)["fingerprint"] == b["fingerprint"]
    aside = [p for p in os.listdir(tmp_path) if p.startswith("o.replaced-")]
    assert len(aside) == 1 and read_meta(str(tmp_path / aside[0]))["fingerprint"] == a["fingerprint"]   # moved, not deleted
    assert verify_dataset(out, deep=True) == []
    # a failed preparation leaves neither a staging directory nor files in the target
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff" * 100)
    with pytest.raises(ValueError):
        prepare_data(str(bad), str(tmp_path / "fresh"), verbose=False)
    assert not os.path.exists(tmp_path / "fresh") and not os.path.exists(tmp_path / "fresh.staging")


# ------------------------------------------------------------------------------- ByteDataset safety
def test_sampler_reaches_the_last_legal_window(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    np.arange(18, dtype=np.uint16).tofile(d / "train.bin")
    ds = ByteDataset(str(d), seed=0)
    starts = set()
    for _ in range(100):
        x, y = ds.get_batch("train", 8, 16, "cpu")
        starts.update(x[:, 0].tolist())
        assert (y[:, -1] == x[:, 0] + 16).all()                       # last target is a real token
    assert starts == {0, 1}                                          # valid starts: 0 and 1 (18 tokens, block 16)


def test_replacing_a_dataset_file_is_noticed(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    np.arange(100, dtype=np.uint16).tofile(d / "train.bin")
    ds = ByteDataset(str(d), seed=0)
    assert int(ds.array(str(d), "train")[0]) == 0
    (np.arange(100, dtype=np.uint16) + 7).tofile(d / "new.bin")
    os.replace(d / "new.bin", d / "train.bin")
    with pytest.warns(UserWarning, match="changed on disk"):
        assert int(ds.array(str(d), "train")[0]) == 7
    assert ds.n_remaps == 1
    ds2 = ByteDataset(str(d), on_change="error")
    ds2.array(str(d), "train")
    np.arange(100, dtype=np.uint16).tofile(d / "n2.bin")
    os.replace(d / "n2.bin", d / "train.bin")
    with pytest.raises(RuntimeError, match="must not change during a run"):
        ds2.array(str(d), "train")
    ds2.close()
    assert int(ds2.array(str(d), "train")[0]) == 0                    # explicit close() re-opens cleanly
    with pytest.raises(ValueError, match="on_change"):
        ByteDataset(str(d), on_change="ignore")


def test_audit_reports_fingerprint_examples_and_problems(tmp_path):
    src = write_corpus(tmp_path / "c.txt", make_docs(300))
    out = str(tmp_path / "o")
    m = prepare_data(src, out, doc_sep=SEP, verbose=False)
    rep = audit_dataset(out, n_examples=2)
    assert m["fingerprint"] in rep["text"] and "Doc " in rep["text"] and rep["problems"] == []
    assert rep["stats"]["train"]["control_fraction"] == 0.0 and len(rep["examples"]["val"]) == 2
    with open(os.path.join(out, "val.bin"), "ab") as fh:
        fh.write(b"\x00\x00")
    assert audit_dataset(out)["problems"] and "PROBLEMS" in audit_dataset(out)["text"]


# ------------------------------------------------------------------------------------- downloads
@pytest.fixture
def http_file(tmp_path):
    payload = ("story one. <|endoftext|>" * 3000).encode()
    (tmp_path / "f.txt").write_bytes(payload)
    srv = HTTPServer(("127.0.0.1", 0), partial(_RangeHandler, directory=str(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/f.txt", payload, tmp_path
    srv.shutdown()


def test_downloads_record_their_hash_and_verify_it(http_file):
    import hashlib
    url, payload, tmp = http_file
    dest = str(tmp / "dl" / "x.txt")
    download_file(url, dest, verbose=False)
    rec = read_download_record(dest)
    assert rec["sha256"] == hashlib.sha256(payload).hexdigest() and rec["bytes"] == len(payload) and rec["url"] == url
    # a cached file that no longer matches its record is fetched again (and the bad copy is kept aside)
    open(dest, "ab").write(b"garbage")
    download_file(url, dest, verbose=False)
    assert open(dest, "rb").read() == payload and any(p.startswith("x.txt.corrupt-") for p in os.listdir(tmp / "dl"))
    # expected_sha256: mismatch is an error and leaves nothing behind
    with pytest.raises(RuntimeError, match="does not match the expected"):
        download_file(url, str(tmp / "dl" / "y.txt"), verbose=False, expected_sha256="0" * 64)
    assert not os.path.exists(tmp / "dl" / "y.txt") and not os.path.exists(tmp / "dl" / "y.txt.part")
    download_file(url, str(tmp / "dl" / "z.txt"), verbose=False, expected_sha256=hashlib.sha256(payload).hexdigest())


# ------------------------------------------------------------------------------------------ CLI
def test_cli_prepare_and_audit(tmp_path, capsys):
    from tensegrity.cli import main
    src = write_corpus(tmp_path / "c.txt", make_docs(300))
    out = str(tmp_path / "o")
    main(["prepare", "--data_path", src, "--out_dir", out, "--doc_sep", SEP, "--val_fraction", "0.1"])
    printed = capsys.readouterr().out
    assert "fingerprint" in printed and "document" in printed
    main(["audit", "--data_dir", out, "--deep", "--examples", "1"])
    assert "integrity   : OK (content re-hashed)" in capsys.readouterr().out
    with open(os.path.join(out, "val.bin"), "ab") as fh:
        fh.write(b"\x00\x00")
    with pytest.raises(SystemExit) as e:
        main(["audit", "--data_dir", out])
    assert e.value.code == 1
    with pytest.raises(FileExistsError):
        main(["prepare", "--data_path", write_corpus(tmp_path / "d.txt", make_docs(300, seed=9)), "--out_dir", out,
              "--doc_sep", SEP])


def test_generation_starts_from_the_token_the_data_contains():
    assert ByteTokenizer.DOC_START == EOS != ByteTokenizer.BOS
