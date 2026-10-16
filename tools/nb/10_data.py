#@title 2 · Download, prepare & audit the dataset { display-mode: "form" }
CUSTOM_TEXT_FIELD = ""  #@param {type:"string"}
CUSTOM_RECORD_TEMPLATE = ""  #@param {type:"string"}
import hashlib, os, shutil

PROJECT_DIR = tg.project_dir(USE_GOOGLE_DRIVE, "tensegrity")
SCRATCH = tg.local_scratch("tensegrity_data")
_free = shutil.disk_usage(SCRATCH).free / 2**30
print(f"project dir : {PROJECT_DIR}\nlocal scratch: {SCRATCH}  ({_free:.0f} GB free)")

_prep_kw = {}
if DATASET == "custom":
    if not CUSTOM_DATA_PATH or not os.path.exists(CUSTOM_DATA_PATH):
        raise FileNotFoundError(
            "DATASET='custom' needs CUSTOM_DATA_PATH to point at a .txt/.md/.jsonl/.parquet file or a folder "
            "of them, e.g. /content/drive/MyDrive/my_corpus.txt")
    _src, _sep, _val = CUSTOM_DATA_PATH, (CUSTOM_DOC_SEPARATOR or None), 0.02
    # jsonl / parquet: name the text field, or stitch several fields into ONE document, e.g.
    #   CUSTOM_RECORD_TEMPLATE = "Q: {question}\nContext: {context}\nA: {answer}"
    _prep_kw = dict(text_field=CUSTOM_TEXT_FIELD or None, record_template=CUSTOM_RECORD_TEMPLATE.replace("\\n", "\n") or None)
    _key = "custom-" + hashlib.md5(os.path.abspath(_src).encode()).hexdigest()[:8]
else:
    _mb = DATA_MEGABYTES if DATASET == "tinystories" else None
    _need = (_mb or tg.DATASETS[DATASET]["approx_mb"]) * 5.5 / 1024      # raw text + token stream + train/val copies
    if _free < _need + 1:
        raise RuntimeError(f"need ~{_need:.1f} GB of local disk, only {_free:.1f} GB free")
    _spec = tg.fetch_dataset(DATASET, os.path.join(SCRATCH, "raw"), max_mb=_mb)
    _src, _sep, _val = _spec["path"], _spec["doc_sep"], _spec["val_fraction"]
    _key = f"{DATASET}-{_mb or 'full'}mb"

DATA_DIR = os.path.join(SCRATCH, "prepared", _key)
if not os.path.exists(os.path.join(DATA_DIR, "meta.json")):
    tg.prepare_data(_src, DATA_DIR, val_fraction=_val, doc_sep=_sep, **_prep_kw)
DATA_META = tg.load_meta(DATA_DIR)
DATA_AUDIT = tg.audit_dataset(DATA_DIR, n_examples=2)
print("\n" + DATA_AUDIT["text"])
if DATA_AUDIT["problems"]:
    raise RuntimeError("the prepared dataset is inconsistent: " + "; ".join(DATA_AUDIT["problems"]))
print(f"\ndata ready: {DATA_META['n_train_tokens'] / 1e6:.1f}M train bytes, {DATA_META['n_val_tokens'] / 1e6:.2f}M val bytes, "
      f"{DATA_META['n_documents']:,} documents  ->  {DATA_DIR}")
