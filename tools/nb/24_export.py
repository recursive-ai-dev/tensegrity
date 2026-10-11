#@title 8 · Export (safetensors, model card, optional INT8 / Hugging Face) { display-mode: "form" }
EXPORT_DTYPE = "float16"  #@param ["float32", "float16", "bfloat16"]
ALSO_INT8_CPU_COPY = False  #@param {type:"boolean"}
DOWNLOAD_ZIP = True  #@param {type:"boolean"}
PUSH_TO_HUGGING_FACE = False  #@param {type:"boolean"}
HF_REPO_ID = ""  #@param {type:"string"}
HF_PRIVATE = True  #@param {type:"boolean"}

import shutil
EXPORT_DIR = os.path.join(PROJECT_DIR, "exports", RUN_NAME_)
_w, _c = tg.export_safetensors(BEST, BEST_CFG, EXPORT_DIR, "tensegrity", EXPORT_DTYPE,
                               extra_metadata={"val_bpb": f"{EVAL['bpb']:.4f}", "dataset": DATASET})
tg.write_model_card(EXPORT_DIR, BEST_CFG, RUN_NAME_, metrics={"validation bits/byte": EVAL["bpb"], "perplexity": EVAL["ppl"]},
                    training={"steps": CKPT["step"], "tokens seen": CKPT.get("tokens_seen", "n/a"), "optimizer": OPTIMIZER},
                    dataset=DATASET)
print(f"wrote {_w} ({os.path.getsize(_w) / 1e6:.1f} MB)")
if ALSO_INT8_CPU_COPY:
    _q = tg.quantize_int8_dynamic(BEST)
    if _q is not None:
        _qp = os.path.join(EXPORT_DIR, "tensegrity_int8_cpu.pt")
        torch.save(_q.state_dict(), _qp)
        print(f"wrote {_qp} ({os.path.getsize(_qp) / 1e6:.1f} MB)")
if PUSH_TO_HUGGING_FACE:
    _tok = tg.get_secret("HF_TOKEN")
    if not (_tok and HF_REPO_ID):
        print("To upload: add a Colab secret named HF_TOKEN (🔑 in the left sidebar) and set HF_REPO_ID = 'user/name'.")
    else:
        print("uploaded to", tg.push_to_hub(EXPORT_DIR, HF_REPO_ID, _tok, private=HF_PRIVATE))
if DOWNLOAD_ZIP:
    _zip = shutil.make_archive(os.path.join(SCRATCH, f"{RUN_NAME_}-export"), "zip", EXPORT_DIR)
    tg.download_to_browser(_zip)
