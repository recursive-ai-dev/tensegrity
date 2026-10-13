#@title 5.2 · Generate text (confidence-shaded) { display-mode: "form" }
PROMPT = "Once upon a time, there was a little"  #@param {type:"string"}
MAX_NEW_BYTES = 400  #@param {type:"integer"}
TEMPERATURE = 0.8  #@param {type:"slider", min:0.0, max:1.5, step:0.05}
TOP_K = 40  #@param {type:"integer"}
TOP_P = 0.95  #@param {type:"slider", min:0.1, max:1.0, step:0.05}
MIN_P = 0.02  #@param {type:"slider", min:0.0, max:0.3, step:0.01}
CONFIDENCE_TEMPERATURE = True  #@param {type:"boolean"}
SEED = 0  #@param {type:"integer"}

import codecs
from IPython.display import HTML, display
tok = tg.ByteTokenizer()
torch.manual_seed(SEED)
_ids = [tok.BOS] + tok.encode(PROMPT)
_idx = torch.tensor([_ids], dtype=torch.long, device=DEVICE)
_dec = codecs.getincrementaldecoder("utf-8")("replace")
_toks, _conf = list(_ids), [None] * len(_ids)
print(PROMPT, end="", flush=True)
for _t, _c in BEST.stream(_idx, MAX_NEW_BYTES, temperature=TEMPERATURE, top_k=TOP_K or None, top_p=TOP_P,
                          min_p=MIN_P or None, use_confidence_temp=CONFIDENCE_TEMPERATURE and BEST.confidence_head is not None):
    _toks.append(_t); _conf.append(_c)
    if _t < 256:
        print(_dec.decode(bytes([_t])), end="", flush=True)
print("\n")
if BEST.confidence_head is not None:
    print("Shaded by the model's own confidence (green = sure, red = guessing):")
    display(HTML(tg.confidence_html(_toks[len(_ids):], _conf[len(_ids):])))
