#@title 0.2 · Load the library & inspect this runtime { display-mode: "form" }
import os, sys, warnings
sys.path.insert(0, os.path.abspath("tensegrity_src"))
for _name in [m for m in sys.modules if m == "tensegrity" or m.startswith("tensegrity.")]:
    del sys.modules[_name]                      # re-running this cell always picks up the freshly written source
warnings.filterwarnings("ignore", message=".*(legacy|TypedStorage).*")
import torch  # shared by the cells below
import tensegrity as tg

print(f"Tensegrity {tg.__version__}")
print(tg.runtime_summary())
_gpu = tg.gpu_info()
if not _gpu["available"]:
    print("\n⚠️  No GPU detected. Use Runtime → Change runtime type → GPU. "
          "On CPU only the 1M model with a short context is practical (set SMOKE_TEST in §1 to try the whole notebook).")
elif _gpu["native_bf16"]:
    print("\n✅ Ampere-or-newer GPU: training in bfloat16.")
else:
    print("\n✅ Turing/Volta-class GPU (e.g. the free T4): training in float16 with loss scaling "
          "(bfloat16 would be emulated and slow here; the library knows).")
