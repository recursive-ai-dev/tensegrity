#@title 7 · Continual learning on a second corpus (new-only vs replay vs replay + EWC) { display-mode: "form" }
RUN_CONTINUAL = False  #@param {type:"boolean"}
NEW_DATASET = "shakespeare"  #@param ["shakespeare", "tinystories-valid", "custom"]
NEW_CUSTOM_PATH = ""  #@param {type:"string"}
REPLAY_FRACTION = 0.3  #@param {type:"slider", min:0.0, max:0.9, step:0.05}
EWC_LAMBDA = 100.0  #@param {type:"number"}
CONTINUAL_SEEDS = "0"  #@param {type:"string"}
CONTINUAL_MEGABYTES_OF_TRAINING = 20  #@param {type:"number"}

if RUN_CONTINUAL:
    if NEW_DATASET == "custom":
        _nsrc, _nsep, _nval, _nkey = NEW_CUSTOM_PATH, None, 0.05, "newcustom"
    else:
        _nspec = tg.fetch_dataset(NEW_DATASET, os.path.join(SCRATCH, "raw"))
        _nsrc, _nsep, _nval, _nkey = _nspec["path"], _nspec["doc_sep"], _nspec["val_fraction"], NEW_DATASET
    NEW_DIR = os.path.join(SCRATCH, "prepared", _nkey)
    if not os.path.exists(os.path.join(NEW_DIR, "meta.json")):
        tg.prepare_data(_nsrc, NEW_DIR, val_fraction=_nval, doc_sep=_nsep)

    _tokens_per_step = TC.batch_size * TC.grad_accum_steps * BEST_CFG.block_size
    _steps = max(10, int(CONTINUAL_MEGABYTES_OF_TRAINING * 1e6) // _tokens_per_step)
    CONTINUAL = tg.compare_continual_strategies(
        BEST_PATH, DATA_DIR, NEW_DIR, steps=_steps, replay_fraction=REPLAY_FRACTION, ewc_lambda=EWC_LAMBDA,
        batch_size=TC.batch_size, grad_accum_steps=TC.grad_accum_steps, learning_rate=TC.learning_rate * 0.3,
        optimizer=TC.optimizer, seeds=[int(s) for s in CONTINUAL_SEEDS.split(",") if s.strip()], device=DEVICE,
        run_dir=os.path.join(PROJECT_DIR, "runs", f"{RUN_NAME_}+{_nkey}"))
else:
    print("Set RUN_CONTINUAL = True to compare ways of learning a second corpus (new-only / replay / replay + EWC).")
