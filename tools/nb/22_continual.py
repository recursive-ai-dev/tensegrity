#@title 7 · Continual learning on a second corpus { display-mode: "form" }
RUN_CONTINUAL = False  #@param {type:"boolean"}
NEW_DATASET = "shakespeare"  #@param ["shakespeare", "tinystories-valid", "custom"]
NEW_CUSTOM_PATH = ""  #@param {type:"string"}
REPLAY_FRACTION = 0.3  #@param {type:"slider", min:0.0, max:0.9, step:0.05}
EWC_LAMBDA = 100.0  #@param {type:"number"}
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

    _old_model, _old_cfg, _ = tg.load_model(BEST_PATH, DEVICE)
    def _val(m, d):
        return tg.evaluate(m, d, _old_cfg, DEVICE, DTYPE, 20, 8)["bpb"]
    _before = (_val(_old_model, DATA_DIR), _val(_old_model, NEW_DIR))
    _fisher, _anchor = tg.compute_fisher_diagonal(_old_model, DATA_DIR, DEVICE, n_sequences=48)
    _ewc = tg.make_ewc_loss(_fisher, _anchor, EWC_LAMBDA)

    _tokens = int(CONTINUAL_MEGABYTES_OF_TRAINING * 1e6)
    _cplan = tg.make_plan(_tokens, TC.batch_size * TC.grad_accum_steps * _old_cfg.block_size, "wsd")
    _ctc = tg.TrainConfig(**{**TC.persistent_dict(), "learning_rate": TC.learning_rate * 0.3, "min_lr": TC.min_lr * 0.3,
                             "ema_decay": 0.0, "max_minutes": STOP_AFTER_MINUTES or None})
    _cdir = os.path.join(PROJECT_DIR, "runs", f"{RUN_NAME_}+{_nkey}")
    _model2 = tg.TensegrityLM(_old_cfg)
    _model2.load_state_dict(_old_model.state_dict())
    _fine = tg.Trainer(_model2, _ctc, _cplan, [NEW_DIR, DATA_DIR], _cdir,
                       data_weights=[1.0 - REPLAY_FRACTION, REPLAY_FRACTION] if REPLAY_FRACTION > 0 else [1.0, 0.0],
                       device=DEVICE, extra_loss=_ewc if EWC_LAMBDA > 0 else None, callbacks=[tg.LivePlot()])
    _fine.fit(resume=True)
    _m2, _, _ = tg.load_model(os.path.join(_cdir, "ckpt_best.pt"), DEVICE)
    _after = (_val(_m2, DATA_DIR), _val(_m2, NEW_DIR))
    print(f"\n{'':22s}{'original corpus':>18s}{'new corpus':>14s}   (bits/byte, lower is better)")
    print(f"{'before fine-tuning':22s}{_before[0]:18.3f}{_before[1]:14.3f}")
    print(f"{'after (replay+EWC)':22s}{_after[0]:18.3f}{_after[1]:14.3f}")
else:
    print("Set RUN_CONTINUAL = True to fine-tune on a second corpus with replay + EWC.")
