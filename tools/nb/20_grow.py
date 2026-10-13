#@title 6b · Or grow the model you just trained by ONE rung { display-mode: "form" }
GROW_NOW = False  #@param {type:"boolean"}

if GROW_NOW:
    _nxt = tg.next_preset(MODEL_SIZE)
    if _nxt is None:
        raise SystemExit("already at the top of the ladder (115M)")
    _target = tg.retarget(BEST_CFG, tg.get_preset_config(_nxt.name))
    GROWN, GROWN_CFG = tg.grow_model(BEST, BEST_CFG, target=_target, device=DEVICE)
    tg.check_function_preserved(BEST, GROWN, BEST_CFG, DEVICE)
    tg.check_growth_trainable(GROWN, BEST_CFG)
    _gm = tg.evaluate(GROWN, DATA_DIR, GROWN_CFG, DEVICE, DTYPE, n_batches=20, batch_size=16)
    GROWN_PATH = os.path.join(PROJECT_DIR, "grown", f"{RUN_NAME_}-to-{_nxt.label}.pt")
    tg.save_checkpoint(GROWN_PATH, GROWN, None, GROWN_CFG, 0, float("inf"))
    print(f"\n{BEST_CFG.n_params() / 1e6:.2f}M -> {GROWN_CFG.n_params() / 1e6:.2f}M parameters; "
          f"validation unchanged at {_gm['bpb']:.3f} bits/byte (just grown, not yet trained).")
    print(f"saved {GROWN_PATH}\n\nTo keep training it: in §1 set MODEL_SIZE = '{_nxt.label}' and "
          f"INIT_FROM_WEIGHTS = '{GROWN_PATH}', then re-run §3 and §4 (use a new RUN_NAME or START_FRESH).")
else:
    print("Set GROW_NOW = True to grow the trained model by one rung and save it.")
