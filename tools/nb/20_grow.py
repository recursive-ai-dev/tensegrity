#@title 6b · Or grow the model you just trained by ONE rung (as a verified transaction) { display-mode: "form" }
GROW_NOW = False  #@param {type:"boolean"}

if GROW_NOW:
    _nxt = tg.next_preset(MODEL_SIZE)
    if _nxt is None:
        raise SystemExit("already at the top of the ladder (115M)")
    _target = tg.retarget(BEST_CFG, tg.get_preset_config(_nxt.name))
    _suite = tg.EvalSuite([tg.EvalDomain("data", DATA_DIR, "retain", n_batches=20, batch_size=16)])
    DECISION = tg.grow_transaction(BEST, BEST_CFG, target=_target, suite=_suite, device=DEVICE,
                                   run_dir=RUN_DIR, verbose=False)
    print(DECISION.summary())
    DECISION.raise_if_rejected()                      # a rejected candidate is never saved or trained
    GROWN, GROWN_CFG, _chk = DECISION.candidate, DECISION.candidate_cfg, DECISION.report.checks
    print(f"function preserved: max |dlogit| {_chk['preservation_max_logit_diff']:.2e}; "
          f"reloads identically via {', '.join(_chk['roundtrip_max_logit_diff'])}")
    print(f"trainability: every new head / unit / channel / layer receives gradient "
          f"(smallest max-|grad| {_chk['trainability_min_max_grad']:.2e})")
    GROWN_PATH = os.path.join(PROJECT_DIR, "grown", f"{RUN_NAME_}-to-{_nxt.label}.pt")
    tg.save_checkpoint(GROWN_PATH, GROWN, None, GROWN_CFG, 0, float("inf"),
                       extra={"grown_from": tg.file_record(BEST_PATH), "growth_report": DECISION.report.to_dict()})
    _gm = DECISION.final_eval.domains["data"]
    print(f"\n{BEST_CFG.n_params() / 1e6:.2f}M -> {GROWN_CFG.n_params() / 1e6:.2f}M parameters; "
          f"validation unchanged at {_gm['bpb_bytes']:.3f} bits/byte (just grown, not yet trained).")
    print(f"saved {GROWN_PATH}\n\nTo keep training it: in §1 set MODEL_SIZE = '{_nxt.label}' and "
          f"INIT_FROM_WEIGHTS = '{GROWN_PATH}', then re-run §3 and §4 (use a new RUN_NAME or START_FRESH).")
else:
    print("Set GROW_NOW = True to grow the trained model by one rung and save it.")
