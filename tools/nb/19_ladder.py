#@title 6 · Train → grow → train ... { display-mode: "form" }
RUN_THE_LADDER = False  #@param {type:"boolean"}
LADDER_RUNGS = "1M,2M,5M"  #@param {type:"string"}
HOURS_PER_RUNG = "0.15,0.3,1.0"  #@param {type:"string"}

if RUN_THE_LADDER:
    _rungs = [r.strip() for r in LADDER_RUNGS.split(",") if r.strip()]
    _hours = [float(h) for h in HOURS_PER_RUNG.split(",") if h.strip()]
    if len(_hours) == 1:
        _hours = _hours * len(_rungs)
    LADDER_ROOT = os.path.join(PROJECT_DIR, "ladders", f"{'-'.join(_rungs)}-{DATASET}")
    LADDER = tg.climb_ladder(_rungs, DATA_DIR, LADDER_ROOT, hours=_hours, optimizer=OPTIMIZER,
                             model_overrides=MODEL_OVERRIDES, batch_tokens=BATCH_TOKENS, callbacks=[tg.LivePlot()],
                             tc_overrides=dict(max_minutes=STOP_AFTER_MINUTES or None))
    print(f"\n{'rung':>6s} {'params':>9s} {'bits/byte':>10s} {'steps':>8s}  status")
    for r in LADDER:
        print(f"{r.rung.upper():>6s} {r.params / 1e6:8.2f}M {r.best_val_bpb:10.3f} {r.steps:8,d}  {r.status}"
              f"{'  (resumed from an earlier session)' if r.skipped else ''}")
else:
    print("Set RUN_THE_LADDER = True to train the rungs above in sequence.")
