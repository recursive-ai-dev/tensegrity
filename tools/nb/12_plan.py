#@title 3 · Hardware check, autotune & time budget { display-mode: "form" }
import shutil, time

DEVICE = tg.pick_device("auto")
DTYPE = tg.pick_dtype("auto", DEVICE)
PRESET = tg.get_preset(MODEL_SIZE)
_CPU_MODE = SMOKE_TEST or DEVICE == "cpu"
MODEL_OVERRIDES = dict(block_size=128, sliding_window=32) if _CPU_MODE else {}
BATCH_TOKENS = 4096 if _CPU_MODE else PRESET.batch_tokens      # small optimizer steps keep CPU runs responsive
if _CPU_MODE:
    print(f"(CPU / smoke-test mode: {MODEL_OVERRIDES['block_size']}-byte context, {BATCH_TOKENS}-byte optimizer steps)")

RUN_NAME_ = RUN_NAME.strip() or f"{PRESET.label}-{DATASET}"
RUN_DIR = os.path.join(PROJECT_DIR, "runs", RUN_NAME_)
if START_FRESH and os.path.exists(RUN_DIR):
    _bak = f"{RUN_DIR}.old-{int(time.time())}"
    shutil.move(RUN_DIR, _bak)                      # moved, never deleted
    print(f"START_FRESH: previous run kept at {_bak}")

_saved = tg.load_run_plan(RUN_DIR)
if _saved is not None:
    PLAN, TC, CFG = _saved
    model = tg.TensegrityLM(CFG)
    print(f"Resuming run '{RUN_NAME_}' with its saved plan ({PLAN.max_steps:,} steps).")
elif INIT_FROM_WEIGHTS:
    _src, CFG, _ = tg.load_model(INIT_FROM_WEIGHTS, "cpu")           # architecture comes from the weights
    model = tg.TensegrityLM(CFG)
    model.load_state_dict(_src.state_dict())
    del _src
    print(f"Initialised a {CFG.n_params() / 1e6:.2f}M-parameter model from {INIT_FROM_WEIGHTS}")
    if abs(CFG.n_params() / PRESET.model_kwargs_params() - 1) > 0.15:
        print(f"⚠️  These weights are not a {PRESET.label} model; hyper-parameters below come from the {PRESET.label} preset.")
else:
    CFG = tg.get_preset_config(PRESET.name, **MODEL_OVERRIDES)
    model = tg.TensegrityLM(CFG)
model.to(DEVICE)

if _saved is None:
    _tune = tg.autotune_micro_batch(model, CFG.block_size, DEVICE, DTYPE, BATCH_TOKENS, verbose=False)
    model.grad_checkpoint = _tune["grad_checkpoint"]
    _tps_step = _tune["micro_batch"] * _tune["grad_accum"] * CFG.block_size
    _tps = tg.measure_throughput(model, _tune["micro_batch"], CFG.block_size, DEVICE, DTYPE, steps=4, warmup=2)
    PLAN = tg.plan_for_time_budget(_tps, TRAIN_HOURS, _tps_step, max_tokens=PRESET.default_tokens)
    TC, PLAN = tg.train_setup(PRESET.name, _tune["micro_batch"], _tune["grad_accum"], plan=PLAN, optimizer=OPTIMIZER)
    _mfu = tg.mfu(CFG, _tps, CFG.block_size, tg.gpu_info()["peak_tflops"])
    print(f"measured throughput : {_tps / 1e3:,.1f}k bytes/s" + (f"  (~{_mfu * 100:.0f}% MFU)" if _mfu else ""))
    print(f"micro-batch         : {_tune['micro_batch']} x {_tune['grad_accum']} accumulation x {CFG.block_size} bytes"
          f"{'  [gradient checkpointing on: memory-bound]' if _tune['grad_checkpoint'] else ''}")
if _saved is None:                                  # new run: prove the optimizer path is numerically sound on THIS device
    TC = tg.preflight_or_fallback(model, TC, DEVICE)  # falls back to AdamW automatically if e.g. Muon misbehaves
TC.max_minutes = STOP_AFTER_MINUTES or None         # per-session limit, never persisted
LINEAGE = ({"init_from": tg.file_record(INIT_FROM_WEIGHTS)} if (INIT_FROM_WEIGHTS and _saved is None) else None)
_epochs = PLAN.total_tokens / max(1, DATA_META["n_train_tokens"])
print(f"model               : {CFG.describe()}")
def _human(n):
    return f"{n / 1e6:,.1f}M" if n >= 1e6 else f"{n / 1e3:,.0f}K"
print(f"training plan       : {PLAN.max_steps:,} steps x {PLAN.tokens_per_step:,} bytes = {_human(PLAN.total_tokens)} bytes "
      f"({_epochs:.2f} passes over your data) | peak LR {TC.learning_rate:.1e} | EMA decay {TC.ema_decay:.4f}")
print(f"schedule            : {PLAN.warmup_steps} warmup -> {PLAN.stable_steps:,} stable -> {PLAN.decay_steps:,} cool-down")
if _epochs > 8:
    print("⚠️  This plan repeats your data many times; consider more data (§1) or fewer hours to avoid memorisation.")

PREFLIGHT = tg.preflight_report(model, TC, PLAN, DATA_DIR, DEVICE)
print("\n" + tg.format_preflight(PREFLIGHT))
if not PREFLIGHT["ok"]:
    raise RuntimeError("preflight found blockers (listed above); fix them before training")
