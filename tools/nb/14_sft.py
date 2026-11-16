#@title 5B · SFT Fine-Tune (optional) { display-mode: "form" }
BASE_CKPT = ""  #@param {type:"string"}  # e.g. runs/5m/ckpt_best.pt — leave empty to use the run you just trained
SFT_HOURS = 2.0  #@param {type:"number"}  # wall-clock budget
SFT_LR = 1e-4  #@param {type:"number"}  # peak LR (AdamW)
SFT_BATCH = 8  #@param {type:"integer"}  # micro-batch
SFT_GRAD_ACC = 4  #@param {type:"integer"}  # grad accumulation

import os, time
if not BASE_CKPT:
    BASE_CKPT = os.path.join(RUN_DIR, "ckpt_best.pt")
if not os.path.exists(BASE_CKPT):
    raise FileNotFoundError(f"base checkpoint not found: {BASE_CKPT}")

SFT_DIR = os.path.join(PROJECT_DIR, "runs", f"{RUN_NAME_}-sft")
if START_FRESH and os.path.exists(SFT_DIR):
    _bak = f"{SFT_DIR}.old-{int(time.time())}"
    shutil.move(SFT_DIR, _bak)
    print(f"START_FRESH: previous SFT run kept at {_bak}")

print(f"[sft] loading base from {BASE_CKPT}")
model, CFG = tg.prepare_sft_from_base(BASE_CKPT, new_vocab_size=265, new_block_size=2048, new_rope_base=50000.0, device=DEVICE)
model.to(DEVICE)

_tune = tg.autotune_micro_batch(model, CFG.block_size, DEVICE, DTYPE, SFT_BATCH * SFT_GRAD_ACC * CFG.block_size, verbose=False)
model.grad_checkpoint = _tune["grad_checkpoint"]
_tps_step = SFT_BATCH * SFT_GRAD_ACC * CFG.block_size
_tps = tg.measure_throughput(model, SFT_BATCH, CFG.block_size, DEVICE, DTYPE, steps=4, warmup=2)
PLAN_SFT = tg.plan_for_time_budget(_tps, SFT_HOURS, _tps_step, max_tokens=100_000_000)
TC_SFT, PLAN_SFT = tg.train_setup(MODEL_SIZE.replace("-chat", ""), SFT_BATCH, SFT_GRAD_ACC, plan=PLAN_SFT, optimizer="adamw")
TC_SFT.learning_rate = SFT_LR
TC_SFT.min_lr = SFT_LR / 10
TC_SFT.schedule = "cosine"
TC_SFT.dropout = 0.05
TC_SFT.max_minutes = None
print(f"[sft] {PRESET.label} -> SFT: vocab {CFG.vocab_size}, ctx {CFG.block_size}, rope_base {CFG.rope_base:.0f}")
print(f"[sft] micro-batch {SFT_BATCH} x {SFT_GRAD_ACC} = {SFT_BATCH * SFT_GRAD_ACC * CFG.block_size:,} bytes/step")

_saved = tg.load_run_plan(SFT_DIR)
if _saved is not None:
    PLAN_SFT, TC_SFT, _ = _saved
    print(f"[sft] resuming SFT run ({PLAN_SFT.max_steps:,} steps)")

trainer_sft = tg.Trainer(model, TC_SFT, PLAN_SFT, DATA_DIR, SFT_DIR, device=DEVICE, lineage={"init_from": tg.file_record(BASE_CKPT)})
RESULT_SFT = trainer_sft.fit(resume=bool(_saved))
print(f"[sft] done: {RESULT_SFT['status']}, step={RESULT_SFT['step']}, best_val_loss={RESULT_SFT['best_val_loss']:.4f}")