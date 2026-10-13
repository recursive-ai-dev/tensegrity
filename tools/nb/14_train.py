#@title 4 · Train (resumable) { display-mode: "form" }
import math
live = tg.LivePlot()
trainer = tg.Trainer(model, TC, PLAN, DATA_DIR, RUN_DIR, device=DEVICE, callbacks=[live], lineage=LINEAGE)
RESULT = trainer.fit(resume=True)
print("\n" + "=" * 70)
if RESULT["status"] == "completed":
    print(f"✅ Finished. Best validation: {RESULT['best_val_loss'] / math.log(2):.3f} bits/byte "
          f"({RESULT['best_source']} weights) after {RESULT['tokens_seen'] / 1e6:,.2f}M bytes.")
else:
    print(f"⏸  Stopped by your STOP_AFTER_MINUTES limit at step {RESULT['step']:,} of {PLAN.max_steps:,}. "
          f"Run this cell again to continue exactly where it left off.")
