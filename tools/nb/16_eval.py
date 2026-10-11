#@title 5.1 · Load the best checkpoint & measure it { display-mode: "form" }
import math
BEST_PATH = os.path.join(RUN_DIR, "ckpt_best.pt")
BEST, BEST_CFG, CKPT = tg.load_model(BEST_PATH, DEVICE)
BEST.eval()
EVAL = tg.evaluate(BEST, DATA_DIR, BEST_CFG, DEVICE, DTYPE, n_batches=50, batch_size=16)
print(f"checkpoint from step {CKPT['step']:,} ({CKPT.get('best_source', 'raw')} weights)")
print(f"validation: {EVAL['loss']:.4f} nats/byte | {EVAL['bpb']:.3f} bits/byte | perplexity {EVAL['ppl']:.2f}  "
      f"(chance = {math.log(259):.2f} nats = {math.log2(259):.2f} bits)")
if BEST.confidence_head is not None:
    _cal = tg.confidence_calibration(BEST, DATA_DIR, DEVICE, n_batches=20, batch_size=8, dtype="auto")
    print(f"confidence head: next-byte accuracy {_cal['base_accuracy']:.3f}, calibration error (ECE) {_cal['ece']:.3f}")
    tg.plot_reliability(_cal)
