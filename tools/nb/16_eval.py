#@title 5.1 · Load the best checkpoint & measure it { display-mode: "form" }
RUN_CONTEXT_PROBE = True  #@param {type:"boolean"}
import math
BEST_PATH = os.path.join(RUN_DIR, "ckpt_best.pt")
BEST, BEST_CFG, CKPT = tg.load_model(BEST_PATH, DEVICE)
BEST.eval()
EVAL = tg.evaluate(BEST, DATA_DIR, BEST_CFG, DEVICE, DTYPE, n_batches=50, batch_size=16)
print(f"checkpoint from step {CKPT['step']:,} ({CKPT.get('best_source', 'raw')} weights)")
print(f"validation: {EVAL['bpb_bytes']:.3f} bits/byte | {EVAL['loss']:.4f} nats/token | perplexity {EVAL['ppl']:.2f}  "
      f"(chance = {math.log2(256):.2f} bits/byte; {EVAL['n_bytes']:,} bytes scored, "
      f"{EVAL['n_special']:,} document-boundary tokens at {EVAL['bits_per_special']:.2f} bits each)"
      if EVAL["n_special"] else
      f"validation: {EVAL['bpb_bytes']:.3f} bits/byte | {EVAL['loss']:.4f} nats/token | perplexity {EVAL['ppl']:.2f}  "
      f"(chance = {math.log2(256):.2f} bits/byte; {EVAL['n_bytes']:,} bytes scored)")
print(BEST_CFG.describe())
if RUN_CONTEXT_PROBE:
    _dist = [d for d in (0, 16, 64, 256, 1024) if 2 * 12 + d <= BEST_CFG.block_size]
    print("\ncopy/retrieval probe (does the model use bytes from d positions back?):")
    print(tg.format_probe(tg.context_probe(BEST, _dist, DEVICE, key_len=12, n_trials=64)))
if BEST.confidence_head is not None:
    _cal = tg.confidence_calibration(BEST, DATA_DIR, DEVICE, n_batches=20, batch_size=8, dtype="auto")
    print(f"\nconfidence head: next-byte accuracy {_cal['base_accuracy']:.3f}, calibration error (ECE) {_cal['ece']:.3f}")
    tg.plot_reliability(_cal)
