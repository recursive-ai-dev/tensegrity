## Troubleshooting & FAQ

**Colab disconnected.** Reconnect, run §0 → §4 again. Datasets are re-downloaded to local disk (cached copies were on the lost VM); the run itself resumes from Drive with the same schedule, optimizer state, EMA and data order.

**"CUDA out of memory".** §3 autotunes the micro-batch and falls back to gradient checkpointing, so this should only happen if another process holds GPU memory — Runtime → Restart session, then run again. Smaller model or shorter `block_size` always fits.

**Which size should I pick?** 5M is the sweet spot on a free T4 (a few hours → clearly coherent stories). 1M/2M are for fast experiments. 10M if you have ~4 h. 50M/115M need an L4/A100 or many hours; use the ladder (§6) to get there incrementally.

**Why bytes?** No tokenizer to train or ship, any language or file type works, and there are no out-of-vocabulary failures — at the cost of ~4× longer sequences than subword models for the same text.

**Is the loss comparable to other models'?** Bits per byte is the tokenizer-independent number; compare *that*, on the same held-out text.

**My own data.** `DATASET = "custom"` with `CUSTOM_DATA_PATH` (file or folder; `.txt .md .jsonl .parquet`). For documents joined by a marker in one big file, set `CUSTOM_DOC_SEPARATOR`; for jsonl/parquet each record becomes one document automatically.

**Files in Drive.** `MyDrive/tensegrity/runs/<run>/` holds `ckpt_latest.pt` (full, resumable), `ckpt_best.pt` (weights only), `train_log.jsonl`, `run_plan.json`. Exports go to `exports/`.

**Reporting a bug / contributing.** https://github.com/recursive-ai-dev/tensegrity
