## 8 · Export, share & write the run report
Writes `tensegrity.safetensors` (a single self-describing file — the config is embedded in its header), a `config.json`, and an honest model card. Float16 halves the size for deployment; the INT8 option produces a CPU-inference copy. Export works on a copy of the weights; the live model is not moved or changed.

Then it writes **`run_report.md` / `run_report.json`** next to the checkpoints: the exact configuration, code and data fingerprints, the updates and bytes actually consumed, hardware and software, held-out results, lineage and growth history, and a check that the exported file reproduces the best model. Keep it with the weights — it is what makes the run reproducible and the claim checkable.
