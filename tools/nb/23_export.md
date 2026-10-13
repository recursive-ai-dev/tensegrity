## 8 · Export & share
Writes `tensegrity.safetensors` (a single self-describing file — the config is embedded in its header), a `config.json`, and an honest model card. Float16 halves the size for deployment; the INT8 option produces a CPU-inference copy.
