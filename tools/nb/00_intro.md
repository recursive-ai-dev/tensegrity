# 🧬 Tensegrity LM — a byte-level language model, 1M → 115M parameters, on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/recursive-ai-dev/tensegrity/blob/main/notebooks/Tensegrity_Colab.ipynb)

**One notebook, six model sizes, no tokenizer.** Tensegrity reads and writes raw UTF-8 bytes, so it handles any language, code or markup out of the box. Pick a size, press *Run all*, and walk away: the notebook auto-detects your GPU, sizes the run to the hours you have, checkpoints to Google Drive, and picks up where it left off if Colab disconnects.

### Quick start
1. **Runtime → Change runtime type → GPU** (a free **T4** is enough for 1M–10M; 50M and 115M want an L4/A100 or patience).
2. Set **§1 · Configure** (model size and the hours you can spare). Everything else has a sensible default.
3. **Runtime → Run all.** Watch the live loss curve and the sample text improve.

### The size ladder

{{PRESET_TABLE}}

Every rung has `head_dim = 32`, GQA 2:1 and the same attention pattern, so a trained rung can be **grown into the next one with exact function preservation** (§6): the bigger model starts with *identical* outputs and a verified-trainable set of new parameters, and the notebook audits every carried weight and checks both properties (plus save → reload) every time it grows a model. Growth is a tool for *reuse and incremental extension* — keep what a small model learned, extend a run when you get more time. It is not a speed-up: in the small equal-compute test in the repository README, training the bigger model from scratch did as well or better.

### What's inside
* **Architecture** — RoPE · grouped-query attention · alternating banded-local / global layers (one fused kernel, no Python chunk loop) · SwiGLU · Q/K-norm · learnable residual gains · logit soft-cap + z-loss · optional ByteMix (causal FIR over recent byte embeddings) · confidence head that drives adaptive sampling temperature.
* **Training** — AdamW or Muon · warmup-stable-decay schedule sized to your time budget · weight EMA · bf16 on Ampere+, fp16 + loss-scaling on T4 (no accidental emulated bf16) · exact resume (weights, optimizer, EMA, every RNG, data order) · wall-clock stop · non-finite-step protection.
* **Trustworthy data** — strict readers (no silent fallbacks), a document-level train/validation split, a hashed **manifest + fingerprint** for every prepared dataset, and a printed **data audit** with decoded examples before you train.
* **Reproducibility** — a **preflight report** before the run (real size, attention reach, data identity, finite numerics on *your* device, memory headroom, save → reload) and a **run report** after it (code/data fingerprints, updates and bytes actually consumed, held-out results, lineage). Resume is exact and refuses to continue a run under different settings or data.
* **Measured claims** — exact bits-per-byte accounting, named evaluation domains, a copy/retrieval **context probe**, growth as a **verified transaction**, and a matched-budget comparison of *new-only vs replay vs replay + EWC* that reports forgetting *and* gain.
* **Extras** — live dashboard · confidence-shaded text · streaming-cache generation (O(1) per byte past the context window) · safetensors / INT8 export with a model card · optional Hugging Face upload.

> **Honest scope.** These are small research models. The individual mechanisms are established in the literature; the *combination* is not benchmarked against other architectures here — the ablations behind the defaults are in the repository README. Expect fluent-looking, narrow-domain text, not a general assistant.
