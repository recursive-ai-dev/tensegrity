# Tensegrity

**Byte-level language models from 1M to 115M parameters, a Colab notebook that runs them end-to-end, and a growth ladder that lets a small trained model become a bigger one without losing what it learned.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/recursive-ai-dev/tensegrity/blob/main/notebooks/Tensegrity_Colab.ipynb)

Tensegrity reads and writes raw UTF-8 bytes (vocabulary 259), so there is no tokenizer to train, ship or break. Everything — the model, optimizers, data pipeline, resumable trainer, function-preserving growth, EWC, export — is a small, dependency-light PyTorch package with a test suite that runs on CPU.

```
Open the notebook in Colab → pick a size → Run all.
```

## The size ladder

| size | parameters | layers × width | heads (q/kv) | FFN | context | default byte budget |
|---|---:|---|---|---:|---:|---:|
| **1M** | 1.02M | 5 × 128 | 4/2 | 384 | 1024 | 300M |
| **2M** | 1.99M | 5 × 192 | 6/3 | 480 | 1024 | 400M |
| **5M** | 5.06M | 7 × 256 | 8/4 | 672 | 1024 | 600M |
| **10M** | 10.04M | 9 × 320 | 10/5 | 832 | 1024 | 800M |
| **50M** | 50.29M | 15 × 512 | 16/8 | 1664 | 1024 | 1,200M |
| **115M** | 115.69M | 18 × 768 | 24/12 | 2016 | 1024 | 1,750M |

(`python -m tensegrity presets` prints this table from the code; a test asserts every row is within 2% of its name.)

All rungs share `head_dim = 32`, a 2:1 GQA ratio and the same attention pattern, and `d_model`, depth and FFN width never decrease from one rung to the next. That makes the ladder **walkable**: a trained rung can be grown into the next one *exactly* (see below).

## Quick start

**Colab (recommended).** Open `notebooks/Tensegrity_Colab.ipynb`, choose `MODEL_SIZE` and `TRAIN_HOURS`, *Runtime → Run all*. The notebook is self-contained (it writes the package to disk itself), detects your GPU, autotunes the batch, sizes the run to your time budget, checkpoints to Drive, and resumes exactly after a disconnect. `SMOKE_TEST` runs the whole thing in about a minute to check your setup.

**Command line / Python.**

```bash
pip install -e .                                   # torch>=2.3, numpy, safetensors
python -m tensegrity presets
python -m tensegrity prepare --data_path corpus.txt --out_dir data/mine --doc_sep '<|endoftext|>'
python -m tensegrity train --preset 5m --data_dir data/mine --out_dir runs/5m --hours 2
python -m tensegrity generate --ckpt runs/5m/ckpt_best.pt --prompt "Once upon a time"
python -m tensegrity grow --ckpt runs/5m/ckpt_best.pt --to 10m --out runs/10m_init.pt
python -m tensegrity train --preset 10m --data_dir data/mine --out_dir runs/10m --init_weights runs/10m_init.pt
python -m tensegrity export --ckpt runs/10m/ckpt_best.pt --out_dir export --dtype float16
```

```python
import tensegrity as tg
cfg = tg.get_preset_config("5m")
model = tg.TensegrityLM(cfg)
ids = tg.ByteTokenizer().encode("Hello", add_bos=True)
```

Multi-GPU: `torchrun --nproc_per_node=2 -m tensegrity train ...` (DDP is detected from the environment).

## Growing instead of restarting

```python
small, scfg, _ = tg.load_model("runs/2m/ckpt_best.pt")
big, bcfg = tg.grow_model(small, scfg, target=tg.retarget(scfg, tg.get_preset_config("5m")))
tg.check_function_preserved(small, big, scfg, "cpu")   # max |Δlogit| ≈ 1e-6
tg.check_growth_trainable(big, scfg)                    # every new head/unit/channel/layer receives gradient
```

`climb_ladder(["1m","2m","5m"], data_dir, root, hours=[0.2, 0.5, 2])` does train → grow → train → … in one resumable call.

Growth works along four axes (depth, FFN width, residual width/heads, context length) and is verified for `embed_scale` on/off, biases, ByteMix, x0-mix, soft-capping and all axes combined. Optimizer state is not carried over (a fresh optimizer is built for the grown model).

## Architecture

```
bytes ─► embedding (tied with the output head) ─► [ByteMix: causal FIR + gated bilinear n-gram mixer over K-1 bytes]
      ─► (n_loops × N) × Block ─► RMSNorm ─► logits (soft-capped)    └► confidence head (stop-gradient probe)

Block:  x ← x + g_x0   · x0                          Attn = GQA, RoPE, Q/K-RMSNorm, v0 value residual;
        x ← x + g_attn · Attn(RMSNorm(x), v0)               layer i is GLOBAL causal iff (i+1) % 3 == 0, else a
        x ← x + g_ffn  · SwiGLU(RMSNorm(x))                 256-byte sliding window ("banded", one fused SDPA call)
```

* **Low-parameter expressivity (`ByteMix` bilinear n-grams, `x0_mix`, `v0_mix`, `n_loops`)**:
  * **Gated bilinear `ByteMix`** (`byte_mix_bilinear=True`) augments the causal FIR filter with elementwise multiplicative byte-pair interactions `(w_j + bi_j · tanh(e_t / scale)) ⊙ e_{t-j}`, giving Layer 0 direct n-gram selectivity at `<0.04%` parameter cost while staying strictly channel-local (exact under width growth).
  * **Value residual (`v0_mix=True`)** and **embedding re-injection (`x0_mix=True`)** route Layer 0's value projections `v0` and byte embeddings `x0` into every block via learnable gains (`n_kv_head` and `1` parameter per block).
  * **Recursive block looping (`n_loops >= 1`)** executes the block stack `n_loops` times with per-pass residual gains (`loop_attn_gains`, `loop_ffn_gains`, `loop_x0_gains`), multiplying effective compositional depth with almost zero extra parameters and supporting function-preserving loop growth (`grow_model(..., new_n_loops=...)`).
* **Learnable residual gains** (`g_attn`, `g_ffn`, init 1/√(2·L·n_loops)) plus an activation-RMS **"homeostasis" loss** that nudges each sublayer's output RMS towards a set-point.
* **Banded local attention** is computed as one fused call — queries are cut into window-sized chunks that attend to a `[previous | current]` key strip under a band mask, with chunks folded into the batch dimension. It equals dense masked attention exactly (tested against a dense reference, forward and backward) and replaces the original Python loop of per-chunk kernels.
* **GQA** uses SDPA's native `enable_gqa` only where it reaches the FlashAttention kernel (Ampere+, fp16/bf16). On a T4 the memory-efficient kernel has no GQA support, so K/V are expanded instead — otherwise SDPA silently falls back to the O(T²)-memory math kernel.
* **LayerDrop** draws all layer-skip decisions from a private CPU generator once per forward — no GPU sync, no per-block collective, identical across DDP ranks, and checkpointable.
* **Generation** has an exact KV cache (token-for-token identical to uncached decoding, including past the context window), streaming, top-k / top-p / min-p, and a **confidence head** whose scores can lower or raise the sampling temperature per token and shade the text in the notebook.

### Honest scope

Every mechanism above is established in the literature (RoPE, GQA, SwiGLU, RMSNorm, LayerDrop, LayerScale-style gains, Gemma-style soft-capping, Muon, WSD). The *combination*, and in particular the homeostasis loss and ByteMix, are this project's own recombination. They are **not** benchmarked against other architectures here; the small ablations below tell you what each knob did at ~1M parameters, nothing more.

## What was measured

Everything below was measured on CPU in this repository's build environment, at **~1M parameters** (the 1M rung's shape, 256-byte context), on the 19 MB TinyStories validation split, **500 optimizer steps × 16 × 256 bytes ≈ 2M bytes of training**, 2 seeds, scored as validation bits-per-byte on 384 fixed windows. Raw logs: [`tools/ablation_round1.log`](tools/ablation_round1.log), [`tools/ablation_round2.log`](tools/ablation_round2.log); re-run with `tools/ablate.py`. These numbers measure *early learning speed at tiny scale*. They are evidence for "does this help or hurt here", not scaling laws; the seed-to-seed standard deviation is ≈0.01–0.02 bpb for AdamW runs and ≈0.001–0.009 for Muon runs, so only larger gaps mean anything.

**Optimizer** (round 1, AdamW unless noted)

| run | val bpb | vs AdamW @ 4e-3 |
|---|---:|---:|
| AdamW lr 2e-3 / **4e-3** / 8e-3 / 1.6e-2 | 1.801 / **1.691** / 1.639 / 1.670 | — |
| **Muon** (hidden matrices) + AdamW (embeddings, norms, scalars) | **1.493 – 1.501** | **−0.19 to −0.20** |

Muon beat the *best* AdamW learning rate by ≈0.15 bpb. Round 2 (all Muon, base = 1.4916): weight decay on the Muon matrices 0.1 → 0 gave **−0.015** (σ 0.0007); Muon LR 0.04 was neutral (+0.003) and 0.01 clearly worse (+0.052).

**Architecture knobs** (Δ vs the AdamW base 1.691 unless noted)

| change | AdamW Δ | Muon Δ | decision |
|---|---:|---:|---|
| ByteMix (causal FIR over the previous K−1 byte embeddings), K=4 | **−0.063** (σ 0.0001) | −0.010 | **on** (K=8 ≈ same) |
| token-embedding ×√d scale **off** | **−0.047** | **−0.026** (on top of ByteMix) | **off** |
| remove Q/K-norm | +0.057 | – | keep |
| remove the homeostasis (activation-RMS) loss | +0.061 | −0.0006 | keep (clearly helps AdamW, neutral under Muon) |
| homeostasis weight ×10 | −0.029 | – | kept at 0.01 (single setting, not swept) |
| x0-mix (learnable re-injection of the embedding stream) | −0.005 | – | off (within noise) |
| logit soft-cap off | +0.0005 | – | kept (stability insurance for fp16) |
| `global_every` 2 / 4 / every layer (vs 3) | −0.006 / −0.010 / +0.004 | – | 3 (within noise) |

The ladder presets therefore use Muon (peak LR 0.02, no decay on the matrices), ByteMix K=4, embedding scale off, Q/K-norm, soft-cap 30, z-loss 1e-4 and the homeostasis loss at 0.01.

**Does growing beat training the big model from scratch?** (`tools/ladder_vs_scratch.py`, 1M → 2M, equal FLOPs: 400 steps of the 2M model vs 383 steps of 1M + 200 steps of 2M, Muon, 2 seeds)

| arm | val bpb |
|---|---:|
| 2M from scratch | **1.456 ± 0.009** |
| 1M → grow → 2M (each stage fully cooled down) | 1.480 ± 0.004 |
| 1M → grow → 2M (small stage barely cooled down) | 1.468 ± 0.006 |

Loss is perfectly continuous across the growth step (e.g. parent 1.512 bpb → grown model 1.512 bpb, then training continues smoothly), but **in this setting growth did not beat training the larger model from scratch**; skipping the intermediate cool-down narrows the gap to about one standard deviation. Treat the ladder as a tool for reuse, sanity-checking cheaply on a small rung, and extending runs incrementally — not as a proven speed-up. Whether that changes at 10M–115M or with multi-hour budgets is untested here (`climb_ladder(..., intermediate_cooldown=0.03)` exposes the cheaper schedule).

**Verified properties** (tests): banded local attention equals dense masked attention (forward and backward); cached decoding equals uncached decoding token-for-token, including past the context window; growth preserves logits (measured ~1e-6, asserted ≤ 2e-5) on every axis and flag combination; every new head/unit/channel/layer receives gradient; resume is bit-exact; the full notebook executes headlessly.

## What changed from v1 (and how each fix was verified)

The original release was a single notebook that wrote two modules to disk. I kept its model, checkpoint format and ideas, and changed the following. Each item was reproduced against the original code first.

| # | Problem in v1 | Effect | Fix / verification |
|---|---|---|---|
| 1 | **Width growth was untrainable.** `transplant_block` zero-filled *both* factors of every new product (new heads q=k=v=o=0; new hidden units gate=up=down=0; new channels reader=writer=0). | `grow_model(new_n_embd=…)` passed its logit check but every new weight had exactly 0.0 gradient — the extra width could never learn. | Reader-random / writer-zero rule (`growth.py`). `check_growth_trainable` probes two fwd/bwd passes and is exercised on 5 axes × 4 flag sets, and detects a re-created v1-style growth (`test_check_growth_trainable_detects_the_v1_dead_capacity_bug`). |
| 2 | **Resume double-decayed the LR.** Base LRs were read *after* `optimizer.load_state_dict`. | After a resume the logged LR was 4.6e-4 where the schedule says 2.85e-3 — below the schedule's own floor. | Base LRs captured from the fresh optimizer; `test_lr_after_resume_follows_the_schedule` (mutation-checked: re-introducing the bug fails it). |
| 3 | **bf16 chosen on a T4.** `torch.cuda.is_bf16_supported()` is `True` on Turing via emulation. | The free-tier GPU trained (and ran Muon's Newton–Schulz) in slow emulated bf16. | Compute-capability-based `pick_dtype`; Newton–Schulz uses fp16 on pre-Ampere. Tested by faking device capabilities. |
| 4 | **GQA feature detection always false.** `inspect.signature(F.scaled_dot_product_attention)` raises on real builds. | `enable_gqa` was never used; K/V were always expanded. | Probe by calling it, and gate native GQA by hardware (see Architecture). |
| 5 | **LayerDrop synced the GPU** (`.item()` + `dist.broadcast`) in *every block*, every step. | Throughput loss; breaks `torch.compile` graphs. | One CPU draw per forward; DDP lock-step verified with a 2-process gloo test. |
| 6 | `compute_fisher_diagonal(model="<path string>")` in the notebook; Fisher from an 8-sequence *batch mean* gradient. | Crash; Fisher under-estimated by ~batch size. | Per-sequence empirical Fisher in fp32; the notebook rewritten. |
| 7 | RMSNorm `eps` handling in growth ignored `embed_scale`. | Small residual mismatch hidden by a loose `atol=1e-3`. | Derived the exact rule (ε unchanged when `embed_scale`, scaled by D0/D1 otherwise), stored in `norm_eps`; growth now preserves logits to ~1e-6. |
| 8 | Notebook: Kaggle paths, hard-coded Drive checkpoint names, a `torch.cuda.memory._dump_snapshot` without recording, duplicated/outdated `grow_model` cell, monkey-patched RNG restore, TinyStories `<|endoftext|>` spelled as 13 bytes. | Not runnable top-to-bottom; the EOS token was never trained. | Replaced by a generated, tested notebook; `doc_sep` maps separators to the real EOS token. |
| 9 | Checkpoints required full unpickling (`weights_only=False`). | Arbitrary-code risk when loading a shared checkpoint. | v2 checkpoints store RNG state as tensors/ints and load with `weights_only=True`; v1 files still load (with a warning). |
| 10 | Validation drew fresh random windows each time; sampling used the global NumPy RNG. | Noisy "best checkpoint" selection; eval frequency changed training batches. | Fixed validation windows; private, checkpointable sampler. Resume is **bit-exact** (tested with LayerDrop, dropout, EMA, ByteMix). |

**Compatibility.** Parameter names and shapes are unchanged for every v1 feature and every new config field defaults to the v1 behaviour, so v1 checkpoints load as the same network. A real checkpoint written by the original code (Muon optimizer, numpy RNG objects) is a test asset (`tests/assets/legacy_v1.pt`): logits, confidence and sampled text match to float tolerance. The original named presets keep their exact parameter counts (15,867,401 / 133,200,409 / 403,107,361).

## Repository layout

```
tensegrity/            the package (notebook embeds these files verbatim)
  config.py layers.py model.py optim.py         architecture, Muon/AdamW
  data.py checkpoint.py schedule.py hw.py       streaming prepare, resumable state, WSD, dtype/autotune
  train.py growth.py ladder.py presets.py       Trainer+EMA+preflight, growth+EWC, ladder, size table
  analysis.py export.py viz.py colab.py cli.py  calibration, safetensors, dashboard, Colab helpers, CLI
notebooks/Tensegrity_Colab.ipynb                generated from tools/nb/* + the package (do not edit by hand)
tools/build_notebook.py                         regenerate / --check the notebook
tools/ablate.py tools/ladder_vs_scratch.py      the experiments quoted above (+ their raw logs)
tests/                                          214 tests: attention vs dense reference, KV-cache exactness,
                                                bit-exact resume, growth, v1 compatibility, DDP, notebook execution
```

```bash
pip install -e ".[dev]"
python -m pytest -q                       # includes executing the notebook headlessly on CPU, offline
python tools/build_notebook.py --check    # CI: the committed notebook must equal the generator's output
```

## Limitations

* Small research models on narrow corpora: fluent-looking text, not a general assistant; no safety tuning.
* Byte-level modelling costs ~4× the sequence length of a subword model for the same text.
* The GPU-only code paths (fp16 loss scaling on a T4, native GQA on Ampere+, fused AdamW, `torch.compile`) were written against the documented behaviour but could not be exercised in the CPU-only environment this release was built in. `preflight` runs real optimizer steps on your device before every new run and the notebook falls back to AdamW if Muon misbehaves; please report anything that surprises you.
* Hyper-parameter defaults per size are scaling heuristics plus the small ablations above, not per-size sweeps.
* No licence file is included; add one before redistributing.
