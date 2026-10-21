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

and the instruments around them:

```bash
python -m tensegrity audit     --data_dir data/mine --deep          # manifest, integrity (re-hash), decoded examples
python -m tensegrity preflight --preset 5m --data_dir data/mine     # size, attention reach, data, numerics, memory, reload; exit 1 on blockers
python -m tensegrity report    --run_dir runs/5m --data_dir data/mine   # run_report.md / .json: what the run actually was
python -m tensegrity probe     --ckpt runs/5m/ckpt_best.pt          # attention reach + copy/retrieval at increasing distances
python -m tensegrity bench     --ckpt runs/5m/ckpt_best.pt          # generation latency per cache mode, SDPA backends
python -m tensegrity continual --ckpt runs/5m/ckpt_best.pt --old_dir data/mine --new_dir data/other --seeds 0,1,2
python -m tensegrity generate  --ckpt runs/5m/ckpt_best.pt --cache_mode streaming --max_new_tokens 5000
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

### Growth as a transaction

```python
suite = tg.EvalSuite([tg.EvalDomain("stories", dir_a, "retain"), tg.EvalDomain("plays", dir_b, "adapt")])
d = tg.grow_transaction(small, scfg, target=tg.retarget(scfg, tg.get_preset_config("5m")), suite=suite,
                        recovery=tg.RecoverySpec([dir_b, dir_a], [0.7, 0.3], steps=200, distill_weight=1.0,
                                                 ewc=tg.EWCState.estimate(small, dir_a, lam=100.0)),
                        gates=tg.GrowthGates(max_retain_regression_bpb=0.02, min_adapt_gain_bpb=0.05))
print(d.summary())       # ACCEPTED / REJECTED, with every number it was based on
d.raise_if_rejected()    # a rejected candidate is never used; the parent is verified unchanged either way
```

Propose → construct (parent untouched) → **audit** that every parent state entry is present in the candidate, in the right corner, at the right scale (`verify_growth_mapping`) → function preserved → survives save → reload through config + state dict, a checkpoint *and* safetensors → new capacity receives gradient → optional recovery training (new data + replay + distillation from the parent + EWC carried through growth) → retention and adaptation gates on named domains → accept or revert.

### Trustworthy data, exact resume

* `prepare_data` is strict: unknown extensions, non-UTF-8 text, or a jsonl/parquet record without a usable text field are errors or counted rejections (never silently glued together), and more than `max_reject_fraction` rejected fails the run. `record_template="Q: {question}\nContext: {context}\nA: {answer}"` keeps a structured record one document; `group_field="context_id"` keeps related records on one side of the split.
* Documents are split **before** tokens are packed, by a content hash (so duplicates cannot straddle the split and a document is never cut in two); a corpus with too few documents falls back to a contiguous tail split — recorded and printed, never silent. The result is staged and published atomically; a different dataset is never written over an existing one without `overwrite=True` (the old files are moved aside).
* `meta.json` is a manifest: SHA-256 of both token files, split policy, tokenizer, counts, rejected-record accounting and a **fingerprint**. A run records the fingerprints it trained on; `Trainer.fit()` verifies datasets against their manifests, refuses to run if a token file changes underneath it, and refuses to resume under a different training policy (`TrainConfig.TRAJECTORY` fields, the LR plan), different data/mixture weights, a different EWC state or teacher, or a different evaluation set — unless you pass `on_policy_change="adopt"` / `on_data_change="adopt"`, which records the change as a phase in the checkpoint's `lineage`. Resume is bit-exact in the same environment (weights, optimizer moments, EMA, LR trace, sampler state).
* Start generation from `ByteTokenizer.DOC_START` (EOS): the data is `bytes + EOS` per document and contains no BOS, so BOS is an input the model has never seen (see "What was measured").

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
* **Generation** has an exact KV cache (token-for-token identical to uncached decoding, including past the context window), streaming output, top-k / top-p / min-p, and a **confidence head** whose scores can lower or raise the sampling temperature per token and shade the text in the notebook. Past the context window `cache_mode="window"` (default) stays exact by re-encoding the window every byte; `cache_mode="streaming"` keeps a bounded KV cache at absolute RoPE positions and costs O(1) per byte, but is only an approximation there (stale deeper-layer states, ByteMix left context) — `long_stream_nll` measures the gap on your text.
* **Reach is reported, not assumed.** `TensegrityConfig.receptive_field()` is the longest dependency the layout can express (a stack of local layers only sees `1 + n_layer·(window−1)` bytes; one global layer sees the whole context). `describe()`, `preflight` and the notebook print it and flag local-only layouts.

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

**Verified properties** (tests): banded local attention equals dense masked attention (forward and backward); cached decoding equals uncached decoding token-for-token, including past the context window; growth preserves logits (measured ~1e-6, asserted ≤ 2e-5) on every axis and flag combination, every carried weight is found in the grown model, and the grown model reloads identically through config + state dict, checkpoint and safetensors; every new head/unit/channel/layer receives gradient; an interrupted run equals an uninterrupted one in weights, optimizer moments, EMA, LR trace and sampler state, and a resume under a different policy/data/EWC state/eval set is refused; document splits keep documents whole and duplicates on one side; the full notebook executes headlessly.

**New measurements in 2.1** (CPU, this repository's build environment; small scale, so evidence for mechanisms rather than benchmarks):

* *Start-of-document context.* After training a 3-layer model for 500 steps on a synthetic story corpus, the first 12 bytes of unseen documents cost **3.13 bits/byte** after a BOS prefix and **0.28** after an EOS prefix: BOS is never an input in the data, so it is out of distribution. Generation, sampling and the model card therefore start from EOS. (One toy model; the size of the effect on real models is unmeasured, the mechanism is not in doubt.)
* *Ingestion cost.* Document-level split + content hashing + manifest + source hashing: 100 MB of text in ~4.0 s (25 MB/s) vs ~2.1 s (48 MB/s) for the old tail split. About 1.9× slower, in exchange for the guarantees above.
* *Streaming decode.* On a tiny CPU model (context 128) decoding past the window runs 1.8× faster with `cache_mode="streaming"` than with the exact window mode; the gap grows with context length. For one-layer models whose attention does not reach the window start it is *exactly* equal to window mode at every position (tested to 2e-4 bits); for deeper models it is an approximation.
* *Context probe on real weights.* A 2-layer all-global model trained for 6000 steps on the copy task went from chance (4%) to 40–58% accuracy at distances 0–24 with 1.5–2.7 bits of gain; the probe's bookkeeping is also tested against a deterministic induction oracle with a known reach. The local-only counter-experiment was **not** run to convergence, so no claim is made about it.
* *RoPE precision.* A float32 `position × frequency` product has a measured maximum phase error of ~2e-4…3e-4 rad at 10⁴ positions, 1e-3…5e-3 at 10⁵ and 2e-2…5e-2 at 10⁶ (head_dim 32–128); tables are now computed in float64.
* *EWC through width growth.* Without the rescale map, a freshly grown model (function identical to its parent) carries an EWC penalty of 6.5e-4 that pulls the rescaled norm weights back; with it, exactly 0, and equal perturbations of the function cost equal penalty.
* *Sampled vs empirical Fisher.* With labels sampled from the model's own distribution the mean gradient over 96 sequences is 0.10 of its RMS (≈ 1/√96, i.e. zero-mean score); with the data's labels it is 0.54.

## What changed in 2.1 — the review ledger

An external review of the original notebook (v0.7.6) listed 14 probes and a plan. Every item was re-checked against this package before anything was changed; most of the model-side findings had already been fixed by the v2 rebuild below. Status of each:

| finding | status |
|---|---|
| preset sizes misleading; no global layer ⇒ short reach | fixed in v2 (sizes within 2% of their names, global every 3rd layer); **now** also reported: `receptive_field()`, `describe()`, preflight warning |
| LR decays twice on resume | fixed in v2; **now** also persisted (`lr_state`) and policy-checked |
| checkpoint step off by one / best-val ordering | fixed in v2; **now** explicit counters `completed_updates`, `applied_updates`, `tokens_seen` |
| depth growth drops Q/K-norm scales | fixed in v2; **now** independently audited for *every* state entry (`verify_growth_mapping`) |
| width growth + `embed_scale` mismatch; dead new capacity; norm-eps lost on reload | fixed in v2; **now** also gated by a three-route save → reload check in `grow_transaction` |
| GQA detection always false | fixed in v2 |
| compiled-wrapper `_orig_mod.` keys in checkpoints | **fixed now** (`unwrap_model`, `normalize_state_dict`) |
| sampler misses last window | already correct in v2; regression test added |
| dataset replaced on disk, stale memmap | **fixed now** (inode/size/mtime tracking; Trainer treats it as an error) |
| wrong download URL; parquet read as text | fixed in v2; **now** download hashes recorded, HF revision pin |
| silent fallbacks in readers; columns glued together | **fixed now** (strict readers, counted rejects, `record_template`) |
| tail split cuts documents / leaks duplicates | **fixed now** (document-level hash split, `group_field`, `dedup`) |
| no dataset identity | **fixed now** (manifest + fingerprint, atomic publish, immutability, `audit`) |
| unknown config field silently accepted | **fixed now** for every loader (`from_dict(strict=True)`) |
| resume mixes exact restoration with a new policy | **fixed now** (validation + explicit `adopt`; lineage). Torch's restore of checkpoint weight-decay over the session's config was reproduced and fixed |
| EWC reference not carried correctly through growth | **fixed now** (θ′ = aθ, F′ = F/a²) |
| "empirical Fisher" not labelled | **now** labelled with its unit; `kind="sampled"` added |
| continual learning not evaluated | **now** `compare_continual_strategies` (matched budget, forgetting *and* gain) — the *tool*; no claim that EWC helps is made here |
| validation on `data_dirs[0]` only | **fixed now** (`EvalSuite`, per-domain logs, fingerprinted) |
| bits-per-byte conflates specials | **fixed now** (`bpb_bytes`; legacy `bpb` kept) |
| context behaviour unmeasured | **now** `context_probe` |
| homeostasis loss never instrumented | **instrumentation only** (`residual_report`: ρ = RMS(g·f(x))/RMS(x)); the ablation the review proposes was **not run** |
| teacher-based recovery after growth | **now** `DistillLoss` + `RecoverySpec` |
| cache lost past the window | **now** `cache_mode="streaming"` (approximation, measured) |
| profile before optimising | **now** `profile_training_step` (host-sync count), `sdpa_backend_report`, `benchmark_generation` |
| compile discarded between cells | not applicable: the compiled wrapper was already separate from the serialised module |
| notebook as safe workflow | **now** audit → preflight → train → evaluate → optional branches → run report cells |

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
  manifest.py ingest.py data.py                 dataset identity + manifests, strict ingestion + document split, batching + downloads
  checkpoint.py schedule.py hw.py               resumable state, WSD + legal plan changes, dtype/autotune
  evaluation.py                                 exact bits/byte, named-domain EvalSuite, context probe, residual diagnostics
  train.py growth.py continual.py               Trainer+EMA+preflight (exact-resume rules), growth + audit, EWCState + distillation
  workflow.py ladder.py presets.py              grow_transaction + continual comparison, ladder, size table
  report.py perf.py                             preflight / run reports, generation + SDPA + host-sync profiling
  analysis.py export.py viz.py colab.py cli.py  calibration, safetensors, dashboard, Colab helpers, CLI
notebooks/Tensegrity_Colab.ipynb                generated from tools/nb/* + the package (do not edit by hand)
tools/build_notebook.py                         regenerate / --check the notebook
tools/ablate.py tools/ladder_vs_scratch.py      the experiments quoted above (+ their raw logs)
tests/                                          305 tests: attention vs dense reference, KV-cache + streaming exactness,
                                                bit-exact resume + resume refusals, growth audit + transaction, data integrity,
                                                evaluation, v1/2.0 compatibility, DDP, notebook execution
```

```bash
pip install -e ".[dev]"
python -m pytest -q                       # includes executing the notebook headlessly on CPU, offline
python tools/build_notebook.py --check    # CI: the committed notebook must equal the generator's output
```

## Limitations

* Small research models on narrow corpora: fluent-looking text, not a general assistant; no safety tuning.
* The ablation and ladder numbers above were measured before 2.1 and report the older `bpb` (loss/ln 2 per target token, document markers included); the exact `bpb_bytes` differs from it by roughly the share of document-boundary tokens among the targets (about 0.1% for TinyStories-style documents, by arithmetic — those runs were not re-scored).
* Continual-learning tooling (`compare_continual_strategies`, `EWCState`, `DistillLoss`, `grow_transaction`) is verified for correctness at toy scale. Whether replay, EWC or distillation helps at 10M–115M on real corpora is unmeasured here; the tools exist so that it can be measured under matched budgets. Retention gates (`GrowthGates`) are thresholds *you* choose for the domains *you* evaluate.
* `cache_mode="streaming"` is an approximation past the context window; the homeostasis loss has been instrumented but not ablated.
* Bit-exact resume is tested for a fixed environment; PyTorch does not promise identical results across releases or hardware, and a resume under a different torch version warns.
* Byte-level modelling costs ~4× the sequence length of a subword model for the same text.
* The GPU-only code paths (fp16 loss scaling on a T4, native GQA on Ampere+, fused AdamW, `torch.compile`) were written against the documented behaviour but could not be exercised in the CPU-only environment this release was built in. `preflight` runs real optimizer steps on your device before every new run and the notebook falls back to AdamW if Muon misbehaves; please report anything that surprises you.
* Hyper-parameter defaults per size are scaling heuristics plus the small ablations above, not per-size sweeps.
* No licence file is included; add one before redistributing.
