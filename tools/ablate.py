#!/usr/bin/env python3
"""Small, reproducible architecture / optimiser ablations for TensegrityLM.

Trains each named variant from the same data order for a fixed number of steps (several seeds,
run in parallel worker processes) and reports final validation bits-per-byte (mean +- std).
This is how the defaults in ``presets.py`` were chosen -- see README "What was measured".

    python tools/ablate.py --data data/tinystories --steps 400 --seeds 2 --workers 4 \
        --variants base,byte_mix,x0_mix,both

NOTE: short, small-scale runs measure *early learning speed* at ~1M parameters. They are good
evidence for "does this help or hurt at this scale and budget", not a scaling-law proof.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

BASE_MODEL: Dict[str, Any] = dict(
    n_layer=5, n_embd=128, n_head=4, n_kv_head=2, ffn_mult=3.0, ffn_align=32,
    block_size=256, sliding_window=64, global_every=3, layerdrop_max=0.0, grad_checkpoint=False,
    logit_soft_cap=30.0, z_loss_weight=1e-4, confidence_detach=True)

BASE_TRAIN: Dict[str, Any] = dict(optimizer="adamw", learning_rate=4e-3, min_lr=4e-4, weight_decay=0.1)

# name -> (model overrides, train overrides)
VARIANTS: Dict[str, Any] = {
    "base": ({}, {}),
    "byte_mix": ({"byte_mix_kernel": 4}, {}),
    "byte_mix8": ({"byte_mix_kernel": 8}, {}),
    "x0_mix": ({"x0_mix": True}, {}),
    "both": ({"byte_mix_kernel": 4, "x0_mix": True}, {}),
    "no_homeo": ({"homeo_weight": 0.0}, {}),
    "homeo_x10": ({"homeo_weight": 0.1}, {}),
    "no_qknorm": ({"use_qk_norm": False}, {}),
    "no_embscale": ({"embed_scale": False}, {}),
    "no_softcap": ({"logit_soft_cap": None, "z_loss_weight": 0.0}, {}),
    "global2": ({"global_every": 2}, {}),
    "global4": ({"global_every": 4}, {}),
    "all_global": ({"global_every": 1}, {}),
    "muon_lr1e-3": ({}, {"optimizer": "muon", "learning_rate": 1e-3, "min_lr": 1e-4}),
    "muon_lr4e-3": ({}, {"optimizer": "muon", "learning_rate": 4e-3, "min_lr": 4e-4}),
    # --- Muon (hidden matrices) + AdamW (embeddings/norms/scalars); aux AdamW lr = learning_rate
    "muon": ({}, {"optimizer": "muon", "learning_rate": 2e-3, "min_lr": 2e-4}),
    "muon_wd0": ({}, {"optimizer": "muon", "learning_rate": 2e-3, "min_lr": 2e-4, "muon_weight_decay": 0.0}),
    "muon_wd.01": ({}, {"optimizer": "muon", "learning_rate": 2e-3, "min_lr": 2e-4, "muon_weight_decay": 0.01}),
    "muon_mlr.01": ({}, {"optimizer": "muon", "learning_rate": 2e-3, "min_lr": 2e-4, "muon_lr": 0.01}),
    "muon_mlr.04": ({}, {"optimizer": "muon", "learning_rate": 2e-3, "min_lr": 2e-4, "muon_lr": 0.04}),
    "muon_bm": ({"byte_mix_kernel": 4}, {"optimizer": "muon", "learning_rate": 2e-3, "min_lr": 2e-4}),
    "muon_bm_noscale": ({"byte_mix_kernel": 4, "embed_scale": False},
                        {"optimizer": "muon", "learning_rate": 2e-3, "min_lr": 2e-4}),
    "muon_bm_x0": ({"byte_mix_kernel": 4, "x0_mix": True}, {"optimizer": "muon", "learning_rate": 2e-3, "min_lr": 2e-4}),
    "muon_bm_nohomeo": ({"byte_mix_kernel": 4, "homeo_weight": 0.0},
                        {"optimizer": "muon", "learning_rate": 2e-3, "min_lr": 2e-4}),
    "lr2e-3": ({}, {"learning_rate": 2e-3, "min_lr": 2e-4}),
    "lr8e-3": ({}, {"learning_rate": 8e-3, "min_lr": 8e-4}),
    "lr1.6e-2": ({}, {"learning_rate": 1.6e-2, "min_lr": 1.6e-3}),
}


def run_one(job: Dict[str, Any]) -> Dict[str, Any]:
    import tempfile
    import torch
    torch.set_num_threads(job["threads"])
    from tensegrity.config import TensegrityConfig
    from tensegrity.model import TensegrityLM
    from tensegrity.schedule import make_plan
    from tensegrity.train import TrainConfig, Trainer

    m_over, t_over = VARIANTS[job["variant"]]
    model_kw = {**BASE_MODEL, **m_over, **job.get("model_extra", {})}
    train_kw = {**BASE_TRAIN, **t_over}
    torch.manual_seed(job["seed"])
    cfg = TensegrityConfig(**model_kw)
    model = TensegrityLM(cfg)
    T, B = cfg.block_size, job["batch"]
    tc = TrainConfig(batch_size=B, block_size=T, eval_interval=job["steps"], eval_batches=24, eval_batch_size=16,
                     log_interval=10 ** 9, sample_every_evals=0, seed=job["seed"], verbose=False,
                     ckpt_interval_minutes=1e9, **train_kw)
    plan = make_plan(job["steps"] * B * T, B * T, "wsd", warmup_frac=0.05, decay_frac=0.25, min_warmup=10)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as out:
        res = Trainer(model, tc, plan, job["data"], out, device="cpu").fit(resume=False)
    return {"variant": job["variant"], "seed": job["seed"], "val_loss": res["best_val_loss"],
            "val_bpb": res["best_val_loss"] / math.log(2), "params": cfg.n_params(), "secs": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="prepared data dir (train.bin/val.bin)")
    ap.add_argument("--variants", default="base,byte_mix,x0_mix,both")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None, help="write raw results as JSON")
    args = ap.parse_args()
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        sys.exit(f"unknown variants {unknown}; available: {sorted(VARIANTS)}")
    jobs = [dict(variant=n, seed=s, steps=args.steps, batch=args.batch, data=args.data, threads=args.threads)
            for s in range(args.seeds) for n in names]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results: List[Dict[str, Any]] = []
        for r in pool.imap_unordered(run_one, jobs):
            results.append(r)
            print(f"  done {r['variant']:14s} seed {r['seed']}  val_bpb {r['val_bpb']:.4f}  ({r['secs']:.0f}s)", flush=True)
    print(f"\n{'variant':16s}{'val bpb (mean)':>16s}{'std':>9s}{'  vs base':>10s}   seeds")
    base = [r["val_bpb"] for r in results if r["variant"] == names[0]]
    base_m = sum(base) / len(base) if base else float("nan")
    for n in names:
        v = [r["val_bpb"] for r in results if r["variant"] == n]
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5 if len(v) > 1 else float("nan")
        print(f"{n:16s}{m:16.4f}{sd:9.4f}{(m - base_m):+10.4f}   {[round(x, 4) for x in v]}")
    print(f"\n{len(jobs)} runs, {time.time() - t0:.0f}s wall")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
