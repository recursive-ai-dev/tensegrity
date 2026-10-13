#!/usr/bin/env python3
"""Does growing beat training the bigger model from scratch at EQUAL COMPUTE?

Arm S (scratch): the target rung, trained for S steps.
Arm L (ladder) : the previous rung for N1 steps, grown (function-preserving) into the target rung,
                 then trained for N2 steps, with  N1*flops(small) + N2*flops(big) == S*flops(big).

Both arms use the same optimiser, schedule family (warmup-stable-decay, each stage cools down at
its own end), batch size and data stream, and are scored by validation bits-per-byte on the same
fixed windows. Several seeds; reports mean +- std. Context is shortened (default 256) so it runs on a CPU.

    python tools/ladder_vs_scratch.py --data data/tinystories --small 1m --big 2m --steps 400 --seeds 2

This is a *small-scale measurement*, not a scaling-law result; read it as "does the mechanism help
at all, and does the loss stay continuous across growth?".
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def run_arm(job: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    torch.set_num_threads(1)
    from tensegrity import (TensegrityLM, TrainConfig, Trainer, evaluate, get_preset_config, grow_model, load_model,
                            make_plan, retarget, check_function_preserved, check_growth_trainable)
    ovr = dict(block_size=job["ctx"], sliding_window=job["ctx"] // 4)
    B, T = job["batch"], job["ctx"]
    opt = job["optimizer"]
    small_cfg = get_preset_config(job["small"], **ovr)
    big_cfg = retarget(small_cfg, get_preset_config(job["big"], **ovr))

    def tc(lr_scale=1.0):
        from tensegrity import get_preset
        lr = get_preset(job["big"]).learning_rate * lr_scale
        return TrainConfig(batch_size=B, block_size=T, learning_rate=lr, min_lr=lr / 10, optimizer=opt,
                           eval_interval=10 ** 9, eval_batches=24, eval_batch_size=16, log_interval=10 ** 9,
                           sample_every_evals=0, verbose=False, seed=job["seed"], ckpt_interval_minutes=1e9)

    def plan(steps, decay=0.25):
        return make_plan(steps * B * T, B * T, "wsd", warmup_frac=0.05, decay_frac=decay, min_warmup=10)

    t0 = time.time()
    torch.manual_seed(job["seed"])
    out: Dict[str, Any] = {"arm": job["arm"], "seed": job["seed"]}
    with tempfile.TemporaryDirectory() as d:
        if job["arm"] == "scratch":
            m = TensegrityLM(big_cfg)
            r = Trainer(m, tc(), plan(job["s"]), job["data"], os.path.join(d, "big"), device="cpu").fit(resume=False)
            out["val_bpb"] = r["best_val_loss"] / math.log(2)
        else:
            small = TensegrityLM(small_cfg)
            r1 = Trainer(small, tc(), plan(job["n1"], job["stage1_decay"]), job["data"], os.path.join(d, "small"), device="cpu").fit(resume=False)
            parent, pcfg, _ = load_model(os.path.join(d, "small", "ckpt_best.pt"), "cpu")
            big, bcfg = grow_model(parent, pcfg, target=big_cfg)
            check_function_preserved(parent, big, pcfg, "cpu", verbose=False)
            check_growth_trainable(big, pcfg, verbose=False)
            v_parent = evaluate(parent, job["data"], pcfg, "cpu", torch.float32, 24, 16)["bpb"]
            v_grown = evaluate(big, job["data"], bcfg, "cpu", torch.float32, 24, 16)["bpb"]
            r2 = Trainer(big, tc(), plan(job["n2"]), job["data"], os.path.join(d, "big"), device="cpu").fit(resume=False)
            out.update(val_bpb=r2["best_val_loss"] / math.log(2), stage1_bpb=r1["best_val_loss"] / math.log(2),
                       parent_bpb=v_parent, grown_bpb=v_grown)
    out["secs"] = time.time() - t0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--small", default="1m")
    ap.add_argument("--big", default="2m")
    ap.add_argument("--steps", type=int, default=400, help="scratch steps S for the big model")
    ap.add_argument("--small_fraction", type=float, default=0.5, help="share of the compute budget spent on the small rung")
    ap.add_argument("--stage1_decay", type=float, default=0.25,
                    help="cool-down fraction of the small rung's schedule; a small value (e.g. 0.03) spends that "
                         "compute on the big rung instead, since its weights are re-warmed after growth anyway")
    ap.add_argument("--arms", default="scratch,ladder")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--optimizer", default="muon", choices=["adamw", "muon"])
    args = ap.parse_args()

    from tensegrity import get_preset_config
    f_small = get_preset_config(args.small, block_size=args.ctx, sliding_window=args.ctx // 4).flops_per_token(args.ctx)
    f_big = get_preset_config(args.big, block_size=args.ctx, sliding_window=args.ctx // 4).flops_per_token(args.ctx)
    budget = args.steps * f_big
    n1 = max(10, int(args.small_fraction * budget / f_small))
    n2 = max(10, int((1 - args.small_fraction) * budget / f_big))
    print(f"FLOPs/token small={f_small / 1e6:.2f}M big={f_big / 1e6:.2f}M  ->  scratch: {args.steps} steps of {args.big}; "
          f"ladder: {n1} steps of {args.small} + {n2} steps of {args.big} (equal compute)")
    jobs = [dict(arm=a, seed=s, small=args.small, big=args.big, s=args.steps, n1=n1, n2=n2, ctx=args.ctx, batch=args.batch,
                 data=args.data, optimizer=args.optimizer, stage1_decay=args.stage1_decay)
            for s in range(args.seeds) for a in args.arms.split(",")]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = []
        for r in pool.imap_unordered(run_arm, jobs):
            res.append(r)
            extra = (f"  (small {r['stage1_bpb']:.3f} -> parent {r['parent_bpb']:.3f} == grown {r['grown_bpb']:.3f})"
                     if r["arm"] == "ladder" else "")
            print(f"  {r['arm']:8s} seed {r['seed']}: val {r['val_bpb']:.4f} bpb ({r['secs']:.0f}s){extra}", flush=True)
    print()
    for arm in args.arms.split(","):
        v = [r["val_bpb"] for r in res if r["arm"] == arm]
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5 if len(v) > 1 else float("nan")
        print(f"{arm:8s} {m:.4f} +- {sd:.4f}   {[round(x, 4) for x in v]}")


if __name__ == "__main__":
    main()
