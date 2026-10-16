"""Command line: ``python -m tensegrity <command>``.

    presets                          list the size ladder
    prepare   --data_path P --out_dir D [--doc_sep '<|endoftext|>']
    train     --preset 5m --data_dir D --out_dir R [--hours 3 | --tokens 5e8] [--optimizer muon]
    generate  --ckpt R/ckpt_best.pt --prompt "Once upon a time"
    eval      --ckpt R/ckpt_best.pt --data_dir D
    grow      --ckpt R/ckpt_best.pt --to 10m --out R2/ckpt_latest.pt
    export    --ckpt R/ckpt_best.pt --out_dir export [--dtype float16]

Multi-GPU: ``torchrun --nproc_per_node=2 -m tensegrity train ...`` (DDP is picked up from the
environment).
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import torch

from . import __version__
from .checkpoint import load_model, save_checkpoint
from .data import prepare_data
from .export import export_safetensors, write_model_card
from .growth import check_function_preserved, check_growth_trainable, grow_model, retarget
from .hw import autotune_micro_batch, measure_throughput, pick_device, pick_dtype
from .model import TensegrityLM
from .presets import (LADDER_NAMES, describe_presets, get_preset, get_preset_config, train_setup)
from .schedule import make_plan, plan_for_time_budget
from .tokenizer import ByteTokenizer
from .train import TrainConfig, Trainer, evaluate, load_run_plan, preflight_or_fallback


def _bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y")


def cmd_presets(args) -> None:
    print(describe_presets())


def cmd_prepare(args) -> None:
    prepare_data(args.data_path, args.out_dir, args.val_fraction, args.seed, doc_sep=args.doc_sep,
                 max_bytes=int(args.max_mb * 1e6) if args.max_mb else None)


def cmd_train(args) -> None:
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    os.makedirs(args.out_dir, exist_ok=True)

    saved = load_run_plan(args.out_dir)
    overrides = {k: getattr(args, k) for k in ("dropout", "block_size", "sliding_window", "layerdrop_max",
                                               "byte_mix_kernel", "byte_mix_bilinear", "x0_mix", "v0_mix",
                                               "n_loops", "homeo_weight")
                 if getattr(args, k, None) is not None}
    if saved is not None and args.resume:
        plan, tc, cfg = saved
        print(f"[cli] resuming run in {args.out_dir} with its saved plan ({plan.max_steps:,} steps)")
        model = TensegrityLM(cfg)
    else:
        if args.init_weights:
            # architecture comes from the checkpoint; only training-time knobs may be overridden
            arch = sorted(set(overrides) - {"dropout", "layerdrop_max", "homeo_weight"})
            if arch:
                raise SystemExit(f"--init_weights fixes the architecture; remove {['--' + a for a in arch]}")
            src, src_cfg, _ = load_model(args.init_weights, "cpu")
            cfg = src_cfg.replace(**overrides)
            model = TensegrityLM(cfg)
            model.load_state_dict(src.state_dict())
            print(f"[cli] initialised {cfg.n_params() / 1e6:.2f}M-parameter model from {args.init_weights}")
        else:
            cfg = get_preset_config(args.preset, **overrides)
            model = TensegrityLM(cfg)
        preset = get_preset(args.preset) if args.preset in LADDER_NAMES else None
        target_tokens = preset.batch_tokens if preset else 131_072
        if args.micro_batch:
            mb, accum = args.micro_batch, args.grad_accum or 1
        else:
            model.to(device)
            tune = autotune_micro_batch(model, cfg.block_size, device, dtype, target_tokens)
            mb, accum = tune["micro_batch"], tune["grad_accum"]
            model.grad_checkpoint = tune["grad_checkpoint"]
        tokens_per_step = mb * accum * cfg.block_size
        total = int(args.tokens) if args.tokens else (preset.default_tokens if preset else 100_000_000)
        if args.hours:
            tps = measure_throughput(model.to(device), mb, cfg.block_size, device, dtype)
            plan = plan_for_time_budget(tps, args.hours, tokens_per_step, max_tokens=total)
            print(f"[cli] measured {tps / 1e3:.1f}k tok/s -> {plan.total_tokens / 1e6:.0f}M tokens in ~{args.hours}h")
        else:
            plan = make_plan(total, tokens_per_step)
        if preset:
            tc, plan = train_setup(preset.name, mb, accum, plan=plan, optimizer=args.optimizer,
                                   dtype=args.dtype, seed=args.seed, max_minutes=args.max_minutes)
        else:
            tc = TrainConfig(batch_size=mb, grad_accum_steps=accum, learning_rate=args.learning_rate or 3e-4,
                             min_lr=(args.learning_rate or 3e-4) / 10, optimizer=args.optimizer,
                             dtype=args.dtype, seed=args.seed, max_minutes=args.max_minutes)
        if args.learning_rate:
            tc.learning_rate, tc.min_lr = args.learning_rate, args.learning_rate / 10
    if not (saved is not None and args.resume):
        tc = preflight_or_fallback(model, tc, args.device)
    trainer = Trainer(model, tc, plan, args.data_dir, args.out_dir, device=args.device)
    res = trainer.fit(resume=args.resume)
    print({k: res[k] for k in ("status", "step", "best_val_loss", "tokens_seen")})


def cmd_generate(args) -> None:
    device = pick_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device, use_ema=args.ema)
    model.eval()
    tok = ByteTokenizer()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    ids = ([ByteTokenizer.BOS] if (args.add_bos or not args.prompt) else []) + tok.encode(args.prompt or "")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                         top_p=args.top_p, min_p=args.min_p, use_confidence_temp=args.use_confidence_temp,
                         stop_on_eos=not args.no_stop_on_eos, use_cache=not args.no_kv_cache)
    print(tok.decode(out[0].tolist()))


def cmd_eval(args) -> None:
    device = pick_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device, use_ema=args.ema)
    r = evaluate(model, args.data_dir, cfg, device, pick_dtype(args.dtype, device), args.eval_batches)
    print(f"val_loss = {r['loss']:.4f}  bits/byte = {r['bpb']:.4f}  perplexity = {r['ppl']:.3f}")


def cmd_grow(args) -> None:
    device = pick_device(args.device)
    old, old_cfg, ckpt = load_model(args.ckpt, device, use_ema=args.ema)
    target = retarget(old_cfg, get_preset_config(args.to))   # size from the preset, function-defining fields from the checkpoint
    new, new_cfg = grow_model(old, old_cfg, target=target, device=device)
    check_function_preserved(old, new, old_cfg, device)
    check_growth_trainable(new, old_cfg)
    path = args.out
    save_checkpoint(path, new, None, new_cfg, 0, float("inf"), extra={"grown_from": os.path.abspath(args.ckpt)})
    print(f"[grow] {old_cfg.n_params() / 1e6:.2f}M -> {new_cfg.n_params() / 1e6:.2f}M saved to {path} "
          f"(weights only; pass --init_weights to `train`)")


def cmd_export(args) -> None:
    model, cfg, ckpt = load_model(args.ckpt, "cpu", use_ema=args.ema)
    w, c = export_safetensors(model, cfg, args.out_dir, args.name, args.dtype)
    write_model_card(args.out_dir, cfg, args.name, metrics={"best_val_loss (nats/byte)": ckpt["best_val_loss"],
                                                             "best_val_bpb": ckpt["best_val_loss"] / math.log(2)})
    print(f"[export] {w}\n[export] {c}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tensegrity", description=f"TensegrityLM {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("presets", help="list the model-size ladder").set_defaults(fn=cmd_presets)

    p = sub.add_parser("prepare", help="tokenise text/markdown/parquet/jsonl into train.bin/val.bin")
    p.add_argument("--data_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--doc_sep", default=None, help="literal separator mapped to EOS, e.g. '<|endoftext|>'")
    p.add_argument("--max_mb", type=float, default=None)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("train", help="train or resume")
    p.add_argument("--preset", default="5m", help=f"ladder {list(LADDER_NAMES)} or legacy name")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hours", type=float, default=None, help="size the run to this wall-clock budget")
    p.add_argument("--tokens", type=float, default=None, help="byte budget (default: preset's)")
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--micro_batch", type=int, default=None, help="skip autotune")
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--max_minutes", type=float, default=None, help="stop cleanly after this long")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--init_weights", default=None, help="start from these weights (e.g. a grown checkpoint)")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=1337)
    for name, typ in (("dropout", float), ("block_size", int), ("sliding_window", int), ("layerdrop_max", float),
                      ("byte_mix_kernel", int), ("n_loops", int), ("homeo_weight", float)):
        p.add_argument(f"--{name}", type=typ, default=None)
    for bname in ("x0_mix", "v0_mix", "byte_mix_bilinear"):
        p.add_argument(f"--{bname}", type=_bool, default=None)
    p.set_defaults(fn=cmd_train, resume=True)

    p = sub.add_parser("generate", help="sample text from a checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--add_bos", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--min_p", type=float, default=None)
    p.add_argument("--use_confidence_temp", action="store_true")
    p.add_argument("--no_stop_on_eos", action="store_true")
    p.add_argument("--no_kv_cache", action="store_true")
    p.add_argument("--ema", action="store_true", help="use EMA weights if the checkpoint has them")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("eval", help="validation loss / bits-per-byte of a checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--ema", action="store_true")
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("grow", help="grow a checkpoint to the next ladder rung (function-preserving)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--to", required=True, choices=list(LADDER_NAMES))
    p.add_argument("--out", required=True)
    p.add_argument("--ema", action="store_true")
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_grow)

    p = sub.add_parser("export", help="write safetensors + config + model card")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--name", default="tensegrity")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--ema", action="store_true")
    p.set_defaults(fn=cmd_export)
    return ap


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
