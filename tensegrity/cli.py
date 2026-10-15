"""Command line: ``python -m tensegrity <command>``.

    presets                          list the size ladder
    prepare   --data_path P --out_dir D [--doc_sep '<|endoftext|>'] [--record_template ... --split ...]
    audit     --data_dir D [--deep]
    train     --preset 5m --data_dir D --out_dir R [--hours 3 | --tokens 5e8] [--optimizer muon]
    generate  --ckpt R/ckpt_best.pt --prompt "Once upon a time"
    eval      --ckpt R/ckpt_best.pt --data_dir D
    grow      --ckpt R/ckpt_best.pt --to 10m --out R2/ckpt_latest.pt
    export    --ckpt R/ckpt_best.pt --out_dir export [--dtype float16]
    preflight --preset 5m --data_dir D         decisive pre-run report (exit 1 on blockers)
    report    --run_dir R [--data_dir D]       write run_report.json / run_report.md
    probe     --ckpt R/ckpt_best.pt            attention reach + copy/retrieval probe at increasing distances
    bench     --ckpt R/ckpt_best.pt            generation latency (window vs streaming) + SDPA backends
    continual --ckpt C --old_dir A --new_dir B new-only vs replay vs replay+EWC, matched budget

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
from .data import audit_dataset, prepare_data
from .export import export_safetensors, write_model_card
from .evaluation import context_probe, format_probe
from .growth import check_function_preserved, check_growth_trainable, grow_model, retarget
from .hw import autotune_micro_batch, measure_throughput, pick_device, pick_dtype
from .manifest import file_record, render_manifest_summary
from .model import TensegrityLM
from .perf import benchmark_generation, format_generation_benchmark, format_sdpa_report, sdpa_backend_report
from .presets import (LADDER_NAMES, describe_presets, get_preset, get_preset_config, train_setup)
from .report import format_preflight, preflight_report, write_run_report
from .schedule import make_plan, plan_for_time_budget
from .tokenizer import ByteTokenizer
from .train import TrainConfig, Trainer, evaluate, load_run_plan, preflight_or_fallback
from .workflow import compare_continual_strategies


def _bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y")


def cmd_presets(args) -> None:
    print(describe_presets())


def cmd_prepare(args) -> None:
    meta = prepare_data(args.data_path, args.out_dir, args.val_fraction, args.seed, doc_sep=args.doc_sep,
                        max_bytes=int(args.max_mb * 1e6) if args.max_mb else None, format=args.format,
                        text_field=args.text_field, record_template=args.record_template,
                        group_field=args.group_field, split=args.split, min_documents=args.min_documents,
                        dedup=args.dedup, validate_utf8=not args.no_validate_utf8,
                        max_reject_fraction=args.max_reject_fraction, overwrite=args.overwrite)
    print(render_manifest_summary(meta))


def cmd_audit(args) -> None:
    rep = audit_dataset(args.data_dir, n_examples=args.examples, deep=args.deep)
    print(rep["text"])
    if rep["problems"]:
        raise SystemExit(1)


def cmd_train(args) -> None:
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    os.makedirs(args.out_dir, exist_ok=True)

    saved = load_run_plan(args.out_dir)
    overrides = {k: getattr(args, k) for k in ("dropout", "block_size", "sliding_window", "layerdrop_max",
                                               "byte_mix_kernel", "x0_mix", "homeo_weight")
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
    lineage = {"init_from": file_record(args.init_weights)} if (args.init_weights and not (saved is not None and args.resume)) else None
    trainer = Trainer(model, tc, plan, args.data_dir, args.out_dir, device=args.device, lineage=lineage)
    res = trainer.fit(resume=args.resume)
    print({k: res[k] for k in ("status", "step", "best_val_loss", "tokens_seen")})


def cmd_generate(args) -> None:
    device = pick_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device, use_ema=args.ema)
    model.eval()
    tok = ByteTokenizer()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    start = "bos" if args.add_bos else args.start_token
    prefix = {"eos": [ByteTokenizer.DOC_START], "bos": [ByteTokenizer.BOS], "none": []}[start]
    ids = prefix + tok.encode(args.prompt or "")
    if not ids:
        raise SystemExit("nothing to condition on: give a --prompt or use --start_token eos")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                         top_p=args.top_p, min_p=args.min_p, use_confidence_temp=args.use_confidence_temp,
                         stop_on_eos=not args.no_stop_on_eos, use_cache=not args.no_kv_cache,
                         cache_mode=args.cache_mode)
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
    save_checkpoint(path, new, None, new_cfg, 0, float("inf"), extra={"grown_from": file_record(args.ckpt)})
    print(f"[grow] {old_cfg.n_params() / 1e6:.2f}M -> {new_cfg.n_params() / 1e6:.2f}M saved to {path} "
          f"(weights only; pass --init_weights to `train`)")


def cmd_preflight(args) -> None:
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    preset = get_preset(args.preset)
    cfg = get_preset_config(preset.name)
    model = TensegrityLM(cfg)
    if args.micro_batch:
        mb, accum = args.micro_batch, args.grad_accum or 1
    else:
        model.to(device)
        tune = autotune_micro_batch(model, cfg.block_size, device, dtype, preset.batch_tokens, verbose=False)
        mb, accum = tune["micro_batch"], tune["grad_accum"]
        model.grad_checkpoint = tune["grad_checkpoint"]
    plan = make_plan(int(args.tokens) if args.tokens else preset.default_tokens, mb * accum * cfg.block_size)
    tc, plan = train_setup(preset.name, mb, accum, plan=plan, optimizer=args.optimizer, dtype=args.dtype, seed=args.seed)
    rep = preflight_report(model, tc, plan, args.data_dir, device, deep_data=args.deep)
    print(format_preflight(rep))
    if not rep["ok"]:
        raise SystemExit(1)


def cmd_report(args) -> None:
    rep = write_run_report(args.run_dir, data_dirs=args.data_dir, export_dir=args.export_dir,
                           evaluate=bool(args.data_dir))
    print(f"wrote {os.path.join(args.run_dir, 'run_report.md')} and run_report.json")
    p = rep.get("progress")
    if p:
        print(f"{'finished' if p['finished'] else 'in progress'}: {p['completed_updates']:,}/{p['planned_updates']:,} updates, "
              f"{p['tokens_seen'] / 1e6:,.2f}M bytes")


def cmd_probe(args) -> None:
    device = pick_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device, use_ema=args.ema)
    print(cfg.describe())
    rows = context_probe(model, tuple(int(d) for d in args.distances.split(",")), device, key_len=args.key_len,
                         n_trials=args.trials)
    print(format_probe(rows))


def cmd_bench(args) -> None:
    device = pick_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device)
    print(format_generation_benchmark(benchmark_generation(model, device, new_tokens=args.new_tokens)))
    print(format_sdpa_report(sdpa_backend_report(cfg, device)))


def cmd_continual(args) -> None:
    compare_continual_strategies(args.ckpt, args.old_dir, args.new_dir, steps=args.steps, replay_fraction=args.replay_fraction,
                                 ewc_lambda=args.ewc_lambda, ewc_kind=args.ewc_kind, batch_size=args.batch_size,
                                 learning_rate=args.learning_rate, seeds=[int(s) for s in args.seeds.split(",")],
                                 device=pick_device(args.device), run_dir=args.out_dir)


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
    p.add_argument("--format", choices=["text", "jsonl", "parquet"], default=None,
                   help="force a reader for files with an unrecognised extension")
    p.add_argument("--text_field", default=None, help="jsonl key / parquet column holding the text (default: auto)")
    p.add_argument("--record_template", default=None,
                   help="stitch several fields into ONE document, e.g. 'Q: {question}\\nA: {answer}'")
    p.add_argument("--group_field", default=None,
                   help="records sharing this field's value always land in the same split (e.g. a context id)")
    p.add_argument("--split", choices=["auto", "document", "tail"], default="auto",
                   help="document: hash each document into train/val; tail: last val_fraction of the stream; "
                        "auto: document when there are enough documents")
    p.add_argument("--min_documents", type=int, default=50)
    p.add_argument("--dedup", action="store_true", help="drop exact duplicate documents (document split only)")
    p.add_argument("--no_validate_utf8", action="store_true")
    p.add_argument("--max_reject_fraction", type=float, default=0.01,
                   help="fail if more than this fraction of a file's records is unusable")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing different dataset in out_dir (the old files are moved aside)")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("audit", help="show a prepared dataset's manifest, integrity and decoded examples")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--examples", type=int, default=2)
    p.add_argument("--deep", action="store_true", help="re-hash the token files against the manifest")
    p.set_defaults(fn=cmd_audit)

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
                      ("byte_mix_kernel", int), ("homeo_weight", float)):
        p.add_argument(f"--{name}", type=typ, default=None)
    p.add_argument("--x0_mix", type=_bool, default=None)
    p.set_defaults(fn=cmd_train, resume=True)

    p = sub.add_parser("generate", help="sample text from a checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--start_token", choices=["eos", "bos", "none"], default="eos",
                   help="context before the prompt. eos (default) is what the model saw before every training "
                        "document; bos was never seen as input and degrades the first bytes")
    p.add_argument("--add_bos", action="store_true", help="deprecated alias for --start_token bos")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--min_p", type=float, default=None)
    p.add_argument("--use_confidence_temp", action="store_true")
    p.add_argument("--no_stop_on_eos", action="store_true")
    p.add_argument("--no_kv_cache", action="store_true")
    p.add_argument("--cache_mode", choices=["window", "streaming"], default="window",
                   help="past the context window: window = exact (re-encodes), streaming = bounded KV cache, O(1) per byte")
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

    p = sub.add_parser("preflight", help="decisive pre-run report: size, attention reach, data, numerics, memory, reload")
    p.add_argument("--preset", default="5m", choices=list(LADDER_NAMES))
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tokens", type=float, default=None)
    p.add_argument("--micro_batch", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--deep", action="store_true", help="re-hash the dataset files against their manifest")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=1337)
    p.set_defaults(fn=cmd_preflight)

    p = sub.add_parser("report", help="write run_report.json / run_report.md for a run directory")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", default=None, help="evaluate ckpt_best.pt on this dataset's validation split")
    p.add_argument("--export_dir", default=None, help="check that exported safetensors reproduce the best model")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("probe", help="copy/retrieval probe at increasing distances, next to the layout's reach")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--distances", default="0,16,64,256,1024")
    p.add_argument("--key_len", type=int, default=12)
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--ema", action="store_true")
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("bench", help="generation latency per cache_mode and SDPA backend availability")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--new_tokens", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("continual", help="new-only vs replay vs replay+EWC from the same weights, matched budget")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--old_dir", required=True)
    p.add_argument("--new_dir", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--replay_fraction", type=float, default=0.3)
    p.add_argument("--ewc_lambda", type=float, default=100.0)
    p.add_argument("--ewc_kind", choices=["empirical", "sampled"], default="empirical")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_continual)

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
