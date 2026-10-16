"""Tensegrity: a byte-level language-model family with a verified, trainable growth ladder.

Quick tour::

    from tensegrity import get_preset_config, TensegrityLM, ByteTokenizer
    cfg = get_preset_config("5m")          # 1m 2m 5m 10m 50m 115m
    model = TensegrityLM(cfg)
"""

__version__ = "2.0.0"

from .checkpoint import (CHECKPOINT_FORMAT_VERSION, capture_rng_state, load_checkpoint, load_model,
                         restore_rng_state, save_checkpoint)
from .config import TensegrityConfig, ffn_hidden_dim
from .data import (DATASETS, ByteDataset, download_file, fetch_dataset, fetch_hf_streaming, load_meta,
                   prepare_data)
from .analysis import benchmark_presets, confidence_calibration, plot_reliability
from .colab import (download_to_browser, free_disk_gb, get_secret, in_colab, in_kaggle, local_scratch, mount_drive,
                    project_dir, push_to_hub, runtime_summary)
from .export import (export_safetensors, load_safetensors, quantize_int8_dynamic, write_model_card)
from .growth import (check_function_preserved, check_growth_trainable, compute_fisher_diagonal,
                     expand_reference_state, ewc_penalty, grow_model, grow_optimizer_state, make_ewc_loss, retarget)
from .hw import (autotune_micro_batch, gpu_info, measure_throughput, mfu, pick_device, pick_dtype)
from .ladder import RungResult, climb_ladder
from .model import TensegrityLM
from .presets import (LADDER, LADDER_NAMES, MODEL_PRESETS, describe_presets, get_preset, get_preset_config,
                      next_preset, train_setup)
from .schedule import RunPlan, extend_plan, get_lr, get_lr_wsd, make_plan, plan_for_time_budget
from .tokenizer import ByteTokenizer
from .train import (EMA, TrainConfig, Trainer, default_ema_decay, evaluate, evaluate_batches,
                    load_run_plan, preflight, preflight_or_fallback)
from .viz import LivePlot, confidence_html

__all__ = [n for n in dir() if not n.startswith("_")]
