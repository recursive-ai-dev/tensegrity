"""Tensegrity: a byte-level language-model family with a verified, trainable growth ladder.

Quick tour::

    from tensegrity import get_preset_config, TensegrityLM, ByteTokenizer
    cfg = get_preset_config("5m")          # 1m 2m 5m 10m 50m 115m
    model = TensegrityLM(cfg)
"""

__version__ = "2.2.0"

# Keep help, version and installation diagnostics usable before optional runtime
# dependencies are available. Public Python APIs retain their original names.
from importlib import import_module as _import_module

_EXPORTS = {
    'installation_report': ('diagnostics', 'installation_report'),
    'format_installation_report': ('diagnostics', 'format_installation_report'),
    'CHECKPOINT_FORMAT_VERSION': ('checkpoint', 'CHECKPOINT_FORMAT_VERSION'),
    'capture_rng_state': ('checkpoint', 'capture_rng_state'),
    'load_checkpoint': ('checkpoint', 'load_checkpoint'),
    'load_model': ('checkpoint', 'load_model'),
    'restore_rng_state': ('checkpoint', 'restore_rng_state'),
    'save_checkpoint': ('checkpoint', 'save_checkpoint'),
    'TensegrityConfig': ('config', 'TensegrityConfig'),
    'ffn_hidden_dim': ('config', 'ffn_hidden_dim'),
    'DATASETS': ('data', 'DATASETS'),
    'ByteDataset': ('data', 'ByteDataset'),
    'audit_dataset': ('data', 'audit_dataset'),
    'download_file': ('data', 'download_file'),
    'fetch_dataset': ('data', 'fetch_dataset'),
    'fetch_hf_streaming': ('data', 'fetch_hf_streaming'),
    'load_meta': ('data', 'load_meta'),
    'prepare_data': ('data', 'prepare_data'),
    'read_download_record': ('data', 'read_download_record'),
    'DatasetError': ('manifest', 'DatasetError'),
    'data_identity': ('manifest', 'data_identity'),
    'file_record': ('manifest', 'file_record'),
    'verify_dataset': ('manifest', 'verify_dataset'),
    'benchmark_presets': ('analysis', 'benchmark_presets'),
    'confidence_calibration': ('analysis', 'confidence_calibration'),
    'plot_reliability': ('analysis', 'plot_reliability'),
    'download_to_browser': ('colab', 'download_to_browser'),
    'free_disk_gb': ('colab', 'free_disk_gb'),
    'get_secret': ('colab', 'get_secret'),
    'in_colab': ('colab', 'in_colab'),
    'in_kaggle': ('colab', 'in_kaggle'),
    'local_scratch': ('colab', 'local_scratch'),
    'mount_drive': ('colab', 'mount_drive'),
    'project_dir': ('colab', 'project_dir'),
    'push_to_hub': ('colab', 'push_to_hub'),
    'runtime_summary': ('colab', 'runtime_summary'),
    'EvalDomain': ('evaluation', 'EvalDomain'),
    'EvalSuite': ('evaluation', 'EvalSuite'),
    'SuiteResult': ('evaluation', 'SuiteResult'),
    'context_probe': ('evaluation', 'context_probe'),
    'format_probe': ('evaluation', 'format_probe'),
    'long_stream_nll': ('evaluation', 'long_stream_nll'),
    'residual_report': ('evaluation', 'residual_report'),
    'export_safetensors': ('export', 'export_safetensors'),
    'load_safetensors': ('export', 'load_safetensors'),
    'quantize_int8_dynamic': ('export', 'quantize_int8_dynamic'),
    'write_model_card': ('export', 'write_model_card'),
    'DistillLoss': ('continual', 'DistillLoss'),
    'EWCState': ('continual', 'EWCState'),
    'GrowthError': ('growth', 'GrowthError'),
    'GrowthReport': ('growth', 'GrowthReport'),
    'check_function_preserved': ('growth', 'check_function_preserved'),
    'check_growth_trainable': ('growth', 'check_growth_trainable'),
    'compute_fisher_diagonal': ('growth', 'compute_fisher_diagonal'),
    'expand_reference_state': ('growth', 'expand_reference_state'),
    'ewc_penalty': ('growth', 'ewc_penalty'),
    'grow_model': ('growth', 'grow_model'),
    'grow_optimizer_state': ('growth', 'grow_optimizer_state'),
    'make_ewc_loss': ('growth', 'make_ewc_loss'),
    'retarget': ('growth', 'retarget'),
    'verify_growth_mapping': ('growth', 'verify_growth_mapping'),
    'autotune_micro_batch': ('hw', 'autotune_micro_batch'),
    'gpu_info': ('hw', 'gpu_info'),
    'measure_throughput': ('hw', 'measure_throughput'),
    'mfu': ('hw', 'mfu'),
    'pick_device': ('hw', 'pick_device'),
    'pick_dtype': ('hw', 'pick_dtype'),
    'RungResult': ('ladder', 'RungResult'),
    'climb_ladder': ('ladder', 'climb_ladder'),
    'TensegrityLM': ('model', 'TensegrityLM'),
    'benchmark_generation': ('perf', 'benchmark_generation'),
    'format_generation_benchmark': ('perf', 'format_generation_benchmark'),
    'format_sdpa_report': ('perf', 'format_sdpa_report'),
    'profile_training_step': ('perf', 'profile_training_step'),
    'sdpa_backend_report': ('perf', 'sdpa_backend_report'),
    'LADDER': ('presets', 'LADDER'),
    'LADDER_NAMES': ('presets', 'LADDER_NAMES'),
    'MODEL_PRESETS': ('presets', 'MODEL_PRESETS'),
    'describe_presets': ('presets', 'describe_presets'),
    'get_preset': ('presets', 'get_preset'),
    'get_preset_config': ('presets', 'get_preset_config'),
    'next_preset': ('presets', 'next_preset'),
    'train_setup': ('presets', 'train_setup'),
    'RunPlan': ('schedule', 'RunPlan'),
    'extend_plan': ('schedule', 'extend_plan'),
    'get_lr': ('schedule', 'get_lr'),
    'get_lr_wsd': ('schedule', 'get_lr_wsd'),
    'make_plan': ('schedule', 'make_plan'),
    'plan_for_time_budget': ('schedule', 'plan_for_time_budget'),
    'ByteTokenizer': ('tokenizer', 'ByteTokenizer'),
    'EMA': ('train', 'EMA'),
    'TrainConfig': ('train', 'TrainConfig'),
    'Trainer': ('train', 'Trainer'),
    'default_ema_decay': ('train', 'default_ema_decay'),
    'evaluate': ('train', 'evaluate'),
    'evaluate_batches': ('train', 'evaluate_batches'),
    'load_run_plan': ('train', 'load_run_plan'),
    'preflight': ('train', 'preflight'),
    'preflight_or_fallback': ('train', 'preflight_or_fallback'),
    'format_preflight': ('report', 'format_preflight'),
    'package_fingerprint': ('report', 'package_fingerprint'),
    'preflight_report': ('report', 'preflight_report'),
    'render_run_report': ('report', 'render_run_report'),
    'write_run_report': ('report', 'write_run_report'),
    'LivePlot': ('viz', 'LivePlot'),
    'confidence_html': ('viz', 'confidence_html'),
    'ArmResult': ('workflow', 'ArmResult'),
    'ContinualReport': ('workflow', 'ContinualReport'),
    'GrowthDecision': ('workflow', 'GrowthDecision'),
    'GrowthGates': ('workflow', 'GrowthGates'),
    'RecoverySpec': ('workflow', 'RecoverySpec'),
    'compare_continual_strategies': ('workflow', 'compare_continual_strategies'),
    'grow_transaction': ('workflow', 'grow_transaction'),
    'verify_roundtrips': ('workflow', 'verify_roundtrips'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = _EXPORTS[name]
    value = getattr(_import_module(f".{module}", __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
