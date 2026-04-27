"""YAML-first configuration loader for trajectory prediction runs.

This module intentionally avoids defining runtime argument defaults in Python.
All experiment/runtime values should come from YAML configuration files.
"""

from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class TrajectoryFormat(Enum):
    """Enumeration of supported trajectory coordinate formats."""
    
    ABSOLUTE = "absolute"
    """Original (x, y) positions in absolute coordinates."""
    
    REL_TO_ORIGIN = "rel_to_origin"
    """Relative to the first frame position."""
    
    REL_TO_T = "rel_to_t"
    """Relative to the last observed frame."""
    
    REL_DELTA = "rel_delta"
    """Frame-to-frame differences (default). Most commonly used."""


class ModelType(Enum):
    """Enumeration of supported trajectory prediction models."""
    
    MULTIMODALLSTM = "multimodallstm"
    """LSTM with Stochastic Embedding (main production model)."""


SECTION_NAMES = {"experiment", "data", "model", "training", "inference", "logging", "runtime"}

# Accept old flat keys while migrating.
LEGACY_ALIASES: Dict[str, str] = {
    "learning_rate": "lr",
    "is_deterministic": "is_deter",
    "model_type": "model",
    "checkpoint_path": "ckpt_path",
}

# Required runtime keys are intentionally explicit (no Python defaults).
REQUIRED_RUNTIME_KEYS = {
    "model",
    "train",
    "random_seed",
    "random_seeds",
    "data_dir",
    "traj_format",
    "base_motion",
    "target",
    "use_headeye",
    "use_pod",
    "use_expt",
    "use_person",
    "aux_format",
    "is_normalize_ts",
    "is_normalize_aux",
    "batch_size",
    "lr",
    "lr_milestones",
    "gamma",
    "n_epochs",
    "is_deter",
    "is_infer_mu",
    "num_samples",
    "is_save_config_file",
}


def _apply_legacy_aliases(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)
    for old_key, new_key in LEGACY_ALIASES.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized.pop(old_key)
    return normalized


def _is_hierarchical(config: Dict[str, Any]) -> bool:
    return any(
        key in SECTION_NAMES and isinstance(config.get(key), dict)
        for key in config
    )


def _flatten_hierarchical_config(config: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}

    experiment = config.get("experiment", {}) or {}
    data = config.get("data", {}) or {}
    model = config.get("model", {}) or {}
    training = config.get("training", {}) or {}
    inference = config.get("inference", {}) or {}
    logging = config.get("logging", {}) or {}
    runtime = config.get("runtime", {}) or {}

    flat.update(experiment)
    flat.update(data)

    if "type" in model:
        flat["model"] = model["type"]
    flat.update(model.get("params", {}) or {})

    # Support nested optimizer/scheduler or direct keys.
    optimizer = training.get("optimizer", {}) or {}
    scheduler = training.get("scheduler", {}) or {}
    for key, value in training.items():
        if key not in {"optimizer", "scheduler"}:
            flat[key] = value
    flat.update(optimizer)
    flat.update(scheduler)

    if "is_deterministic" in inference and "is_deter" not in inference:
        inference["is_deter"] = inference.pop("is_deterministic")
    flat.update(inference)
    flat.update(logging)
    flat.update(runtime)

    return _apply_legacy_aliases(flat)


def _validate_enums(flat: Dict[str, Any]) -> None:
    model_value = flat.get("model")
    if model_value is not None:
        try:
            ModelType(model_value)
        except ValueError as exc:
            supported = ", ".join(m.value for m in ModelType)
            raise ValueError(f"Unsupported model: {model_value}. Supported: {supported}") from exc

    traj_format_value = flat.get("traj_format")
    if traj_format_value is not None:
        try:
            TrajectoryFormat(traj_format_value)
        except ValueError as exc:
            supported = ", ".join(t.value for t in TrajectoryFormat)
            raise ValueError(f"Unsupported format: {traj_format_value}. Supported: {supported}") from exc


def _validate_required_keys(flat: Dict[str, Any]) -> None:
    missing = sorted(k for k in REQUIRED_RUNTIME_KEYS if k not in flat)
    if missing:
        raise ValueError(
            "Config is missing required keys (define them in YAML): " + ", ".join(missing)
        )


def load_yaml_config(config_filename: str) -> Dict[str, Any]:
    """Load config YAML and return a flat runtime dictionary."""
    path = Path(config_filename)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_filename}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("YAML config must be a mapping")

    flat = _flatten_hierarchical_config(raw) if _is_hierarchical(raw) else _apply_legacy_aliases(raw)
    _validate_enums(flat)
    _validate_required_keys(flat)
    return flat


def load_runtime_args(config_filename: str, cli_args: Optional[argparse.Namespace] = None) -> argparse.Namespace:
    """Load YAML config as argparse-style namespace.

    CLI args are only optional explicit overrides. Keys with value ``None`` are ignored.
    """
    flat = load_yaml_config(config_filename)
    if cli_args is not None:
        for key, value in vars(cli_args).items():
            if key == "config_filename":
                continue
            if value is not None:
                flat[key] = value

    flat = _apply_legacy_aliases(flat)
    _validate_enums(flat)
    _validate_required_keys(flat)
    return argparse.Namespace(**flat)


def to_hierarchical_config(args_or_dict: Any) -> Dict[str, Any]:
    """Serialize runtime args to hierarchical YAML structure."""
    flat = vars(args_or_dict) if isinstance(args_or_dict, argparse.Namespace) else dict(args_or_dict)
    flat = _apply_legacy_aliases(flat)

    hierarchy: Dict[str, Any] = {
        "experiment": {
            "train": flat.get("train"),
            "random_seed": flat.get("random_seed"),
            "random_seeds": flat.get("random_seeds"),
            "ckpt_path": flat.get("ckpt_path"),
        },
        "data": {
            "data_dir": flat.get("data_dir"),
            "traj_format": flat.get("traj_format"),
            "base_motion": flat.get("base_motion"),
            "target": flat.get("target"),
            "use_headeye": flat.get("use_headeye"),
            "use_pod": flat.get("use_pod"),
            "use_expt": flat.get("use_expt"),
            "use_person": flat.get("use_person"),
            "aux_format": flat.get("aux_format"),
            "is_normalize_ts": flat.get("is_normalize_ts"),
            "is_normalize_aux": flat.get("is_normalize_aux"),
        },
        "model": {
            "type": flat.get("model"),
            "params": {},
        },
        "training": {
            "n_epochs": flat.get("n_epochs"),
            "batch_size": flat.get("batch_size"),
            "optimizer": {
                "lr": flat.get("lr"),
            },
            "scheduler": {
                "lr_milestones": flat.get("lr_milestones"),
                "gamma": flat.get("gamma"),
            },
        },
        "inference": {
            "is_deter": flat.get("is_deter"),
            "is_infer_mu": flat.get("is_infer_mu"),
            "num_samples": flat.get("num_samples"),
        },
        "logging": {
            "is_save_config_file": flat.get("is_save_config_file"),
        },
        "runtime": {},
    }

    reserved = {
        "config_filename", "model", "train", "random_seed", "random_seeds", "ckpt_path",
        "data_dir", "traj_format", "base_motion", "target", "use_headeye", "use_pod",
        "use_expt", "use_person", "aux_format", "is_normalize_ts", "is_normalize_aux",
        "n_epochs", "batch_size", "lr", "lr_milestones", "gamma",
        "is_deter", "is_infer_mu", "num_samples", "is_save_config_file",
    }
    for key, value in flat.items():
        if key not in reserved:
            hierarchy["model"]["params"][key] = value

    return hierarchy
