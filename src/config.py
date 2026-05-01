"""
Configuration loader for the LLM project.
Loads from YAML and allows CLI overrides.
"""

import os
import yaml
import argparse
from dataclasses import dataclass, field
from typing import Optional


def load_config(config_path: str = None, overrides: dict = None) -> dict:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML config file. Defaults to config/default.yaml
        overrides: Dict of key=value overrides (supports nested keys with dots)
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.path.join("config", "default.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Apply overrides (support dot notation: "finetune.learning_rate=1e-4")
    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            # Try to cast to appropriate type
            d[keys[-1]] = _auto_cast(value)
    
    # Resolve "auto" values
    config = _resolve_auto(config)
    
    return config


def _auto_cast(value: str):
    """Auto-cast string values to appropriate Python types."""
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _resolve_auto(config: dict) -> dict:
    """Resolve 'auto' device and dtype settings."""
    # FIX: torch may not be available in all contexts (e.g. pure data-prep scripts)
    try:
        import torch
    except ImportError:
        if config.get("device") == "auto":
            config["device"] = "cpu"
        if config.get("dtype") == "auto":
            config["dtype"] = "float32"
        return config

    if config.get("device") == "auto":
        if torch.cuda.is_available():
            config["device"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            config["device"] = "mps"
        else:
            config["device"] = "cpu"
    
    if config.get("dtype") == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            config["dtype"] = "bfloat16"
        else:
            config["dtype"] = "float32"
    
    return config


def get_config_from_args() -> dict:
    """
    Parse command-line arguments and load config.
    
    Usage:
        python -m src.training.train --config config/default.yaml --set max_iters=50000
    """
    parser = argparse.ArgumentParser(description="LLM Training Pipeline")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: config/default.yaml)"
    )
    parser.add_argument(
        "--set", nargs="*", default=[],
        help="Override config values: --set key1=val1 key2=val2"
    )
    
    args, _ = parser.parse_known_args()
    
    overrides = {}
    for s in args.set:
        if "=" in s:
            k, v = s.split("=", 1)
            overrides[k] = v
    
    return load_config(config_path=args.config, overrides=overrides)
