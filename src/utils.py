"""
============================================================
Utilities module
============================================================

Provides:
    - set_seed(): Reproducibility
    - load_config(): Load YAML config
    - save_json(): Save metrics/results
    - get_device(): GPU/CPU detection
    - count_parameters(): Model size
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Set seed cho reproducibility trên Python, NumPy, PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Để training deterministic (chậm hơn chút)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, path: Union[str, Path], indent: int = 2):
    """Save dict/list vào JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_device() -> torch.device:
    """Detect available device và print info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[CPU] No CUDA GPU found — training will be slow")
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Đếm parameters của model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def format_time(seconds: float) -> str:
    """Format seconds → 'HHh MMm SSs'."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
