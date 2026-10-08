"""Shared reproducibility and configuration helpers."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed):
    """Seed Python, NumPy and torch; use deterministic torch operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def read_config(path):
    """Load an experiment YAML configuration."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fingerprint(value):
    """Return a stable hash of JSON-compatible cache metadata."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    """Write readable JSON, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def synchronize(device):
    """Wait for CUDA work before measuring elapsed time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
