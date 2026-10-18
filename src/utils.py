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


def resolve_device(name):
    """Validate cpu/cuda/auto and return a torch device without silent fallback."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda or auto")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "NVIDIA GPU is unavailable to this Python environment. "
            "Use .venv\\Scripts\\python.exe and install CUDA-enabled PyTorch "
            "as described in README.md. Check nvidia-smi for the GPU driver."
        )
    return torch.device(name)


def select_device(setting):
    """Ask for 1=CPU or 2=NVIDIA GPU, or validate an explicit configuration."""
    if setting != "prompt":
        return str(resolve_device(setting))
    print("Select execution device:\n  1: CPU\n  2: NVIDIA GPU", flush=True)
    while True:
        try:
            choice = input("Enter 1 or 2: ").strip()
        except EOFError as error:
            raise RuntimeError(
                "No device selection received. Enter 1 or 2 in a terminal, "
                "or pass --device cpu / --device cuda."
            ) from error
        if choice not in {"1", "2"}:
            print("Please enter 1 or 2.", flush=True)
            continue
        try:
            return str(resolve_device("cpu" if choice == "1" else "cuda"))
        except RuntimeError as error:
            print(f"{error}\nSelect 1 to use CPU, or Ctrl+C to exit.", flush=True)
