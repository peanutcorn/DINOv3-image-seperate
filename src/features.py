"""Extract frozen features and reuse only matching caches."""

import time
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from src.utils import fingerprint, synchronize


def extract_features(model, loader, device, cache_dir, metadata):
    """Return feature dataset, original extraction seconds and cache-hit flag."""
    path = Path(cache_dir) / (fingerprint(metadata) + ".pt")
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["metadata"] == metadata:
            return (
                TensorDataset(saved["features"], saved["labels"]),
                saved["seconds"],
                True,
            )
    model.eval()
    features, labels = [], []
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="Extract features", leave=False):
            features.append(model(images.to(device)).cpu())
            labels.append(targets.cpu())
    synchronize(device)
    seconds = time.perf_counter() - started
    saved = dict(
        features=torch.cat(features),
        labels=torch.cat(labels),
        metadata=metadata,
        seconds=seconds,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(path)
    return TensorDataset(saved["features"], saved["labels"]), seconds, False
