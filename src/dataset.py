"""Balanced, disjoint CIFAR-10 splits and nested labeled subsets."""

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def split_indices(labels, validation_fraction, seed):
    """Return stratified train/validation indices from training labels only."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        count = max(1, int(len(indices) * validation_fraction))
        if count >= len(indices):
            raise ValueError(
                "Each class needs at least one train and validation sample"
            )
        validation.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return train, validation


def nested_indices(labels, train_indices, ratio, seed):
    """Select a deterministic per-class prefix, nested across ratios."""
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    labels = np.asarray(labels)
    indices = np.asarray(train_indices, dtype=int)
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels[indices]):
        group = rng.permutation(indices[labels[indices] == label])
        selected.extend(group[: max(1, int(len(group) * ratio))].tolist())
    return selected


class ImageSubset(Dataset):
    """Apply a model-specific transform to fixed source image indices."""

    def __init__(self, dataset, indices, transform):
        self.dataset, self.indices, self.transform = dataset, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return self.transform(image), label


def load_cifar(data_dir, download):
    """Load official CIFAR-10 train and independent test datasets."""
    return (
        CIFAR10(data_dir, train=True, download=download),
        CIFAR10(data_dir, train=False, download=download),
    )
