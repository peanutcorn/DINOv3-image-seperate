import numpy as np
from src.dataset import split_indices, nested_indices


def test_disjoint_balanced_reproducible_nested_splits():
    labels = np.repeat(np.arange(10), 100)
    train, val = split_indices(labels, 0.1, 42)
    assert (train, val) == split_indices(labels, 0.1, 42)
    assert not set(train) & set(val)
    assert len(set(train) | set(val)) == len(labels)
    previous = set()
    for ratio in [0.1, 0.2, 0.5, 1.0]:
        selected = nested_indices(labels, train, ratio, 7)
        assert previous <= set(selected)
        assert not set(selected) & set(val)
        assert len(set(np.bincount(labels[selected]))) == 1
        previous = set(selected)
