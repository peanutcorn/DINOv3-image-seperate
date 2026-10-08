import torch
from torch.utils.data import TensorDataset, DataLoader
from src.train import fit
from src.features import extract_features
from src.utils import seed_everything


def test_validation_restores_best_state_and_reproducibility():
    def train_once():
        seed_everything(4)
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).repeat(10, 1)
        y = torch.tensor([0, 1]).repeat(10)
        loader = DataLoader(TensorDataset(x, y), batch_size=10)
        model = torch.nn.Linear(2, 2)
        _, history = fit(model, loader, loader, torch.device("cpu"), 8, 0.1, 0.0, 2)
        model.eval()
        loss = torch.nn.functional.cross_entropy(model(x), y).item()
        assert abs(loss - min(row["val_loss"] for row in history)) < 1e-6
        assert (model(x).argmax(1) == y).all()
        return model.state_dict()

    first, second = train_once(), train_once()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_feature_cache_reuse_and_invalidation(tmp_path):
    model = torch.nn.Linear(3, 2).requires_grad_(False)
    data = TensorDataset(torch.randn(4, 3), torch.arange(4))
    loader = DataLoader(data, batch_size=2)
    metadata = {"indices": [0, 1, 2, 3], "weights": "test-only"}
    first, seconds, hit = extract_features(
        model, loader, torch.device("cpu"), tmp_path, metadata
    )
    assert not hit
    second, original_seconds, hit = extract_features(
        model, loader, torch.device("cpu"), tmp_path, metadata
    )
    assert hit and original_seconds == seconds
    assert torch.equal(first.tensors[0], second.tensors[0])
    assert torch.equal(first.tensors[1], data.tensors[1])
    _, _, hit = extract_features(
        model, loader, torch.device("cpu"), tmp_path, dict(metadata, weights="changed")
    )
    assert not hit
