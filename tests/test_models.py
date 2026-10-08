import torch
from src.models import build_resnet, training_mode


def test_resnet_freeze_and_feature_shape():
    model, _, size = build_resnet(pretrained=False)
    assert not any(p.requires_grad for p in model.parameters())
    model.eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, size)


def test_partial_finetuning_keeps_frozen_batchnorm_buffers():
    model, _, _ = build_resnet(frozen=False, pretrained=False)
    assert all(
        p.requires_grad == name.startswith(("layer4.", "fc."))
        for name, p in model.named_parameters()
    )
    before = model.bn1.running_mean.clone()
    training_mode(model)
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    assert torch.equal(before, model.bn1.running_mean)
    assert model.fc.weight.grad is not None
