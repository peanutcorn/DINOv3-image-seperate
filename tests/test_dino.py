"""Offline API wiring test; random tiny model is never used for experiments."""

import pytest
import torch
from PIL import Image
from src.models import build_dino


def test_local_dino_automodel_processor_and_pooled_features(tmp_path):
    transformers = pytest.importorskip("transformers", minversion="4.56.2")
    config = transformers.DINOv3ViTConfig(
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=3,
        image_size=32,
        patch_size=16,
        num_register_tokens=4,
    )
    backbone = transformers.DINOv3ViTModel(config)
    backbone.save_pretrained(tmp_path)
    processor = transformers.DINOv3ViTImageProcessorFast(
        size={"height": 32, "width": 32}
    )
    processor.save_pretrained(tmp_path)
    model, transform, size = build_dino(str(tmp_path))
    pixels = transform(Image.new("RGB", (32, 32))).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        assert model(pixels).shape == (1, size)
    assert not any(p.requires_grad for p in model.parameters())
