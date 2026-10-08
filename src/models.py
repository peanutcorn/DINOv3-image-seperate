"""Pretrained ResNet18 backbone and configurable transfer learning."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet(frozen=True, layers=("layer4", "fc"), pretrained=True):
    """Return ResNet18, its official transform and feature size (512)."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights if pretrained else None)
    model.requires_grad_(False)
    if frozen:
        model.fc = nn.Identity()
    else:
        valid = {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"}
        if not set(layers) <= valid or "fc" not in layers:
            raise ValueError(
                "finetune_layers must include fc and valid ResNet layer names"
            )
        model.fc = nn.Linear(512, 10)
        for name in layers:
            getattr(model, name).requires_grad_(True)
    return model, weights.transforms(), 512


def training_mode(model):
    """Enable training but keep frozen BatchNorm buffers unchanged."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def trainable_parameters(model):
    """Count parameters optimized during training."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DINOFeatures(nn.Module):
    """Expose the official DINOv3 pooled CLS feature as a tensor."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)

    def forward(self, pixels):
        return self.backbone(pixel_values=pixels).pooler_output


class DINOTransform:
    """Apply the checkpoint's official processor to a PIL image."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]

    def __repr__(self):
        return self.processor.to_json_string()


def build_dino(model_id, revision="main"):
    """Load authorized Hugging Face DINOv3 weights and checkpoint preprocessing."""
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) < Version("4.56.2"):
        raise RuntimeError(
            "DINOv3 requires transformers>=4.56.2. Install requirements.txt in a new venv."
        )
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        model_id, revision=revision, use_fast=True
    )
    backbone = AutoModel.from_pretrained(model_id, revision=revision)
    if backbone.config.model_type != "dinov3_vit":
        raise ValueError("Expected a DINOv3 ViT checkpoint")
    return DINOFeatures(backbone), DINOTransform(processor), backbone.config.hidden_size
