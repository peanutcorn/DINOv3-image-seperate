"""Independent test evaluation and confusion-matrix artifacts."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from src.dataset import CLASSES
from src.utils import save_json


def evaluate(model, loader, device, output_dir):
    """Evaluate once on held-out test data and save all-class metrics."""
    model.eval()
    actual, predicted = [], []
    with torch.inference_mode():
        for inputs, targets in loader:
            predicted.extend(model(inputs.to(device)).argmax(1).cpu().tolist())
            actual.extend(targets.tolist())
    labels = list(range(10))
    matrix = confusion_matrix(actual, predicted, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(
        ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title("Held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png")
    plt.close(fig)
    save_json(output_dir / "confusion_matrix.json", matrix.tolist())
    save_json(
        output_dir / "classification_report.json",
        classification_report(
            actual,
            predicted,
            labels=labels,
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
    )
    return dict(
        accuracy=accuracy_score(actual, predicted),
        macro_f1=f1_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        ),
    )
