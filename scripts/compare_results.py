"""Aggregate one experiment run and plot actual measured results."""

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def compare(csv_path):
    """Save seed counts, means, sample standard deviations and comparison plots."""
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    if frame.duplicated(["model", "seed", "ratio"]).any():
        raise ValueError(
            "Duplicate model/seed/ratio rows cannot be treated as independent repeats"
        )
    for column in ["smoke", "split_id", "validation_samples", "test_samples"]:
        if frame[column].nunique() != 1:
            raise ValueError(
                f"Mixed {column} conditions; compare one consistent run at a time"
            )
    metrics = [
        "accuracy",
        "macro_f1",
        "training_seconds",
        "total_compute_seconds",
        "feature_extraction_seconds",
        "linear_training_seconds",
        "finetuning_seconds",
        "trainable_parameters",
    ]
    grouped = frame.groupby(["model", "ratio"])
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary["seed_count"] = grouped["seed"].nunique()
    summary = summary.reset_index()
    summary.to_csv(csv_path.parent / "summary.csv", index=False)
    condition = (
        "SMOKE TEST — tiny CIFAR-10 subsets, 1 epoch"
        if bool(frame["smoke"].iloc[0])
        else "CIFAR-10 — fixed validation/test split"
    )
    for metric, ylabel in [
        ("accuracy", "Test accuracy"),
        ("macro_f1", "Test macro F1"),
        ("training_seconds", "Head training / fine-tuning time (s)"),
        ("total_compute_seconds", "Training + original feature extraction time (s)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for model, rows in summary.groupby("model"):
            rows = rows.sort_values("ratio")
            # Undefined single-seed standard deviations stay blank in CSV.
            ax.errorbar(
                rows["ratio"] * 100,
                rows[metric + "_mean"],
                yerr=rows[metric + "_std"].fillna(0),
                marker="o",
                capsize=3,
                label=model,
            )
        ax.set_xlabel("Labeled fraction of training split (%)")
        ax.set_ylabel(ylabel)
        if metric in ["accuracy", "macro_f1"]:
            ax.set_ylim(0, 1)
        ax.set_title(condition + "\nMean; sample SD shown only for multiple seeds")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(csv_path.parent / f"{metric}_comparison.png", dpi=150)
        plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    compare(args.input)
