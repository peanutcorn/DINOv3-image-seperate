"""Run reproducible CIFAR-10 transfer-learning experiments."""

# ruff: noqa: E402 -- configure CUDA determinism and project imports before torch.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import importlib.metadata
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from src.dataset import ImageSubset, load_cifar, split_indices, nested_indices
from src.models import build_resnet, build_dino, trainable_parameters
from src.features import extract_features
from src.train import fit
from src.evaluate import evaluate
from src.utils import read_config, save_json, seed_everything, fingerprint


def run(config, smoke=False):
    """Execute selected models/seeds/ratios; write an isolated result directory."""
    config = dict(config)
    if smoke:
        config.update(
            epochs=1,
            batch_size=10,
            num_workers=0,
            ratios=[0.1],
            seeds=[42],
            device="cpu",
        )
    if config["epochs"] < 1 or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("epochs must be positive and seeds must be distinct")
    allowed = {"resnet18_frozen", "dinov3_frozen", "resnet18_finetune"}
    if not set(config["models"]) <= allowed or not config["models"]:
        raise ValueError(f"models must be selected from {allowed}")
    for ratio in config["ratios"]:
        if not 0 < ratio <= 1:
            raise ValueError("ratios must be in (0, 1]")
    torch.set_num_threads(config["cpu_threads"])
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config["device"] == "auto"
        else torch.device(config["device"])
    )
    output = (
        ROOT
        / config["output_dir"]
        / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ("_smoke" if smoke else ""))
    )
    output.mkdir(parents=True)
    save_json(output / "config.json", dict(config, smoke=smoke))
    save_json(
        output / "environment.json",
        dict(
            python=platform.python_version(),
            device=str(device),
            hardware=(
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            cpu_threads=config["cpu_threads"],
            versions={
                name: importlib.metadata.version(name)
                for name in [
                    "torch",
                    "torchvision",
                    "transformers",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                ]
            },
        ),
    )
    print(f"Results: {output}", flush=True)
    train_data, test_data = load_cifar(ROOT / config["data_dir"], config["download"])
    train_indices, val_indices = split_indices(
        train_data.targets, config["validation_fraction"], config["split_seed"]
    )
    test_indices = list(range(len(test_data)))
    if smoke:
        train_indices = nested_indices(
            train_data.targets,
            train_indices,
            100 / len(train_indices),
            config["split_seed"],
        )
        val_indices = nested_indices(
            train_data.targets, val_indices, 20 / len(val_indices), config["split_seed"]
        )
        test_indices = nested_indices(
            test_data.targets,
            test_indices,
            20 / len(test_indices),
            config["split_seed"],
        )
    save_json(
        output / "splits.json",
        dict(train=train_indices, validation=val_indices, test=test_indices),
    )
    rows, failures = [], []

    def loader(dataset, shuffle=False, seed=42):
        """Construct a seeded loader, safe with Windows and CPU execution."""
        return DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=shuffle,
            num_workers=config["num_workers"],
            generator=torch.Generator().manual_seed(seed),
        )

    for name in config["models"]:
        frozen = name.endswith("_frozen")
        seed_everything(config["seeds"][0])
        try:
            if name == "dinov3_frozen":
                model, transform, size = build_dino(
                    config["dinov3_id"], config["dinov3_revision"]
                )
            else:
                model, transform, size = build_resnet(
                    frozen=frozen, layers=config["finetune_layers"]
                )
        except (OSError, RuntimeError, ImportError, ValueError) as error:
            if name != "dinov3_frozen":
                raise
            reason = str(error)
            failures.append(dict(model=name, stage="load", reason=reason))
            save_json(output / "failures.json", failures)
            print(
                f'DINOv3 skipped: {reason}\nVisit https://huggingface.co/{config["dinov3_id"]}, accept access/license terms,\nthen set HF_TOKEN or run hf auth login. Install requirements.txt if the API is unavailable.',
                flush=True,
            )
            continue
        model = model.to(device)
        resolved_revision = getattr(
            getattr(getattr(model, "backbone", None), "config", None),
            "_commit_hash",
            None,
        )
        save_json(
            output / f"{name}_backbone.json",
            dict(
                transform=str(transform),
                weights=(
                    config["dinov3_id"] if name == "dinov3_frozen" else "IMAGENET1K_V1"
                ),
                revision=resolved_revision,
                feature_size=size,
            ),
        )
        datasets = [
            ImageSubset(train_data, train_indices, transform),
            ImageSubset(train_data, val_indices, transform),
            ImageSubset(test_data, test_indices, transform),
        ]
        feature_seconds, train_feature_seconds, hits = 0.0, 0.0, []
        if frozen:
            cached = []
            for split, dataset in zip(["train", "validation", "test"], datasets):
                metadata = dict(
                    schema=1,
                    model=name,
                    weights=(
                        config["dinov3_id"]
                        if name == "dinov3_frozen"
                        else "IMAGENET1K_V1"
                    ),
                    revision=(
                        resolved_revision or config["dinov3_revision"]
                        if name == "dinov3_frozen"
                        else None
                    ),
                    transformers=(
                        importlib.metadata.version("transformers")
                        if name == "dinov3_frozen"
                        else None
                    ),
                    transform=str(transform),
                    split=split,
                    indices=dataset.indices,
                    labels=[dataset.dataset.targets[i] for i in dataset.indices],
                    dataset="CIFAR10",
                    device=str(device),
                    hardware=(
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda"
                        else platform.processor()
                    ),
                    cpu_threads=config["cpu_threads"],
                    torch=str(torch.__version__),
                    torchvision=importlib.metadata.version("torchvision"),
                )
                feature_data, seconds, hit = extract_features(
                    model, loader(dataset), device, ROOT / config["cache_dir"], metadata
                )
                cached.append(feature_data)
                feature_seconds += seconds
                if split == "train":
                    train_feature_seconds = seconds
                hits.append(hit)
            datasets = cached
        for seed in config["seeds"]:
            for ratio in config["ratios"]:
                seed_everything(seed)
                selected = nested_indices(
                    train_data.targets, train_indices, ratio, seed
                )
                positions = {
                    index: position for position, index in enumerate(train_indices)
                }
                training_data = Subset(datasets[0], [positions[i] for i in selected])
                run_dir = output / f"{name}_seed{seed}_ratio{ratio:g}"
                run_dir.mkdir()
                save_json(run_dir / "train_indices.json", selected)
                if frozen:
                    classifier = nn.Linear(size, 10).to(device)
                    lr = config["linear_lr"]
                else:
                    classifier, _, _ = build_resnet(
                        frozen=False, layers=config["finetune_layers"]
                    )
                    classifier = classifier.to(device)
                    lr = config["finetune_lr"]
                print(
                    f"{name}, seed={seed}, ratio={ratio}, train={len(selected)}",
                    flush=True,
                )
                seconds, history = fit(
                    classifier,
                    loader(training_data, True, seed),
                    loader(datasets[1]),
                    device,
                    config["epochs"],
                    lr,
                    config["weight_decay"],
                    config["patience"],
                )
                save_json(run_dir / "history.json", history)
                torch.save(
                    dict(
                        state_dict={
                            k: v.cpu() for k, v in classifier.state_dict().items()
                        },
                        model=name,
                        feature_size=size,
                        config=config,
                        revision=resolved_revision,
                    ),
                    run_dir / "best_model.pt",
                )
                metrics = evaluate(classifier, loader(datasets[2]), device, run_dir)
                row = dict(
                    model=name,
                    seed=seed,
                    ratio=ratio,
                    smoke=smoke,
                    train_samples=len(selected),
                    validation_samples=len(val_indices),
                    test_samples=len(test_indices),
                    split_id=fingerprint(
                        dict(train=train_indices, val=val_indices, test=test_indices)
                    ),
                    trainable_parameters=trainable_parameters(classifier),
                    epochs=len(history),
                    feature_extraction_seconds=feature_seconds,
                    train_feature_seconds=train_feature_seconds,
                    linear_training_seconds=seconds if frozen else 0.0,
                    finetuning_seconds=0.0 if frozen else seconds,
                    training_seconds=seconds,
                    total_compute_seconds=seconds + feature_seconds,
                    cache_hit=all(hits) if frozen else False,
                    **metrics,
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(output / "results.csv", index=False)
                save_json(run_dir / "metrics.json", row)
        del model, datasets
    save_json(output / "failures.json", failures)
    if rows:
        from scripts.compare_results import compare

        compare(output / "results.csv")
    else:
        raise RuntimeError(f'No experiments completed. See {output / "failures.json"}')
    return output


def main():
    """Parse CLI overrides without editing the configuration file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Real CIFAR-10/pretrained weights; CPU 1 epoch, 10 train/20 val/20 test",
    )
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--ratios", nargs="+", type=float)
    args = parser.parse_args()
    config = read_config(args.config)
    for key in ["models", "seeds", "ratios"]:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    try:
        run(config, args.smoke)
    except Exception as error:
        print(f"Experiment failed ({type(error).__name__}): {error}", file=sys.stderr)
        print(
            "Check dependencies, data/model download access and available memory.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
