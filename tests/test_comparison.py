import pandas as pd
import pytest
from scripts.compare_results import compare


def result(seed):
    return dict(
        model="unit_test",
        seed=seed,
        ratio=0.1,
        smoke=True,
        split_id="same",
        validation_samples=20,
        test_samples=20,
        accuracy=0.5,
        macro_f1=0.4,
        training_seconds=1.0,
        total_compute_seconds=3.0,
        feature_extraction_seconds=2.0,
        linear_training_seconds=1.0,
        finetuning_seconds=0.0,
        trainable_parameters=10,
    )


def test_single_seed_has_no_invented_standard_deviation(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame([result(42)]).to_csv(path, index=False)
    summary = compare(path)
    assert summary.iloc[0]["seed_count"] == 1
    assert pd.isna(summary.iloc[0]["accuracy_std"])
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_duplicate_repeats_are_rejected(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame([result(42), result(42)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        compare(path)
