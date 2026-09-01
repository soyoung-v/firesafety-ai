"""Dataset validation 규칙 테스트 - training/dataset/validation.py."""

from training.dataset.generator import DatasetGenerator
from training.dataset.validation import validate_dataset


def _valid_df():
    return DatasetGenerator(runs_per_scenario=1, samples_per_run=60, seed=1).generate()


def test_validate_passes_on_freshly_generated_dataset():
    assert validate_dataset(_valid_df()) == []


def test_validate_detects_negative_current():
    df = _valid_df().copy()
    df.loc[0, "current"] = -1.0
    issues = validate_dataset(df)
    assert any("current" in issue for issue in issues)


def test_validate_detects_circuit_out_of_range():
    df = _valid_df().copy()
    df.loc[0, "circuit"] = 11
    issues = validate_dataset(df)
    assert any("circuit" in issue for issue in issues)


def test_validate_detects_unknown_risk_level():
    df = _valid_df().copy()
    df.loc[0, "risk_level"] = "CRITICAL"
    issues = validate_dataset(df)
    assert any("risk_level" in issue for issue in issues)


def test_validate_detects_unknown_scenario():
    df = _valid_df().copy()
    df.loc[0, "scenario"] = "UNKNOWN"
    issues = validate_dataset(df)
    assert any("scenario" in issue for issue in issues)


def test_validate_detects_null_values():
    df = _valid_df().copy()
    df.loc[0, "temperature"] = None
    issues = validate_dataset(df)
    assert any("null" in issue for issue in issues)


def test_validate_detects_missing_column():
    df = _valid_df().drop(columns=["current"])
    issues = validate_dataset(df)
    assert any("필수 컬럼 누락" in issue for issue in issues)
