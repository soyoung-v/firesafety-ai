"""DatasetGenerator 재현성/스키마 테스트 - training/dataset/generator.py.

전체 대용량 Dataset을 만들지 않고 작은 run/sample 설정을 사용한다.
"""

from training.dataset.columns import ALL_COLUMNS, SCENARIOS
from training.dataset.generator import DatasetGenerator
from training.dataset.validation import validate_dataset


def _small_generator(seed: int = 42) -> DatasetGenerator:
    return DatasetGenerator(runs_per_scenario=2, samples_per_run=60, seed=seed)


def test_generate_reproducible_with_same_seed():
    df1 = _small_generator(seed=7).generate()
    df2 = _small_generator(seed=7).generate()
    assert df1.equals(df2)


def test_generate_different_seed_produces_different_values():
    df1 = _small_generator(seed=7).generate()
    df2 = _small_generator(seed=8).generate()
    assert not df1["current"].equals(df2["current"])


def test_generate_contains_all_scenarios():
    df = _small_generator().generate()
    assert set(df["scenario"].unique()) == set(SCENARIOS)


def test_generate_has_expected_columns_in_order():
    df = _small_generator().generate()
    assert list(df.columns) == ALL_COLUMNS


def test_generate_passes_validation():
    df = _small_generator().generate()
    assert validate_dataset(df) == []


def test_generate_run_id_maps_to_single_scenario():
    df = _small_generator(seed=9).generate()
    counts = df.groupby("run_id")["scenario"].nunique()
    assert (counts == 1).all()


def test_generate_row_count_matches_runs_times_samples():
    df = _small_generator().generate()
    expected = len(SCENARIOS) * 2 * 60
    assert len(df) == expected
