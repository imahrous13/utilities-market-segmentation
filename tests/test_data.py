"""Tests for data loading, cleaning and standardisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utilities_clustering import clean, load_and_prepare, load_raw
from utilities_clustering.config import DATA_PATH, FEATURES


@pytest.fixture(scope="module")
def prepared():
    return load_and_prepare(DATA_PATH)


def test_source_file_is_readable_and_untouched():
    raw = load_raw(DATA_PATH)
    assert len(raw) == 22
    # Headers arrive with spaces and mixed case; cleaning is what fixes them.
    assert "Company" in raw.columns
    assert "Fixed_charge" in raw.columns


def test_cleaning_normalises_headers_and_index():
    cleaned, report = clean(load_raw(DATA_PATH))
    assert list(cleaned.columns) == list(FEATURES)
    assert cleaned.index.name is None
    assert all(not str(name) != str(name).strip() for name in cleaned.index)
    assert report["columns_renamed"] is True
    assert report["id_whitespace_stripped"] is True


def test_cleaning_is_a_no_op_on_this_dataset():
    """Cleaning runs but must not silently drop good rows."""
    _, report = clean(load_raw(DATA_PATH))
    assert report["rows_before"] == 22
    assert report["rows_after"] == 22
    assert report["null_rows_dropped"] == 0
    assert report["duplicate_rows_dropped"] == 0


def test_missing_expected_column_raises(tmp_path):
    broken = pd.DataFrame({"company": ["A"], "ror": [1.0]})
    path = tmp_path / "broken.csv"
    broken.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        clean(pd.read_csv(path))


def test_null_and_duplicate_rows_are_dropped(tmp_path):
    frame = pd.DataFrame(
        {
            "Company": ["A", "A", "B", "C"],
            **{c: [1.0, 1.0, 2.0, None] for c in ("Fixed charge",)},
        }
    )
    for column in FEATURES:
        if column != "fixed_charge":
            frame[column] = [1.0, 1.0, 2.0, 3.0]
    _, report = clean(frame)
    assert report["null_rows_dropped"] == 1
    assert report["duplicate_rows_dropped"] == 1
    assert report["rows_after"] == 2


def test_quality_report_shape_and_definitions(prepared):
    _, _, quality = prepared
    assert quality["shape"] == [22, 8]
    assert quality["n_observations"] == 22
    assert quality["null_cells"] == 0
    assert quality["duplicate_rows"] == 0
    assert set(quality["describe"]) == set(FEATURES)
    assert quality["constant_features"] == []
    assert set(quality["describe"]["sales"]) == {
        "mean", "std", "min", "q25", "median", "q75", "max"
    }


def test_sales_dominates_raw_distance(prepared):
    """The empirical basis for insisting on standardisation."""
    _, _, quality = prepared
    share = quality["raw_squared_distance_share_pct"]
    assert share["sales"] > 99.0
    others = sum(v for c, v in share.items() if c != "sales")
    assert others < 1.0


def test_standardisation_is_zero_mean_unit_variance(prepared):
    standardised, _, _ = prepared
    assert np.allclose(standardised.values.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(standardised.values.std(axis=0), 1.0, atol=1e-12)


def test_standardised_values_match_manual_transform(prepared):
    """Guards against a silently wrong scaler definition."""
    standardised, _, _ = prepared
    raw = standardised.raw
    expected = (raw.to_numpy() - raw.to_numpy().mean(axis=0)) / raw.to_numpy().std(axis=0)
    assert np.allclose(standardised.values, expected)


def test_standardisation_preserves_shape_and_order(prepared):
    standardised, _, _ = prepared
    assert standardised.values.shape == (22, 8)
    assert standardised.features == FEATURES
    assert len(standardised.labels) == 22
    assert standardised.n == 22


def test_nuclear_is_effectively_binary(prepared):
    standardised, _, _ = prepared
    nuclear = standardised.raw["nuclear"]
    assert (nuclear == 0).sum() == 12
    assert nuclear.median() == 0.0


def test_scaler_can_transform_new_rows(prepared):
    """StandardScaler must expose a reusable transform, not just a fitted array."""
    standardised, _, _ = prepared
    raw = standardised.raw
    # Refitting on the same frame must reproduce the stored matrix exactly.
    assert np.allclose(standardised.scaler.transform(raw.to_numpy()), standardised.values)
    # An unseen batch must transform without error and keep the same scale convention.
    probe = raw.iloc[:3]
    transformed = standardised.scaler.transform(probe.to_numpy())
    assert transformed.shape == (3, 8)
    # Values are centred on the *fitted* mean, not the probe's own mean.
    assert not np.allclose(
        transformed,
        (probe.to_numpy() - probe.to_numpy().mean(axis=0)) / probe.to_numpy().std(axis=0),
    )
