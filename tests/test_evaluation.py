"""Tests for evaluation, k-selection and the reported conclusions.

These lock in the numbers the README quotes. If a dependency upgrade changes a
conclusion, a test should fail rather than the write-up silently going stale.
"""

from __future__ import annotations

import numpy as np
import pytest

from utilities_clustering import (
    K_PRIMARY,
    agreement,
    cluster_profiles,
    load_and_prepare,
    recommended_k,
    run_analysis,
    seed_sensitivity,
    select_k,
    separation,
    subsample_stability,
    to_serialisable,
)
from utilities_clustering.config import DATA_PATH, FEATURES


@pytest.fixture(scope="module")
def result():
    return run_analysis(DATA_PATH)


@pytest.fixture(scope="module")
def standardised():
    return load_and_prepare(DATA_PATH)[0]


def test_k_is_selected_by_evidence_not_assumption(result):
    """Two of the three indices must agree on k = 5."""
    recommendation = result.recommendation
    assert recommendation["recommended"] == K_PRIMARY
    assert recommendation["votes"]["silhouette"] == K_PRIMARY
    assert recommendation["votes"]["calinski_harabasz"] == K_PRIMARY


def test_silhouette_peaks_at_selected_k(result):
    table = result.k_selection
    best = table.loc[table["silhouette"].idxmax(), "k"]
    assert int(best) == K_PRIMARY
    at_k5 = float(table.loc[table["k"] == K_PRIMARY, "silhouette"].iloc[0])
    at_k6 = float(table.loc[table["k"] == 6, "silhouette"].iloc[0])
    assert at_k5 > 2 * at_k6


def test_silhouette_is_low_in_absolute_terms(result):
    """Guards against over-claiming: 8 dimensions on 22 points cannot separate cleanly."""
    value = result.primary_separation["silhouette"]
    assert 0.0 < value < 0.25


def test_a_material_share_of_points_sit_on_the_wrong_side(result):
    separation_stats = result.primary_separation
    assert separation_stats["share_below_zero"] > 0.1
    assert separation_stats["points_below_zero"] >= 3


def test_k5_and_k6_are_the_same_segmentation_at_different_resolution(result):
    scores = agreement(result.primary.labels, result.sensitivity.labels)
    assert scores["ari"] > 0.75
    assert scores["nmi"] > 0.85


def test_selected_k_keeps_nuclear_operators_in_one_cluster(result, standardised):
    raw = standardised.raw
    nuclear_heavy = raw.index[raw["nuclear"] > 30]
    assert len(nuclear_heavy) == 4
    assigned = {
        result.primary.labels[list(raw.index).index(name)] for name in nuclear_heavy
    }
    assert len(assigned) == 1


def test_k6_splits_those_nuclear_operators(result, standardised):
    raw = standardised.raw
    nuclear_heavy = raw.index[raw["nuclear"] > 30]
    assigned = {
        result.sensitivity.labels[list(raw.index).index(name)] for name in nuclear_heavy
    }
    assert len(assigned) == 2


def test_preferred_linkages_collapse_onto_one_partition(result):
    cross = result.linkage_agreement.attrs["cross_linkage"]
    assert cross, "expected at least one preferred-linkage comparison"
    for scores in cross.values():
        assert scores["ari"] > 0.999


def test_single_linkage_disagrees_with_everyone(result):
    single = result.primary_vs_linkage.set_index("linkage").loc["single", "ari"]
    others = result.primary_vs_linkage[result.primary_vs_linkage["linkage"] != "single"]["ari"]
    assert single < 0.2
    assert (others > 0.3).all()


def test_kmedoids_is_reproducible_and_kmeans_is_not(result):
    assert result.seed_probe["seed_independent"] is True
    assert result.seed_probe["min_ari"] > 0.999
    assert result.kmeans_contrast["kmeans_seed_min_ari"] < 0.9


def test_kmeans_centroids_are_not_observations(result, standardised):
    """Contrast case: K-Means cannot name an operator for a segment."""
    assert result.kmeans_contrast["centroid_is_observation"] is False
    assert result.kmeans_contrast["min_distance_point_to_own_centroid"] > 0.5


def test_subsample_stability_is_reported_honestly(result):
    """22 points is a thin sample; stability must be reported, not hidden."""
    stability = result.stability
    assert not stability.empty
    assert (stability["mean_ari"] < 1.0).all()
    assert stability["mean_ari"].is_monotonic_increasing


def test_cluster_profiles_are_standardised_deviations(standardised):
    labels = np.zeros(standardised.n, dtype=int)
    profiles = cluster_profiles(standardised.raw, labels, FEATURES)
    for column in FEATURES:
        assert np.isclose(profiles.loc[0, f"z_{column}"], 0.0, atol=1e-9)
        assert np.isclose(profiles.loc[0, f"mean_{column}"], standardised.raw[column].mean())


def test_cluster_profiles_row_count_equals_k(result):
    assert len(result.profiles) == result.primary.k
    assert result.profiles["n_members"].sum() == 22


def test_pca_reports_honest_variance_coverage(result):
    coverage = result.pca["cumulative_explained_variance"]
    assert 0.3 < coverage < 0.8


def test_segmentation_frame_is_internally_consistent(result):
    frame = result.segmentation_frame()
    assert len(frame) == 22
    assert frame["company"].is_unique
    assert int(frame["is_medoid"].sum()) == result.primary.k
    assert frame["distance_to_medoid"].min() == pytest.approx(0.0, abs=1e-9)
    assert set(frame["medoid"]) == set(result.primary.names)


def test_every_medoid_row_has_zero_distance(result):
    frame = result.segmentation_frame()
    assert (frame.loc[frame["is_medoid"], "distance_to_medoid"] < 1e-9).all()


def test_selection_table_shape_and_columns(standardised):
    table = select_k(standardised.values, standardised.labels, 2, 8, 0)
    assert list(table["k"]) == list(range(2, 9))
    for column in ("silhouette", "calinski_harabasz", "davies_bouldin", "wcss_total"):
        assert np.isfinite(table[column]).all()


def test_recommended_k_tallies_votes(standardised):
    table = select_k(standardised.values, standardised.labels, 2, 8, 0)
    outcome = recommended_k(table)
    assert sum(outcome["tally"].values()) == 3


def test_separation_scores_are_in_range(standardised):
    labels = np.repeat(np.arange(4), 6)[: standardised.n]
    stats = separation(standardised.values, labels)
    assert -1.0 <= stats["silhouette"] <= 1.0
    assert stats["davies_bouldin"] >= 0.0
    assert stats["calinski_harabasz"] >= 0.0


def test_seed_sensitivity_helper_agrees_with_full_pipeline(result, standardised):
    probe = seed_sensitivity(
        standardised.values, standardised.labels, K_PRIMARY, range(1, 9), 0
    )
    assert probe["seed_independent"] == result.seed_probe["seed_independent"]


def test_subsample_helper_reduces_on_tiny_samples(standardised):
    table = subsample_stability(
        standardised.values, standardised.labels, K_PRIMARY, sizes=(10,), repeats=20, seed=0
    )
    assert len(table) == 1
    assert table.loc[0, "mean_ari"] <= 1.0


def test_segmentation_quality_exposes_the_weakest_cluster(result):
    """A good mean must not hide a badly mixed segment."""
    table = result.segmentation_quality()
    assert len(table) == result.primary.k
    assert table["n"].sum() == 22
    assert table.index.tolist() == [f"cluster_{c}" for c in range(result.primary.k)]

    means = table["mean_silhouette"]
    assert means.idxmin() == "cluster_0"
    assert means.idxmax() == "cluster_1"
    assert means.loc["cluster_0"] < 0.0
    assert means.loc["cluster_1"] > 0.4

    # The nuclear-heavy segment has no member closer to another cluster than its own.
    assert table.loc["cluster_1", "members_below_zero"] == 0
    # The weakest segment does.
    assert table.loc["cluster_0", "members_below_zero"] == 3


def test_kmedoids_import_path_is_actually_importable():
    """Guards the real import path, which is what breaks - not the top-level package.

    ``import sklearn_extra`` succeeds even when ``sklearn_extra.cluster`` cannot be
    imported, so a shallow smoke test would miss the failure.
    """
    from sklearn_extra.cluster import KMedoids  # noqa: F401


def test_setuptools_declared_for_python_312():
    """scikit-learn-extra needs the distutils shim that setuptools provides on 3.12+.

    Without this, CI fails at collection on 3.12 with a confusing
    ``ModuleNotFoundError: No module named 'distutils'`` raised from inside a
    dependency. See the marker in pyproject.toml.
    """
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    dependencies = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
    assert any(
        dep.startswith("setuptools") and "python_version >= '3.12'" in dep
        for dep in dependencies
    ), (
        "setuptools must be declared for python>=3.12: scikit-learn-extra 0.3.0 imports "
        "distutils, removed from the stdlib in 3.12"
    )


def test_metrics_payload_is_json_serialisable(result):
    import json

    from utilities_clustering.cli import _metrics_payload

    text = json.dumps(to_serialisable(_metrics_payload(result, 0)))
    assert json.loads(text)["meta"]["k_selected"] == result.primary.k


def test_to_serialisable_handles_numpy(result):
    payload = to_serialisable(
        {"a": np.int64(3), "b": np.float64(1.5), "c": np.array([1, 2]), "d": np.True_}
    )
    assert payload == {"a": 3, "b": 1.5, "c": [1, 2], "d": True}
