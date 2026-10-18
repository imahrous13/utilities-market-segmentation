"""Tests for the clustering estimators and the invariants that define them."""

from __future__ import annotations

import numpy as np
import pytest

from utilities_clustering import (
    K_PRIMARY,
    SEED,
    agreement,
    hierarchical,
    kmedoids,
    load_and_prepare,
    run_analysis,
    within_cluster_ss,
)
from utilities_clustering.clustering import canonicalise, linkage_span
from utilities_clustering.config import DATA_PATH, LINKAGES


@pytest.fixture(scope="module")
def standardised():
    return load_and_prepare(DATA_PATH)[0]


@pytest.fixture(scope="module")
def result():
    return run_analysis(DATA_PATH)


def test_kmedoids_returns_requested_cluster_count(standardised):
    part = kmedoids(standardised.values, standardised.labels, K_PRIMARY, SEED)
    assert part.k == K_PRIMARY
    assert set(np.unique(part.labels)) == set(range(K_PRIMARY))
    assert part.sizes().sum() == standardised.n


def test_every_medoid_is_a_real_observation(standardised):
    """The defining property of K-Medoids: centres cannot be invented."""
    values = standardised.values
    labels = {n.strip() for n in standardised.labels}
    for k in range(2, 8):
        part = kmedoids(values, standardised.labels, k, SEED)
        for name, center in zip(part.names, part.centers):
            assert name in labels
            assert np.isclose(center, values[standardised.labels.index(
                next(n for n in standardised.labels if n.strip() == name)
            )]).all()


def test_kmedoids_is_seed_independent(standardised):
    """Verified empirically across seeds; the published result must not rest on luck."""
    reference = kmedoids(standardised.values, standardised.labels, K_PRIMARY, SEED)
    for seed in range(1, 9):
        other = kmedoids(standardised.values, standardised.labels, K_PRIMARY, seed)
        assert agreement(reference.labels, other.labels)["ari"] > 0.999


def test_canonical_labelling_is_stable_under_relabelling(standardised):
    """Applying a permutation to cluster ids must not change the canonical result."""
    part = kmedoids(standardised.values, standardised.labels, K_PRIMARY, SEED)
    permutation = list(range(1, part.k)) + [0]  # cluster i is relabelled permutation[i]

    permuted_labels = np.empty_like(part.labels)
    for old, new in enumerate(permutation):
        permuted_labels[part.labels == old] = new
    # The representative name and its centre must both travel with the cluster.
    inverse = [permutation.index(i) for i in range(part.k)]
    permuted_names = [part.names[inverse[j]] for j in range(part.k)]
    permuted_centers = part.centers[inverse]

    recanonicalised = canonicalise(
        permuted_labels, permuted_names, permuted_centers, part.method
    )
    assert recanonicalised.names == part.names
    assert np.array_equal(recanonicalised.labels, part.labels)
    assert np.allclose(recanonicalised.centers, part.centers)


def test_canonical_names_are_alphabetically_ordered(standardised):
    part = kmedoids(standardised.values, standardised.labels, K_PRIMARY, SEED)
    assert list(part.names) == sorted(part.names, key=str.lower)


def test_canonical_names_are_unique(standardised):
    part = kmedoids(standardised.values, standardised.labels, K_PRIMARY, SEED)
    assert len(set(part.names)) == part.k


@pytest.mark.parametrize("method", LINKAGES)
def test_hierarchical_cut_yields_requested_groups(standardised, method):
    _, cut = hierarchical(standardised.values, standardised.labels, method, cut_k=K_PRIMARY)
    assert len(set(cut.labels.tolist())) == K_PRIMARY
    assert cut.sizes().sum() == standardised.n


@pytest.mark.parametrize("method", LINKAGES)
def test_hierarchical_group_names_are_real_observations(standardised, method):
    _, cut = hierarchical(standardised.values, standardised.labels, method, cut_k=K_PRIMARY)
    labels = {n.strip() for n in standardised.labels}
    for name in cut.names:
        assert name.strip() in labels


def test_unknown_linkage_is_rejected(standardised):
    with pytest.raises(ValueError, match="unsupported linkage"):
        hierarchical(standardised.values, standardised.labels, "not-a-linkage", cut_k=3)


def test_single_linkage_keeps_merging_longest(standardised):
    """The chaining signature: single linkage is still merging when the others stop.

    Its final merge height is the lowest of the four criteria, which means it has not
    found a point at which the groups are clearly separated - exactly the failure mode
    that makes its flat cut unreliable.
    """
    spans = {
        method: linkage_span(
            hierarchical(standardised.values, standardised.labels, method, cut_k=2)[0]
        )
        for method in LINKAGES
    }
    single = spans["single"]["final_merge_height"]
    assert single == min(s["final_merge_height"] for s in spans.values())
    assert single < spans["average"]["final_merge_height"]
    assert single < spans["ward"]["final_merge_height"]
    # Every criterion merges the same closest pair first.
    firsts = {s["first_merge_height"] for s in spans.values()}
    assert len(firsts) == 1


def test_ward_requires_euclidean_inputs(standardised):
    """Ward is only valid on Euclidean data; the guard should be explicit."""
    matrix, _ = hierarchical(standardised.values, standardised.labels, "ward", cut_k=K_PRIMARY)
    assert matrix.shape == (standardised.n - 1, 4)


def test_within_cluster_ss_matches_manual_computation(standardised):
    part = kmedoids(standardised.values, standardised.labels, K_PRIMARY, SEED)
    fit = within_cluster_ss(standardised.values, part.labels, part.centers)
    manual = 0.0
    for c in range(part.k):
        rows = standardised.values[part.labels == c]
        manual += ((rows - part.centers[c]) ** 2).sum()
    assert np.isclose(fit["total"], manual)


def test_within_cluster_ss_per_cluster_sums_to_total(standardised):
    part = kmedoids(standardised.values, standardised.labels, K_PRIMARY, SEED)
    fit = within_cluster_ss(standardised.values, part.labels, part.centers)
    assert np.isclose(sum(fit["per_cluster"].values()), fit["total"])
    assert set(fit["per_cluster"]) == set(range(part.k))


def test_wcss_decreases_monotonically_in_k(standardised, result):
    """Monotone WCSS is forced by the objective, so it cannot justify k on its own."""
    wcss = result.k_selection["wcss_total"].to_numpy()
    assert np.all(np.diff(wcss) < 0)
