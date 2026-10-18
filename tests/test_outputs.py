"""Tests that the CLI and figure generation actually produce the committed artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utilities_clustering import run_analysis
from utilities_clustering.cli import main
from utilities_clustering.config import DATA_PATH
from utilities_clustering.plotting import render_all

EXPECTED_FIGURES = {
    "01_attribute_distributions.png",
    "02_raw_distance_decomposition.png",
    "03_outlier_screen.png",
    "04_neighbour_isolation.png",
    "05_k_selection_criteria.png",
    "06_wcss_elbow.png",
    "07_dendrograms.png",
    "08_linkage_agreement.png",
    "09_pca_projection.png",
    "10_per_point_silhouette.png",
    "11_kmedoids_vs_kmeans.png",
    "12_cluster_profile_heatmap.png",
    "13_k_selection_contrast.png",
    "14_subsample_stability.png",
    "15_k_agreement_matrix.png",
}

EXPECTED_ARTEFACTS = {
    "metrics.json",
    "k_selection.csv",
    "segmentation.csv",
    "cluster_profiles.csv",
    "linkage_comparison.csv",
    "stability.csv",
}


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("outputs")
    assert main(["--data", str(DATA_PATH), "--output", str(out), "--quick"]) == 0
    return out


def test_cli_writes_every_expected_artefact(cli_run):
    produced = {p.name for p in cli_run.iterdir() if p.is_file()}
    assert EXPECTED_ARTEFACTS <= produced


def test_cli_writes_every_expected_figure(cli_run):
    figures = cli_run / "figures"
    assert figures.is_dir()
    produced = {p.name for p in figures.iterdir() if p.is_file()}
    assert produced == EXPECTED_FIGURES


def test_figures_are_non_trivial_files(cli_run):
    """Guards against a silent all-blank render."""
    for path in (cli_run / "figures").iterdir():
        assert path.stat().st_size > 5_000, f"{path.name} is suspiciously small"


def test_metrics_json_is_valid_and_complete(cli_run):
    payload = json.loads((cli_run / "metrics.json").read_text(encoding="utf-8"))
    assert payload["meta"]["n_observations"] == 22
    assert payload["meta"]["k_selected"] == 5
    assert payload["k_recommendation"]["recommended"] == 5
    assert set(payload["primary"]["members"]) == set(payload["primary"]["names"])
    assert sum(payload["primary"]["sizes"]) == 22
    assert len(payload["primary"]["members"]) == 5
    assert payload["seed_sensitivity"]["seed_independent"] is True
    assert len(payload["linkages"]) == 4


def test_segmentation_csv_covers_every_company(cli_run):
    import csv

    with (cli_run / "segmentation.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 22
    assert {r["company"].strip() for r in rows} == {
        "Arizona", "Boston", "Central", "Commonwealth", "NY", "Florida", "Hawaiian",
        "Idaho", "Kentucky", "Madison", "Nevada", "New England", "Northern", "Oklahoma",
        "Pacific", "Puget", "San Diego", "Southern", "Texas", "Wisconsin", "United",
        "Virginia",
    }
    assert sum(1 for r in rows if r["is_medoid"] == "True") == 5


def test_k_selection_csv_matches_metrics(cli_run):
    import csv

    with (cli_run / "k_selection.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["k"]) for r in rows] == list(range(2, 9))
    metrics = json.loads((cli_run / "metrics.json").read_text(encoding="utf-8"))
    best = max(rows, key=lambda r: float(r["silhouette"]))
    assert int(best["k"]) == metrics["meta"]["k_selected"]


def test_render_all_returns_written_paths(tmp_path):
    result = run_analysis(DATA_PATH)
    written = render_all(result, tmp_path / "figures")
    assert set(written) >= {
        "exploration/outlier_screen",
        "k_selection/criteria",
        "dendrograms/dendrograms",
        "projection/projection",
        "profiles/heatmap",
        "stability/stability",
    }
    for path in written.values():
        assert Path(path).is_file()


def test_no_figure_uses_a_hardcoded_dataset_path():
    """Figures must come from the result object, never from a file on disk."""
    from utilities_clustering import plotting

    source = Path(plotting.__file__).read_text(encoding="utf-8")
    assert "read_csv" not in source
    assert "Utilities.csv" not in source


def test_pipeline_is_deterministic_across_runs():
    first = run_analysis(DATA_PATH, k_primary=5, seed=0)
    second = run_analysis(DATA_PATH, k_primary=5, seed=0)
    assert np.array_equal(first.primary.labels, second.primary.labels)
    assert first.primary.names == second.primary.names
    assert first.primary_fit["total"] == second.primary_fit["total"]
    assert first.primary_separation["silhouette"] == second.primary_separation["silhouette"]


def test_reported_silhouette_is_stable_across_runs():
    values = {
        run_analysis(DATA_PATH, k_primary=5, seed=0).primary_separation["silhouette"]
        for _ in range(2)
    }
    assert len(values) == 1
    assert 0.14 < values.pop() < 0.15
