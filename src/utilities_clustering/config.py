"""Paths, constants and shared configuration for the utilities segmentation analysis."""

from __future__ import annotations

from pathlib import Path

# src/utilities_clustering/config.py -> src/utilities_clustering -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = REPO_ROOT / "data" / "Utilities.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
SEGMENTATION_PATH = OUTPUT_DIR / "segmentation.csv"
SELECTION_PATH = OUTPUT_DIR / "k_selection.csv"
PROFILES_PATH = OUTPUT_DIR / "cluster_profiles.csv"

#: Identifier column; becomes the DataFrame index so clusters are named, not numbered.
ID_COLUMN = "company"

#: Modelling attributes, in the order they appear in the source file.
FEATURES = (
    "fixed_charge",
    "ror",
    "cost",
    "load_factor",
    "demand_growth",
    "sales",
    "nuclear",
    "fuel_cost",
)

#: Master seed. Every stochastic step in the pipeline is derived from this.
SEED = 0

#: Cluster counts evaluated by the k-selection sweep.
K_MIN = 2
K_MAX = 8

#: Selected by silhouette + Calinski-Harabasz (see README "Choosing k").
K_PRIMARY = 5

#: The k originally reported in the submitted report, retained as a sensitivity check.
K_SENSITIVITY = 6

#: Linkage criteria compared. ``single`` is included deliberately: it demonstrates the
#: chaining pathology that motivates preferring ``average``.
LINKAGES = ("single", "complete", "average", "ward")

#: Linkages that are expected to agree; used to quantify agreement with K-Medoids.
LINKAGES_PREFERRED = ("complete", "average", "ward")

#: Number of random starts used to probe seed sensitivity.
SEED_PROBE_RANGE = tuple(range(1, 9))

#: Resampling sizes (as absolute counts) and repeats for the stability probe.
STABILITY_SAMPLE_SIZES = (16, 18, 20)
STABILITY_REPEATS = 200

#: Colour-blind-safe qualitative palette, used consistently across every figure.
PALETTE = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

CLUSTER_LABELS = tuple("ABCDEFGH")
