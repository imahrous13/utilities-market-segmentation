"""Unsupervised segmentation of utility companies by operating profile.

Public surface:

    from utilities_clustering import load_and_prepare, kmedoids, run_analysis

``run_analysis`` executes the whole study and returns an :class:`AnalysisResult`
holding every table, metric and figure input the write-up needs.
"""

from __future__ import annotations

from .clustering import Partition, hierarchical, kmedoids, linkage_span
from .config import FEATURES, K_PRIMARY, SEED
from .data import (
    StandardisedData,
    clean,
    load_and_prepare,
    load_raw,
    quality_report,
    standardize,
)
from .evaluation import (
    agreement,
    cluster_profiles,
    kmeans_contrast,
    pca_projection,
    recommended_k,
    seed_sensitivity,
    select_k,
    separation,
    subsample_stability,
    within_cluster_ss,
)
from .pipeline import AnalysisResult, run_analysis, to_serialisable

__version__ = "1.0.0"

__all__ = [
    "AnalysisResult",
    "FEATURES",
    "K_PRIMARY",
    "Partition",
    "SEED",
    "StandardisedData",
    "__version__",
    "agreement",
    "clean",
    "cluster_profiles",
    "hierarchical",
    "kmedoids",
    "kmeans_contrast",
    "linkage_span",
    "load_and_prepare",
    "load_raw",
    "pca_projection",
    "quality_report",
    "recommended_k",
    "run_analysis",
    "seed_sensitivity",
    "select_k",
    "separation",
    "standardize",
    "subsample_stability",
    "to_serialisable",
    "within_cluster_ss",
]
