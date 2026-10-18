"""End-to-end orchestration: run every stage and return one structured result.

Both the CLI runner and the notebook call :func:`run_analysis`, so the published
figures, tables and metrics can never drift from the code that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import clustering, evaluation
from .config import (
    FEATURES,
    K_MAX,
    K_MIN,
    K_PRIMARY,
    K_SENSITIVITY,
    LINKAGES,
    LINKAGES_PREFERRED,
    SEED,
    SEED_PROBE_RANGE,
    STABILITY_REPEATS,
    STABILITY_SAMPLE_SIZES,
)
from .data import load_and_prepare


@dataclass
class AnalysisResult:
    """Everything the analysis produces, in one object."""

    standardised: Any
    quality: dict
    cleaning: dict
    k_selection: pd.DataFrame
    recommendation: dict
    partitions: dict[int, clustering.Partition]
    primary: clustering.Partition
    sensitivity: clustering.Partition
    primary_fit: dict
    primary_separation: dict
    sensitivity_fit: dict
    profiles: pd.DataFrame
    sensitivity_profiles: pd.DataFrame
    k_agreement: pd.DataFrame
    linkage_matrices: dict[str, np.ndarray]
    linkage_cuts: dict[str, clustering.Partition]
    linkage_spans: dict[str, dict]
    linkage_agreement: pd.DataFrame
    primary_vs_linkage: pd.DataFrame
    seed_probe: dict
    stability: pd.DataFrame
    kmeans_contrast: dict
    pca: dict
    pairwise_agreement: pd.DataFrame = field(default_factory=pd.DataFrame)

    def segmentation_frame(self) -> pd.DataFrame:
        """One row per company: assignment, medoid, and nearest-centre distance."""
        distances = np.linalg.norm(
            self.standardised.values[:, None, :] - self.primary.centers[None, :, :], axis=2
        )
        nearest = distances.min(axis=1)
        return pd.DataFrame(
            {
                "company": self.standardised.labels,
                "cluster": self.primary.labels,
                "cluster_name": [f"Cluster {chr(65 + c)}" for c in self.primary.labels],
                "medoid": [self.primary.names[c] for c in self.primary.labels],
                "is_medoid": [
                    self.standardised.labels[i].strip() == self.primary.names[c]
                    for i, c in enumerate(self.primary.labels)
                ],
                "distance_to_medoid": nearest,
                "silhouette": self.primary_separation["per_point"],
            }
        )


    def segmentation_quality(self) -> pd.DataFrame:
        """Per-segment cohesion, so a strong average cannot hide a weak cluster.

        A single mean silhouette over the whole dataset will happily average a very tidy
        segment against a badly mixed one. This table is the diagnostic that stops that.
        """
        frame = self.segmentation_frame()
        table = frame.groupby("cluster").agg(
            n=("company", "size"),
            mean_silhouette=("silhouette", "mean"),
            min_silhouette=("silhouette", "min"),
            members_below_zero=("silhouette", lambda s: int((s < 0).sum())),
            mean_distance_to_medoid=("distance_to_medoid", "mean"),
        )
        table.index = [f"cluster_{c}" for c in table.index]
        return table.round(4)


def run_analysis(data_path, k_primary: int = K_PRIMARY, seed: int = SEED) -> AnalysisResult:
    """Run the complete analysis and return a populated :class:`AnalysisResult`."""
    standardised, cleaning, quality = load_and_prepare(data_path)
    values = standardised.values
    labels = standardised.labels

    # -- cluster-count selection -------------------------------------------------
    k_selection = evaluation.select_k(values, labels, K_MIN, K_MAX, seed)
    recommendation = evaluation.recommended_k(k_selection)

    # Every candidate partition is kept so agreement can be reported pairwise.
    partitions = {
        int(k): clustering.kmedoids(values, labels, int(k), seed)
        for k in range(K_MIN, K_MAX + 1)
    }

    primary_k = k_primary
    sensitivity_k = K_SENSITIVITY if K_SENSITIVITY in partitions else primary_k
    primary = partitions[primary_k]
    sensitivity = partitions[sensitivity_k]

    primary_fit = evaluation.within_cluster_ss(values, primary.labels, primary.centers)
    primary_separation = evaluation.separation(values, primary.labels)
    sensitivity_fit = evaluation.within_cluster_ss(values, sensitivity.labels, sensitivity.centers)

    profiles = evaluation.cluster_profiles(standardised.raw, primary.labels, FEATURES)
    sensitivity_profiles = evaluation.cluster_profiles(
        standardised.raw, sensitivity.labels, FEATURES
    )

    # -- agreement between candidate k ------------------------------------------
    k_agreement = evaluation.sensitivity_to_k(primary, partitions)

    pairwise_rows = []
    ks = sorted(partitions)
    for i, ki in enumerate(ks):
        for kj in ks[i + 1 :]:
            pairwise_rows.append(
                {
                    "k_a": ki,
                    "k_b": kj,
                    **evaluation.agreement(partitions[ki].labels, partitions[kj].labels),
                }
            )
    pairwise_agreement = pd.DataFrame(pairwise_rows)

    # -- hierarchical clustering ------------------------------------------------
    linkage_matrices: dict[str, np.ndarray] = {}
    linkage_cuts: dict[str, clustering.Partition] = {}
    linkage_spans: dict[str, dict] = {}
    for method in LINKAGES:
        matrix, cut = clustering.hierarchical(values, labels, method, cut_k=primary_k)
        linkage_matrices[method] = matrix
        linkage_cuts[method] = cut
        linkage_spans[method] = clustering.linkage_span(matrix)

    linkage_agreement = pd.DataFrame(
        [
            {
                "linkage": method,
                **clustering.linkage_span(linkage_matrices[method]),
                **evaluation.agreement(linkage_cuts[method].labels, primary.labels),
            }
            for method in LINKAGES
        ]
    )
    primary_vs_linkage = pd.DataFrame(
        [
            {"linkage": method, **evaluation.agreement(primary.labels, linkage_cuts[method].labels)}
            for method in LINKAGES
        ]
    )

    # Do the preferred linkages actually collapse onto the same partition?
    preferred = [m for m in LINKAGES_PREFERRED if m in linkage_cuts]
    cross_linkage = {}
    if len(preferred) >= 2:
        for other in preferred[1:]:
            cross_linkage[f"{preferred[0]} vs {other}"] = evaluation.agreement(
                linkage_cuts[preferred[0]].labels, linkage_cuts[other].labels
            )

    # -- robustness -------------------------------------------------------------
    seed_probe = evaluation.seed_sensitivity(values, labels, primary_k, SEED_PROBE_RANGE, seed)
    stability = evaluation.subsample_stability(
        values, labels, primary_k, STABILITY_SAMPLE_SIZES, STABILITY_REPEATS, seed
    )
    kmeans_contrast = evaluation.kmeans_contrast(values, labels, primary_k, seed)
    pca = evaluation.pca_projection(values, FEATURES)

    linkage_agreement.attrs["cross_linkage"] = cross_linkage

    return AnalysisResult(
        standardised=standardised,
        quality=quality,
        cleaning=cleaning,
        k_selection=k_selection,
        recommendation=recommendation,
        partitions=partitions,
        primary=primary,
        sensitivity=sensitivity,
        primary_fit=primary_fit,
        primary_separation=primary_separation,
        sensitivity_fit=sensitivity_fit,
        profiles=profiles,
        sensitivity_profiles=sensitivity_profiles,
        k_agreement=k_agreement,
        pairwise_agreement=pairwise_agreement,
        linkage_matrices=linkage_matrices,
        linkage_cuts=linkage_cuts,
        linkage_spans=linkage_spans,
        linkage_agreement=linkage_agreement,
        primary_vs_linkage=primary_vs_linkage,
        seed_probe=seed_probe,
        stability=stability,
        kmeans_contrast=kmeans_contrast,
        pca=pca,
    )


def to_serialisable(obj):
    """Best-effort conversion of numpy/pandas scalars into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, np.ndarray):
        return to_serialisable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return to_serialisable(obj.to_dict(orient="records"))
    if obj is None or isinstance(obj, (int, float, str)):
        return obj
    return str(obj)
