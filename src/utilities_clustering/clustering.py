"""Clustering estimators: K-Medoids (PAM) and agglomerative hierarchical clustering.

Both operate on the same z-scored matrix so their outputs are directly comparable.
Cluster ids produced by both methods are canonicalised (see :func:`canonicalise`)
because raw estimator labels are arbitrary and would otherwise make every reported
result depend on internal ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn_extra.cluster import KMedoids


@dataclass(frozen=True)
class Partition:
    """A labelling of the observations with the centres that generated it."""

    labels: np.ndarray
    names: tuple[str, ...]
    centers: np.ndarray
    #: ``True`` when every centre is an actual observation rather than a computed centroid.
    centers_are_observations: bool
    method: str

    @property
    def k(self) -> int:
        return len(self.names)

    def sizes(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=self.k)


def canonicalise(
    labels: np.ndarray, names: list[str], centers: np.ndarray, method: str
) -> Partition:
    """Relabel clusters so that cluster *i* is always the one whose representative is
    ``names``-sorted at position *i*.

    Estimator labels carry no meaning beyond identity, and both ``KMedoids`` and
    ``fcluster`` are free to renumber between versions or runs. Sorting by the
    representative's name gives a stable, human-readable ordering that makes the
    committed results directly comparable with a fresh run.
    """
    names = list(names)
    order = np.array(sorted(range(len(names)), key=lambda i: names[i].lower()))
    remap = {old: new for new, old in enumerate(order)}
    relabelled = np.array([remap[int(label)] for label in labels], dtype=int)
    return Partition(
        labels=relabelled,
        names=tuple(names[i] for i in order),
        centers=np.asarray(centers)[order],
        centers_are_observations=True,
        method=method,
    )


def kmedoids(data: np.ndarray, labels: tuple[str, ...], k: int, seed: int) -> Partition:
    """Partition with K-Medoids (PAM), assigning each row to its nearest medoid.

    K-Medoids is preferred over K-Means here because the medoid must be an actual
    company, so every segment is anchored to a real, nameable operator.
    """
    model = KMedoids(n_clusters=k, random_state=seed).fit(data)
    medoid_names = [labels[i].strip() for i in model.medoid_indices_]
    return canonicalise(model.labels_, medoid_names, model.cluster_centers_, f"kmedoids(k={k})")


def hierarchical(
    data: np.ndarray, labels: tuple[str, ...], method: str, cut_k: int
) -> tuple[np.ndarray, Partition]:
    """Agglomerative clustering under one linkage criterion, cut into ``cut_k`` groups.

    Returns the raw SciPy linkage matrix (for dendrogram plotting) and the canonical
    flat partition.

    The cut is taken with ``criterion="maxclust"`` rather than by reading colours off a
    dendrogram threshold. A colour threshold is a rendering choice, not a partition, and
    its implied grouping changes between SciPy releases.
    """
    if method not in {"single", "complete", "average", "weighted", "centroid", "ward"}:
        raise ValueError(f"unsupported linkage: {method!r}")

    linkage_matrix = linkage(data, method=method)
    raw = fcluster(linkage_matrix, t=cut_k, criterion="maxclust") - 1

    groups = sorted(set(raw.tolist()))
    # Name each cut group after the alphabetically first member: hierarchical clustering
    # yields no centres, so the member stands in as the group's representative. The
    # centroid stored alongside is the group mean, used only for dispersion bookkeeping.
    group_names = [min(labels[i] for i in np.flatnonzero(raw == g)) for g in groups]
    centers = np.vstack([data[np.flatnonzero(raw == g)].mean(axis=0) for g in groups])
    partition = canonicalise(raw, group_names, centers, f"hierarchical({method}, k={cut_k})")
    return linkage_matrix, partition


def linkage_span(linkage_matrix: np.ndarray) -> dict:
    """Summary of where merges happen, which is what makes chaining visible."""
    heights = linkage_matrix[:, 2]
    return {
        "first_merge_height": float(heights.min()),
        "final_merge_height": float(heights.max()),
        "n_single_step_merges": int(np.sum(np.diff(np.sort(heights)) < 1e-9)),
    }
