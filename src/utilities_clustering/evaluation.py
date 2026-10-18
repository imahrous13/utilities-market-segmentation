"""Evaluation: dispersion, internal validity indices, agreement and stability.

Three independent questions are answered here, because no single one is sufficient:

1. Does the partition *fit*?           -> within-cluster sum of squares
2. Is it *well separated*?             -> silhouette, Calinski-Harabasz, Davies-Bouldin
3. Is it *robust and reproducible*?     -> subsample resampling, multi-seed refitting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)

from .clustering import Partition, kmedoids


def within_cluster_ss(data: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> dict:
    """Total and per-cluster squared distance to the nearest centre."""
    distances = cdist(data, centers, metric="euclidean")
    nearest_sq = distances.min(axis=1) ** 2
    per_cluster = {
        int(c): float(nearest_sq[labels == c].sum()) for c in sorted(set(labels.tolist()))
    }
    return {
        "total": float(nearest_sq.sum()),
        "per_cluster": per_cluster,
        "mean_per_cluster": float(nearest_sq.sum() / len(np.unique(labels))),
    }


def separation(data: np.ndarray, labels: np.ndarray) -> dict:
    """Internal validity indices, all of which should be read together.

    Silhouette ranges [-1, 1] and is scale-free; Calinski-Harabasz and Davies-Bouldin
    both assume roughly spherical, convex clusters, which is exactly the assumption
    K-Medoids on 8 dimensions does *not* guarantee. Disagreement between them is
    therefore information, not noise.
    """
    scores = silhouette_samples(data, labels, metric="euclidean")
    return {
        "silhouette": float(silhouette_score(data, labels, metric="euclidean")),
        "calinski_harabasz": float(calinski_harabasz_score(data, labels)),
        "davies_bouldin": float(davies_bouldin_score(data, labels)),
        "points_below_zero": int((scores < 0).sum()),
        "share_below_zero": float((scores < 0).mean()),
        "per_point": scores,
    }


def select_k(
    data: np.ndarray,
    labels: tuple[str, ...],
    k_min: int,
    k_max: int,
    seed: int,
) -> pd.DataFrame:
    """Sweep the cluster count and score every candidate on every criterion."""
    rows = []
    for k in range(k_min, k_max + 1):
        part = kmedoids(data, labels, k, seed)
        fit = within_cluster_ss(data, part.labels, part.centers)
        sep = separation(data, part.labels)
        sizes = part.sizes()
        rows.append(
            {
                "k": k,
                "silhouette": sep["silhouette"],
                "calinski_harabasz": sep["calinski_harabasz"],
                "davies_bouldin": sep["davies_bouldin"],
                "wcss_total": fit["total"],
                "wcss_mean": fit["mean_per_cluster"],
                "share_below_zero": sep["share_below_zero"],
                "min_cluster_size": int(sizes.min()),
                "n_singleton": int((sizes == 1).sum()),
            }
        )
    table = pd.DataFrame(rows)
    table["marginal_wcss_gain"] = table["wcss_total"].diff().abs().round(6)
    return table


def recommended_k(table: pd.DataFrame) -> dict:
    """Pick k by majority vote across the three indices.

    Silhouette is weighted as the tie-breaker because it is the only one of the three
    that is scale-free and does not assume cluster convexity.
    """
    votes = {}
    for column, mode in (
        ("silhouette", "max"),
        ("calinski_harabasz", "max"),
        ("davies_bouldin", "min"),
    ):
        idx = table[column].idxmax() if mode == "max" else table[column].idxmin()
        votes[column] = int(table.loc[idx, "k"])
    tally: dict[int, int] = {}
    for value in votes.values():
        tally[value] = tally.get(value, 0) + 1
    winner = max(sorted(tally), key=lambda k: tally[k])
    return {
        "votes": votes,
        "tally": tally,
        "recommended": int(winner),
        "silhouette_at_recommended": float(table.loc[table["k"] == winner, "silhouette"].iloc[0]),
    }


def agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """How much two partitions agree, independent of cluster numbering."""
    return {
        "ari": float(adjusted_rand_score(labels_a, labels_b)),
        "nmi": float(normalized_mutual_info_score(labels_a, labels_b)),
    }


def sensitivity_to_k(reference: Partition, candidates: dict[int, Partition]) -> pd.DataFrame:
    """Agreement of the selected solution with every other candidate k."""
    rows = []
    for k, part in sorted(candidates.items()):
        ari = agreement(reference.labels, part.labels)
        rows.append({"k": k, "is_reference": k == reference.k, **ari})
    return pd.DataFrame(rows)


def seed_sensitivity(
    data: np.ndarray, labels: tuple[str, ...], k: int, seeds, seed: int = 0
) -> dict:
    """Refit across seeds and check whether the partition changes at all.

    A strong result here means ``random_state`` is not load-bearing for this dataset,
    so the published segmentation does not rest on a lucky initialisation.
    """
    reference = kmedoids(data, labels, k, seed)
    ari = [agreement(reference.labels, kmedoids(data, labels, k, s).labels)["ari"] for s in seeds]
    return {
        "k": int(k),
        "reference_seed": int(seed),
        "seeds_tested": list(seeds),
        "min_ari": float(min(ari)),
        "mean_ari": float(np.mean(ari)),
        "seed_independent": bool(min(ari) > 0.999),
    }


def subsample_stability(
    data: np.ndarray, labels: tuple[str, ...], k: int, sizes, repeats: int, seed: int
) -> pd.DataFrame:
    """Refit on random subsets and score agreement with the full-data solution.

    This is the harshest honest test available: it measures how much of the
    segmentation is a property of the 22 companies versus of the particular sample.
    """
    reference = kmedoids(data, labels, k, seed)
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        if n > data.shape[0]:
            continue
        scores = []
        for _ in range(repeats):
            idx = rng.choice(data.shape[0], size=n, replace=False)
            subset_labels = tuple(labels[i] for i in idx)
            try:
                part = kmedoids(data[idx], subset_labels, k, seed)
            except Exception:
                continue
            scores.append(agreement(reference.labels[idx], part.labels)["ari"])
        scores = np.asarray(scores)
        rows.append(
            {
                "subsample_size": int(n),
                "repeats": len(scores),
                "mean_ari": float(scores.mean()),
                "median_ari": float(np.median(scores)),
                "min_ari": float(scores.min()),
                "share_ari_above_0_5": float((scores > 0.5).mean()),
            }
        )
    return pd.DataFrame(rows)


def kmeans_contrast(data: np.ndarray, labels: tuple[str, ...], k: int, seed: int) -> dict:
    """Quantify what K-Medoids buys over K-Means on this dataset.

    Two claims are checked rather than asserted: that K-Means centroids are not real
    observations, and that K-Means is seed-sensitive where K-Medoids is not.
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(data)
    distances = np.linalg.norm(data[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    seeds = range(1, 9)
    kmeans_aris = [
        adjusted_rand_score(
            km.labels_, KMeans(n_clusters=k, n_init=10, random_state=s).fit(data).labels_
        )
        for s in seeds
    ]
    return {
        "ari_vs_kmedoids": float(
            adjusted_rand_score(kmedoids(data, labels, k, seed).labels, km.labels_)
        ),
        "min_distance_point_to_own_centroid": float(np.median(distances.min(axis=1))),
        "centroid_is_observation": bool(np.isclose(distances.min(), 0.0, atol=1e-9).any()),
        "kmeans_seed_min_ari": float(min(kmeans_aris)),
    }


def pca_projection(data: np.ndarray, features: tuple[str, ...]) -> dict:
    """Two-component PCA, used purely as a 2-D rendering of an 8-D solution."""
    pca = PCA(n_components=2, random_state=0).fit(data)
    return {
        "coordinates": pca.transform(data),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": float(pca.explained_variance_ratio_[:2].sum()),
        "loadings": {f: [float(v) for v in row] for f, row in zip(features, pca.components_)},
    }


def cluster_profiles(raw: pd.DataFrame, labels: np.ndarray, features) -> pd.DataFrame:
    """Per-cluster raw means and their deviation from the dataset mean, in sd units.

    The z-score columns are the interpretable ones: they answer "how unusual is this
    cluster on this attribute" without the reader having to know the raw units, which
    the source data does not document.
    """
    frame = raw.copy()
    frame["cluster"] = labels
    grouped = frame.groupby("cluster")
    means = grouped[list(features)].mean()
    overall = raw[list(features)].mean()
    sd = raw[list(features)].std()
    deviation = (means - overall) / sd

    columns = (
        ["n_members"]
        + [f"mean_{c}" for c in features]
        + [f"z_{c}" for c in features]
    )
    return pd.concat(
        [grouped.size().rename("n_members"), means.add_prefix("mean_"),
         deviation.add_prefix("z_")],
        axis=1,
    ).reindex(columns=columns)
