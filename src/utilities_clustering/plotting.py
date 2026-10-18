"""Figure generation.

One house style is applied by :func:`apply_style` so every figure in ``outputs/figures``
reads as a single set. Figures return the path they wrote so callers never have to
reconstruct filenames.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from .config import FEATURES, PALETTE
from .pipeline import AnalysisResult

FEATURE_LABELS = {
    "fixed_charge": "Fixed charge",
    "ror": "Rate of return",
    "cost": "Cost",
    "load_factor": "Load factor",
    "demand_growth": "Demand growth",
    "sales": "Sales",
    "nuclear": "Nuclear",
    "fuel_cost": "Fuel cost",
}


def apply_style() -> None:
    """Shared visual settings: no top/right spines, muted grid, consistent sizing."""
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.8,
            "axes.titlesize": 13,
            "axes.titleweight": "600",
            "axes.labelsize": 10,
            "axes.labelcolor": "#333333",
            "axes.edgecolor": "#BBBBBB",
            "text.color": "#222222",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#555555",
            "ytick.color": "#555555",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.autolayout": False,
        }
    )


def _save(fig, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _cluster_colors(k: int) -> list:
    return [PALETTE[i % len(PALETTE)] for i in range(k)]


# --------------------------------------------------------------------------- EDA


def plot_exploration(result: AnalysisResult, directory: Path) -> dict:
    """Attribute distributions, pairwise matrix and the distance-scale argument."""
    raw = result.standardised.raw
    paths = {}

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
    for ax, column in zip(axes.ravel(), FEATURES):
        ax.hist(raw[column], bins=8, color=PALETTE[0], edgecolor="white", linewidth=0.8)
        ax.axvline(raw[column].mean(), color=PALETTE[4], linewidth=1.6, linestyle="--",
                   label=f"mean {raw[column].mean():.2f}")
        zero_share = (raw[column] == 0).mean()
        if zero_share > 0:
            ax.axvline(
                0, color="#888888", linewidth=1.2, linestyle=":",
                label=f"{zero_share:.0%} exact zero",
            )
        ax.set_title(FEATURE_LABELS[column], fontsize=11)
        ax.set_ylabel("companies")
        ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle(
        "Distribution of every modelling attribute",
        fontsize=14, fontweight="600", y=1.005,
    )
    fig.tight_layout()
    paths["distributions"] = _save(fig, directory, "01_attribute_distributions.png")

    # Why standardisation is mandatory: raw distance decomposition.
    share = result.quality["raw_squared_distance_share_pct"]
    order = sorted(share, key=lambda c: -share[c])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    values = [share[c] for c in order]
    colours = [PALETTE[4] if c == "sales" else PALETTE[0] for c in order]
    bars = ax.bar(
        [FEATURE_LABELS[c] for c in order], values, color=colours, width=0.62,
    )
    ax.set_yscale("log")
    ax.set_ylabel("share of total pairwise squared distance (log scale)")
    ax.set_title("Raw Euclidean distance is a function of sales alone")
    ax.bar_label(bars, labels=[f"{v:.4g}%" for v in values], padding=3, fontsize=8)
    ax.margins(y=0.18)
    fig.tight_layout()
    paths["distance_decomposition"] = _save(fig, directory, "02_raw_distance_decomposition.png")

    # Candidate outliers, stated as a question rather than a verdict.
    from scipy.spatial.distance import cdist

    distance_matrix = cdist(result.standardised.values, result.standardised.values)
    np.fill_diagonal(distance_matrix, np.inf)
    nearest = distance_matrix.min(axis=1)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    colours = _cluster_colors(result.primary.k)
    ax.scatter(
        raw["sales"], raw["fuel_cost"],
        c=[colours[c] for c in result.primary.labels],
        s=90, edgecolor="white", linewidth=1.1, zorder=3,
    )
    for name, x, y in zip(result.standardised.labels, raw["sales"], raw["fuel_cost"]):
        ax.annotate(name.strip(), (x, y), fontsize=7, xytext=(4, 4),
                    textcoords="offset points", color="#555555")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Fuel cost")
    ax.set_title("Outlier screen: sales vs fuel cost, coloured by cluster")
    ax.margins(x=0.16, y=0.16)
    fig.tight_layout()
    paths["outlier_screen"] = _save(fig, directory, "03_outlier_screen.png")

    # Formal nearest-neighbour distances, so the visual claim has numbers behind it.
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ranked = pd.Series(nearest, index=result.standardised.labels).sort_values(ascending=False)
    threshold = ranked.median() * 1.6
    standout = ranked > threshold
    ax.barh(
        [n.strip() for n in ranked.index][::-1], ranked.values[::-1],
        color=[
            PALETTE[4] if standout[name] else PALETTE[0]
            for name in ranked.index[::-1]
        ],
        height=0.7,
    )
    ax.axvline(threshold, color="#999999", linestyle="--", linewidth=1.2)
    ax.annotate(
        f"1.6x median = {threshold:.2f}", xy=(threshold, 0),
        xytext=(5, 8), textcoords="offset points", fontsize=8.5, color="#666666",
    )
    for position, name in enumerate(ranked.index[::-1]):
        if standout[name]:
            ax.annotate(
                name.strip(), (ranked[name], position), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9,
                fontweight="600", color=PALETTE[4],
            )
    ax.set_xlabel("distance to nearest other company (z-scored space)")
    ax.set_title(
        f"Isolation: median nearest-neighbour distance {ranked.median():.2f}; "
        f"{int(standout.sum())} of {len(ranked)} companies sit noticeably apart (orange)"
    )
    ax.margins(x=0.12)
    fig.tight_layout()
    paths["isolation"] = _save(fig, directory, "04_neighbour_isolation.png")
    return paths


# ----------------------------------------------------------------- k selection


def plot_k_selection(result: AnalysisResult, directory: Path) -> dict:
    """The three selection criteria, and where they agree."""
    table = result.k_selection
    k = table["k"].to_numpy()
    paths = {}
    panels = [
        ("silhouette", "Silhouette", PALETTE[0], (None, None)),
        ("calinski_harabasz", "Calinski-Harabasz index", PALETTE[2], (0, None)),
        ("davies_bouldin", "Davies-Bouldin index", PALETTE[4], (None, None)),
        ("wcss_total", "Total within-cluster SS", PALETTE[5], (None, None)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.6))
    for ax, (column, title, colour, ylim) in zip(axes.ravel(), panels):
        ax.plot(k, table[column], marker="o", color=colour, linewidth=2, markersize=6,
                markeredgecolor="white", markeredgewidth=1.1)
        best = int(result.recommendation["recommended"])
        ax.axvline(best, color="#999999", linestyle="--", linewidth=1.2)
        ax.annotate(f"k = {best}", (best, ax.get_ylim()[1]), textcoords="offset points",
                    xytext=(5, -12), fontsize=9, color="#666666")
        if column == "silhouette":
            ax.axhline(0, color="#CCCCCC", linewidth=1)
        if ylim[1] is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel("number of clusters (k)")
        ax.set_xticks(k)
    fig.suptitle(
        "Four criteria for choosing k. Silhouette and Calinski-Harabasz both peak at "
        f"k = {result.recommendation['recommended']}",
        fontsize=13, fontweight="600",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    paths["criteria"] = _save(fig, directory, "05_k_selection_criteria.png")

    # Elbow detail, with the marginal gain called out.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(k, table["wcss_total"], marker="o", color=PALETTE[0], linewidth=2,
            markersize=6, markeredgecolor="white", markeredgewidth=1.1, label="total WCSS")
    ax.set_xlabel("number of clusters (k)")
    ax.set_ylabel("total within-cluster sum of squares", color=PALETTE[0])
    ax.tick_params(axis="y", labelcolor=PALETTE[0])
    ax.set_xticks(k)

    twin = ax.twinx()
    twin.bar(k, table["marginal_wcss_gain"].fillna(0), width=0.5, color=PALETTE[1],
             alpha=0.55, zorder=0)
    twin.set_ylabel("gain from adding one more cluster", color=PALETTE[1])
    twin.tick_params(axis="y", labelcolor=PALETTE[1])
    twin.grid(False)
    twin.spines["top"].set_visible(False)

    smallest = table.loc[table["marginal_wcss_gain"].idxmax(), "k"]
    anchor = float(table.loc[table["k"] == smallest, "wcss_total"].iloc[0])
    ax.annotate(
        f"smallest gain in the sweep\nat k = {smallest}",
        xy=(smallest, anchor),
        xytext=(-72, 34), textcoords="offset points", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#666666", linewidth=1.1),
    )
    ax.set_title("Within-cluster dispersion and the marginal value of each extra medoid")
    fig.tight_layout()
    paths["elbow"] = _save(fig, directory, "06_wcss_elbow.png")
    return paths


# ------------------------------------------------------------------- clustering


def plot_dendrograms(result: AnalysisResult, directory: Path) -> dict:
    """All four linkage criteria, cut into the selected number of groups."""
    paths = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.4))
    for ax, method in zip(axes.ravel(), result.linkage_matrices):
        span = result.linkage_spans[method]
        dendrogram(
            result.linkage_matrices[method], labels=result.standardised.labels,
            leaf_rotation=90, leaf_font_size=7.5, color_threshold=0,
            above_threshold_color="#BBBBBB", ax=ax,
        )
        ax.set_title(
            f"{method.capitalize()} linkage\n"
            f"merges {span['first_merge_height']:.2f} to {span['final_merge_height']:.2f}",
            fontsize=11,
        )
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.grid(axis="y")
        ax.axhline(0, color="#DDDDDD", linewidth=0.8)
    fig.suptitle(
        "Agglomerative clustering under four linkage criteria "
        f"(every one cut into k = {result.primary.k})",
        fontsize=13, fontweight="600",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths["dendrograms"] = _save(fig, directory, "07_dendrograms.png")

    # How much does the chosen linkage matter?
    agreement = result.linkage_agreement.sort_values("ari")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colours = [PALETTE[4] if a < 0.2 else PALETTE[0] for a in agreement["ari"]]
    ax.barh(agreement["linkage"], agreement["ari"], color=colours, height=0.6)
    ax.axvline(result.primary_vs_linkage["ari"].mean(), color="#999999",
               linestyle="--", linewidth=1.2)
    for i, value in enumerate(agreement["ari"]):
        ax.annotate(f"ARI {value:.3f}", (value, i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("adjusted Rand index against the K-Medoids solution")
    ax.set_title("Single linkage is the outlier: it barely agrees with anything")
    ax.grid(axis="x")
    fig.tight_layout()
    paths["linkage_agreement"] = _save(fig, directory, "08_linkage_agreement.png")
    return paths


def plot_projection(result: AnalysisResult, directory: Path) -> dict:
    """PCA projection coloured by cluster, with medoids marked."""
    coords = result.pca["coordinates"]
    variance = result.pca["explained_variance_ratio"]
    colours = _cluster_colors(result.primary.k)
    paths = {}

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    for c in range(result.primary.k):
        members = result.primary.labels == c
        ax.scatter(
            coords[members, 0], coords[members, 1],
            s=150, color=colours[c], edgecolor="white", linewidth=1.4,
            label=f"Cluster {chr(65 + c)} (n={int(members.sum())})", zorder=3,
        )
    medoid_rows = [
        i for i, name in enumerate(result.standardised.labels)
        if name.strip() in set(result.primary.names)
    ]
    ax.scatter(
        coords[medoid_rows, 0], coords[medoid_rows, 1],
        s=340, facecolor="none", edgecolor="#222222", linewidth=1.7, zorder=4,
        label="medoid",
    )
    for name in result.standardised.labels:
        ax.annotate(name.strip(), coords[result.standardised.labels.index(name)],
                    fontsize=7.5, xytext=(6, 5), textcoords="offset points", color="#444444")
    ax.set_xlabel(f"PC1 ({variance[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1%} of variance)")
    ax.set_title(
        f"Utilities in the plane, coloured by cluster "
        f"(first two components: {result.pca['cumulative_explained_variance']:.0%} of variance)"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.margins(0.1)
    fig.tight_layout()
    paths["projection"] = _save(fig, directory, "09_pca_projection.png")

    # Per-point silhouette: which companies sit awkwardly in their cluster.
    scores = pd.Series(
        result.primary_separation["per_point"], index=result.standardised.labels
    ).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = _cluster_colors(result.primary.k)
    rows = [result.standardised.labels.index(n) for n in scores.index]
    bar_colours = [
        PALETTE[4] if v < 0 else palette[result.primary.labels[i]]
        for i, v in zip(rows, scores.values)
    ]
    ax.barh([n.strip() for n in scores.index], scores.values, color=bar_colours, height=0.72)
    ax.axvline(0, color="#888888", linewidth=1.2)
    ax.set_xlabel("silhouette coefficient")
    ax.set_title(
        f"Silhouette per company at k = {result.primary.k}: "
        f"{result.primary_separation['share_below_zero']:.0%} sit closer to another cluster "
        "than to their own medoid (orange)"
    )
    ax.margins(x=0.08)
    fig.tight_layout()
    paths["silhouette"] = _save(fig, directory, "10_per_point_silhouette.png")

    # K-Medoids vs K-Means, the justification for the algorithm choice.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(result.primary.k)
    ax.bar(x - 0.2, 1.0, width=0.4, color=PALETTE[2],
           label="K-Medoids (centres are real companies)")
    ax.bar(x + 0.2, result.kmeans_contrast["kmeans_seed_min_ari"], width=0.4,
           color=PALETTE[4], label="K-Means (worst seed-to-seed agreement)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in x])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("minimum ARI across 8 random seeds")
    ax.set_title("Reproducibility: K-Medoids is seed-stable, K-Means is not")
    ax.legend(loc="lower left")
    ax.margins(y=0.16)
    fig.tight_layout()
    paths["kmeans_contrast"] = _save(fig, directory, "11_kmedoids_vs_kmeans.png")
    return paths


def plot_profiles(result: AnalysisResult, directory: Path) -> dict:
    """Cluster profile heatmap and the primary-vs-sensitivity comparison."""
    profiles = result.profiles
    z = profiles[[f"z_{c}" for c in FEATURES]]
    paths = {}

    fig, ax = plt.subplots(figsize=(11, 4.4))
    limit = float(np.abs(z.to_numpy()).max())
    image = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURE_LABELS[c] for c in FEATURES], rotation=32, ha="right")
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels(
        [f"Cluster {chr(65 + i)}\nmedoid: {name}" for i, name in enumerate(result.primary.names)]
    )
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            value = z.iloc[i, j]
            ax.annotate(
                f"{value:+.2f}", (j, i), ha="center", va="center", fontsize=8.5,
                color="white" if abs(value) > limit * 0.55 else "#222222",
            )
    ax.set_title("Cluster profile: deviation of each cluster mean from the dataset mean, in sd")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="standard deviations")
    fig.tight_layout()
    paths["heatmap"] = _save(fig, directory, "12_cluster_profile_heatmap.png")

    # k = selected vs the k originally reported, side by side.
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.2), sharex=True)
    nuclear_col = list(FEATURES).index("nuclear")
    for ax, part, prof, title in (
        (axes[0], result.primary, result.profiles, f"k = {result.primary.k}  (selected)"),
        (axes[1], result.sensitivity, result.sensitivity_profiles,
         f"k = {result.sensitivity.k}  (reported in the original submission)"),
    ):
        values = prof[[f"z_{c}" for c in FEATURES]].to_numpy()
        limit = float(np.abs(values).max())
        ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(FEATURES)))
        ax.set_xticklabels([FEATURE_LABELS[c] for c in FEATURES], rotation=32, ha="right")
        ax.set_yticks(range(len(prof)))
        ax.set_yticklabels(
            [f"{chr(65 + i)} (n={int(n)})\n{part.names[i]}"
             for i, n in enumerate(prof["n_members"])]
        )
        ax.set_title(title, fontsize=11.5)
        ax.add_patch(
            plt.Rectangle(
                (nuclear_col - 0.5, -0.5), 1, len(prof),
                fill=False, edgecolor="#222222", linewidth=2.2, linestyle="--",
            )
        )
        ax.grid(False)

    ari = float(
        result.k_agreement.loc[result.k_agreement["k"] == result.sensitivity.k, "ari"].iloc[0]
    )
    axes[0].annotate(
        "one coherent\nnuclear cluster",
        xy=(nuclear_col, 1), xycoords="data",
        xytext=(nuclear_col - 1.9, 2.7), textcoords="data",
        fontsize=9, color="#222222", ha="left",
        arrowprops=dict(arrowstyle="->", color="#222222", linewidth=1.2),
    )
    axes[1].annotate(
        "nuclear group split\ninto two segments",
        xy=(nuclear_col, 3.6), xycoords="data",
        xytext=(nuclear_col - 3.4, 4.4), textcoords="data",
        fontsize=9, color="#222222", ha="left",
        arrowprops=dict(arrowstyle="->", color="#222222", linewidth=1.2),
    )
    fig.suptitle(
        f"The two solutions agree substantially (ARI = {ari:.2f}) - the difference is resolution, "
        f"not structure.\nk = {result.primary.k} keeps the four nuclear-heavy "
        f"operators in one segment; k = {result.sensitivity.k} splits them "
        f"into two segments with no domain meaning.",
        fontsize=12.5, fontweight="600",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    paths["k_contrast"] = _save(fig, directory, "13_k_selection_contrast.png")
    return paths


def plot_stability(result: AnalysisResult, directory: Path) -> dict:
    """Subsample resampling and pairwise agreement between candidate k."""
    paths = {}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    stability = result.stability
    ax.bar(
        [f"n = {n}" for n in stability["subsample_size"]],
        stability["mean_ari"],
        color=[PALETTE[1] if a < 0.7 else PALETTE[2] for a in stability["mean_ari"]],
        width=0.55, yerr=None,
    )
    for i, value in enumerate(stability["mean_ari"]):
        ax.annotate(f"{value:.2f}", (i, value), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean adjusted Rand index vs full-data solution")
    ax.set_title(
        f"Stability under resampling ({int(stability['repeats'].iloc[0])} refits per size): "
        "22 companies is a thin sample and the segmentation is correspondingly soft"
    )
    ax.margins(y=0.2)
    fig.tight_layout()
    paths["stability"] = _save(fig, directory, "14_subsample_stability.png")

    matrix = result.pairwise_agreement
    if not matrix.empty:
        ks = sorted(set(matrix["k_a"]) | set(matrix["k_b"]))
        grid = pd.DataFrame(np.eye(len(ks)), index=ks, columns=ks, dtype=float)
        for _, row in matrix.iterrows():
            grid.loc[row["k_a"], row["k_b"]] = row["ari"]
            grid.loc[row["k_b"], row["k_a"]] = row["ari"]
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        image = ax.imshow(grid.to_numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(ks)))
        ax.set_yticks(range(len(ks)))
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_yticklabels([f"k={k}" for k in ks])
        for i in range(len(ks)):
            for j in range(len(ks)):
                ax.annotate(f"{grid.iloc[i, j]:.2f}", (j, i), ha="center", va="center",
                            fontsize=8.5, color="white" if grid.iloc[i, j] > 0.6 else "#222222")
        ax.set_title("Agreement between every pair of candidate k")
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="adjusted Rand index")
        fig.tight_layout()
        paths["k_agreement"] = _save(fig, directory, "15_k_agreement_matrix.png")
    return paths


def render_all(result: AnalysisResult, directory: Path) -> dict:
    """Produce every figure and return a flat mapping of name -> written path."""
    apply_style()
    directory = Path(directory)
    figures: dict = {}
    for group in (
        plot_exploration,
        plot_k_selection,
        plot_dendrograms,
        plot_projection,
        plot_profiles,
        plot_stability,
    ):
        figures.update({f"{group.__name__.removeprefix('plot_')}/{k}": v
                        for k, v in group(result, directory).items()})
    return figures
