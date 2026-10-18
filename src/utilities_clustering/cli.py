"""Command-line interface.

Installed as the ``utilities-segment`` console script, and also reachable directly via
``python scripts/run_analysis.py``. Every artefact the write-up quotes is written here,
so nothing in the README has to be transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config, pipeline, plotting
from .config import DATA_PATH, OUTPUT_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="utilities-segment",
        description="Segment utility companies with hierarchical clustering and K-Medoids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="input CSV path")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="output directory")
    parser.add_argument("--k", type=int, default=config.K_PRIMARY, help="cluster count to report")
    parser.add_argument("--seed", type=int, default=config.SEED, help="master random seed")
    parser.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    parser.add_argument(
        "--quick", action="store_true", help="reduce resampling repeats for a fast smoke run"
    )
    return parser


def _metrics_payload(result, seed: int) -> dict:
    return {
        "meta": {
            "seed": seed,
            "k_selected": result.primary.k,
            "k_sensitivity": result.sensitivity.k,
            "features": list(result.standardised.features),
            "n_observations": result.standardised.n,
        },
        "cleaning": result.cleaning,
        "quality": result.quality,
        "k_recommendation": result.recommendation,
        "primary": {
            "method": result.primary.method,
            "names": list(result.primary.names),
            "sizes": [int(v) for v in result.primary.sizes()],
            "members": {
                result.primary.names[c]: sorted(
                    n.strip()
                    for n, lab in zip(result.standardised.labels, result.primary.labels)
                    if lab == c
                )
                for c in range(result.primary.k)
            },
            "fit": result.primary_fit,
            "separation": {
                key: value
                for key, value in result.primary_separation.items()
                if key != "per_point"
            },
        },
        "sensitivity": {
            "method": result.sensitivity.method,
            "names": list(result.sensitivity.names),
            "sizes": [int(v) for v in result.sensitivity.sizes()],
            "fit": result.sensitivity_fit,
            "agreement_with_primary": result.k_agreement.to_dict(orient="records"),
        },
        "linkages": result.linkage_agreement.to_dict(orient="records"),
        "segmentation_quality": result.segmentation_quality().to_dict(orient="index"),
        "cross_linkage_agreement": result.linkage_agreement.attrs.get("cross_linkage", {}),
        "seed_sensitivity": result.seed_probe,
        "subsample_stability": result.stability.to_dict(orient="records"),
        "kmeans_contrast": result.kmeans_contrast,
        "pca": {
            "explained_variance_ratio": result.pca["explained_variance_ratio"],
            "cumulative_explained_variance": result.pca["cumulative_explained_variance"],
            "loadings": result.pca["loadings"],
        },
    }


def _display(path: Path) -> str:
    """Repo-relative path when possible, absolute otherwise (e.g. ``--output /tmp/x``)."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    print(f"[1/4] loading and standardising  {args.data}")
    result = pipeline.run_analysis(args.data, k_primary=args.k, seed=args.seed)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("[2/4] writing tables")
    result.k_selection.round(6).to_csv(out / "k_selection.csv", index=False)
    result.segmentation_frame().round(6).to_csv(out / "segmentation.csv", index=False)
    result.segmentation_quality().to_csv(out / "segmentation_quality.csv")
    result.profiles.round(4).to_csv(out / "cluster_profiles.csv")
    result.sensitivity_profiles.round(4).to_csv(
        out / f"cluster_profiles_k{result.sensitivity.k}.csv"
    )
    result.linkage_agreement.round(4).to_csv(out / "linkage_comparison.csv", index=False)
    result.stability.round(4).to_csv(out / "stability.csv", index=False)

    print("[3/4] writing metrics.json")
    (out / "metrics.json").write_text(
        json.dumps(pipeline.to_serialisable(_metrics_payload(result, args.seed)), indent=2),
        encoding="utf-8",
    )

    if args.no_figures:
        print("[4/4] figures skipped (--no-figures)")
        return 0

    print("[4/4] rendering figures")
    for name, path in plotting.render_all(result, out / "figures").items():
        print(f"      {name:40s} -> {_display(path)}")

    print("\ndone.")
    silhouette = result.primary_separation["silhouette"]
    print(f"  selected k : {result.primary.k}   silhouette {silhouette:.4f}")
    print(f"  medoids    : {', '.join(result.primary.names)}")
    print(f"  total WCSS : {result.primary_fit['total']:.3f}")
    print(f"  outputs    : {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
