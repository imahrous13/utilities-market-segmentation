"""Loading, cleaning and standardisation of the utilities dataset.

The raw file arrives with spaced, inconsistently cased headers and with trailing
whitespace inside the company names. Everything in this module exists to make that
file trustworthy before any distance is computed from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, ID_COLUMN


def _snake(name: str) -> str:
    """Normalise a raw header such as ``"Fixed charge"`` to ``"fixed_charge"``."""
    return name.strip().replace(" ", "_").lower()


def load_raw(path) -> pd.DataFrame:
    """Read the source CSV exactly as delivered, with headers untouched.

    No cleaning happens here so the cleaning report can state what it changed.
    """
    return pd.read_csv(path)


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean and shape the dataset for analysis.

    Steps: normalise headers, coerce features to ``float64``, drop null rows, drop
    duplicate rows, and promote the identifier column to the index.

    Returns the cleaned frame and a dict recording what each step removed, so the
    cleaning is auditable rather than asserted.
    """
    report: dict = {}

    cleaned = frame.rename(columns=lambda c: _snake(str(c)))
    report["columns_renamed"] = list(frame.columns) != list(cleaned.columns)

    missing = [c for c in (ID_COLUMN, *FEATURES) if c not in cleaned.columns]
    if missing:
        raise ValueError(f"expected columns missing from dataset: {missing}")

    # Only the modelling attributes are numeric-coerced; the identifier stays a string.
    for column in FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.loc[:, [ID_COLUMN, *FEATURES]]

    raw_ids = cleaned[ID_COLUMN].astype("string")
    cleaned[ID_COLUMN] = raw_ids.str.strip()
    report["id_whitespace_stripped"] = bool((raw_ids != cleaned[ID_COLUMN]).any())

    nulls_per_row = cleaned[list(FEATURES)].isna().any(axis=1)
    report["rows_before"] = int(len(cleaned))
    report["null_rows_dropped"] = int(nulls_per_row.sum())
    cleaned = cleaned.loc[~nulls_per_row]

    dupes = cleaned.duplicated()
    report["duplicate_rows_dropped"] = int(dupes.sum())
    cleaned = cleaned.loc[~dupes]

    report["rows_after"] = int(len(cleaned))

    indexed = cleaned.set_index(ID_COLUMN)
    indexed.columns.name = None
    indexed.index.name = None
    return indexed, report


def quality_report(frame: pd.DataFrame) -> dict:
    """Descriptive statistics plus the per-feature share of raw pairwise distance.

    The distance decomposition is the evidence that justifies standardising: it shows
    how completely the raw feature scales swamp every other attribute.
    """
    per_feature = {
        column: float((cdist(frame[[column]].to_numpy(), frame[[column]].to_numpy()) ** 2).sum())
        for column in FEATURES
    }
    total = sum(per_feature.values())
    return {
        "n_observations": int(len(frame)),
        "n_features": int(len(FEATURES)),
        "shape": list(frame.shape),
        "null_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "describe": {
            column: {
                "mean": float(frame[column].mean()),
                "std": float(frame[column].std()),
                "min": float(frame[column].min()),
                "q25": float(frame[column].quantile(0.25)),
                "median": float(frame[column].median()),
                "q75": float(frame[column].quantile(0.75)),
                "max": float(frame[column].max()),
            }
            for column in FEATURES
        },
        "raw_squared_distance_share_pct": {
            column: 100.0 * value / total for column, value in per_feature.items()
        },
        "constant_features": [c for c in FEATURES if frame[c].nunique() <= 1],
        "zero_inflation": {
            column: int((frame[column] == 0).sum())
            for column in FEATURES
            if (frame[column] == 0).any()
        },
    }


@dataclass(frozen=True)
class StandardisedData:
    """A z-scored design matrix paired with its labels and fitted scaler."""

    values: np.ndarray
    labels: tuple[str, ...]
    features: tuple[str, ...]
    scaler: StandardScaler
    raw: pd.DataFrame

    @property
    def n(self) -> int:
        return len(self.labels)


def standardize(frame: pd.DataFrame) -> StandardisedData:
    """Z-score every attribute with zero mean and unit population variance.

    ``StandardScaler`` is used deliberately rather than a hand-rolled
    ``(x - mean) / std``: it is the conventional definition, it divides by the
    *population* standard deviation, and it is safe to re-apply to unseen rows.
    """
    scaler = StandardScaler().fit(frame.to_numpy(dtype=float))
    values = scaler.transform(frame.to_numpy(dtype=float))
    return StandardisedData(
        values=values,
        labels=tuple(frame.index.astype(str)),
        features=tuple(frame.columns),
        scaler=scaler,
        raw=frame,
    )


def load_and_prepare(path):
    """Full ingest: read, clean, report quality, standardise.

    Returns ``(standardised, clean_report, quality_report)``.
    """
    raw = load_raw(path)
    cleaned, clean_info = clean(raw)
    return standardize(cleaned), clean_info, quality_report(cleaned)
