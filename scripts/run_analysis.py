#!/usr/bin/env python
"""Reproduce the entire analysis in one command.

Equivalent to the installed ``utilities-segment`` console script; this wrapper exists so
the pipeline can be run from a fresh clone without installing the package first.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utilities_clustering.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
