# Development and verification tasks. Run `make help` for the list.
#
#   make install           editable install with dev + notebook extras
#   make run               reproduce every figure, table and metric into outputs/
#   make test              full test suite
#   make verify            assert the numbers published in README.md still hold
#   make check             lint + test + verify, what CI runs
#   make notebook          execute the analysis notebook in place
#   make clean             remove generated artefacts

PYTHON ?= python

.PHONY: help install run test lint verify check notebook notebook-rebuild clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:  ## editable install including dev and notebook extras
	$(PYTHON) -m pip install -e ".[dev,notebook]"

run:  ## regenerate outputs/ (figures, tables, metrics.json)
	$(PYTHON) scripts/run_analysis.py

test:  ## run the test suite
	$(PYTHON) -m pytest

lint:  ## style and static checks
	$(PYTHON) -m ruff check src tests scripts

verify:  ## assert the numbers published in README.md still hold
	$(PYTHON) scripts/verify_published_metrics.py

check: lint test verify  ## what CI runs

notebook:  ## execute the analysis notebook in place
	jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

notebook-rebuild:  ## regenerate the notebook from scripts/build_notebook.py, then execute
	$(PYTHON) scripts/build_notebook.py --execute

clean:  ## remove generated artefacts
	rm -rf outputs .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +