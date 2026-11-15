PYTHON ?= python3

.PHONY: check software-checks verify paper release spectral-checks context-checks extension-checks legacy-checks bounded-checks review-checks
check: software-checks verify

software-checks: bounded-checks review-checks
	PYTHONPATH=src $(PYTHON) tests/run_cpu_checks.py
	$(PYTHON) -m unittest discover -s studies/spectral_finite_bank_v1/tests -v
	$(PYTHON) -m unittest discover -s studies/context_generalization_v1/tests -v
	$(PYTHON) -m unittest discover -s studies/spectral_extensions_v1/tests -v

verify:
	$(PYTHON) scripts/verify_bounded_revision.py
	$(PYTHON) scripts/verify_frozen.py
	$(PYTHON) scripts/verify_prospective.py
	$(PYTHON) scripts/verify_characterization.py
	$(PYTHON) scripts/verify_spectral.py
	$(PYTHON) scripts/verify_tmlr_studies.py
	$(PYTHON) scripts/check_release.py

paper:
	bash scripts/build_paper.sh

release:
	$(PYTHON) scripts/package_release.py

spectral-checks:
	$(PYTHON) -m unittest discover -s studies/spectral_finite_bank_v1/tests -v

context-checks:
	$(PYTHON) -m unittest discover -s studies/context_generalization_v1/tests -v

extension-checks:
	$(PYTHON) -m unittest discover -s studies/spectral_extensions_v1/tests -v

# Software checks of older workflows only; never starts a historical study.
legacy-checks:
	PYTHONPATH=src $(PYTHON) tests/run_prospective_checks.py
	$(PYTHON) -m pytest -q studies/characterization_v1/tests studies/compact_context_v1/tests

# Reformat audited archived outcomes only, without experiments or resampling.
followup-assets:
	$(PYTHON) scripts/build_tmlr_results.py

# Synthetic engineering checks and saved-output checks only; no family full run.
bounded-checks:
	$(PYTHON) -m unittest discover -s analysis/main30_bounded_v1/tests -v
	$(PYTHON) analysis/rank_sensitivity_v1/test_sensitivity.py

# Reformat already derived tables only, without recalculating scientific outcomes.
bounded-assets:
	$(PYTHON) analysis/main30_paper_v1/render_tables.py

# Only saved-endpoint capsule engineering checks; no models or LOFO.
review-checks:
	$(PYTHON) -m unittest discover -s review_reproduction/tests -v
