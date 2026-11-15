PYTHON ?= python3

.PHONY: check assets paper release verify
check:
	PYTHONPATH=src $(PYTHON) tests/run_cpu_checks.py
	$(PYTHON) scripts/verify_frozen.py
	$(PYTHON) scripts/check_release.py

assets:
	$(PYTHON) scripts/build_paper_assets.py
	$(PYTHON) scripts/build_characterization_status.py
	$(PYTHON) scripts/build_characterization_assets.py
	$(PYTHON) scripts/build_reproduction_assets.py

paper:
	bash scripts/build_paper.sh

release:
	$(PYTHON) scripts/package_release.py

verify:
	$(PYTHON) scripts/verify_frozen.py
	$(PYTHON) scripts/check_release.py

.PHONY: prospective-checks prospective-plan prospective-verify
prospective-checks:
	PYTHONPATH=src $(PYTHON) tests/run_prospective_checks.py
prospective-plan:
	PYTHONPATH=src $(PYTHON) -m bsg_next plan
prospective-verify:
	$(PYTHON) scripts/verify_prospective.py

.PHONY: characterization-checks
characterization-checks:
	$(PYTHON) scripts/verify_characterization.py
	$(PYTHON) -m pytest -q studies/characterization_v1/tests
