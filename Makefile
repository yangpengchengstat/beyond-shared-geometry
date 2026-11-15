PYTHON ?= python3
.PHONY: check verify package
check verify:
	$(PYTHON) -B scripts/verify_delivery.py
	$(PYTHON) -B scripts/verify_submission_surface.py
# Explicit packaging only; ../dist must be fresh. This does not compile or run science.
package: check
	$(PYTHON) -B scripts/package_submission.py --output ../dist
# Historical build and scientific-check targets are retained in
# release/history/before_completion_v2/Makefile; they are not automatic prerequisites.
