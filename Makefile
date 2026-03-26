.PHONY: test demo sources crosswalk fdic-institutions insured-panel all-insured-panel parent-panel event-study site-data reproduce

PYTHON ?= python

test:
	$(PYTHON) -B -m pytest -q

demo:
	$(PYTHON) -m slr_watch.cli demo-headroom

sources:
	$(PYTHON) -m slr_watch.cli print-sources

crosswalk:
	$(PYTHON) -m slr_watch.cli build-crosswalk

fdic-institutions:
	$(PYTHON) -m slr_watch.cli build-fdic-institutions

insured-panel:
	$(PYTHON) -m slr_watch.cli build-insured-panel

all-insured-panel:
	$(PYTHON) -m slr_watch.cli build-all-insured-panel

parent-panel:
	$(PYTHON) -m slr_watch.cli build-parent-panel

event-study:
	$(PYTHON) -m slr_watch.cli run-event-study

site-data:
	$(PYTHON) -m slr_watch.cli build-site-data

reproduce:
	$(PYTHON) -m slr_watch.cli build-crosswalk
	$(PYTHON) -m slr_watch.cli build-fdic-institutions
	$(PYTHON) -m slr_watch.cli build-all-insured-panel
	$(PYTHON) -m slr_watch.cli build-insured-panel
	$(PYTHON) -m slr_watch.cli build-parent-panel
	$(PYTHON) -m slr_watch.cli run-event-study
	$(PYTHON) -m slr_watch.cli run-reallocation-report
	$(PYTHON) -m slr_watch.cli run-safe-asset-absorption-report
	$(PYTHON) -m slr_watch.cli run-parent-transmission-report
	$(PYTHON) -m slr_watch.cli run-treasury-intermediation-report
	$(PYTHON) -m slr_watch.cli run-policy-regime-panel-report
	$(PYTHON) -m slr_watch.cli run-constraint-decomposition-report
	$(PYTHON) -m slr_watch.cli build-site-data
