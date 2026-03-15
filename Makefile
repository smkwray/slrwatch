.PHONY: test demo sources plan crosswalk insured-panel parent-panel event-study

PYTHON ?= python

test:
	$(PYTHON) -B -m pytest -q

demo:
	$(PYTHON) -m slr_watch.cli demo-headroom

sources:
	$(PYTHON) -m slr_watch.cli print-sources

plan:
	$(PYTHON) -m slr_watch.cli print-plan

crosswalk:
	$(PYTHON) -m slr_watch.cli build-crosswalk

insured-panel:
	$(PYTHON) -m slr_watch.cli build-insured-panel

parent-panel:
	$(PYTHON) -m slr_watch.cli build-parent-panel

event-study:
	$(PYTHON) -m slr_watch.cli run-event-study
