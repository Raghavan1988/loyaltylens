# LoyaltyLens convenience targets. CPU-only targets run locally; GPU work goes through Modal.
PY ?= python

.PHONY: help install install-probing test audit select transfer oracle figures pipeline clean-pyc

help:
	@echo "install          full pinned stack (requirements.txt)"
	@echo "install-probing  CPU-only probing/analysis deps"
	@echo "test             run pytest suite"
	@echo "audit            identity-confound audit"
	@echo "select           freeze probe on the P-M pair only"
	@echo "transfer         frozen transfer matrix incl. Caldera"
	@echo "oracle           oracle diagnostic (reported as confounded)"
	@echo "figures          regenerate report figures"
	@echo "pipeline         audit -> select -> transfer -> oracle -> figures"

install:
	$(PY) -m pip install -r requirements.txt

install-probing:
	$(PY) -m pip install -r requirements-probing.txt

test:
	$(PY) -m pytest -q

audit:
	$(PY) -m probing.audit_methodology

select:
	$(PY) -m probing.paired_select

transfer:
	$(PY) -m probing.paired_transfer --include-caldera

oracle:
	$(PY) -m probing.paired_oracle

figures:
	$(PY) -m analysis.make_figures

pipeline: audit select transfer oracle figures

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
