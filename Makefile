PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTHON_BIN := $(VENV)/bin/python

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@touch $@

.PHONY: install
install: $(VENV)/bin/activate

.PHONY: run
run: $(VENV)/bin/activate
	$(PYTHON_BIN) src/compare_detectors.py

.PHONY: test
test: $(VENV)/bin/activate
	$(PYTHON_BIN) -m pytest

.PHONY: clean
clean:
	rm -rf $(VENV)
