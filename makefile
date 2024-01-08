# Default
all: test
.PHONY: all

OS := $(shell uname | tr '[:upper:]' '[:lower:]')
CURRENT_DIR=$(shell pwd)

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

###
# Package
###
install:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda env create -f environment_$(OS).yml
	@echo ">>> Conda env created."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif

## Update conda environment
update_env:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, exporting conda environment."
	conda env update --name picoclvr --file environment_$(OS).yml --prune
	@echo ">>> Conda env exported."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif

## Export conda environment
export_env:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, exporting conda environment."
	conda env export -n picoclvr | grep -v "^prefix: " > environment_$(OS).yml

	@echo ">>> Conda env exported."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif

.PHONY: install update_env export_env


docs: ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
.PHONY: docs

##
# CI
###
lint:
	ruff check src

lint-fix:
	ruff check src --fix

lint-fix-unsafe:
	ruff check src --fix --unsafe-fixes

format:
	ruff format src

typecheck:
	mypy $(CURRENT_DIR)/src

test:
	pytest --disable-pytest-warnings src

ci: lint typecheck test

.PHONY: typecheck yapf lint test ci
