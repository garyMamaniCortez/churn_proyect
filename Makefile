#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = churn_proyect
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Extract raw client data from Postgres into data/raw/ (requires a working .env)
.PHONY: extract-data
extract-data: requirements
	$(PYTHON_INTERPRETER) -m churn_detection.dataset extract-all

## Build the client segmentation table from data/raw/ -> data/processed/clientes_segmentacion.csv
.PHONY: build-features
build-features: requirements
	$(PYTHON_INTERPRETER) -m churn_detection.features

## Version the freshly extracted raw data with DVC
.PHONY: version-data
version-data:
	dvc add data/raw/personas.csv data/raw/inscripciones.csv data/raw/servicios.csv \
		data/raw/registros_acceso.csv data/raw/ventas_servicios.csv data/raw/pagos_pendientes.csv
	@echo ">>> Now git add the resulting data/raw/*.dvc files and commit."

## run EDA analisis
.PHONY: eda-analisis
eda-analisis:
	$(PYTHON_INTERPRETER) -m churn_detection.plots

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
