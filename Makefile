.PHONY: help create_environment install_requirements download_quickdraw filter_quickdraw clean

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = seqsketch
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Display this help message
help:
	@echo Project name $(PROJECT_NAME)
	@echo Python version $(PYTHON_VERSION)
	@echo Python interpreter $(PYTHON_INTERPRETER)

## Set up python interpreter environment
create_environment:
	conda env remove --name $(PROJECT_NAME) --yes --quiet || true
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

## Install Python Dependencies
install_requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel --no-cache-dir
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Download data
download_quickdraw:
	mkdir -p data/raw/quickdraw
	gsutil -m cp 'gs://quickdraw_dataset/full/simplified/*.ndjson' data/raw/quickdraw/

## Filter data
filter_quickdraw:
	sh scripts/filter_quickdraw/filter_data_all.sh
	
## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
