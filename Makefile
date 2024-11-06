.PHONY: help create_environment install_requirements download_quickdraw filter_quickdraw delete_quickdraw clean

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

create_venv:
	python3 -m venv cs492d
	source cs492d/bin/activate


## Set up python interpreter environment
create_environment:
	conda env remove --name $(PROJECT_NAME) --yes --quiet || true
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

## Install Python Dependencies
install_requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel --no-cache-dir
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt --no-cache-dir
	$(PYTHON_INTERPRETER) -m pip install -e . --no-cache-dir

## Download data
download_quickdraw:
	mkdir -p data/quickdraw/raw/
	gsutil -m cp 'gs://quickdraw_dataset/full/simplified/*.ndjson' data/quickdraw/raw/

## Filter data
process_rawdata:
#sh scripts/filter_quickdraw/filter_data_all.sh
	sh scripts/filter_quickdraw/filter_data_seq_all.sh

create_datasplit:
	sh scripts/filter_quickdraw/create_datasplit_json_all.sh


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

train
	train --config baseline.yaml
	inference --model_folder baseline/v0_20241101_195027