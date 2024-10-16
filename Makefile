.PHONY: create_environment clean

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = seqsketch
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel --no-cache-dir
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
