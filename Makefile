PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = $(PROJECT_DIR)/.venv/bin/python

data/bank__data_pipeline.plt: data/bank_data.csv
	$(PYTHON_INTERPRETER) src/data/make_pipeline.py $< $@

