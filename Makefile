PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = $(PROJECT_DIR)/.venv/bin/python

data/bank_data_dval.npy: data/bank_data_pipeline.joblib data/bank_data.csv
	$(PYTHON_INTERPRETER) src/data/make_dataset.py $^ $@ 'val'

data/bank_data_dtrain.npy: data/bank_data_pipeline.joblib data/bank_data.csv
	$(PYTHON_INTERPRETER) src/data/make_dataset.py $^ $@ 'train'

data/bank_data_pipeline.joblib: data/bank_data.csv
	$(PYTHON_INTERPRETER) src/data/make_pipeline.py $< $@

