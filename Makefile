PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = $(PROJECT_DIR)/.venv/bin/python

all: data pipeline model

.PHONY: all data pipeline model

model: model/bank_model.pt

model/bank_model.pt: data/bank_data_dtrain.npy data/bank_data_dval.npy
	$(PYTHON_INTERPRETER) src/model/train_model.py $^ $@

data: data/bank_data_dval.npy data/bank_data_dtrain.npy

data/bank_data_dval.npy: data/bank_data_pipeline.joblib data/bank_data.csv
	$(PYTHON_INTERPRETER) src/data/make_dataset.py $^ $@ 'val'

data/bank_data_dtrain.npy: data/bank_data_pipeline.joblib data/bank_data.csv
	$(PYTHON_INTERPRETER) src/data/make_dataset.py $^ $@ 'train'

pipeline: data/bank_data_pipeline.joblib

data/bank_data_pipeline.joblib: data/bank_data.csv
	$(PYTHON_INTERPRETER) src/data/make_pipeline.py $< $@

