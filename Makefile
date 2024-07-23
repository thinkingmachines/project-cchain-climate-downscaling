.PHONY: clean clean-test clean-pyc clean-build dev venv help
.DEFAULT_GOAL := help
-include .env

help:
	@awk -F ':.*?## ' '/^[a-zA-Z]/ && NF==2 {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

env:
	conda env create -f environment.yml --no-default-packages

setup:
	conda install -c conda-forge gdal esmpy -y
	pip install uv
	uv pip sync requirements.txt
	pre-commit install

requirements:
	uv pip compile pyproject.toml -o requirements.txt -v
	echo "esmpy==8.6.1" >> requirements.txt
	uv pip sync requirements.txt

test:
	pytest -v tests/
