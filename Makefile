 lam.PHONY: help build dev-install install test

help:
	@echo "-----------------------"
	@echo "Available make targets:"
	@echo ""
	@echo "help 		- show this help"
	@echo "build 		- build the package"
	@echo "install		- install the package"
	@echo "dev-install 	- install the package in development mode"
	@echo "test 		- run the tests"
	@echo "-----------------------"

build:
	python3.12 -m build

dev-install:
	pip3.12 install -e .

install:
	pip3.12 install .

test:
	python3.12 -m pytest

.DEFAULT_GOAL := help