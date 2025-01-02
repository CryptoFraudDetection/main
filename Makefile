 lam.PHONY: help install test

help:
	@echo "-----------------------"
	@echo "Available make targets:"
	@echo ""
	@echo "help 		- show this help"
	@echo "install 		- install all dependencies"
	@echo "test 		- run all tests"
	@echo "-----------------------"

install:
	uv sync
test:
	pytest

.DEFAULT_GOAL := help