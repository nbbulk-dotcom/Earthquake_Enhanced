# Makefile for Earthquake Enhanced - Deterministic Measurement System

.PHONY: help install test lint format run-historical run-experiment clean docker-build docker-run

help:
	@echo "Earthquake Enhanced - Deterministic Measurement System"
	@echo ""
	@echo "Available targets:"
	@echo "  install          - Install dependencies"
	@echo "  test             - Run all tests"
	@echo "  lint             - Run linting checks"
	@echo "  format           - Format code with black"
	@echo "  run-historical   - Run historical data analysis"
	@echo "  run-experiment   - Run Tokyo experiment"
	@echo "  clean            - Clean generated files"
	@echo "  docker-build     - Build Docker image"
	@echo "  docker-run       - Run in Docker container"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=backend --cov-report=term-missing --cov-report=html

lint:
	flake8 backend/ tests/
	mypy backend/

format:
	black backend/ tests/

run-historical:
	python backend/physics_engine/historical_runner.py

run-experiment:
	cd experiments/tokyo && ./run_experiment.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov/

docker-build:
	docker build -t earthquake-enhanced:latest .

docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data earthquake-enhanced:latest
