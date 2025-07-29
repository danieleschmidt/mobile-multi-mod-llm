.PHONY: help install install-dev test lint format clean build docs
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e .[dev]
	pre-commit install

test: ## Run tests
	pytest tests/ --cov=src --cov-report=term-missing

test-fast: ## Run tests without coverage
	pytest tests/ -x

lint: ## Run all linting tools
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/
	bandit -r src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

security: ## Run security checks
	bandit -r src/
	safety check

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

build: ## Build package
	python -m build

docs: ## Build documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

benchmark: ## Run performance benchmarks
	pytest tests/ -m benchmark --benchmark-only

mobile-export: ## Export models for mobile deployment
	python scripts/export_models.py --all

profile: ## Profile model performance
	python scripts/profile_models.py