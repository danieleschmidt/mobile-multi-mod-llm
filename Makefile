# Enhanced Makefile for Mobile Multi-Modal LLM
# Comprehensive build system with advanced SDLC automation

.PHONY: help install install-dev test lint format clean build docs docker monitoring
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

# Project configuration
PROJECT_NAME := mobile-multimodal-llm
PYTHON_VERSION := 3.11
DOCKER_IMAGE := $(PROJECT_NAME)
VERSION := $(shell python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")

help: ## Show this help message
	@echo '$(BLUE)Mobile Multi-Modal LLM - Enhanced Build System$(NC)'
	@echo ''
	@echo '$(GREEN)Usage:$(NC) make [target]'
	@echo ''
	@echo '$(GREEN)Development Targets:$(NC)'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# Installation and Setup
# =============================================================================

install: ## Install package for production
	@echo "$(BLUE)Installing package...$(NC)"
	pip install -e .

install-dev: ## Install package with development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install -e .[dev,test,docs,mobile]
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

setup-hooks: ## Setup pre-commit hooks
	@echo "$(BLUE)Setting up pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg

# =============================================================================
# Testing and Quality Assurance
# =============================================================================

test: ## Run comprehensive test suite
	@echo "$(BLUE)Running comprehensive test suite...$(NC)"
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml

test-fast: ## Run tests without coverage (fast)
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest tests/ -x --disable-warnings

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/ -m "unit" --cov=src --cov-report=term-missing

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration/ -m "integration" -v

test-security: ## Run security tests
	@echo "$(BLUE)Running security tests...$(NC)"
	pytest tests/security/ -m "security" -v

test-mobile: ## Run mobile-specific tests
	@echo "$(BLUE)Running mobile tests...$(NC)"
	pytest tests/mobile/ -m "mobile" --run-mobile -v

test-gpu: ## Run GPU tests (requires CUDA)
	@echo "$(BLUE)Running GPU tests...$(NC)"
	pytest tests/ -m "gpu" --run-gpu -v

test-all-environments: ## Test across all Python environments
	@echo "$(BLUE)Testing across environments...$(NC)"
	tox

# =============================================================================
# Code Quality and Linting
# =============================================================================

lint: ## Run all linting tools
	@echo "$(BLUE)Running code quality checks...$(NC)"
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/
	bandit -r src/

lint-fix: ## Fix linting issues automatically
	@echo "$(BLUE)Fixing code formatting...$(NC)"
	black src/ tests/
	isort src/ tests/

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy src/ --strict

# =============================================================================
# Security and Compliance
# =============================================================================

security: ## Run comprehensive security checks
	@echo "$(BLUE)Running security analysis...$(NC)"
	bandit -r src/ -f json -o reports/bandit-report.json || true
	bandit -r src/
	safety check --json --output reports/safety-report.json || true
	safety check
	pip-audit --format=json --output=reports/pip-audit-report.json || true
	pip-audit

security-scan-deps: ## Scan dependencies for vulnerabilities
	@echo "$(BLUE)Scanning dependencies...$(NC)"
	safety check --json --output reports/safety-report.json
	pip-audit --format=json --output=reports/pip-audit-report.json

generate-sbom: ## Generate Software Bill of Materials
	@echo "$(BLUE)Generating SBOM...$(NC)"
	mkdir -p reports/
	cyclonedx-py -o reports/sbom.json
	cyclonedx-py -o reports/sbom.xml --format xml

compliance-check: ## Run compliance validation
	@echo "$(BLUE)Running compliance checks...$(NC)"
	$(MAKE) security
	$(MAKE) generate-sbom
	@echo "$(GREEN)Compliance check completed!$(NC)"

# =============================================================================
# Performance and Benchmarking
# =============================================================================

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	pytest tests/benchmarks/ -m benchmark --benchmark-only --benchmark-json=reports/benchmark-results.json

benchmark-memory: ## Run memory profiling benchmarks
	@echo "$(BLUE)Running memory benchmarks...$(NC)"
	pytest tests/benchmarks/ -m benchmark -k "memory" --benchmark-only

benchmark-inference: ## Benchmark inference performance
	@echo "$(BLUE)Benchmarking inference...$(NC)"
	pytest tests/benchmarks/ -m benchmark -k "inference" --benchmark-only

profile: ## Profile application performance
	@echo "$(BLUE)Profiling application...$(NC)"
	python -m cProfile -o reports/profile.stats scripts/profile_models.py
	python -c "import pstats; p = pstats.Stats('reports/profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# =============================================================================
# Build and Package Management
# =============================================================================

clean: ## Clean all build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .tox/
	rm -rf reports/ .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed!$(NC)"

build: ## Build package for distribution
	@echo "$(BLUE)Building package...$(NC)"
	python -m build
	twine check dist/*

build-wheel: ## Build wheel package only
	@echo "$(BLUE)Building wheel...$(NC)"
	python -m build --wheel

build-sdist: ## Build source distribution only
	@echo "$(BLUE)Building source distribution...$(NC)"
	python -m build --sdist

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	mkdocs build --strict

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	mkdocs serve --dev-addr 0.0.0.0:8000

docs-clean: ## Clean documentation build
	@echo "$(BLUE)Cleaning documentation...$(NC)"
	rm -rf site/

# =============================================================================
# Docker Operations
# =============================================================================

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(VERSION) -t $(DOCKER_IMAGE):latest .

docker-build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	docker build --target development -t $(DOCKER_IMAGE):dev .

docker-build-gpu: ## Build GPU-enabled Docker image
	@echo "$(BLUE)Building GPU Docker image...$(NC)"
	docker build --target gpu -t $(DOCKER_IMAGE):gpu .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm -p 8080:8080 $(DOCKER_IMAGE):latest

docker-run-dev: ## Run development Docker container
	@echo "$(BLUE)Running development container...$(NC)"
	docker run -it --rm -v $(PWD):/app -p 8000:8000 $(DOCKER_IMAGE):dev bash

docker-test: ## Run tests in Docker container
	@echo "$(BLUE)Running tests in Docker...$(NC)"
	docker build --target testing -t $(DOCKER_IMAGE):test .
	docker run --rm $(DOCKER_IMAGE):test

docker-compose-up: ## Start all services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	docker-compose up -d

docker-compose-down: ## Stop all docker-compose services
	@echo "$(BLUE)Stopping docker-compose services...$(NC)"
	docker-compose down

# =============================================================================
# Mobile Development
# =============================================================================

mobile-export: ## Export models for mobile deployment
	@echo "$(BLUE)Exporting mobile models...$(NC)"
	python scripts/export_models.py --all

mobile-android: ## Build Android-specific exports
	@echo "$(BLUE)Building Android exports...$(NC)"
	python scripts/export_models.py --platform android --quantization int2

mobile-ios: ## Build iOS-specific exports  
	@echo "$(BLUE)Building iOS exports...$(NC)"
	python scripts/export_models.py --platform ios --quantization int2

mobile-test-export: ## Test mobile model exports
	@echo "$(BLUE)Testing mobile exports...$(NC)"
	python scripts/test_mobile_exports.py

# =============================================================================
# Monitoring and Observability
# =============================================================================

monitoring-up: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	docker-compose --profile monitoring up -d prometheus grafana

monitoring-down: ## Stop monitoring stack
	@echo "$(BLUE)Stopping monitoring stack...$(NC)"
	docker-compose --profile monitoring down

logs: ## Show application logs
	@echo "$(BLUE)Showing application logs...$(NC)"
	docker-compose logs -f app-prod

metrics: ## Display key metrics
	@echo "$(BLUE)Fetching key metrics...$(NC)"
	curl -s http://localhost:9090/api/v1/query?query=up | jq '.data.result'

# =============================================================================
# Release Management  
# =============================================================================

version: ## Show current version
	@echo "$(GREEN)Current version: $(VERSION)$(NC)"

version-bump-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(NC)"
	bump2version patch

version-bump-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(NC)"
	bump2version minor

version-bump-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(NC)"
	bump2version major

release-check: ## Check if ready for release
	@echo "$(BLUE)Checking release readiness...$(NC)"
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) security
	$(MAKE) build
	@echo "$(GREEN)Ready for release!$(NC)"

# =============================================================================
# CI/CD Support
# =============================================================================

ci-install: ## Install dependencies for CI
	@echo "$(BLUE)Installing CI dependencies...$(NC)"
	pip install -e .[dev,test,mobile]

ci-test: ## Run CI test suite
	@echo "$(BLUE)Running CI tests...$(NC)"
	pytest tests/ --cov=src --cov-report=xml --junitxml=reports/junit.xml

ci-lint: ## Run linting for CI
	@echo "$(BLUE)Running CI linting...$(NC)"
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/ --format=junit-xml --output-file=reports/flake8.xml
	mypy src/ --junit-xml reports/mypy.xml

ci-security: ## Run security checks for CI
	@echo "$(BLUE)Running CI security checks...$(NC)"
	bandit -r src/ -f json -o reports/bandit-report.json
	safety check --json --output reports/safety-report.json

# =============================================================================
# Development Utilities
# =============================================================================

setup-dev: ## Complete development environment setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	$(MAKE) install-dev
	$(MAKE) setup-hooks
	mkdir -p reports/
	@echo "$(GREEN)Development environment ready!$(NC)"

dev-server: ## Start development server
	@echo "$(BLUE)Starting development server...$(NC)"
	python -m mobile_multimodal.server --dev

jupyter: ## Start Jupyter lab for development
	@echo "$(BLUE)Starting Jupyter Lab...$(NC)"
	jupyter lab --ip=0.0.0.0 --no-browser --allow-root

health-check: ## Run comprehensive health check
	@echo "$(BLUE)Running health check...$(NC)"
	$(MAKE) test-fast
	$(MAKE) lint
	$(MAKE) security-scan-deps
	@echo "$(GREEN)Health check completed!$(NC)"

# =============================================================================
# Reporting
# =============================================================================

reports: ## Generate all reports
	@echo "$(BLUE)Generating comprehensive reports...$(NC)"
	mkdir -p reports/
	$(MAKE) ci-test
	$(MAKE) benchmark
	$(MAKE) security
	$(MAKE) generate-sbom
	@echo "$(GREEN)All reports generated in reports/ directory$(NC)"