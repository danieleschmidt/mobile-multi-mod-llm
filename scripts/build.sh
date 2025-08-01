#!/bin/bash

# Build script for Mobile Multi-Modal LLM
# Comprehensive build automation with security and optimization

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_NAME="mobile-multimodal-llm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Build configuration
BUILD_TYPE="${BUILD_TYPE:-release}"
TARGET_PLATFORM="${TARGET_PLATFORM:-all}"
ENABLE_OPTIMIZATIONS="${ENABLE_OPTIMIZATIONS:-true}"
ENABLE_SECURITY_SCAN="${ENABLE_SECURITY_SCAN:-true}"
SKIP_TESTS="${SKIP_TESTS:-false}"

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command '$1' not found"
        exit 1
    fi
}

# =============================================================================
# Dependency Checks
# =============================================================================

check_dependencies() {
    log_info "Checking build dependencies..."
    
    check_command python
    check_command pip
    check_command docker
    
    # Check Python version
    python_version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    required_version="3.10"
    
    if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
        log_error "Python $required_version or higher required (found $python_version)"
        exit 1
    fi
    
    log_success "Dependencies check passed"
}

# =============================================================================
# Environment Setup
# =============================================================================

setup_environment() {
    log_info "Setting up build environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create build directories
    mkdir -p build/
    mkdir -p dist/
    mkdir -p reports/
    mkdir -p models/quantized/
    
    # Set environment variables
    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
    export BUILD_NUMBER="${BUILD_NUMBER:-$(date +%Y%m%d%H%M%S)}"
    export BUILD_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    
    log_success "Environment setup complete"
}

# =============================================================================
# Code Quality Checks
# =============================================================================

run_quality_checks() {
    if [[ "$BUILD_TYPE" == "debug" ]]; then
        log_info "Skipping quality checks for debug build"
        return 0
    fi
    
    log_info "Running code quality checks..."
    
    # Format check
    log_info "Checking code formatting..."
    if ! black --check src/ tests/; then
        log_error "Code formatting check failed. Run 'make format' to fix."
        exit 1
    fi
    
    # Import sorting check
    log_info "Checking import sorting..."
    if ! isort --check-only src/ tests/; then
        log_error "Import sorting check failed. Run 'make format' to fix."
        exit 1
    fi
    
    # Linting
    log_info "Running linting..."
    flake8 src/ tests/ || exit 1
    
    # Type checking
    log_info "Running type checking..."
    mypy src/ || exit 1
    
    log_success "Code quality checks passed"
}

# =============================================================================
# Security Scanning
# =============================================================================

run_security_scan() {
    if [[ "$ENABLE_SECURITY_SCAN" != "true" ]]; then
        log_info "Security scanning disabled"
        return 0
    fi
    
    log_info "Running security scans..."
    
    # Dependency vulnerability scan
    log_info "Scanning dependencies for vulnerabilities..."
    safety check --json --output reports/safety-report.json || log_warning "Safety check found vulnerabilities"
    
    # Code security scan
    log_info "Scanning code for security issues..."
    bandit -r src/ -f json -o reports/bandit-report.json || log_warning "Bandit found security issues"
    
    # Audit Python packages
    if command -v pip-audit &> /dev/null; then
        log_info "Running pip-audit..."
        pip-audit --format=json --output=reports/pip-audit-report.json || log_warning "pip-audit found issues"
    fi
    
    log_success "Security scans completed"
}

# =============================================================================
# Testing
# =============================================================================

run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_info "Tests skipped"
        return 0
    fi
    
    log_info "Running test suite..."
    
    # Create test results directory
    mkdir -p reports/test-results/
    
    # Run tests with coverage
    pytest tests/ \
        --cov=src \
        --cov-report=html:reports/htmlcov \
        --cov-report=xml:reports/coverage.xml \
        --cov-report=term-missing \
        --junitxml=reports/test-results/pytest.xml \
        --cov-fail-under=85
    
    log_success "Tests passed"
}

# =============================================================================
# Package Building
# =============================================================================

build_python_package() {
    log_info "Building Python package..."
    
    # Clean previous builds
    rm -rf build/ dist/ *.egg-info/
    
    # Build wheel and source distribution
    python -m build
    
    # Verify build
    twine check dist/*
    
    log_success "Python package built successfully"
}

# =============================================================================
# Docker Image Building
# =============================================================================

build_docker_images() {
    log_info "Building Docker images..."
    
    local version
    version=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
    
    # Build development image
    log_info "Building development image..."
    docker build --target development -t "${PROJECT_NAME}:dev" .
    
    # Build production image
    log_info "Building production image..."
    docker build --target production -t "${PROJECT_NAME}:${version}" -t "${PROJECT_NAME}:latest" .
    
    # Build GPU image if requested
    if [[ "$TARGET_PLATFORM" == "all" ]] || [[ "$TARGET_PLATFORM" == "gpu" ]]; then
        log_info "Building GPU image..."
        docker build --target gpu -t "${PROJECT_NAME}:gpu" .
    fi
    
    # Build mobile development image
    if [[ "$TARGET_PLATFORM" == "all" ]] || [[ "$TARGET_PLATFORM" == "mobile" ]]; then
        log_info "Building mobile development image..."
        docker build --target mobile-dev -t "${PROJECT_NAME}:mobile-dev" .
    fi
    
    log_success "Docker images built successfully"
}

# =============================================================================
# Mobile Model Export
# =============================================================================

export_mobile_models() {
    if [[ "$TARGET_PLATFORM" != "all" ]] && [[ "$TARGET_PLATFORM" != "mobile" ]]; then
        log_info "Skipping mobile model export"
        return 0
    fi
    
    log_info "Exporting models for mobile deployment..."
    
    # Create mobile model directories
    mkdir -p models/android/
    mkdir -p models/ios/
    
    # Export Android models (TensorFlow Lite)
    if command -v python &> /dev/null; then
        log_info "Exporting Android TensorFlow Lite models..."
        python scripts/export_models.py \
            --platform android \
            --quantization int2 \
            --output models/android/ || log_warning "Android export failed"
    fi
    
    # Export iOS models (Core ML)
    if command -v python &> /dev/null; then
        log_info "Exporting iOS Core ML models..."
        python scripts/export_models.py \
            --platform ios \
            --quantization int2 \
            --output models/ios/ || log_warning "iOS export failed"
    fi
    
    log_success "Mobile models exported"
}

# =============================================================================
# Performance Benchmarking
# =============================================================================

run_benchmarks() {
    if [[ "$BUILD_TYPE" == "debug" ]]; then
        log_info "Skipping benchmarks for debug build"
        return 0
    fi
    
    log_info "Running performance benchmarks..."
    
    # Create benchmark results directory
    mkdir -p reports/benchmarks/
    
    # Run benchmark tests
    pytest tests/benchmarks/ \
        -m benchmark \
        --benchmark-only \
        --benchmark-json=reports/benchmarks/results.json \
        --benchmark-html=reports/benchmarks/results.html || log_warning "Benchmarks failed"
    
    log_success "Benchmarks completed"
}

# =============================================================================
# Build Artifacts
# =============================================================================

package_artifacts() {
    log_info "Packaging build artifacts..."
    
    local version
    version=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
    
    # Create artifacts directory
    mkdir -p artifacts/
    
    # Package Python distribution
    if [[ -d "dist/" ]]; then
        cp -r dist/ artifacts/python-dist/
    fi
    
    # Package mobile models
    if [[ -d "models/android/" ]] || [[ -d "models/ios/" ]]; then
        mkdir -p artifacts/mobile-models/
        [[ -d "models/android/" ]] && cp -r models/android/ artifacts/mobile-models/
        [[ -d "models/ios/" ]] && cp -r models/ios/ artifacts/mobile-models/
    fi
    
    # Package reports
    if [[ -d "reports/" ]]; then
        cp -r reports/ artifacts/reports/
    fi
    
    # Create build manifest
    cat > artifacts/build-manifest.json << EOF
{
    "build": {
        "number": "${BUILD_NUMBER}",
        "timestamp": "${BUILD_TIMESTAMP}",
        "type": "${BUILD_TYPE}",
        "platform": "${TARGET_PLATFORM}",
        "version": "${version}",
        "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
        "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    },
    "artifacts": {
        "python_package": $([ -d "artifacts/python-dist/" ] && echo "true" || echo "false"),
        "mobile_models": $([ -d "artifacts/mobile-models/" ] && echo "true" || echo "false"),
        "docker_images": true,
        "reports": $([ -d "artifacts/reports/" ] && echo "true" || echo "false")
    }
}
EOF
    
    log_success "Build artifacts packaged"
}

# =============================================================================
# Build Summary
# =============================================================================

print_build_summary() {
    local version
    version=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || echo "unknown")
    
    echo ""
    echo "=========================================="
    echo "         BUILD SUMMARY"
    echo "=========================================="
    echo "Project: $PROJECT_NAME"
    echo "Version: $version"
    echo "Build Type: $BUILD_TYPE"
    echo "Target Platform: $TARGET_PLATFORM"
    echo "Build Number: $BUILD_NUMBER"
    echo "Build Timestamp: $BUILD_TIMESTAMP"
    echo ""
    echo "Artifacts Location: artifacts/"
    echo "Reports Location: reports/"
    echo ""
    
    if [[ -f "artifacts/build-manifest.json" ]]; then
        echo "Build Manifest:"
        cat artifacts/build-manifest.json | python -m json.tool
    fi
    
    echo "=========================================="
}

# =============================================================================
# Main Build Flow
# =============================================================================

main() {
    log_info "Starting build process for $PROJECT_NAME"
    log_info "Build type: $BUILD_TYPE, Target: $TARGET_PLATFORM"
    
    # Build steps
    check_dependencies
    setup_environment
    run_quality_checks
    run_security_scan
    run_tests
    build_python_package
    build_docker_images
    export_mobile_models
    run_benchmarks
    package_artifacts
    print_build_summary
    
    log_success "Build completed successfully!"
}

# =============================================================================
# Script Entry Point
# =============================================================================

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --platform)
            TARGET_PLATFORM="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --no-security)
            ENABLE_SECURITY_SCAN="false"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --type TYPE        Build type (debug|release) [default: release]"
            echo "  --platform PLATFORM Target platform (all|mobile|gpu) [default: all]"
            echo "  --skip-tests       Skip running tests"
            echo "  --no-security      Skip security scans"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Run main build process
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi