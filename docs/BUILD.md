# Build System Documentation

Comprehensive documentation for building, containerizing, and deploying the Mobile Multi-Modal LLM project.

## Overview

The Mobile Multi-Modal LLM project uses a sophisticated build system that supports:

- **Multi-stage Docker builds** for different environments
- **Comprehensive testing** and quality assurance
- **Mobile model export** for Android and iOS
- **Security scanning** and vulnerability assessment
- **Performance benchmarking** and optimization
- **Automated release management** with semantic versioning

## Quick Start

### Prerequisites

```bash
# Required tools
- Python 3.10+
- Docker 20.10+
- Make 4.0+
- Git 2.30+

# Optional tools (for mobile development)
- Android SDK & NDK
- Xcode 15+ (macOS only)
```

### Basic Build Commands

```bash
# Full development setup
make setup-dev

# Run tests and build
make test && make build

# Build Docker images
make docker-build

# Build mobile models
make mobile-export
```

## Build System Components

### 1. Makefile Targets

The project includes a comprehensive Makefile with 40+ targets organized into categories:

#### Development Targets
```bash
make install-dev      # Install with development dependencies
make setup-hooks      # Setup pre-commit hooks
make format          # Format code with black and isort
make lint            # Run all linting tools
make test            # Run comprehensive test suite
```

#### Build Targets
```bash
make build           # Build Python package
make docker-build    # Build Docker images
make mobile-export   # Export mobile models
make clean          # Clean all build artifacts
```

#### Quality Assurance
```bash
make security        # Run security scans
make benchmark       # Run performance benchmarks
make compliance-check # Run compliance validation
make reports         # Generate all reports
```

### 2. Docker Multi-Stage Build

The Dockerfile supports multiple build targets optimized for different use cases:

#### Available Targets

1. **Base** - Common foundation with Python and system dependencies
2. **Development** - Full development environment with tools
3. **Testing** - Testing environment with additional test tools
4. **Production** - Minimal production runtime
5. **Mobile-Dev** - Mobile development with Android SDK
6. **GPU** - CUDA-enabled environment for training

#### Build Examples

```bash
# Development environment
docker build --target development -t mobile-mm-llm:dev .

# Production optimized
docker build --target production -t mobile-mm-llm:prod .

# GPU-enabled for training
docker build --target gpu -t mobile-mm-llm:gpu .
```

### 3. Docker Compose Services

Comprehensive service orchestration with docker-compose.yml:

```bash
# Development services
docker-compose up app-dev docs

# Testing environment
docker-compose up app-test

# Full monitoring stack
docker-compose --profile monitoring up
```

#### Available Services

- **app-dev**: Development environment with hot reload
- **app-test**: Automated testing environment
- **app-prod**: Production deployment
- **app-gpu**: GPU-enabled training environment
- **mobile-dev**: Mobile development environment
- **docs**: Documentation server
- **postgres**: Database for metrics
- **redis**: Caching layer
- **prometheus**: Metrics collection
- **grafana**: Metrics visualization

## Build Scripts

### Automated Build Script

The `scripts/build.sh` provides comprehensive build automation:

```bash
# Full release build
./scripts/build.sh --type release --platform all

# Debug build without tests
./scripts/build.sh --type debug --skip-tests

# Mobile-only build
./scripts/build.sh --platform mobile
```

#### Build Script Features

- **Dependency validation** - Checks all required tools
- **Code quality enforcement** - Linting, formatting, type checking
- **Security scanning** - Vulnerability detection and SBOM generation
- **Comprehensive testing** - Unit, integration, and mobile tests
- **Multi-platform support** - Python, Docker, mobile exports
- **Artifact packaging** - Structured output with manifests

## Mobile Build Pipeline

### Android (TensorFlow Lite)

```bash
# Build Android-specific models
make mobile-android

# Manual export with custom settings
python scripts/export_models.py \
    --platform android \
    --quantization int2 \
    --output models/android/ \
    --optimization speed
```

### iOS (Core ML)

```bash
# Build iOS-specific models
make mobile-ios

# Manual export with custom settings
python scripts/export_models.py \
    --platform ios \
    --quantization int2 \
    --output models/ios/ \
    --target neural_engine
```

### Mobile Testing

```bash
# Test mobile model exports
make mobile-test-export

# Run mobile-specific tests
pytest tests/ -m mobile --run-mobile
```

## Build Optimization

### Docker Build Optimization

1. **Multi-stage builds** minimize final image size
2. **Layer caching** optimizes rebuild times
3. **.dockerignore** excludes unnecessary files
4. **Security scanning** in CI/CD pipeline

### Build Performance Tips

```bash
# Use BuildKit for improved performance
export DOCKER_BUILDKIT=1

# Parallel testing
pytest -n auto

# Skip slow tests during development
pytest -m "not slow"

# Use local caching
docker build --cache-from mobile-mm-llm:latest .
```

## Security and Compliance

### Security Scanning

The build system includes comprehensive security scanning:

```bash
# Run all security checks
make security

# Individual security tools
bandit -r src/              # Code security analysis
safety check                # Dependency vulnerability scan
pip-audit                   # Python package audit
```

### SBOM Generation

Software Bill of Materials (SBOM) generation for compliance:

```bash
# Generate SBOM
make generate-sbom

# Outputs:
# - reports/sbom.json (CycloneDX JSON format)
# - reports/sbom.xml (CycloneDX XML format)
```

### Compliance Validation

```bash
# Run full compliance check
make compliance-check

# Includes:
# - Security scanning
# - License compliance
# - SBOM generation
# - Dependency auditing
```

## Performance Benchmarking

### Benchmark Execution

```bash
# Run all benchmarks
make benchmark

# Specific benchmark categories
make benchmark-memory        # Memory usage benchmarks
make benchmark-inference     # Inference performance
pytest tests/benchmarks/ -k "latency"  # Latency benchmarks
```

### Benchmark Outputs

- **JSON Results**: `reports/benchmark-results.json`
- **HTML Report**: `reports/benchmarks/results.html`
- **Performance Profiles**: `reports/profile.stats`

## Release Management

### Semantic Release

The project uses semantic-release for automated versioning:

```bash
# Trigger release (CI/CD)
semantic-release

# Manual version bump
make version-bump-minor
```

### Release Process

1. **Commit Analysis** - Determines version bump from commit messages
2. **Changelog Generation** - Updates CHANGELOG.md
3. **Version Update** - Updates pyproject.toml and __init__.py
4. **Build Artifacts** - Creates packages and mobile models
5. **GitHub Release** - Creates release with artifacts
6. **Tag Creation** - Tags the release commit

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

**Types**: feat, fix, docs, style, refactor, perf, test, build, ci, chore

## Build Artifacts

### Output Structure

```
artifacts/
├── python-dist/          # Python packages (.whl, .tar.gz)
├── mobile-models/        # Mobile model exports
│   ├── android/         # TensorFlow Lite models
│   └── ios/            # Core ML models
├── reports/             # Build reports and analysis
│   ├── coverage.xml    # Test coverage
│   ├── benchmark-results.json
│   ├── security-reports/
│   └── sbom.json       # Software Bill of Materials
└── build-manifest.json  # Build metadata
```

### Artifact Upload

For CI/CD pipelines:

```bash
# Upload to artifact storage
aws s3 cp artifacts/ s3://mobile-mm-artifacts/ --recursive

# Upload to GitHub Releases (automated via semantic-release)
gh release upload v1.0.0 dist/*.whl dist/*.tar.gz
```

## Environment Configuration

### Build Environment Variables

```bash
# Build configuration
export BUILD_TYPE=release          # release, debug
export TARGET_PLATFORM=all         # all, mobile, gpu
export ENABLE_OPTIMIZATIONS=true   # Enable build optimizations
export SKIP_TESTS=false            # Skip test execution
export ENABLE_SECURITY_SCAN=true   # Enable security scanning

# Docker configuration
export DOCKER_BUILDKIT=1           # Enable BuildKit
export COMPOSE_DOCKER_CLI_BUILD=1  # Use Docker CLI build

# Mobile development
export ANDROID_SDK_ROOT=/opt/android-sdk
export ANDROID_NDK_ROOT=/opt/android-ndk
export IOS_DEPLOYMENT_TARGET=14.0
```

### CI/CD Integration

#### GitHub Actions Example

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Environment
        run: |
          make ci-install
          
      - name: Run Build
        run: |
          ./scripts/build.sh --type release
          
      - name: Upload Artifacts
        uses: actions/upload-artifact@v3
        with:
          name: build-artifacts
          path: artifacts/
```

## Troubleshooting

### Common Build Issues

#### Docker Build Fails

```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t mobile-mm-llm:latest .
```

#### Mobile Export Fails

```bash
# Check mobile SDK installation
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import coremltools; print(coremltools.__version__)"

# Verify mobile SDKs
echo $ANDROID_SDK_ROOT
xcrun --show-sdk-path  # macOS only
```

#### Test Failures

```bash
# Run tests with verbose output
pytest -v --tb=long

# Run specific test category
pytest tests/unit/ -v

# Skip slow tests
pytest -m "not slow"
```

### Build Performance Issues

```bash
# Use parallel builds
make -j$(nproc)

# Enable Docker BuildKit
export DOCKER_BUILDKIT=1

# Use local pip cache
pip install --cache-dir ~/.cache/pip -r requirements.txt
```

## Advanced Build Scenarios

### Cross-Platform Builds

```bash
# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t mobile-mm-llm:latest .

# Mobile cross-compilation
python scripts/export_models.py --target-arch arm64 --platform android
```

### Custom Build Configurations

```bash
# Custom quantization settings
python scripts/export_models.py \
    --quantization-config configs/custom_quant.yaml \
    --calibration-samples 5000

# Development build with debug symbols
BUILD_TYPE=debug make build
```

### Integration with External Systems

```bash
# MLflow model registry
python scripts/register_model.py --model-uri models/best_model.pth

# Kubernetes deployment
kubectl apply -f k8s/deployment.yaml
```

---

## Support and Maintenance

### Build System Maintenance

- **Monthly**: Update base Docker images
- **Quarterly**: Review and update dependencies
- **Annually**: Major build system upgrades

### Getting Help

1. **Documentation**: Check this BUILD.md for detailed instructions
2. **Issues**: Create GitHub issues for build problems
3. **Discussions**: Use GitHub Discussions for questions
4. **CI Logs**: Check CI/CD logs for detailed error information

**Build System Version**: 2.0.0  
**Last Updated**: January 2025