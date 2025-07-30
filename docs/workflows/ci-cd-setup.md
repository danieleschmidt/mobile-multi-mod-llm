# GitHub Actions CI/CD Setup Guide

This document provides comprehensive GitHub Actions workflow templates for the mobile multimodal LLM project.

## Required Workflows

### 1. Main CI Workflow (`ci.yml`)

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        make ci-install
    
    - name: Run tests
      run: |
        make ci-test
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: 3.11
    - name: Install dependencies
      run: make ci-install
    - name: Run linting
      run: make ci-lint

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: 3.11
    - name: Install dependencies
      run: make ci-install
    - name: Run security checks
      run: make ci-security
```

### 2. Mobile Build Workflow (`mobile-build.yml`)

```yaml
name: Mobile Model Build

on:
  push:
    branches: [ main ]
    paths:
      - 'src/mobile_multimodal/**'
      - 'scripts/export_models.py'
  workflow_dispatch:

jobs:
  mobile-export:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        make ci-install
        pip install -e .[mobile]
    
    - name: Export mobile models
      run: |
        make mobile-export
    
    - name: Test mobile exports
      run: |
        make mobile-test-export
    
    - name: Upload mobile artifacts
      uses: actions/upload-artifact@v3
      with:
        name: mobile-models
        path: |
          models/mobile/
          exports/
```

### 3. Security Scan Workflow (`security.yml`)

```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: make ci-install
    
    - name: Run Bandit security scan
      run: |
        bandit -r src/ -f sarif -o bandit-results.sarif
    
    - name: Upload Bandit results to GitHub
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: bandit-results.sarif
    
    - name: Run dependency vulnerability scan
      run: |
        safety check --json --output safety-results.json
        pip-audit --format=json --output=pip-audit-results.json
    
    - name: Generate SBOM
      run: |
        make generate-sbom
    
    - name: Upload security artifacts
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-results.sarif
          safety-results.json
          pip-audit-results.json
          reports/sbom.json
```

### 4. Performance Benchmarking (`benchmark.yml`)

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'tests/benchmarks/**'
  schedule:
    - cron: '0 4 * * 1'  # Weekly on Monday at 4 AM

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        make ci-install
        pip install pytest-benchmark
    
    - name: Run benchmarks
      run: |
        make benchmark
    
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: reports/benchmark-results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '200%'
        fail-on-alert: true
```

### 5. Docker Build and Push (`docker.yml`)

```yaml
name: Docker Build and Push

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to Container Registry
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

### 6. SLSA Provenance (`slsa.yml`)

```yaml
name: SLSA Provenance

on:
  push:
    tags: [ 'v*' ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      artifact-name: ${{ steps.build.outputs.artifact-name }}
      artifact-sha256: ${{ steps.build.outputs.artifact-sha256 }}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: make ci-install
    
    - name: Build package
      id: build
      run: |
        make build
        echo "artifact-name=$(ls dist/*.whl | head -1)" >> $GITHUB_OUTPUT
        echo "artifact-sha256=$(sha256sum dist/*.whl | cut -d' ' -f1)" >> $GITHUB_OUTPUT
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: python-package
        path: dist/

  provenance:
    needs: build
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.5.0
    with:
      base64-subjects: "${{ needs.build.outputs.artifact-sha256 }} ${{ needs.build.outputs.artifact-name }}"
    secrets:
      registry-password: ${{ secrets.GITHUB_TOKEN }}
```

### 7. Chaos Engineering (`chaos.yml`)

```yaml
name: Chaos Engineering

on:
  schedule:
    - cron: '0 3 * * 2'  # Weekly on Tuesday at 3 AM
  workflow_dispatch:

jobs:
  chaos-testing:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: make ci-install
    
    - name: Run chaos engineering tests
      run: |
        pytest tests/chaos/ -v --tb=short
    
    - name: Generate chaos report
      run: |
        python scripts/chaos-testing-runner.py --report-only
    
    - name: Upload chaos report
      uses: actions/upload-artifact@v3
      with:
        name: chaos-engineering-report
        path: reports/chaos-report.json
```

## Setup Instructions

1. **Create workflows directory:**
   ```bash
   mkdir -p .github/workflows
   ```

2. **Add workflow files:**
   Copy the above YAML content into separate files in `.github/workflows/`

3. **Configure secrets:**
   - `CODECOV_TOKEN` for coverage reporting
   - `DOCKER_HUB_USERNAME` and `DOCKER_HUB_ACCESS_TOKEN` for Docker Hub
   - GitHub token is automatically provided

4. **Branch protection rules:**
   - Require status checks to pass
   - Require branches to be up to date
   - Require review from code owners
   - Include administrators in restrictions

5. **Enable GitHub Advanced Security:**
   - Code scanning with CodeQL
   - Secret scanning
   - Dependency review

## Required GitHub Secrets

- `CODECOV_TOKEN`: For coverage reporting
- `DOCKER_HUB_USERNAME`: Docker Hub username
- `DOCKER_HUB_ACCESS_TOKEN`: Docker Hub access token

## Monitoring and Alerts

The workflows include:
- Performance regression detection
- Security vulnerability alerts
- Build failure notifications
- Coverage decrease warnings
- Chaos engineering failure alerts

## Customization

Adjust these workflows based on:
- Repository specific requirements
- Mobile deployment targets
- Security compliance needs
- Performance benchmarking requirements
- Team workflow preferences