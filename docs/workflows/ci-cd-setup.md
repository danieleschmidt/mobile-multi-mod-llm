# CI/CD Workflow Setup Guide

This document outlines the required GitHub Actions workflows for the Mobile Multi-Modal LLM project.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)

**Triggers**: Push to main/develop, Pull Requests
**Purpose**: Code quality, testing, security validation

```yaml
# Recommended workflow structure (create manually):

name: CI
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .[dev]
    
    - name: Run tests
      run: |
        make test
    
    - name: Run linting
      run: |
        make lint
    
    - name: Security scan
      run: |
        make security
```

### 2. Mobile Build Validation (`mobile.yml`)

**Triggers**: PR with mobile/ changes, Release
**Purpose**: Validate mobile export and app builds

```yaml
# Mobile validation workflow (create manually):

name: Mobile
on:
  pull_request:
    paths: ['mobile-app-*/**', 'src/mobile_multimodal/export/**']
  release:
    types: [published]

jobs:
  android:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Setup Android
      uses: android-actions/setup-android@v2
    
    - name: Export TFLite model
      run: |
        python tools/export_tflite.py --model demo --int2
    
    - name: Build Android app
      run: |
        cd mobile-app-android
        ./gradlew assembleDebug

  ios:
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v4
    - name: Setup Xcode
      uses: maxim-lobanov/setup-xcode@v1
      with:
        xcode-version: latest-stable
    
    - name: Export Core ML model
      run: |
        python tools/export_coreml.py --model demo --int2
    
    - name: Build iOS app
      run: |
        cd mobile-app-ios
        xcodebuild -scheme MultiModalDemo build
```

### 3. Security Scanning (`security.yml`)

**Triggers**: Daily, PR with dependency changes
**Purpose**: Dependency scanning, secret detection

```yaml
# Security workflow (create manually):

name: Security
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  pull_request:
    paths: ['requirements*.txt', 'pyproject.toml']

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Dependency scan
      run: |
        pip install safety
        safety check -r requirements.txt
    
    - name: Secret scan
      uses: gitguardian/ggshield-action@v1
      env:
        GITGUARDIAN_API_KEY: ${{ secrets.GITGUARDIAN_API_KEY }}
```

### 4. Release Automation (`release.yml`)

**Triggers**: Tag creation (v*)
**Purpose**: Build and publish packages

```yaml
# Release workflow (create manually):

name: Release
on:
  push:
    tags: ['v*']

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Build package
      run: |
        pip install build
        python -m build
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@v1.8.10
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

## Required Secrets

Configure these in GitHub Settings > Secrets:

| Secret | Purpose | Required For |
|--------|---------|--------------|
| `PYPI_API_TOKEN` | PyPI package publishing | Release workflow |
| `GITGUARDIAN_API_KEY` | Secret scanning | Security workflow |
| `CODECOV_TOKEN` | Code coverage reporting | CI workflow |

## Repository Settings

### Branch Protection Rules

Configure for `main` branch:
- ✅ Require status checks to pass
- ✅ Require branches to be up to date
- ✅ Required status checks:
  - `test (3.10)`
  - `test (3.11)`
  - `test (3.12)`
  - `lint`
  - `security`
- ✅ Require review from code owners
- ✅ Dismiss stale reviews
- ✅ Require signed commits

### Security Settings

- ✅ Enable Dependabot alerts
- ✅ Enable Dependabot security updates
- ✅ Enable secret scanning
- ✅ Enable code scanning (CodeQL)

## Workflow Triggers

| Event | Workflows |
|-------|-----------|
| Push to main | CI, Security |
| Pull Request | CI, Mobile (conditional) |
| Tag creation | Release |
| Schedule (daily) | Security |
| Dependency updates | Security |

## Performance Considerations

### Optimization Strategies
- **Matrix builds**: Parallel testing across Python versions
- **Conditional execution**: Mobile workflows only on relevant changes
- **Caching**: pip cache, model weights, build artifacts
- **Artifacts**: Store build outputs for debugging

### Resource Usage
- **Standard runners**: Most workflows
- **macOS runners**: iOS builds only (expensive)
- **Self-hosted**: Consider for heavy ML workloads

## Integration Points

### External Services
- **PyPI**: Package distribution
- **TestPyPI**: Pre-release testing
- **Codecov**: Coverage reporting
- **GitGuardian**: Security scanning
- **Sonar**: Code quality analysis

### Notifications
- **Slack**: Critical workflow failures
- **Email**: Release notifications
- **GitHub**: PR status updates

## Monitoring

### Key Metrics
- Workflow success rate
- Build time trends
- Test coverage percentage
- Security scan results
- Deployment frequency

### Alerts
- Failed security scans
- Dependency vulnerabilities
- Build failures on main branch
- Test coverage drops

## Manual Setup Instructions

1. **Create workflow files** in `.github/workflows/`
2. **Configure secrets** in repository settings
3. **Set up branch protection** rules
4. **Enable Dependabot** and security features
5. **Test workflows** with test branches

## Troubleshooting

### Common Issues
- **Permission errors**: Check GITHUB_TOKEN permissions
- **Missing secrets**: Verify secret configuration
- **Build failures**: Check dependency compatibility
- **Mobile builds**: Verify SDK versions

### Debug Steps
1. Check workflow logs
2. Verify secret availability
3. Test locally with same commands
4. Check runner environment differences

## Best Practices

- **Fail fast**: Quick feedback on obvious issues
- **Parallel execution**: Maximize concurrency
- **Minimal permissions**: Use least privilege principle
- **Clear naming**: Descriptive workflow and job names
- **Status badges**: Display build status in README
- **Regular updates**: Keep actions up to date