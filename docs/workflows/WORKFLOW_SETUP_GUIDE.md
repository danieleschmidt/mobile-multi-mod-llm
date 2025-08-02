# GitHub Actions Workflow Setup Guide

This guide provides step-by-step instructions for implementing the Mobile Multi-Modal LLM CI/CD workflows in your repository.

## Overview

Our workflow automation provides:
- **10 comprehensive workflows** covering all aspects of SDLC
- **Mobile-first approach** with Android, iOS, and cross-platform validation
- **Enterprise-grade security** with SLSA L3 compliance
- **Performance monitoring** with regression detection
- **Automated dependency management** with vulnerability patching

## Prerequisites

### 1. Repository Permissions

Ensure your repository has the following permissions configured:

```yaml
# In repository settings > Actions > General
permissions:
  contents: write      # For creating releases and updating files
  packages: write      # For publishing Docker images
  security-events: write  # For uploading SARIF security scan results
  id-token: write      # For SLSA provenance generation
  issues: write        # For commenting on PRs
  pull-requests: write # For PR status updates
```

### 2. Required Secrets

Add the following secrets in repository settings:

#### Core Secrets
```bash
# PyPI publishing (required for release automation)
PYPI_API_TOKEN="pypi-AgEIcHlwaS5vcmcC..."

# Code coverage (optional but recommended)
CODECOV_TOKEN="your-codecov-token"

# Container registry (for Docker publishing)
GHCR_TOKEN="${{ secrets.GITHUB_TOKEN }}"  # Automatically available
```

#### Security Scanning Secrets (optional)
```bash
# For enhanced security scanning
SNYK_TOKEN="your-snyk-token"
SONAR_TOKEN="your-sonarcloud-token"
```

#### Mobile Development Secrets (optional)
```bash
# For mobile app store publishing
ANDROID_KEYSTORE_PASSWORD="your-keystore-password"
IOS_CERTIFICATE_PASSWORD="your-certificate-password"
```

### 3. Branch Protection Rules

Configure branch protection for `main` branch:

1. Go to repository Settings > Branches
2. Add rule for `main` branch:
   ```
   ✅ Require status checks to pass before merging
   ✅ Require branches to be up to date before merging
   ✅ Require review from code owners
   ✅ Restrict pushes that create files larger than 100MB
   ✅ Include administrators
   ```

3. Required status checks:
   ```
   - CI Success
   - Security Scan
   - Mobile Build Validation
   - Performance Benchmarks (for releases)
   ```

## Implementation Steps

### Phase 1: Core Infrastructure (Week 1)

#### Step 1: Create Workflow Directory

```bash
mkdir -p .github/workflows
```

#### Step 2: Implement Core CI/CD Pipeline

Copy the following workflows from `docs/workflows/examples/`:

```bash
# Core CI/CD pipeline
cp docs/workflows/examples/ci.yml .github/workflows/

# Mobile build validation
cp docs/workflows/examples/mobile-build.yml .github/workflows/

# Security scanning
cp docs/workflows/examples/security.yml .github/workflows/
```

#### Step 3: Test Basic Pipeline

1. Create a test PR
2. Verify all workflows trigger
3. Check workflow status in Actions tab
4. Fix any configuration issues

### Phase 2: Advanced Features (Week 2)

#### Step 4: Add Performance Monitoring

```bash
# Performance regression detection
cp docs/workflows/examples/performance.yml .github/workflows/

# Container security
cp docs/workflows/examples/docker.yml .github/workflows/
```

#### Step 5: Implement Dependency Management

```bash
# Automated dependency updates
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

#### Step 6: Configure Notifications

Add Slack/Teams webhooks for important events:

```yaml
# In workflow files, add notification step
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Phase 3: Production Ready (Week 3-4)

#### Step 7: Add SLSA Compliance

```bash
# SLSA build provenance
cp docs/workflows/examples/slsa.yml .github/workflows/
```

#### Step 8: Implement Chaos Engineering

```bash
# Resilience testing
cp docs/workflows/examples/chaos.yml .github/workflows/
```

#### Step 9: Add Release Automation

```bash
# Automated releases
cp docs/workflows/examples/release.yml .github/workflows/
```

#### Step 10: Mobile Fleet Monitoring

```bash
# Mobile deployment monitoring
cp docs/workflows/examples/mobile-monitoring.yml .github/workflows/
```

## Workflow Configuration

### Environment Variables

Set the following environment variables in your workflows:

```yaml
env:
  # Python configuration
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'
  
  # Mobile development
  ANDROID_API_LEVEL: '34'
  IOS_DEPLOYMENT_TARGET: '14.0'
  
  # Performance thresholds
  MAX_MODEL_SIZE_MB: '35'
  MAX_INFERENCE_TIME_MS: '100'
  MIN_ACCURACY_THRESHOLD: '0.90'
  
  # Security configuration
  ENABLE_SECURITY_SCANNING: 'true'
  SECURITY_SCAN_SEVERITY: 'high'
```

### Workflow Triggers

Configure appropriate triggers for each workflow:

```yaml
# Core CI - run on every change
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

# Performance - run on schedule and releases
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  release:
    types: [published]

# Security - run daily and on security-related changes
on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM
  push:
    paths:
      - 'requirements*.txt'
      - 'pyproject.toml'
      - 'Dockerfile'
```

## Customization Guide

### Adding Custom Checks

To add custom validation steps:

1. Create a new job in the appropriate workflow:

```yaml
custom-validation:
  name: Custom Validation
  runs-on: ubuntu-latest
  steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Run custom check
      run: |
        # Your custom validation logic
        python scripts/custom_validation.py
```

2. Add it to the dependency chain:

```yaml
ci-success:
  needs: [lint, test, mobile-build, custom-validation]
```

### Modifying Mobile Platforms

To add support for additional mobile platforms:

1. Add platform-specific export job:

```yaml
webassembly-export:
  name: WebAssembly Export
  runs-on: ubuntu-latest
  steps:
    - name: Export WebAssembly model
      run: |
        python scripts/export_models.py --platform wasm
```

2. Update cross-platform validation:

```yaml
cross-platform-validation:
  needs: [android-export, ios-export, onnx-export, webassembly-export]
```

### Performance Threshold Configuration

Adjust performance thresholds in `performance.yml`:

```yaml
- name: Check performance thresholds
  run: |
    python scripts/check_performance.py \
      --max-inference-time 100 \
      --max-memory-usage 512 \
      --min-accuracy 0.90
```

## Monitoring and Maintenance

### Workflow Health Monitoring

Monitor workflow health using:

1. **GitHub Actions Dashboard**: View success rates and trends
2. **Workflow Run Analytics**: Analyze performance over time
3. **Alert Configuration**: Set up notifications for failures

### Regular Maintenance Tasks

#### Weekly
- [ ] Review failed workflow runs
- [ ] Update performance baselines
- [ ] Check security scan results
- [ ] Validate mobile compatibility matrix

#### Monthly
- [ ] Update workflow dependencies
- [ ] Review and adjust performance thresholds
- [ ] Audit security configurations
- [ ] Optimize workflow performance

#### Quarterly
- [ ] Review workflow architecture
- [ ] Update mobile platform support
- [ ] Assess new GitHub Actions features
- [ ] Update documentation

### Performance Optimization

Optimize workflow performance by:

1. **Caching Dependencies**:
```yaml
- name: Cache pip dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
```

2. **Parallel Execution**:
```yaml
strategy:
  matrix:
    platform: [android, ios, onnx]
  max-parallel: 3
```

3. **Conditional Execution**:
```yaml
if: contains(github.event.head_commit.message, '[mobile]')
```

## Troubleshooting

### Common Issues

#### Workflow Not Triggering
1. Check workflow file syntax with `yamllint`
2. Verify file is in `.github/workflows/` directory
3. Check branch protection rules
4. Validate trigger conditions

#### Permission Errors
1. Verify repository permissions in Settings > Actions
2. Check token scopes for external services
3. Validate secrets configuration

#### Mobile Build Failures
1. Check mobile SDK availability
2. Verify export script compatibility
3. Review model size constraints
4. Validate platform-specific dependencies

#### Performance Test Failures
1. Review performance thresholds
2. Check test environment consistency
3. Validate benchmark data quality
4. Consider baseline adjustment

### Debug Commands

```bash
# Test workflow locally (using act)
act -j ci-test

# Validate workflow syntax
yamllint .github/workflows/*.yml

# Check workflow logs
gh run list --limit 10
gh run view <run-id> --log

# Test mobile exports locally
python scripts/export_models.py --platform all --test-only
```

### Getting Help

If you encounter issues:

1. **Check Documentation**: Review this guide and workflow comments
2. **Search Issues**: Look for similar problems in repository issues
3. **Create Issue**: Provide workflow logs and configuration details
4. **Community Support**: Ask in GitHub Actions community forums

## Success Metrics

Track these metrics to measure workflow effectiveness:

### Technical Metrics
- **Build Success Rate**: Target >98%
- **Average Build Time**: Target <15 minutes
- **Security Scan Coverage**: 100%
- **Performance Regression Detection**: <24 hours

### Mobile Metrics
- **Mobile Export Success Rate**: Target >99%
- **Cross-Platform Compatibility**: Target 95%
- **Model Size Compliance**: 100%
- **Performance Threshold Compliance**: Target >95%

### Developer Experience
- **Time to Feedback**: Target <10 minutes for PR checks
- **False Positive Rate**: Target <5%
- **Developer Productivity**: Measured via surveys
- **Workflow Adoption**: Track usage across team

---

## Next Steps

After implementing these workflows:

1. **Monitor Performance**: Track metrics for 2-4 weeks
2. **Gather Feedback**: Survey development team
3. **Optimize Configuration**: Adjust based on real usage
4. **Expand Coverage**: Add additional platforms or checks
5. **Document Learnings**: Update this guide with insights

**Need help?** Create an issue with the `workflow-support` label for assistance.