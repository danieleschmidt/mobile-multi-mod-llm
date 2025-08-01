# Mobile Multi-Modal LLM Test Suite

Comprehensive testing infrastructure for the Mobile Multi-Modal LLM project, covering unit tests, integration tests, performance benchmarks, and mobile platform validation.

## Test Structure

```
tests/
├── fixtures/           # Test fixtures and utilities
│   ├── model_fixtures.py    # Model-related fixtures
│   ├── data_fixtures.py     # Data and dataset fixtures
│   └── mobile_fixtures.py   # Mobile platform fixtures
├── unit/              # Unit tests (fast, isolated)
│   └── test_model_components.py
├── integration/       # Integration tests (cross-component)
│   └── test_end_to_end.py
├── benchmarks/        # Performance benchmarks
│   └── test_performance.py
├── security/          # Security tests
│   └── test_security.py
├── chaos/             # Chaos engineering tests
│   └── test_resilience.py
├── performance/       # Performance regression tests
│   └── test_regression.py
├── data/              # Test data files
└── logs/              # Test execution logs
```

## Test Categories

### 🚀 Unit Tests
Fast, isolated tests for individual components.

```bash
# Run all unit tests
pytest -m unit

# Run specific component tests
pytest tests/unit/test_model_components.py -v
```

**Coverage:**
- Vision encoder functionality
- Text encoder functionality  
- Multi-task decoder heads
- Model utilities and interfaces
- Configuration handling

### 🔗 Integration Tests
Tests that verify component interactions and end-to-end workflows.

```bash
# Run integration tests
pytest -m integration

# Run end-to-end pipeline tests
pytest tests/integration/test_end_to_end.py -v
```

**Coverage:**
- Training to deployment pipeline
- Multi-modal task workflows
- Data processing pipelines
- Mobile platform integration
- Error handling and edge cases

### 📱 Mobile Tests
Tests for mobile platform deployment and optimization.

```bash
# Run mobile tests (requires SDKs)
pytest -m mobile --run-mobile

# Test specific mobile platforms
pytest -k "tflite" -m mobile
pytest -k "coreml" -m mobile
pytest -k "hexagon" -m mobile
```

**Coverage:**
- TensorFlow Lite integration
- Core ML integration
- Hexagon NPU acceleration
- Quantization validation
- Mobile performance benchmarks

### ⚡ Performance Tests
Benchmarking and performance validation tests.

```bash
# Run performance tests
pytest -m benchmark --run-slow

# Run with performance profiling
pytest tests/benchmarks/test_performance.py --benchmark-only
```

**Coverage:**
- Inference latency benchmarks
- Memory usage profiling
- Throughput measurements
- Quantization impact analysis
- Hardware-specific optimizations

### 🔒 Security Tests
Security-focused tests for vulnerability detection.

```bash
# Run security tests
pytest tests/security/test_security.py -v

# Run adversarial testing
pytest -k "adversarial" -m security
```

**Coverage:**
- Input validation
- Adversarial robustness
- Model security
- Data privacy
- Dependency vulnerabilities

### 🌪️ Chaos Tests
Chaos engineering tests for system resilience.

```bash
# Run chaos tests
pytest tests/chaos/test_resilience.py -v --run-slow
```

**Coverage:**
- Network failures
- Memory pressure
- Hardware failures
- Resource exhaustion
- Graceful degradation

## Running Tests

### Quick Test Commands

```bash
# Run all tests (fast subset)
pytest

# Run all tests including slow ones
pytest --run-slow

# Run tests with coverage
pytest --cov

# Run tests in parallel
pytest -n auto

# Run specific test categories
pytest -m "unit or integration"
pytest -m "not slow"
pytest -m "mobile and not gpu"
```

### Advanced Test Execution

```bash
# Run tests on specific hardware
pytest --run-gpu -m gpu        # GPU tests
pytest --run-mobile -m mobile  # Mobile SDK tests

# Performance testing
pytest -m benchmark --benchmark-only --benchmark-sort=mean

# Debugging tests
pytest --pdb -s -v tests/unit/test_model_components.py::TestVisionEncoder::test_vision_encoder_forward

# Generate detailed coverage report
pytest --cov --cov-report=html
open htmlcov/index.html
```

### Continuous Integration

```bash
# CI test pipeline (fast tests only)
pytest -m "not slow and not mobile and not gpu" --tb=short --durations=10

# Nightly test pipeline (all tests)
pytest --run-slow --run-mobile --run-gpu --tb=short
```

## Test Configuration

### Environment Variables

```bash
# Test execution control
export TESTING=1                    # Enable testing mode
export ENABLE_CUDA_TESTS=1         # Enable GPU tests
export RUN_SLOW_TESTS=1            # Enable slow tests
export RUN_MOBILE_TESTS=1          # Enable mobile tests

# Test data and artifacts
export TEST_DATA_DIR=/tmp/test_data
export MODEL_CACHE_DIR=/tmp/models
export TORCH_HOME=/tmp/torch_models

# Logging
export PYTEST_LOG_LEVEL=INFO
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

### Pytest Configuration

Key settings in `pytest.ini`:

- **Coverage threshold**: 85% minimum
- **Test timeout**: 300 seconds
- **Parallel execution**: Auto-scaling
- **Comprehensive markers**: 15+ test categories
- **Strict configuration**: No unmarked tests

## Test Data Management

### Fixtures and Mock Data

```python
# Use pre-built fixtures
from tests.fixtures import create_sample_model, create_test_dataset, mock_mobile_runtime

# Model fixtures
model = create_sample_model()

# Data fixtures  
dataset = create_test_dataset(size=100)

# Mobile fixtures
mobile_runtimes = mock_mobile_runtime()
```

### Test Data Generation

```python
# Generate test images
from tests.fixtures.data_fixtures import DataFixture

images = DataFixture.create_sample_images(count=10)
captions = DataFixture.create_sample_captions(count=10)
vqa_pairs = DataFixture.create_vqa_pairs(count=20)
```

## Performance Benchmarking

### Benchmark Execution

```bash
# Run performance benchmarks
pytest tests/benchmarks/ --benchmark-only

# Compare benchmarks
pytest tests/benchmarks/ --benchmark-compare=baseline.json

# Save benchmark results
pytest tests/benchmarks/ --benchmark-save=current
```

### Benchmark Metrics

- **Inference Latency**: < 15ms on flagship devices
- **Memory Usage**: < 150MB peak runtime
- **Model Size**: < 35MB quantized
- **Accuracy**: > 90% of FP32 baseline
- **Throughput**: > 60 FPS on high-end devices

## Mobile Testing

### Prerequisites

```bash
# Android testing
export ANDROID_SDK_ROOT=/opt/android-sdk
export ANDROID_NDK_ROOT=/opt/android-ndk

# iOS testing  
export XCODE_VERSION=15.0
export IOS_DEPLOYMENT_TARGET=14.0

# Install mobile SDKs
pip install tensorflow-lite coremltools onnxruntime
```

### Mobile Test Execution

```bash
# Test TensorFlow Lite export
pytest tests/integration/test_end_to_end.py::TestMobileIntegration::test_android_tflite_integration

# Test Core ML export
pytest tests/integration/test_end_to_end.py::TestMobileIntegration::test_ios_coreml_integration

# Test Hexagon NPU
pytest tests/integration/test_end_to_end.py::TestMobileIntegration::test_hexagon_npu_integration
```

## Test Debugging

### Debug Test Failures

```bash
# Run with debugger
pytest --pdb tests/unit/test_model_components.py::test_failing_function

# Verbose output
pytest -v -s tests/integration/

# Show local variables in tracebacks
pytest --tb=long --showlocals

# Run only failed tests
pytest --lf
```

### Test Analysis

```bash
# Test execution time analysis
pytest --durations=0

# Coverage analysis
pytest --cov --cov-report=term-missing

# Test dependencies
pytest --collect-only
```

## Continuous Integration

### GitHub Actions Integration

```yaml
# Example CI workflow
- name: Run Tests
  run: |
    pytest -m "not slow and not mobile" --cov --tb=short
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Quality Gates

- **Unit Test Coverage**: ≥ 85%
- **Integration Test Coverage**: ≥ 70%
- **Performance Regression**: < 5% slower
- **Security Vulnerabilities**: Zero critical/high
- **Mobile Compatibility**: 100% passing

## Contributing to Tests

### Adding New Tests

1. **Choose appropriate category**: unit, integration, mobile, etc.
2. **Use existing fixtures** when possible
3. **Follow naming conventions**: `test_[component]_[behavior]`
4. **Add appropriate markers**: `@pytest.mark.unit`
5. **Include docstrings** explaining test purpose

### Test Best Practices

- **Fast unit tests**: < 100ms execution time
- **Isolated tests**: No shared state between tests
- **Deterministic tests**: Same input → same output
- **Clear assertions**: Meaningful error messages
- **Comprehensive coverage**: Happy path + edge cases

### Example Test Structure

```python
import pytest
from tests.fixtures import create_sample_model

class TestModelComponent:
    """Test model component functionality."""
    
    def test_component_basic_functionality(self):
        """Test basic component operation."""
        # Arrange
        model = create_sample_model()
        input_data = create_test_input()
        
        # Act
        result = model.process(input_data)
        
        # Assert
        assert result is not None
        assert result.shape == expected_shape
        assert not torch.isnan(result).any()
    
    @pytest.mark.slow
    def test_component_performance(self):
        """Test component performance characteristics."""
        # Performance validation logic
        pass
    
    @pytest.mark.mobile
    def test_component_mobile_compatibility(self):
        """Test component mobile deployment."""
        # Mobile-specific testing logic
        pass
```

## Test Metrics and Reporting

### Coverage Reports

- **HTML Report**: `htmlcov/index.html`
- **XML Report**: `coverage.xml` (for CI)
- **Terminal Report**: Real-time coverage display

### Performance Reports

- **Benchmark JSON**: Detailed performance metrics
- **Regression Analysis**: Performance trends over time
- **Mobile Performance**: Device-specific benchmarks

### Test Execution Reports

- **JUnit XML**: `test-results.xml` (for CI)
- **Test Logs**: `tests/logs/pytest.log`
- **Failure Analysis**: Detailed failure reports

---

## Support

For test-related questions or issues:

1. **Check existing tests** for similar patterns
2. **Review test documentation** for guidance
3. **Run `pytest --help`** for command options
4. **Create GitHub issues** for test infrastructure problems

**Happy Testing! 🧪**