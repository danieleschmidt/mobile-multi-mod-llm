# Development Guide

This guide covers the development setup and workflows for the Mobile Multi-Modal LLM project.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/terragon-labs/mobile-multimodal-llm.git
cd mobile-multimodal-llm
python -m venv venv && source venv/bin/activate
make install-dev

# Verify setup
make test
```

## Project Structure

```
mobile-multimodal-llm/
├── src/mobile_multimodal/     # Main package
│   ├── core.py               # Core model implementation
│   ├── models/               # Model architectures
│   ├── quantization/         # INT2/INT4/INT8 quantization
│   ├── export/               # Mobile export utilities
│   └── benchmarks/           # Performance benchmarking
├── tests/                    # Test suite
├── scripts/                  # Training and utility scripts
├── tools/                    # Development tools
├── configs/                  # Model and training configurations
├── docs/                     # Documentation
├── mobile-app-android/       # Android demo app
└── mobile-app-ios/          # iOS demo app
```

## Development Commands

| Command | Description |
|---------|-------------|
| `make install` | Install package only |
| `make install-dev` | Install with dev dependencies |
| `make test` | Run full test suite with coverage |
| `make test-fast` | Run tests without coverage |
| `make lint` | Run all code quality checks |
| `make format` | Auto-format code |
| `make security` | Run security scans |
| `make docs` | Build documentation |
| `make clean` | Clean build artifacts |

## Model Development

### Training New Models

```bash
# Neural Architecture Search
python scripts/train_nas.py --config configs/mobile_nas.yaml

# Multi-task training
python scripts/train_multitask.py --arch_checkpoint nas_best.pth

# INT2 quantization
python scripts/quantize_int2.py --model_path checkpoints/best.pth
```

### Export for Mobile

```bash
# Android (TFLite + Hexagon)
python tools/export_tflite.py --use_hexagon --int2

# iOS (Core ML + Neural Engine)
python tools/export_coreml.py --use_ane --compute_precision int2
```

## Testing Strategy

### Unit Tests
- **Location**: `tests/test_*.py`
- **Coverage**: >90% target
- **Run**: `make test`

### Integration Tests
- **Mobile export validation**
- **End-to-end inference testing**
- **Performance regression testing**

### Benchmarking
```bash
# Performance benchmarks
make benchmark

# Device profiling
python tools/benchmark_device.py --device "Pixel 8 Pro"
```

## Code Quality

### Pre-commit Hooks
Automatically run on every commit:
- Black formatting
- isort import sorting
- flake8 linting
- mypy type checking
- bandit security scanning

### Manual Quality Checks
```bash
make lint      # All linting tools
make security  # Security-specific checks
make format    # Auto-formatting
```

### Type Checking
- **Required**: Type hints for all public APIs
- **Tool**: mypy with strict configuration
- **Config**: `pyproject.toml`

## Mobile Development

### Android Development
```bash
cd mobile-app-android
./gradlew assembleDebug
```

**Requirements**:
- Android Studio Arctic Fox+
- NDK r25+
- Qualcomm Hexagon SDK 5.5.0+

### iOS Development
```bash
cd mobile-app-ios
xcodebuild -scheme MultiModalDemo build
```

**Requirements**:
- Xcode 15+
- iOS 14+ deployment target
- Core ML 6.0+

## Hardware-Specific Development

### Qualcomm Hexagon NPU
```bash
# Setup Hexagon SDK
export HEXAGON_SDK_ROOT=/path/to/hexagon_sdk
export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools

# Profile on device
adb shell "cd /data/local/tmp && ./hexagon_profiler model.dlc"
```

### Apple Neural Engine
```bash
# Profile Core ML model
python tools/profile_coreml.py --model model.mlpackage
```

## Debugging

### Model Debugging
```bash
# Trace model execution
python tools/trace_model.py --model checkpoints/model.pth

# Compare quantized vs original
python tools/verify_quantized.py --original model.pth --quantized model.tflite
```

### Mobile Debugging
```bash
# Android logs
adb logcat | grep "MultiModal"

# iOS logs (Xcode Console or)
xcrun simctl spawn booted log stream --predicate 'subsystem CONTAINS "multimodal"'
```

## Performance Optimization

### Profiling Tools
```bash
# Python profiling
python -m cProfile scripts/benchmark.py

# Model optimization
python tools/optimize_model.py --model checkpoints/model.pth
```

### Benchmarking
```bash
# Inference benchmarks
pytest tests/ -m benchmark --benchmark-only

# Memory profiling
python tools/memory_profile.py
```

## Documentation

### Building Docs
```bash
make docs        # Build static docs
make docs-serve  # Serve locally at http://localhost:8000
```

### Documentation Structure
- **User Guide**: Installation, quickstart, tutorials
- **API Reference**: Auto-generated from docstrings
- **Developer Guide**: This document
- **Model Zoo**: Pre-trained model documentation

## Troubleshooting

### Common Issues

**Import errors after installation**
```bash
pip install -e .[dev]  # Editable install
```

**Tests fail with CUDA errors**
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU testing
```

**Pre-commit hooks fail**
```bash
pre-commit run --all-files  # Fix all files
```

### Environment Issues
- **Python 3.10+** required
- **Virtual environment** strongly recommended
- **GPU memory**: 8GB+ recommended for training

### Getting Help
- Check existing [issues](https://github.com/terragon-labs/mobile-multimodal-llm/issues)
- Create detailed bug reports with environment info
- Join [discussions](https://github.com/terragon-labs/mobile-multimodal-llm/discussions)