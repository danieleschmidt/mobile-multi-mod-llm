# Changelog

All notable changes to the Mobile Multi-Modal LLM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced pre-commit hooks with security scanning
- Advanced security audit script with ML-specific checks
- Mobile model export validation system
- Quantization accuracy drift monitoring
- VS Code development environment configuration
- Architecture Decision Records documentation
- CODEOWNERS file for code review governance

### Changed
- Enhanced pre-commit configuration with additional quality tools
- Improved Docker configuration for multi-environment support

### Security
- Added secrets detection baseline configuration
- Implemented advanced security scanning in pre-commit hooks
- Added license header enforcement for copyright protection

## [0.1.0] - 2025-01-29

### Added
- Initial project structure with comprehensive SDLC setup
- Multi-task mobile multimodal LLM architecture
- INT2 quantization support for Qualcomm Hexagon NPU
- Neural Architecture Search for mobile optimization
- Docker-based development environment
- Comprehensive testing framework with security and performance tests
- Monitoring stack with Prometheus and Grafana
- Documentation site with MkDocs
- Mobile app templates for Android and iOS
- Advanced Makefile with 50+ automated tasks

### Features
- **Model Architecture**: Shared vision encoder with task-specific heads
- **Quantization**: INT2 quantization with <35MB model size
- **Multi-Task Support**: Captioning, OCR, VQA, and retrieval
- **Mobile Deployment**: Android TFLite and iOS Core ML export
- **Hardware Optimization**: Qualcomm Hexagon NPU and Apple Neural Engine support

### Technical
- Python 3.10+ with modern ML libraries
- PyTorch-based training pipeline
- ONNX export for cross-platform deployment
- TensorFlow Lite for Android
- Core ML for iOS
- Docker multi-stage builds
- Pre-commit hooks with security scanning
- Comprehensive CI/CD documentation

### Documentation
- Detailed README with quick start guide
- API documentation with MkDocs
- Contributing guidelines and code of conduct
- Security policy and vulnerability reporting
- Deployment guides for mobile platforms

### Security
- Bandit static security analysis
- Safety dependency vulnerability scanning
- GitGuardian secrets detection
- Container security scanning
- SBOM generation for supply chain security

### Performance
- Benchmark suite for model evaluation
- Mobile device performance profiling
- Memory usage optimization
- Inference time monitoring
- Battery usage analysis

[Unreleased]: https://github.com/terragon-labs/mobile-multimodal-llm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/terragon-labs/mobile-multimodal-llm/releases/tag/v0.1.0