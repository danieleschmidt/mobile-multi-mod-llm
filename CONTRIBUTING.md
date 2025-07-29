# Contributing to Mobile Multi-Modal LLM

We welcome contributions to the Mobile Multi-Modal LLM project! This document provides guidelines for contributing.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/mobile-multimodal-llm.git
   cd mobile-multimodal-llm
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   make install-dev
   ```

3. **Verify setup**
   ```bash
   make test
   make lint
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**
   ```bash
   make lint      # Check code style
   make test      # Run tests
   make security  # Security checks
   ```

4. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- **Python**: Follow PEP 8, use Black formatter (88 chars)
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all modules, classes, functions
- **Tests**: pytest with >90% coverage target

## Commit Messages

Use conventional commit format:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` test additions/changes
- `refactor:` code refactoring

## Areas for Contribution

### High Priority
- INT4/INT8 quantization implementations
- Additional mobile hardware support (MediaTek, Samsung Exynos)
- Performance optimizations for mobile devices
- Model compression techniques

### Medium Priority
- New multimodal tasks (visual grounding, video understanding)
- Enhanced documentation and tutorials
- Mobile app improvements
- Benchmark suite expansions

### Getting Started
- Documentation improvements
- Test coverage improvements
- Bug fixes and minor enhancements

## Pull Request Process

1. **Pre-submission checklist**
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Security checks pass
   - [ ] Relevant issue linked

2. **PR Requirements**
   - Clear description of changes
   - Performance impact assessment
   - Breaking changes documented
   - Mobile compatibility verified

3. **Review Process**
   - Automated checks must pass
   - Code review by maintainers
   - Testing on mobile devices
   - Performance benchmarking

## Hardware Requirements

For testing mobile optimizations:
- **Android**: Device with Snapdragon 8 Gen 2+ or equivalent
- **iOS**: iPhone 12+ with A14 Bionic or newer
- **Development**: GPU-enabled machine for training

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/terragon-labs/mobile-multimodal-llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/terragon-labs/mobile-multimodal-llm/discussions)
- **Email**: mobile-ai@terragon.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.