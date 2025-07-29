# Contributing to Liquid Audio Neural Networks

Thank you for your interest in contributing to liquid-audio-nets! This guide will help you get started with contributing to our edge-efficient audio processing library.

## 🚀 Quick Start

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/liquid-audio-nets.git
   cd liquid-audio-nets
   ```

2. **Set up Development Environment**
   ```bash
   make dev-setup
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🏗️ Development Setup

### Prerequisites

- **Python 3.8+** with pip
- **Rust 1.70+** with cargo
- **CMake 3.16+** for C++ builds
- **Git** for version control

### Optional (for embedded development)
- **ARM GCC Toolchain** for embedded targets
- **OpenOCD** for hardware debugging
- **CMSIS-DSP** library for ARM optimizations

### Environment Setup

```bash
# Install development dependencies
make dev-setup

# Run tests to verify setup
make test

# Build all components
make build
```

## 📝 Contributing Guidelines

### Code Style

We use automated formatting and linting:

- **Python**: Black + Ruff + MyPy
- **Rust**: rustfmt + clippy
- **C++**: clang-format + clang-tidy

Run before committing:
```bash
make format
make lint
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add adaptive timestep controller for ARM Cortex-M4
fix: resolve memory leak in ODE solver
docs: update embedded deployment guide
test: add integration tests for keyword spotting
```

### Testing

All contributions must include appropriate tests:

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **Embedded Tests**: Test on target hardware (if applicable)

```bash
# Run all tests
make test

# Run specific test suites
make test-python
make test-rust
make test-cpp
```

## 🎯 Areas for Contribution

### High Priority
- **RISC-V Optimizations**: Port ARM optimizations to RISC-V
- **TensorFlow Lite Micro**: Integration for broader ecosystem support
- **Additional Audio Tasks**: Noise suppression, acoustic event detection
- **Hardware Accelerator Support**: TPU, Neural Processing Units

### Medium Priority
- **Model Compression**: Advanced quantization techniques
- **Power Profiling Tools**: Better embedded power measurement
- **Documentation**: More examples and tutorials
- **Benchmarks**: Comparative performance studies

### Ongoing
- **Bug Fixes**: Always welcome
- **Performance Improvements**: Optimization opportunities
- **Test Coverage**: Expand test suites
- **Platform Support**: New embedded targets

## 🔧 Development Workflows

### Adding a New Feature

1. **Create an Issue** describing the feature
2. **Discuss the Approach** with maintainers
3. **Implement** with tests and documentation
4. **Submit PR** with detailed description

### Bug Fixes

1. **Reproduce the Bug** with a test case
2. **Fix the Issue** while maintaining backwards compatibility
3. **Add Regression Tests** to prevent recurrence
4. **Update Documentation** if needed

### Performance Optimizations

1. **Establish Baseline** with benchmarks
2. **Profile** to identify bottlenecks
3. **Optimize** with measurable improvements
4. **Validate** on target hardware

## 📚 Technical Guidelines

### Python Development

- Use type hints for all public APIs
- Follow PEP 8 style guidelines
- Write docstrings for all public functions
- Use dataclasses for configuration objects

### Rust Development

- Follow Rust API guidelines
- Use `#![no_std]` for embedded components
- Implement proper error handling with `Result<T, E>`
- Write comprehensive documentation comments

### C++ Development

- Use C++17 features appropriately
- Follow RAII principles
- Use smart pointers for memory management
- Optimize for embedded constraints

### Embedded Considerations

- **Memory Usage**: Profile RAM and Flash usage
- **Power Consumption**: Measure and optimize power draw
- **Real-time Constraints**: Ensure deterministic performance
- **Hardware Compatibility**: Test on target microcontrollers

## 🧪 Testing on Hardware

### Supported Platforms

- **STM32F4** series (primary target)
- **nRF52840** (secondary target)
- **ESP32-S3** (experimental)

### Hardware-in-the-Loop Testing

```bash
# Run HIL tests (requires connected hardware)
python tests/hil/test_stm32f4.py --device /dev/ttyUSB0

# Power profiling
liquid-profile --device stm32f407 --model keyword.lnn
```

## 📖 Documentation

### Adding Documentation

- **API Docs**: Document all public interfaces
- **Guides**: Step-by-step tutorials for common tasks
- **Examples**: Complete working examples
- **Architecture**: Design decisions and trade-offs

### Building Documentation

```bash
make docs
# Open docs/_build/index.html
```

## 🔒 Security

### Security Considerations

- Never commit secrets, keys, or credentials
- Follow secure coding practices
- Report security issues privately to maintainers
- Use the security check script: `python scripts/security_check.py`

### Embedded Security

- Validate all inputs from external sources
- Use secure boot and firmware signing
- Implement fail-safe mechanisms
- Consider side-channel attack mitigations

## 🤝 Community

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Technical questions and ideas
- **Pull Requests**: Code contributions

### Code of Conduct

We follow the [Contributor Covenant](CODE_OF_CONDUCT.md). Please be respectful and inclusive in all interactions.

### Getting Help

- Check existing issues and documentation
- Ask questions in GitHub Discussions
- Join our community forums (coming soon)

## 🏆 Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **Academic papers** (for research contributions)

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

Thank you for helping make liquid-audio-nets better! 🎵