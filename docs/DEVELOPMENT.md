# Development Guide

This guide covers the development setup and workflow for liquid-audio-nets.

## Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Rust 1.70+** with cargo  
- **CMake 3.16+** for C++ builds
- **Git** for version control

### One-Command Setup

```bash
# Clone and setup in one go
git clone https://github.com/terragon-labs/liquid-audio-nets.git
cd liquid-audio-nets
make dev-setup
```

This will:
- Install all dependencies (Python, Rust, C++)
- Set up pre-commit hooks
- Configure development tools
- Run initial tests

## Development Environments

### Option 1: Local Development

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake pkg-config

# Set up development environment
make dev-setup

# Verify setup
make test
```

### Option 2: VS Code Dev Container

```bash
# Open in VS Code with dev container
code .
# VS Code will prompt to reopen in container
```

The dev container includes:
- All required tools and dependencies
- Pre-configured VS Code extensions
- ARM embedded toolchain
- Hardware debugging tools

### Option 3: Docker Development

```bash
# Build development image
docker build -f .devcontainer/Dockerfile -t liquid-audio-dev .

# Run development container
docker run -it -v $(pwd):/workspace liquid-audio-dev
```

## Project Structure

```
liquid-audio-nets/
├── python/                 # Python package
│   ├── liquid_audio_nets/  # Main Python module
│   │   ├── lnn.py          # Core LNN implementation
│   │   ├── training.py     # Training utilities
│   │   └── tools/          # CLI tools
│   └── tests/              # Python tests
├── src/                    # Rust/C++ source
│   ├── core/               # Core implementations
│   ├── platform/           # Platform-specific code
│   └── python/             # Python bindings
├── include/                # C++ headers
├── examples/               # Usage examples
├── docs/                   # Documentation
├── tests/                  # Cross-language tests
└── scripts/                # Build and utility scripts
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/awesome-feature

# Make changes...
# Write tests...

# Run pre-commit checks
pre-commit run --all-files

# Run full test suite
make test

# Commit changes
git commit -m "feat: add awesome feature"

# Push and create PR
git push origin feature/awesome-feature
```

### 2. Testing

```bash
# Run all tests
make test

# Run specific language tests
make test-python
make test-rust  
make test-cpp

# Run with coverage
pytest tests/ --cov=liquid_audio_nets
cargo test --all-features

# Run integration tests
pytest tests/integration/ -v

# Run embedded tests (requires hardware)
python tests/embedded/test_stm32f4.py
```

### 3. Code Quality

```bash
# Format all code
make format

# Lint all code
make lint

# Security checks
python scripts/security_check.py
detect-secrets scan --all-files

# Dependency audit
cargo audit
safety check
```

### 4. Documentation

```bash
# Build documentation
make docs

# Serve documentation locally
cd docs/_build && python -m http.server 8000
# Open http://localhost:8000

# Update API documentation
sphinx-apidoc -o docs/api python/liquid_audio_nets
```

## Language-Specific Development

### Python Development

```bash
# Install in development mode
pip install -e ".[dev,training,embedded]"

# Run Python tests
pytest tests/ -v

# Type checking
mypy python/liquid_audio_nets/

# Profile Python code
python -m cProfile -o profile.stats scripts/benchmark.py
snakeviz profile.stats
```

### Rust Development

```bash
# Build Rust library
cargo build --release

# Run Rust tests
cargo test --all-features

# Check without building
cargo check

# Lint Rust code
cargo clippy --all-targets --all-features -- -D warnings

# Format Rust code
cargo fmt

# Generate documentation
cargo doc --open

# Run benchmarks
cargo bench
```

### C++ Development

```bash
# Configure build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
gdb ./tests/test_lnn_core

# ARM embedded build
cmake .. -DENABLE_EMBEDDED_BUILD=ON -DENABLE_ARM_OPTIMIZATIONS=ON
make -j$(nproc)
```

## Hardware Development

### Embedded Targets

Supported platforms:
- **STM32F4** series (primary)
- **nRF52840** (secondary) 
- **ESP32-S3** (experimental)

### Cross-compilation

```bash
# ARM Cortex-M4 (STM32F4)
cargo build --target thumbv7em-none-eabihf --release

# Generate bindings for embedded
cbindgen --config cbindgen.toml --crate liquid-audio-nets --output include/liquid_audio.h

# Flash to hardware
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
        -c "program target/thumbv7em-none-eabihf/release/liquid_audio_embedded.elf verify reset exit"
```

### Hardware-in-the-Loop Testing

```bash
# Connect STM32F4 board via ST-Link
# Run HIL tests
python tests/hardware/test_stm32f4.py --device /dev/ttyUSB0

# Power profiling
liquid-profile --device stm32f407 --model models/keyword.lnn
```

## Performance Optimization

### Profiling

```bash
# Python profiling
python -m cProfile -o profile.stats scripts/benchmark.py
snakeviz profile.stats

# Rust profiling with perf
cargo build --release
perf record --call-graph=dwarf target/release/benchmark
perf report

# Memory profiling
valgrind --tool=massif --stacks=yes ./target/release/benchmark
ms_print massif.out.* | head -30
```

### Optimization Targets

- **Latency**: < 20ms processing time
- **Power**: < 2mW average consumption
- **Memory**: < 64KB RAM usage
- **Accuracy**: > 95% on target datasets

## Debugging

### Python Debugging

```python
# Use debugger
import pdb; pdb.set_trace()

# VS Code debugging - use F5 or create launch.json:
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        }
    ]
}
```

### Rust Debugging

```bash
# GDB debugging
cargo build
gdb target/debug/liquid_audio_test
(gdb) run
(gdb) bt

# LLDB debugging (better Rust support)
rust-lldb target/debug/liquid_audio_test
(lldb) run
(lldb) bt
```

### Embedded Debugging

```bash
# OpenOCD + GDB
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg &
arm-none-eabi-gdb target/thumbv7em-none-eabihf/debug/liquid_audio_embedded.elf
(gdb) target remote localhost:3333
(gdb) monitor reset halt
(gdb) load
(gdb) continue
```

## Troubleshooting

### Common Issues

**Build failures:**
```bash
# Clean all build artifacts
make clean

# Rebuild from scratch
make build
```

**Dependency issues:**
```bash
# Python dependencies
pip install --upgrade --force-reinstall -e ".[dev]"

# Rust dependencies
cargo clean
cargo update
cargo build
```

**Permission errors:**
```bash
# Fix file permissions
find . -name "*.sh" -exec chmod +x {} \;

# Reset git permissions
git config core.fileMode false
```

**Pre-commit failures:**
```bash
# Update pre-commit hooks
pre-commit autoupdate
pre-commit install --install-hooks

# Skip hooks temporarily
git commit --no-verify -m "message"
```

### Getting Help

- 📖 Check the [documentation](https://liquid-audio-nets.readthedocs.io)
- 🐛 Search [existing issues](https://github.com/terragon-labs/liquid-audio-nets/issues)
- 💬 Start a [discussion](https://github.com/terragon-labs/liquid-audio-nets/discussions)
- 📧 Email the maintainers: dev@terragon.dev

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](../LICENSE) for details.