# CI/CD Workflows Documentation

This directory contains documentation and templates for GitHub Actions workflows.

**Note**: Actual workflow files cannot be created automatically due to security policies. The templates below should be manually created in `.github/workflows/` directory.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-python:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    - name: Run tests
      run: pytest tests/ -v --cov=liquid_audio_nets
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  test-rust:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
    - name: Run tests
      run: cargo test --all-features
    - name: Run clippy
      run: cargo clippy --all-targets --all-features -- -D warnings

  test-cpp:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake build-essential
    - name: Build
      run: |
        mkdir build
        cd build
        cmake .. -DBUILD_TESTS=ON
        make -j$(nproc)
    - name: Test
      run: cd build && ctest --output-on-failure
```

### 2. Security Scanning (`security.yml`)

```yaml
name: Security

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run custom security check
      run: python scripts/security_check.py
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  dependency-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Rust security audit
      uses: actions-rs/audit-check@v1
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Python safety check
      run: |
        pip install safety
        safety check
```

### 3. Release Workflow (`release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  build-and-release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine maturin
    
    - name: Build Python package
      run: python -m build
    
    - name: Build Rust package
      run: maturin build --release
    
    - name: Create Release
      uses: softprops/action-gh-release@v1
      with:
        files: |
          dist/*
          target/wheels/*
        generate_release_notes: true
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 4. Documentation (`docs.yml`)

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
    paths: [ 'docs/**', '*.md', 'python/**/*.py' ]
  pull_request:
    branches: [ main ]
    paths: [ 'docs/**', '*.md' ]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install sphinx sphinx-rtd-theme
    
    - name: Build documentation
      run: make docs
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build
```

## Workflow Features

### Multi-language Support
- **Python**: Testing across multiple versions (3.8-3.11)
- **Rust**: Latest stable toolchain with clippy and rustfmt
- **C++**: CMake-based build system with testing

### Security Integration
- Custom security scanning script
- Trivy vulnerability scanning
- Dependency security audits (Rust: cargo-audit, Python: safety)
- Secrets detection via pre-commit hooks

### Quality Assurance
- Code coverage reporting
- Static analysis (clippy, mypy)
- Formatting enforcement (black, rustfmt, clang-format)
- Documentation builds

### Release Management
- Automated releases on tag creation
- Multi-format package building (Python wheels, Rust crates)
- Release notes generation

## Setup Instructions

1. **Create workflow files**: Copy the YAML templates above into `.github/workflows/`

2. **Configure secrets**: Add the following secrets in repository settings:
   - `CODECOV_TOKEN`: For code coverage reporting
   - `PYPI_API_TOKEN`: For Python package publishing
   - `CRATES_IO_TOKEN`: For Rust crate publishing

3. **Enable GitHub Pages**: For documentation deployment

4. **Configure branch protection**: Require CI checks to pass before merging

## Embedded Testing

For hardware-in-the-loop testing, consider additional workflows:

```yaml
# embedded-test.yml (runs on self-hosted runners with hardware)
name: Embedded Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  hardware-test:
    runs-on: [self-hosted, embedded]
    steps:
    - uses: actions/checkout@v4
    - name: Build for STM32F4
      run: |
        mkdir build-stm32
        cd build-stm32
        cmake .. -DENABLE_EMBEDDED_BUILD=ON -DENABLE_ARM_OPTIMIZATIONS=ON
        make -j$(nproc)
    - name: Flash and test on hardware
      run: |
        openocd -f interface/stlink.cfg -f target/stm32f4x.cfg -c "program build-stm32/liquid_audio_test.elf verify reset exit"
        python tests/hardware/test_stm32f4.py
```

## Performance Monitoring

Add performance regression testing:

```yaml
# performance.yml
name: Performance

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run benchmarks
      run: |
        make bench
        python scripts/benchmark_analysis.py results/
    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: machine-learning-apps/pr-comment@master
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        path: benchmark_results.md
```