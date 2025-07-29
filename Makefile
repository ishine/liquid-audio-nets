.PHONY: all build test clean install docs lint format check-format
.DEFAULT_GOAL := build

# Build system detection
PYTHON := $(shell command -v python3 2> /dev/null || echo python)
CARGO := $(shell command -v cargo 2> /dev/null)
CMAKE := $(shell command -v cmake 2> /dev/null)

# Build targets
all: build test

build: build-rust build-python build-cpp

build-rust:
ifdef CARGO
	@echo "Building Rust library..."
	cargo build --release
	@echo "✓ Rust build complete"
else
	@echo "⚠ Cargo not found, skipping Rust build"
endif

build-python:
	@echo "Building Python package..."
	$(PYTHON) -m pip install -e .[dev]
	@echo "✓ Python build complete"

build-cpp:
ifdef CMAKE
	@echo "Building C++ library..."
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
	cd build && make -j$(shell nproc 2>/dev/null || echo 4)
	@echo "✓ C++ build complete"
else
	@echo "⚠ CMake not found, skipping C++ build"
endif

# Testing
test: test-rust test-python test-cpp

test-rust:
ifdef CARGO
	@echo "Running Rust tests..."
	cargo test
	@echo "✓ Rust tests complete"
endif

test-python:
	@echo "Running Python tests..."
	$(PYTHON) -m pytest tests/ -v
	@echo "✓ Python tests complete"

test-cpp:
ifdef CMAKE
	@echo "Running C++ tests..."
	cd build && ctest --output-on-failure
	@echo "✓ C++ tests complete"
endif

# Code quality
lint: lint-python lint-rust lint-cpp

lint-python:
	@echo "Linting Python code..."
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy liquid_audio_nets/

lint-rust:
ifdef CARGO
	@echo "Linting Rust code..."
	cargo clippy -- -D warnings
endif

lint-cpp:
	@echo "Linting C++ code..."
	@find src/ include/ -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" | \
		xargs clang-format --dry-run --Werror 2>/dev/null || \
		echo "⚠ clang-format not available or formatting issues found"

# Formatting
format: format-python format-rust format-cpp

format-python:
	@echo "Formatting Python code..."
	$(PYTHON) -m black .
	$(PYTHON) -m ruff --fix .

format-rust:
ifdef CARGO
	@echo "Formatting Rust code..."
	cargo fmt
endif

format-cpp:
	@echo "Formatting C++ code..."
	@find src/ include/ -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" | \
		xargs clang-format -i 2>/dev/null || \
		echo "⚠ clang-format not available"

# Documentation
docs:
	@echo "Building documentation..."
	$(PYTHON) -m sphinx -b html docs/ docs/_build/
	@echo "Documentation built in docs/_build/"

# Installation
install: build
	@echo "Installing liquid-audio-nets..."
	$(PYTHON) -m pip install .
ifdef CARGO
	cargo install --path .
endif
ifdef CMAKE
	cd build && make install
endif
	@echo "✓ Installation complete"

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf target/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✓ Cleanup complete"

# Development setup
dev-setup:
	@echo "Setting up development environment..."
	$(PYTHON) -m pip install -e .[dev,training,embedded]
	$(PYTHON) -m pre-commit install
ifdef CARGO
	rustup component add clippy rustfmt
endif
	@echo "✓ Development environment ready"

# Benchmarks
bench: bench-rust bench-python

bench-rust:
ifdef CARGO
	@echo "Running Rust benchmarks..."
	cargo bench
endif

bench-python:
	@echo "Running Python benchmarks..."
	$(PYTHON) -m pytest benchmarks/ -v

# Help
help:
	@echo "Available targets:"
	@echo "  build       - Build all components (Rust, Python, C++)"
	@echo "  test        - Run all tests"
	@echo "  lint        - Lint all code"
	@echo "  format      - Format all code"
	@echo "  docs        - Build documentation"
	@echo "  install     - Install the package"
	@echo "  clean       - Clean build artifacts"
	@echo "  dev-setup   - Set up development environment"
	@echo "  bench       - Run benchmarks"
	@echo "  help        - Show this help message"