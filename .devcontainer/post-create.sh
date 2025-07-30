#!/bin/bash
# Post-create script for Liquid Audio Nets development container

set -e

echo "🚀 Setting up Liquid Audio Nets development environment..."

# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    clang \
    clang-format \
    clang-tidy \
    lldb \
    gdb \
    valgrind \
    libasound2-dev \
    libpulse-dev \
    portaudio19-dev \
    libfftw3-dev \
    libblas-dev \
    liblapack-dev \
    gcc-arm-none-eabi \
    openocd

# Install Python development dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev,training,embedded]"
pip install pre-commit detect-secrets

# Install Rust components
echo "🦀 Installing Rust components..."
rustup component add clippy rustfmt llvm-tools-preview
cargo install cargo-audit cargo-watch cargo-expand

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Initialize secrets baseline
echo "🔒 Initializing secrets detection..."
detect-secrets scan --all-files --baseline .secrets.baseline || true

# Build the project
echo "🏗️ Building project..."
make build || echo "⚠️ Build failed - this is expected on first setup"

# Set up git configuration
echo "📝 Configuring git..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf false
git config --global core.eol lf

echo "✅ Development environment setup complete!"
echo ""
echo "🎵 Welcome to Liquid Audio Nets development!"
echo "📚 Quick start:"
echo "  • Run 'make help' to see available commands"
echo "  • Run 'make test' to run all tests"
echo "  • Run 'make dev-setup' for additional development tools"
echo "  • Check CONTRIBUTING.md for contribution guidelines"
echo ""
echo "Happy coding! 🎧"