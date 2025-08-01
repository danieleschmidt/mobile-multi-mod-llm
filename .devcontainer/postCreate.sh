#!/bin/bash

# Post-create script for Mobile Multi-Modal LLM development environment
set -e

echo "🚀 Setting up Mobile Multi-Modal LLM development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libopencv-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libglib2.0-dev \
    libgtk-3-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    wget \
    curl \
    unzip \
    git-lfs

# Initialize git-lfs
echo "📚 Initializing Git LFS..."
git lfs install

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in development mode
echo "📦 Installing package in development mode..."
pip install -e .

# Install pre-commit hooks
echo "🔒 Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p models/checkpoints
mkdir -p models/quantized
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs
mkdir -p experiments
mkdir -p outputs

# Set up Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name mobile-ml --display-name "Mobile ML"

# Download sample data (if available)
echo "📊 Setting up sample data..."
if [ -f "scripts/download_sample_data.py" ]; then
    python scripts/download_sample_data.py --sample-only
fi

# Run initial tests to verify setup
echo "🧪 Running setup verification tests..."
if python -m pytest tests/test_setup.py -v 2>/dev/null; then
    echo "✅ Setup verification passed!"
else
    echo "⚠️  Setup verification tests not found or failed - this is expected for new projects"
fi

# Create VS Code workspace settings if they don't exist
if [ ! -d ".vscode" ]; then
    echo "⚙️  Creating VS Code workspace settings..."
    mkdir -p .vscode
fi

# Set up git configuration recommendations
echo "🔧 Configuring git settings..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Install additional development tools
echo "🛠️  Installing additional development tools..."
pip install \
    jupyterlab \
    jupyterlab-git \
    notebook \
    ipywidgets \
    tensorboard \
    wandb \
    mlflow

# Set up shell aliases for convenience
echo "🔗 Setting up convenient aliases..."
cat >> ~/.zshrc << 'EOF'

# Mobile Multi-Modal LLM Development Aliases
alias mm-test='python -m pytest tests/ -v'
alias mm-lint='pre-commit run --all-files'
alias mm-format='black src/ tests/ && isort src/ tests/'
alias mm-train='python scripts/train.py'
alias mm-eval='python scripts/evaluate.py'
alias mm-export='python scripts/export_model.py'
alias mm-bench='python scripts/benchmark.py'
alias mm-logs='tail -f logs/training.log'
alias mm-tb='tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006'
alias mm-jupyter='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# Quick navigation
alias cdmm='cd /workspaces/mobile-multi-mod-llm'
alias cdmodels='cd /workspaces/mobile-multi-mod-llm/models'
alias cddata='cd /workspaces/mobile-multi-mod-llm/data'
alias cdscripts='cd /workspaces/mobile-multi-mod-llm/scripts'

# Git shortcuts
alias gitmm='git add . && git commit -m'
alias gitpush='git push origin $(git branch --show-current)'
alias gitpull='git pull origin main'
alias gitstatus='git status --short'
EOF

# Source the updated shell configuration
source ~/.zshrc 2>/dev/null || true

echo "✅ Development environment setup complete!"
echo ""
echo "🎉 You can now start developing with:"
echo "   • mm-test     - Run tests"
echo "   • mm-lint     - Run linting"  
echo "   • mm-format   - Format code"
echo "   • mm-train    - Start training"
echo "   • mm-jupyter  - Launch Jupyter Lab"
echo "   • mm-tb       - Launch TensorBoard"
echo ""
echo "📚 Useful directories:"
echo "   • models/     - Model checkpoints and artifacts"
echo "   • data/       - Training and validation data"
echo "   • experiments/ - Experiment tracking and results"
echo "   • logs/       - Training and application logs"
echo ""
echo "Happy coding! 🚀"