#!/bin/bash
set -e

echo "🚀 Setting up Mobile Multi-Modal LLM development environment..."

# Update system packages
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    git-lfs \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    && sudo rm -rf /var/lib/apt/lists/*

# Initialize git LFS
git lfs install

# Upgrade pip and install package management tools
python -m pip install --upgrade pip setuptools wheel uv

# Install the project in development mode with all dependencies
echo "📦 Installing project dependencies..."
pip install -e ".[dev,test,docs,mobile]"

# Install additional development tools
pip install \
    pre-commit \
    jupyter \
    ipywidgets \
    tensorboard \
    wandb \
    mlflow \
    optuna

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create useful aliases
echo "⚙️  Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# Mobile Multi-Modal LLM Development Aliases
alias mm-train='python -m mobile_multimodal.scripts.train'
alias mm-export='python -m mobile_multimodal.scripts.export'
alias mm-benchmark='python -m mobile_multimodal.scripts.benchmark'
alias mm-test='pytest tests/ -v'
alias mm-test-coverage='pytest tests/ --cov=src --cov-report=html'
alias mm-lint='flake8 src/ tests/ && mypy src/'
alias mm-format='black src/ tests/ && isort src/ tests/'
alias mm-security='bandit -r src/ && safety check'
alias mm-docs='mkdocs serve'
alias mm-clean='find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true'

# Jupyter shortcuts
alias jupyter-lab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias tensorboard='tensorboard --logdir=./tensorboard_logs --host=0.0.0.0 --port=6006'

# Git shortcuts
alias gst='git status'
alias gco='git checkout'
alias gbd='git branch -d'
alias glog='git log --oneline --graph --decorate'
EOF

# Set up Jupyter configuration
mkdir -p ~/.jupyter
cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
EOF

# Create development directories
echo "📁 Creating development directories..."
mkdir -p {data,experiments,outputs,checkpoints,models,notebooks}

# Download development datasets (placeholder)
echo "📊 Setting up development data..."
mkdir -p data/{train,val,test,calibration}

# Create example configuration files
echo "⚙️  Creating example configurations..."
mkdir -p configs/examples

# Set correct permissions
sudo chown -R vscode:vscode /workspaces/mobile-multi-mod-llm
chmod +x .devcontainer/setup.sh

echo "✅ Development environment setup complete!"
echo ""
echo "🔧 Available commands:"
echo "  mm-train      - Train models"
echo "  mm-export     - Export models for mobile"
echo "  mm-benchmark  - Run performance benchmarks"
echo "  mm-test       - Run tests"
echo "  mm-lint       - Run linting"
echo "  mm-format     - Format code"
echo "  mm-security   - Security scanning"
echo "  mm-docs       - Serve documentation"
echo ""
echo "🚀 Start developing with:"
echo "  mm-test       # Run tests to verify setup"
echo "  jupyter-lab   # Start Jupyter Lab"
echo "  mm-docs       # Start documentation server"
echo ""