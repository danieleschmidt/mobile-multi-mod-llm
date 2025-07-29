# Multi-stage Dockerfile for Mobile Multi-Modal LLM
# Optimized for development, testing, and production deployment

# =============================================================================
# Base Image with Python and System Dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# =============================================================================
# Development Stage
# =============================================================================
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    tree \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-dev.txt

# Copy source code
COPY --chown=appuser:appuser . /app/

# Install package in development mode
RUN pip install -e .[dev,test,docs,mobile]

# Switch to non-root user
USER appuser

# Expose ports for development servers
EXPOSE 8000 8888

# Default command for development
CMD ["python", "-c", "print('Development container ready. Run: docker exec -it <container> bash')"]

# =============================================================================
# Testing Stage
# =============================================================================
FROM development as testing

# Switch back to root for testing setup
USER root

# Install additional testing tools
RUN pip install tox pytest-xdist pytest-benchmark

# Create test directories
RUN mkdir -p /app/test-results /app/coverage-reports

# Switch back to appuser
USER appuser

# Run tests by default
CMD ["pytest", "--cov=src", "--cov-report=html:/app/coverage-reports", "--junitxml=/app/test-results/pytest.xml"]

# =============================================================================
# Production Builder Stage
# =============================================================================
FROM base as builder

# Copy requirements and install production dependencies only
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --user --no-warn-script-location -r requirements.txt

# Copy source code and build package
COPY . /app/
RUN pip install --user --no-warn-script-location .

# =============================================================================
# Production Stage
# =============================================================================
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    ENVIRONMENT=production

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy installed packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --from=builder --chown=appuser:appuser /app/src /app/src

# Set working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import mobile_multimodal; print('OK')" || exit 1

# Default production command
CMD ["python", "-m", "mobile_multimodal.server"]

# =============================================================================
# Mobile Development Stage (for cross-platform builds)
# =============================================================================
FROM development as mobile-dev

# Switch to root for mobile SDK installation
USER root

# Install Android NDK and tools (conditional on availability)
RUN mkdir -p /opt/android-sdk && \
    curl -o android-commandlinetools.zip https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip && \
    unzip android-commandlinetools.zip -d /opt/android-sdk && \
    rm android-commandlinetools.zip

# Install additional mobile development tools
RUN pip install tensorflow==2.15.0 coremltools>=7.0 onnxruntime>=1.15.0

# Set Android environment variables
ENV ANDROID_SDK_ROOT=/opt/android-sdk \
    PATH="$PATH:/opt/android-sdk/cmdline-tools/latest/bin"

# Switch back to appuser
USER appuser

# Default command for mobile development
CMD ["python", "-c", "print('Mobile development container ready')"]

# =============================================================================
# GPU-Enabled Stage (for ML training and inference)
# =============================================================================
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as gpu

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES=0

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Install Python dependencies with CUDA support
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Copy source code
COPY --chown=appuser:appuser . /app/

# Install package
RUN pip install -e .[dev,mobile]

# Switch to non-root user
USER appuser

# Default command for GPU container
CMD ["python", "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"]