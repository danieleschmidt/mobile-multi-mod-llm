# Multi-stage build for optimized mobile AI deployment
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim

LABEL maintainer="Terragon Labs <info@terragon.com>"
LABEL description="Mobile Multi-Modal LLM for edge deployment"
LABEL version="0.1.0"

# Create non-root user for security
RUN groupadd -r mobile-ai && useradd -r -g mobile-ai mobile-ai

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set up application directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY monitoring/ ./monitoring/
COPY pyproject.toml ./

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/cache /app/exports \
    && chown -R mobile-ai:mobile-ai /app

# Copy startup script
COPY deployment/entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Security and performance settings
ENV LOG_LEVEL=INFO
ENV DEVICE=cpu
ENV MAX_WORKERS=4
ENV TIMEOUT=30
ENV ENABLE_MOCK_MODE=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "
import sys; sys.path.insert(0, 'src')
from mobile_multimodal import MobileMultiModalLLM
import numpy as np
try:
    model = MobileMultiModalLLM()
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _ = model.generate_caption(test_img)
    print('Health check passed')
except Exception as e:
    print(f'Health check failed: {e}')
    sys.exit(1)
"

# Switch to non-root user
USER mobile-ai

# Expose port
EXPOSE 8000

# Start application
ENTRYPOINT ["./entrypoint.sh"]
CMD ["api"]