# -----------------------------------------------------------------------------
# LightRAG Server Dockerfile
# Multi-stage build for production and development
# -----------------------------------------------------------------------------

# =============================================================================
# Base Stage - Common dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements*.txt ./
RUN pip install --upgrade pip setuptools wheel

# =============================================================================
# Development Stage
# =============================================================================
FROM base as development

# Install development dependencies
RUN pip install -r requirements.txt \
    && pip install -r requirements-dev.txt

# Install additional development tools
RUN pip install \
    watchdog \
    jupyter \
    ipython \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Copy source code (for development, this will be overridden by volume mount)
COPY . .

# Expose ports
EXPOSE 8000 8001

# Development command (will be overridden)
CMD ["python", "app.py", "--reload", "--debug"]

# =============================================================================
# Production Dependencies Stage
# =============================================================================
FROM base as prod-deps

# Install only production dependencies
RUN pip install -r requirements.txt

# =============================================================================
# Production Stage
# =============================================================================
FROM prod-deps as production

# Create non-root user
RUN groupadd -r lightrag && useradd -r -g lightrag -d /app -s /sbin/nologin lightrag

# Copy application code
COPY --chown=lightrag:lightrag . .

# Create necessary directories
RUN mkdir -p /app/cache /app/logs /app/tmp && \
    chown -R lightrag:lightrag /app

# Install Gunicorn for production
RUN pip install gunicorn[gevent] uvicorn[standard]

# Switch to non-root user
USER lightrag

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120", "--keepalive", "5", "--max-requests", "1000", "--max-requests-jitter", "100", "app:app"]

# =============================================================================
# Testing Stage
# =============================================================================
FROM development as testing

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini ./

# Run tests
RUN python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Test command
CMD ["python", "-m", "pytest", "tests/", "-v"]