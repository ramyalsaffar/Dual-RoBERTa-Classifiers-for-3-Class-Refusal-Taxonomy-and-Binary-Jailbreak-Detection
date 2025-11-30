# Dual RoBERTa Classifiers - Docker Image
# Multi-stage build for optimal size and flexibility
# Supports both CPU and GPU environments

# =============================================================================
# STAGE 1: BASE IMAGE
# =============================================================================
# WHY: Use official PyTorch image for CUDA support and optimized PyTorch installation
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# WHY: git for transformers model downloads, build-essential for some pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# =============================================================================
# STAGE 2: DEPENDENCIES
# =============================================================================
# WHY: Separate stage for dependencies enables better Docker layer caching
FROM base AS dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# WHY: Install in a separate stage to cache this layer when code changes
RUN pip install --no-cache-dir -r requirements.txt

# Download RoBERTa model weights at build time (optional but recommended)
# WHY: Speeds up first run by pre-downloading 500MB model
RUN python -c "from transformers import RobertaTokenizer, RobertaModel; \
    RobertaTokenizer.from_pretrained('roberta-base'); \
    RobertaModel.from_pretrained('roberta-base')"

# =============================================================================
# STAGE 3: DEVELOPMENT
# =============================================================================
# WHY: Development stage includes additional tools for debugging and testing
FROM dependencies AS development

# Install development tools
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    pytest \
    black \
    flake8

# Copy project files
COPY . .

# Create necessary directories
# WHY: Ensure directories exist before running pipeline
RUN mkdir -p data/{raw,responses,processed,splits} \
    models \
    results \
    visualizations \
    reports \
    logs

# Expose ports for Jupyter and API
EXPOSE 8888 8000

# Default command for development
CMD ["/bin/bash"]

# =============================================================================
# STAGE 4: PRODUCTION (Training)
# =============================================================================
# WHY: Optimized for training - includes all dependencies but minimal dev tools
FROM dependencies AS production

# Copy only necessary project files
# WHY: Smaller image size by excluding unnecessary files
COPY src/ ./src/
COPY README.md LICENSE ./

# Create necessary directories
RUN mkdir -p data/{raw,responses,processed,splits} \
    models \
    results \
    visualizations \
    reports \
    logs

# Set Python path
ENV PYTHONPATH=/app/src

# Default command
CMD ["python", "src/32-Execute.py", "--full"]

# =============================================================================
# STAGE 5: API (Production API)
# =============================================================================
# WHY: Minimal image for serving predictions - excludes training dependencies
FROM python:3.11-slim AS api

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies (curl for healthcheck)
# WHY: curl is needed for Docker HEALTHCHECK command
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install only API dependencies
# WHY: Minimal dependencies for faster startup and smaller image
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    transformers>=4.30.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    pydantic>=2.0.0 \
    numpy>=1.24.0

# Pre-download RoBERTa model
RUN python -c "from transformers import RobertaTokenizer, RobertaModel; \
    RobertaTokenizer.from_pretrained('roberta-base'); \
    RobertaModel.from_pretrained('roberta-base')"

# Copy only API-related files
# WHY: Include all dependencies needed by ProductionAPI
COPY src/01-Imports.py src/02-Setup.py src/03-Utils.py src/04-Config.py \
     src/15-RefusalClassifier.py src/16-JailbreakDetector.py \
     src/34-ProductionAPI.py ./src/

# Copy trained models (these should be mounted as volumes in production)
RUN mkdir -p models

# Expose API port
EXPOSE 8000

# Health check
# WHY: Kubernetes/Docker Swarm can monitor container health
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
# WHY: Use --app-dir to handle numbered Python files (34-ProductionAPI.py)
CMD ["uvicorn", "34-ProductionAPI:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8000"]
