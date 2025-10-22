# Dockerfile for Earthquake Enhanced - Deterministic Measurement System

FROM python:3.11-slim

LABEL maintainer="Earthquake Enhanced Team"
LABEL description="Deterministic seismic stress measurement system"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgeos-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY backend/ ./backend/
COPY tests/ ./tests/
COPY data_manifest.yaml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/checksums

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command
CMD ["python", "backend/physics_engine/historical_runner.py"]
