# ============================================================
# MindSense AI — Dockerfile
# ============================================================
# Multi-stage build:
#   Stage 1 (builder): Install Python dependencies
#   Stage 2 (runtime): Lean production image
#
# Build:  docker build -t mindsense-ai .
# Run:    docker run -p 5000:5000 --env-file .env mindsense-ai
# ============================================================

# --------------- Stage 1: Builder ---------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for PyMuPDF and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python packages into /install prefix
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# --------------- Stage 2: Runtime ---------------
FROM python:3.11-slim AS runtime

LABEL maintainer="MindSense AI Team"
LABEL version="1.0.0"
LABEL description="AI-Powered Mental Health Assistant"

WORKDIR /app

# Install only runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY . .

# Create necessary directories
RUN mkdir -p logs model/faiss model/embedding_model memory/user_profiles

# Create a non-root user for security
RUN useradd -m -u 1000 mindsense && chown -R mindsense:mindsense /app
USER mindsense

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Environment defaults (override with --env-file .env)
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV FLASK_DEBUG=False
ENV LOG_LEVEL=INFO

# Start the Flask application
CMD ["python", "app.py"]
