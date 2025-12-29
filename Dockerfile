# Multi-stage build for optimized DiffML Docker image
# Supports both CPU and GPU (CUDA) environments

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.0 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root --no-dev

# Copy source code
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

# Install the package
RUN poetry install --no-dev

# Runtime stage - CPU version
FROM python:3.11-slim as cpu-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 diffml && \
    mkdir -p /app /data /results && \
    chown -R diffml:diffml /app /data /results

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder --chown=diffml:diffml /app/.venv /app/.venv
COPY --from=builder --chown=diffml:diffml /app/src ./src
COPY --from=builder --chown=diffml:diffml /app/scripts ./scripts
COPY --from=builder --chown=diffml:diffml /app/configs ./configs

# Copy additional files
COPY --chown=diffml:diffml README.md ./
COPY --chown=diffml:diffml notebooks ./notebooks
COPY --chown=diffml:diffml tests ./tests

# Switch to non-root user
USER diffml

# Set volumes for data and results
VOLUME ["/data", "/results"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import diffml; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import diffml; print('DiffML container ready!')"]

# GPU runtime stage - CUDA version
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as gpu-runtime

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    CUDA_VISIBLE_DEVICES=0

# Create non-root user
RUN useradd -m -u 1000 diffml && \
    mkdir -p /app /data /results && \
    chown -R diffml:diffml /app /data /results

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder --chown=diffml:diffml /app/.venv /app/.venv
COPY --from=builder --chown=diffml:diffml /app/src ./src
COPY --from=builder --chown=diffml:diffml /app/scripts ./scripts
COPY --from=builder --chown=diffml:diffml /app/configs ./configs

# Copy additional files
COPY --chown=diffml:diffml README.md ./
COPY --chown=diffml:diffml notebooks ./notebooks
COPY --chown=diffml:diffml tests ./tests

# Switch to non-root user
USER diffml

# Set volumes
VOLUME ["/data", "/results"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || exit 1

# Default command
CMD ["python", "-c", "import torch; import diffml; print(f'DiffML GPU container ready! CUDA: {torch.cuda.is_available()}')"]

# Development stage with Jupyter
FROM cpu-runtime as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and additional packages
RUN /app/.venv/bin/pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly \
    streamlit

# Create Jupyter configuration
RUN mkdir -p /home/diffml/.jupyter && \
    chown -R diffml:diffml /home/diffml/.jupyter

USER diffml

# Expose ports
EXPOSE 8888 8501 6006

# Jupyter command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]