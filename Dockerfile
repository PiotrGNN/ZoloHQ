# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

# Minimal system dependencies for scientific stack and build
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency files for better cache
COPY ZoL0-master/pyproject.toml ./pyproject.toml
COPY ZoL0-master/poetry.lock ./poetry.lock

# Install Poetry (compatible version for 'export' command)
RUN pip install --upgrade pip && \
    pip install "poetry>=1.2.0,<2.0.0"

# Install dependencies to a custom location (for multi-stage minimal image)
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip install --prefix=/install -r requirements.txt

# Final minimal image
FROM python:3.11-slim
RUN apt-get update && \
    apt-get install --no-install-recommends -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /install /usr/local
COPY ZoL0-master/ .
ENV PYTHONUNBUFFERED=1

# (Optional) Run minimal test, do not fail build if tests fail
RUN pip install pytest && pytest || true

# Default command (change to your entrypoint if needed)
CMD ["python", "main.py"]
