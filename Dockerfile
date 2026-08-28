# ---- Stage 1: Builder ----
# WHY multi-stage: The final image doesn't include pip, gcc, or build tools.
# This cuts the image size by ~40%.

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies for native extensions (asyncpg, mediapipe)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.11-slim

WORKDIR /app

# Runtime dependencies only (no gcc, no build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq5 \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Non-root user for security
# WHY: Running as root inside a container is a security risk.
# A compromised app running as root = compromised host.
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# WHY uvicorn with ${PORT:-8000}: Allows Render/Cloud to inject dynamic $PORT (default 8000 for local docker).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

