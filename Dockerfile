# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for scikit-learn / numpy compilation wheels
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Abdulbosit Abdurazzakov <B00979380>"
LABEL description="COM668 AI-Based Intrusion Detection System — REST API"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source (excludes paths in .dockerignore)
COPY src/      ./src/
COPY data/processed/ ./data/processed/
COPY models/   ./models/

# Non-root user for security
RUN useradd -m -u 1001 ids && chown -R ids:ids /app
USER ids

EXPOSE 5000

# Gunicorn: 2 workers × 2 threads is appropriate for inference workloads
# where each request is CPU-bound and short-lived.
CMD ["gunicorn", \
     "--workers", "2", \
     "--threads", "2", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "src.api:app"]
