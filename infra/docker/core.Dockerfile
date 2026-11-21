# Core recommendation service image. Built from the repo root (see docker-compose.prod.yml):
#   docker build -f infra/docker/core.Dockerfile .
#
# Python 3.11 is the project's stack (CLAUDE.md), whatever a developer runs locally.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# implicit (ALS, BPR) needs the OpenMP runtime, which the slim image leaves out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before code: a code-only change reuses this cached layer.
# The constraints file pins every package to the versions this image was tested with.
COPY requirements.txt .
COPY infra/docker/core-constraints.txt .
RUN pip install -r requirements.txt -c core-constraints.txt

COPY alembic.ini .
COPY core ./core

# Not root: a compromised process should not be administrator inside the container.
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000

# Bring the database schema up to date, then serve. Every deploy migrates by itself, so
# there is no manual step to forget. exec makes uvicorn the main process, so it receives
# Docker's stop signal and shuts down cleanly.
# --proxy-headers: trust Caddy's X-Forwarded-* headers for the client address and scheme.
# Only Caddy can reach this port, on the private Docker network.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn core.api.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'"]
