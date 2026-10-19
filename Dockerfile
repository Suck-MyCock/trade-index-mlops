ARG BASE_IMAGE=python:3.12-slim

FROM ${BASE_IMAGE}

# Override the index when PyPI is slow (e.g. mainland China):
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#                -t trade-index:local .
# The default keeps GitHub Actions on ordinary PyPI.
ARG PIP_INDEX_URL=https://pypi.org/simple

# Deployment identity, supplied by the CD pipeline from the .env it generates.
# Keeping these as build args (rather than baking an .env file into a layer)
# means the values are visible in `docker history` but no secret file is ever
# stored in the image.
ARG APP_ENV=local
ARG LOG_LEVEL=INFO

# Keep the image free of .pyc files and buffer-free logs so `docker logs` shows
# output immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=${APP_ENV} \
    LOG_LEVEL=${LOG_LEVEL}

WORKDIR /app

# Dependencies first: this layer stays cached until requirements.txt changes, so
# editing application code does not trigger a reinstall.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir \
        --index-url "${PIP_INDEX_URL}" \
        -r requirements.txt

# Application code and the trained artifact the API loads at startup.
COPY app/ ./app/
COPY models/ ./models/

# Run as a non-root user. UID 10001 matches the convention used elsewhere in
# this course so mounted volumes get consistent ownership.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R 10001:10001 /app
USER 10001

EXPOSE 8000

# The model is loaded once during FastAPI's lifespan hook, so serving starts
# almost immediately after the container is up.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
