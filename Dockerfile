# ImageHarbor watcher image (amd64).
FROM python:3.12-slim

# Non-root runtime user.
RUN useradd --create-home --uid 1000 harbor

WORKDIR /app

# Install the package with the OpenAI-compatible classifier extra.
COPY pyproject.toml README.md ./
COPY imageharbor ./imageharbor
RUN pip install --no-cache-dir ".[openai]"

# Default mount points (see docker-compose.yml).
ENV IMAGEHARBOR_SOURCE=/data/source \
    IMAGEHARBOR_DEST=/data/dest \
    IMAGEHARBOR_CATALOG=/data/catalog/catalog.db

# Create the data mount points and give them to the non-root user. A Docker
# named volume initializes its ownership from the image directory it mounts
# over, so /data/catalog must be owned by 'harbor' for the catalog to be
# writable at runtime (bind mounts for source/dest get their ownership from
# the host).
RUN mkdir -p /data/source /data/dest /data/catalog \
    && chown -R harbor:harbor /data

# Operational dashboard (see docker-compose.yml's `ports`/`healthcheck` and
# `imageharbor watch --dashboard-port`). Documentation only -- EXPOSE does
# not itself publish the port -- but keeps the image's own contract visible
# without cross-referencing compose.
EXPOSE 8080

USER harbor

ENTRYPOINT ["imageharbor"]
CMD ["watch"]
