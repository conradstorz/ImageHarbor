# ImageHarbor watcher image (amd64).
FROM python:3.12-slim

# Non-root runtime user.
RUN useradd --create-home --uid 1000 harbor

WORKDIR /app

# Build from the lockfile so the container ships exactly the dependency set
# the suite was tested against -- `pip install ".[...]"` resolved fresh
# against loose floors on every build. uv is copied from its official image
# (pinned); --frozen refuses a stale lock instead of silently re-resolving.
# 'faces' adds ~261 MB of model weights on the FIRST `faces scan`/`watch`
# run (see imageharbor/faces/download.py) -- not at build time, so this
# layer stays small; docker-compose.yml's `imageharbor-models` volume stops
# that download from repeating on every container recreate.
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

# The build context has no .git, so setuptools-scm cannot derive a version;
# the release workflow passes the tag here. A local `docker build` without
# the arg gets 0.0.0 -- visibly a non-release.
ARG IMAGEHARBOR_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_IMAGEHARBOR=${IMAGEHARBOR_VERSION} \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# LICENSE ships in the image: this is an AGPL network service; the conveyed
# artifact must carry the licence text (README.md "Licence" section).
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY imageharbor ./imageharbor
RUN uv sync --frozen --no-dev --no-editable --extra openai --extra faces
ENV PATH="/opt/venv/bin:${PATH}"

# Default mount points (see docker-compose.yml).
ENV IMAGEHARBOR_SOURCE=/data/source \
    IMAGEHARBOR_DEST=/data/dest \
    IMAGEHARBOR_CATALOG=/data/catalog/catalog.db \
    IMAGEHARBOR_FACE_MODEL_DIR=/data/models

# Create the data mount points and give them to the non-root user. A Docker
# named volume initializes its ownership from the image directory it mounts
# over, so /data/catalog and /data/models must be owned by 'harbor' for the
# catalog and the downloaded model weights to be writable at runtime (bind
# mounts for source/dest get their ownership from the host).
RUN mkdir -p /data/source /data/dest /data/catalog /data/models \
    && chown -R harbor:harbor /data

# Operational dashboard (see docker-compose.yml's `ports`/`healthcheck` and
# `imageharbor watch --dashboard-port`). Documentation only -- EXPOSE does
# not itself publish the port -- but keeps the image's own contract visible
# without cross-referencing compose.
EXPOSE 8080

USER harbor

ENTRYPOINT ["imageharbor"]
CMD ["watch"]
