# ImageHarbor watcher image (amd64).
FROM python:3.12-slim

# Non-root runtime user.
RUN useradd --create-home --uid 1000 harbor

WORKDIR /app

# Install the package with the OpenAI-compatible classifier extra and the
# 'faces' extra (onnxruntime + numpy). 'faces' adds ~261 MB of model weights
# on the FIRST `faces scan`/`watch` run (see FaceStore's model download,
# imageharbor/faces/download.py) -- not at build time, so this layer itself
# stays small. docker-compose.yml's `imageharbor-models` volume is what
# stops that 261 MB download from repeating on every container recreate.
COPY pyproject.toml README.md ./
COPY imageharbor ./imageharbor
RUN pip install --no-cache-dir ".[openai,faces]"

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
