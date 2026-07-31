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

USER harbor

ENTRYPOINT ["imageharbor"]
CMD ["watch"]
