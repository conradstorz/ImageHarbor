# Deploying ImageHarbor as a Docker watcher

ImageHarbor can run as a continuous watcher container on an always-on amd64
Linux host, organizing a NAS photo library and classifying images with a
self-hosted OpenAI-compatible AI server (e.g. a Jetson running Ollama).

## 1. Mount the NAS on the host

The container uses **host bind-mounts**, so mount the NAS shares on the Docker
host first (credentials stay here, never in the container). Example CIFS entries
in `/etc/fstab`:

```
//DS220plus/photos          /mnt/nas/photos            cifs  ro,credentials=/etc/nas.cred,uid=1000,iocharset=utf8  0 0
//DS220plus/photos-organized /mnt/nas/photos-organized  cifs  rw,credentials=/etc/nas.cred,uid=1000,iocharset=utf8  0 0
```

`/etc/nas.cred` holds `username=` / `password=` (chmod 600). Adjust share names
to your NAS. NFS works equally well.

## 2. Point at your AI server

Edit `docker-compose.yml` environment:

- `IMAGEHARBOR_AI: openai`
- `IMAGEHARBOR_AI_BASE_URL`: your server's OpenAI-compatible endpoint
  (e.g. `http://jetson.local:11434/v1` for Ollama).
- `IMAGEHARBOR_AI_MODEL`: a vision model available on that server (e.g. `llava`).
- `IMAGEHARBOR_AI_API_KEY`: usually `not-needed` for local servers.

To run without AI (filename-keyword stub), set `IMAGEHARBOR_AI: stub`.

## 3. Build and run

```
docker compose build
docker compose up -d
docker compose logs -f
```

Each pass logs `watch pass N: processed=.. skipped=.. errors=..`. The catalog is
kept on the local `imageharbor-catalog` volume; organized copies are written to
the NAS. Originals are never modified.

## 4. Smoke test

```
docker run --rm imageharbor:latest --help
docker run --rm imageharbor:latest watch --help
```

Then verify integrity of the organized library at any time:

```
docker compose run --rm imageharbor verify /data/dest
```

## Notes

- Watching is **poll-based** (default 300s via `IMAGEHARBOR_INTERVAL`), because
  filesystem events (inotify) are unreliable over SMB/CIFS. Unchanged files are
  skipped without re-reading them, using a local seen-cache.
- Keep only one watcher instance per catalog (the catalog is single-writer).
