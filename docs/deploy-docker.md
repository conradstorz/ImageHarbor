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

## 5. Reach the dashboard

`docker-compose.yml` already publishes `8080:8080` and points its
`healthcheck` at `/healthz`. Once the container is up, browse to
`http://<docker-host>:8080/` for the operational dashboard: library stats,
evidence-quality tier tables, work queues, pass history, and a projection of
when the remaining backlog will clear (or `stalled`/`unknown` when the AI
backend is unreachable, the watcher is paused, or there isn't enough recent
history to trust a number).

Three controls live on the page:

- **Pause / Resume** — stops the watcher between photos, never mid-photo, in
  both the facts and enrichment passes; persisted, so it survives
  `docker compose restart`.
- **Poll interval** — overrides `IMAGEHARBOR_INTERVAL` at runtime without
  editing compose.
- **AI enrichment on/off** — turns the enrichment pass off while the Jetson (or
  other AI server) is down or busy; the facts pass keeps organizing at full
  speed regardless.

Any control currently overriding a compose env var shows a warning line naming
the value it is overriding (e.g. `⚠ overriding IMAGEHARBOR_INTERVAL=300`), with
a one-click revert. This exists to stop the failure mode where you edit
`docker-compose.yml`, restart, and nothing changes because a months-old
dashboard override is silently still winning.

To disable the dashboard entirely, set `command: watch --no-dashboard` (or add
`--no-dashboard` to the `command:` list) — the watcher organizes exactly the
same either way; a dashboard failure (e.g. the port already bound on the host)
never stops it, it only logs a warning.

## Notes

- Watching is **poll-based** (default 300s via `IMAGEHARBOR_INTERVAL`), because
  filesystem events (inotify) are unreliable over SMB/CIFS. Unchanged files are
  skipped without re-reading them, using a local seen-cache.
- Keep only one watcher instance per catalog (the catalog is single-writer).
- The dashboard is served in-process by the same `watch` container on a
  daemon thread — no separate service to run. It reads through the same
  catalog connection discipline (WAL mode, `busy_timeout`) as the watcher
  itself, so a slow page render never blocks a pass.
