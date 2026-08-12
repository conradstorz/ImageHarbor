# Jetson Photo Workflow Roadmap (Rev. 2)

> **Historical document.** The PCS folder tree and PCS-prefixed filenames
> described below were superseded on 2026-08-11 by the date-derived tree and
> the `[<date>][-<descriptor>]_<digest>` filename grammar. See
> `docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md`. The
> integrity, immutability, and resumability requirements are unchanged and
> still authoritative.

## Vision

Build a deterministic, resumable photo organization system that reads
photos from a read-only source, never modifies originals, uses AI to
classify and describe images, copies them into an organized library,
maintains a searchable SQLite catalog, and uses a stable filename that
remains valid for decades.

## Core Principles

-   Originals are immutable.
-   AI enriches metadata but never changes image bytes.
-   Organization is deterministic and resumable.
-   Every organized file is self-verifying.

## Processing Pipeline

1.  Discover images.
2.  Compute SHA-256 of original bytes.
3.  Read EXIF/XMP.
4.  Run AI classification and captioning.
5.  Map results into the Photo Classification Standard (PCS).
6.  Generate deterministic filename.
7.  Copy image.
8.  Verify copied bytes.
9.  Update SQLite catalog.
10. Optionally write JSON sidecar.

## Photo Classification Standard (PCS)

PCS is a controlled taxonomy owned by the project.

Example ranges: - 100 People - 200 Animals - 300 Places - 400
Transportation - 500 Events - 600 Nature - 700 Documents - 800
Architecture - 900 Miscellaneous

The AI selects from PCS rather than inventing categories.

## Filename Standard

Format:

``` text
<PCS>-<descriptor>_<sha256-base64url>.<extension>
```

Example:

``` text
330-indiana-dunes_qfQ8jnnXIdtn-juMY-1JDqyBLPF6j2MJlbh8sZOIfcI.jpg
```

### PCS

Three-digit controlled taxonomy identifier.

### Descriptor

One to three normalized words. Rules: - lowercase - ASCII
letters/digits - hyphen separators - max 30 characters - human
readable - deterministic

### SHA-256 Identity

Store the COMPLETE SHA-256 digest encoded as unpadded Base64url (43
characters). It is never truncated.

Benefits: - deterministic identity - duplicate detection - integrity
verification - immutable identifier

Any byte change changes the digest.

### Extension

Preserve original extension, normalized to lowercase.

## Filename Constraints

-   target filename under 100 characters
-   filesystem-safe
-   compatible with Windows, Linux, macOS, SMB, Synology

## Verification

1.  Extract digest from filename.
2.  Compute SHA-256.
3.  Encode Base64url (no padding).
4.  Compare exactly.

## SQLite Catalog

Store: - original path - organized path - SHA-256 - PCS version - PCS
primary category - secondary tags - AI caption - objects - OCR - EXIF -
timestamps - model version - processing history

## Directory Layout

``` text
Photos-Original/ (read-only)

Photos-Organized/
    300-places/
        330-beach/
            330-indiana-dunes_<sha>.jpg

Review/
Duplicates/
Logs/
Manifests/
```

## Acceptance Criteria

-   originals never modified
-   processing resumable
-   duplicate detection via full SHA-256
-   every filename contains PCS + descriptor + complete cryptographic
    identity
-   filename alone permits integrity verification
-   rich metadata remains in SQLite and optional sidecars
