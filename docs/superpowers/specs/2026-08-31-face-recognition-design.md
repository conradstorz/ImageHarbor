# Face Recognition

**Date:** 2026-08-31
**Status:** Designed, not built.

## Why this exists

The library holds ~77,000 photographs spanning 1968–2026. Google named people in
about one photo in seven — 10,574 of 70,536 — and named nobody in the rest. Those
names are already ingested: `takeout/ingest.py` writes them into each sidecar's
`people` list with `source: "google_photos_people"`. But they are inert. Nothing
reads them, and nothing extends them to the **85% of photos Google left
untagged**.

This spec builds a third pass that detects faces, groups them, and uses the names
Google already supplied as **seed evidence** to propose who the untagged photos
show. It answers three questions the catalog cannot answer today:

- Which photos show a given person?
- Who are the recurring faces in this library?
- For a face Google named once, where else does that person appear?

**It does not put people in filenames.** Faces never enter the descriptor ladder,
never call `tiers.is_upgrade`, and never cause a rename or a move. This is a
deliberate scope boundary, not an omission — see "Out of scope".

## Scope

**In scope.**

- A `faces` pass: detect, embed, cluster, and propose names. Independent of the
  facts pass and of the enrichment pass, and runnable with the AI backend down.
- Catalog tables for faces, clusters, people, and proposals.
- A People section on the existing dashboard for confirming, rejecting, merging,
  and splitting clusters.
- Confirmed names propagated into each photo's sidecar `people` list.
- A `calibrate` step that measures the clustering threshold against the library's
  own labelled data rather than adopting a constant from elsewhere.

**Out of scope,** deliberately:

- **Faces in filenames or folder placement.** Placement comes from
  `date_resolver.ResolvedDate.folder` and descriptors come from the tier ladder;
  neither learns about faces. A person's name is not a fact about when a photo
  was taken, and the filename grammar is a date-and-descriptor grammar.
- **Automatic identity assertion.** No code path writes a person's name onto a
  photo without a human confirming that cluster. See "Propose, never assert".
- **Video.** No video bytes are copied by any current code path; faces follow the
  same boundary.
- **Face-based deduplication or quality ranking.** Different feature, different
  spec.
- **Recognition as a service.** Inference runs in-process in the ImageHarbor
  container. No new deployable.

## Evidence

Measured directly on 2026-08-31 by parsing **every** per-media JSON sidecar in
all 175 archives. Not a sample:

| Quantity | Value |
| --- | --- |
| Media sidecars read | 70,536 |
| Carrying a non-empty `people[]` | **10,574 (15.0%)** |
| Carrying **exactly one** name | **5,670 (8.0%)** |
| Distinct names | **90** |
| Names per tagged photo | 1: 5,670 · 2: 2,830 · 3: 905 · 4: 613 · 5: 116 · 6: 113 · 7: 63 · 8: 90 |

The eight most-tagged names, with counts: Conrad Storz (3,309), Suzanne Storz
(1,922), pete storz (1,539), Douglas Storz (1,287), Judy Storz (1,198), Michelle
Knight (1,193), Conrad Storz III (980), Cathy Garrett (534).

A 10-archive random sample taken first put the tagged rate at 11.0%; the full
read found 15.0%. The sample understated it by a third, which is why the number
above is a census rather than an extrapolation.

The 2026-08-23 survey counted 70,783 Google JSON sidecars against the 70,536
here. The difference is album metadata: that count is every `.json` member, this
one is only per-media sidecars, selected by the presence of `photoTakenTime` or
`creationTime`.

This settles the design's central risk. `takeout/metadata.py` notes that the
older AlbumArchive schema "has no `people` at all"; had this export been that
generation, the seeding strategy below would have had nothing to seed from. It is
the newer schema, and the names are there.

**The names are dirty, and two of the defects matter.**

- **Suffixes distinguish real people.** `Conrad Storz` (3,309) and `Conrad Storz
  III` (980) are both heavily used. They are different people whose names differ
  only by a suffix. Any fuzzy or string-similarity matching over this vocabulary
  merges a father and a son. Name identity must be **exact**, and any
  consolidation must be human-confirmed.
- **Whitespace and case drift.** `Gladys Blankenbeker ` carries a trailing space
  across all **461** of its occurrences; `pete storz` (1,539) and `claire Storz`
  (442) are lower-cased. Because the sidecar's `people` list is keyed on the
  name, an unnormalized key silently splits one person into two entries — and at
  1,539 photos, `pete storz` is not a rounding error.

The resolution is asymmetric on purpose: **strip and collapse whitespace
automatically; never case-fold automatically.** Whitespace carries no meaning and
its removal cannot merge two people. Case might — and given that this vocabulary
already proves suffixes are load-bearing, a case-insensitive merge is a guess.
Case variants surface in the review UI as a *suggested* merge, which is the same
propose-never-assert rule applied to names instead of faces.

A separate roster exists and cannot be joined: the preserved Picasa face-tag file
names **73 people across 1,496 entries with no photo reference at all**
(`takeout/provenance.py`). It is useless as evidence and valuable as a
vocabulary — it seeds the review UI's autocomplete and catches spelling drift
between Google's tags and the operator's own. Note that Google's 90 names and
Picasa's 73 are different-sized sets, so neither is a subset of the other and the
union is the real vocabulary of this library.

## Design

### Where inference runs

In the ImageHarbor container, via ONNX Runtime, behind a new `faces` extra. No
new service, no network call, no dependency on the Jetson. The existing AI
backend is a vision **chat** model (LLaVA via Ollama) and cannot produce face
embeddings; face work is a different computation with a different failure mode,
and coupling it to the AI backend would mean the backend being down stops face
work that needs nothing from it.

This also keeps the faces pass **deterministic** — the same bytes and the same
model produce the same vector — which is what makes the idempotence and
resumability properties below achievable at all.

### Models

| Role | Model | Licence | Size |
| --- | --- | --- | --- |
| Detector | YuNet (`face_detection_yunet`) | MIT | ~350 KB |
| Embedder | AuraFace v1 (ArcFace ResNet-100, 512-dim) | Apache-2.0 | 261 MB |

Both are permissively licensed, which matters: InsightFace's ArcFace weights are
not, and PhotoPrism refuses to redistribute them for that reason. ImageHarbor
must not acquire a non-free artifact dependency by default.

AuraFace over the lighter SFace (128-dim, 39 MB) is a trade of machine time for
human time. Embedding is a one-time overnight cost; cluster quality converts
directly into how many confirmations the operator clicks, because a weaker model
fragments one person into many clusters and every fracture is manual work under a
propose-never-assert rule.

Weights are **not shipped in the wheel.** They download once, checksum-verified,
into a new `imageharbor-models` Docker volume.

### `imageharbor/faces/` — module layout

Pure logic is split from I/O so the core is testable without a single byte of
model weights, the same split that makes `tiers.py`, `takeout/pairing.py`, and
`takeout/report.py` exhaustively testable.

| Module | Purity | Responsibility |
| --- | --- | --- |
| `models.py` | pure | Registry: artifact, checksum, geometry, **channel order, normalization**, alignment mode, dim, licence, URL. |
| `detect.py` | I/O | YuNet session. Image → `DetectedFace(bbox, score, landmarks)`. |
| `align.py` | **pure** | 5-landmark → ArcFace 112×112 affine warp. |
| `embed.py` | I/O | Aligned crop → L2-normalized vector, stamped with its model. |
| `cluster.py` | **pure** | Embeddings + params → cluster assignments. |
| `attribute.py` | **pure** | (cluster → photos) + (photo → names) → ranked proposals. |
| `calibrate.py` | **pure** | Anchor pairs → measured threshold and a precision/recall curve. |
| `store.py` | I/O | Catalog tables and the crop cache. |
| `runner.py` | I/O | The resumable per-photo pass. |

`models.py` exists for one reason: **channel order and normalization are not
recoverable from an ONNX graph.** A wrong input shape raises immediately; a wrong
channel order loads, runs, and returns plausible embeddings that are quietly
worse. Those fields are declared per model and never inferred — the same
discipline `content_type.py` applies to extensions, applied to model contracts.

`align.py` uses Pillow's `Image.transform(AFFINE)`. **No OpenCV.** Given how
deliberately thin this project's dependency list is, a 60 MB vision library for
one matrix operation is not a trade worth making.

### Catalog schema

```sql
CREATE TABLE faces (
  id             INTEGER PRIMARY KEY,
  sha256_b64url  TEXT    NOT NULL,      -- → photos
  bbox_x, bbox_y, bbox_w, bbox_h INTEGER NOT NULL,
  det_score      REAL    NOT NULL,
  landmarks      TEXT    NOT NULL,      -- JSON, 5 points
  detect_model   TEXT    NOT NULL,      -- provenance: placed these landmarks
  embed_model    TEXT,                  -- provenance: made this vector
  embedding      BLOB,                  -- float32, L2-normalized
  embedding_dim  INTEGER,
  cluster_id     INTEGER,               -- NULL = unclustered
  rejected       TEXT,                  -- quality-gate reason, NULL = kept
  detected_at    TEXT    NOT NULL
);

CREATE TABLE face_scan (                -- work queue + idempotence
  sha256_b64url  TEXT    NOT NULL,
  detect_model   TEXT    NOT NULL,
  face_count     INTEGER NOT NULL,
  scanned_at     TEXT    NOT NULL,
  sidecar_at     TEXT,                  -- last person-propagation write
  PRIMARY KEY (sha256_b64url, detect_model)
);

CREATE TABLE clusters (
  id           INTEGER PRIMARY KEY,
  embed_model  TEXT    NOT NULL,        -- never compared across models
  centroid     BLOB,
  face_count   INTEGER NOT NULL,
  person_id    INTEGER,                 -- NULL until a human confirms
  assigned_at  TEXT,
  created_at   TEXT    NOT NULL
);

CREATE TABLE people (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL UNIQUE,
  source      TEXT NOT NULL,            -- human | google_photos_people | picasa_roster
  created_at  TEXT NOT NULL
);

CREATE TABLE proposals (
  cluster_id   INTEGER NOT NULL,
  name         TEXT    NOT NULL,
  support      INTEGER NOT NULL,        -- photos in cluster tagged this name
  total_tagged INTEGER NOT NULL,        -- photos in cluster tagged anything
  score        REAL    NOT NULL,
  proposed_at  TEXT    NOT NULL,
  decided      TEXT,                    -- NULL | confirmed | rejected
  decided_at   TEXT,
  PRIMARY KEY (cluster_id, name)
);
```

Three properties this shape buys:

- **`clusters.person_id` is many-to-one and moves only on human confirmation.**
  One person legitimately owns several clusters — see "Aging" below.
- **`embed_model` on every face and every cluster**, with cross-model comparison
  refused, is what makes "run a better model in 2027" safe rather than silently
  corrupting.
- **`face_scan` keyed on `(digest, detect_model)`** gives one-photo resumability
  and no-op re-runs, the property `takeout_members` already gives ingestion.

### Stage 1 — detect and embed

Work queue is every organized image with no `face_scan` row for the current
detector. Per photo:

1. Open the organized copy and call `draft('RGB', (640, 640))` **before**
   `load()`. On a 12 MP JPEG this performs the downscale in the DCT domain and
   skips most of the decode. Decode, not inference, dominates this loop; this one
   call is what fits the pass into an overnight window on CPU.
2. YuNet → boxes, scores, five landmarks.
3. **Quality gate**: reject faces below a minimum detection score or box size. A
   20-pixel face in a crowd shot yields an embedding that is noise, and noise
   does not cluster — it smears. Rejected faces are **recorded with a reason, not
   dropped**, following `mark_extras`' precedent of classifying rather than
   omitting.
4. `align.py` warps survivors onto the ArcFace template.
5. AuraFace → 512-dim vector, **L2-normalized where it is produced**, so cosine
   and Euclidean stay equivalent everywhere downstream.
6. One transaction: `faces` rows, the `face_scan` row, the crops.

Interruption costs at most one photo. A corrupt image is recorded in the existing
`failed_files` table with a face-stage marker and **never reaches the circuit
breaker** — the breaker's invariant reserves it strictly for
`AIClassifier.describe()` failures, and this pass makes no AI calls at all.

### Anchors: the library's own ground truth

A photo with **exactly one detected face and exactly one Google name** is an
unambiguous `(face → name)` pair. The evidence above bounds these from above at
**5,670 photos**; after the one-face condition, expect low thousands spread
across the 90 names Google used.

Anchors give three things that guessing cannot:

- **A measured threshold.** Compare the similarity distribution of same-name
  anchor pairs against different-name pairs and cut at a target precision,
  instead of copying a constant out of another project's README.
- **Named seeds.** Emma's cluster begins as Emma's cluster.
- **An honest score.** Hold out a portion of anchors and report measured
  precision and recall. The system reports how well it is working rather than
  asserting that it is.

### Stage 2 — cluster

Pairwise clustering over ~150,000 faces is ~11 billion comparisons. Instead,
**faces are compared against cluster centroids, not against each other** — a few
thousand centroids, one chunked matmul, seconds.

**Phase A, seed from anchors.** Group anchor faces by name; within each name,
mutually-similar faces form that person's initial clusters. One name may seed
several clusters. That is correct, not a defect.

**Phase B, assign the remainder.** For each remaining face in **deterministic
digest order**: cosine similarity against every centroid of the same
`embed_model`; join the best above threshold, else start a cluster. Centroids
update as a running mean, re-normalized.

**Phase B is order-dependent, and it is this design's weakest joint.** Three
things contain it: anchors are placed first, so the clusters that matter exist
before any guessing begins; digest ordering makes a re-cluster reproducible
rather than a fresh roll; and merge/split in the review UI is the actual repair.
`--recluster` rebuilds from stored embeddings and never re-runs a model.

Embeddings stream in chunks of ~10,000 rather than materializing the full
similarity matrix; 150,000 × 512 × float32 is ~300 MB held at once.

### Aging

**No model solves aging, and the design must not pretend otherwise.** This
library spans 58 years. A person at 5 and the same person at 25 will not land in
one cluster under AuraFace, SFace, or anything Immich or PhotoPrism ships.

This is why `clusters.person_id` is many-to-one and why **merge is a first-class
operation rather than an error path**. One person being several clusters is the
expected steady state for a library of this span, not a failure to be tuned away.

### Stage 3 — attribute

For each cluster, let **T** be its photos carrying any Google name:

```
support(n) = photos in T tagged n
score(n)   = support(n) / |T|
```

Propose the top name when `score >= min_score` and `support >= min_support`.
Everything lands in `proposals` with its evidence. **Nothing writes
`clusters.person_id`.**

The evidence surfaced is two numbers, because the second is the entire point:

> **Suzanne Storz** — 14 of 15 named photos in this cluster agree.
> Confirming names **340 photos Google never tagged.**

That ratio is what filling the gap actually looks like, and seeing it before
committing is what makes a proposal judgeable rather than merely trustable.

### Propose, never assert

The rule, stated once so every call site inherits it:

**No code path writes a person's identity onto a photo without a human having
confirmed that cluster.** Proposals live in their own table. `clusters.person_id`
moves only through the confirm endpoint. A rejected proposal is **recorded as
rejected, not deleted**, so the same wrong guess is not re-proposed every pass.

This follows the project's existing posture — `pairing.sidecar_for` returns
`None` rather than guess, `date_resolver` refuses mtime rather than assert a date
it cannot support — applied to identity, where a wrong assertion is both harder
to notice and more personal than a wrong date.

### Sidecar

**The sidecar records what a human decided; the catalog holds what the machine
computed.** Geometry, embeddings, and cluster membership are deterministic and
re-derivable, so they stay in the catalog. A confirmed name is human input that
cannot be re-derived, so it goes in the sidecar.

`KEYED_LISTS["people"]` widens from `("name",)` to `("name", "source")`.

**This needs no migration.** Every existing entry already carries
`source: "google_photos_people"`, so the wider key resolves them unchanged. The
change is necessary because under the narrow key, an ImageHarbor-confirmed
"Suzanne Storz" merging onto Google's entry of the same name would *change* the
`source` field and relocate the old value into history — recording the two as if
they conflicted, when both are simultaneously true. Google tagged her; the
cluster confirms her. Under the wider key they are two entries, both correct.

Face-confirmed entries carry `confirmed_at` and `cluster_ids`. **`confirmed_at`
must be added to `sidecar_schema._ANNOTATION_FIELDS`.** Omitting it means the
entry can never match itself on a later merge and the history list grows on every
`watch` cycle, forever. This exact failure has shipped here once already.

### Dashboard

New routes on the existing server. Its router is a hand-rolled `if/elif` chain on
exact paths, so `/api/face-crop/<id>` is the first route needing a prefix match.

| Route | Purpose |
| --- | --- |
| `GET /api/people` | Review queue: clusters, proposals, evidence, crop ids |
| `GET /api/face-crop/<id>` | `image/jpeg` from the crop cache |
| `POST /api/people/confirm` | `{cluster_id, name}` — the only writer of identity |
| `POST /api/people/reject` | Dismiss a proposal, recorded not deleted |
| `POST /api/people/merge` | `{person_id, cluster_ids}` — the aging repair |
| `POST /api/people/split` | `{cluster_id, face_ids}` — the bad-cluster repair |

**Confirm must not write sidecars synchronously.** Confirming a 340-photo cluster
means 340 sidecar merges over CIFS — minutes, inside an HTTP handler. Confirm
writes the catalog and returns; the next `faces` pass propagates.

That queue needs no new table: a photo needs propagation when any of its faces
belongs to a cluster whose `assigned_at` is later than that photo's
`face_scan.sidecar_at`. Derived, self-healing, correct after a crash.

The faces pass respects the existing pause — between photos, never mid-photo —
and gains a **faces on/off** toggle mirroring the AI enrichment toggle.

Per the dashboard's founding rule, **a dashboard failure never stops organizing.**
A face-review error logs and the passes carry on.

### Crop cache

`<catalog_dir>/face-crops/<ab>/<cd>/<id>.jpg`, sharded two levels, on the
**catalog volume, not the NAS**. At ~150,000 faces × ~4 KB that is ~600 MB;
writing that many small files to a CIFS mount would be slow to write and worse to
serve.

Crops are a **derived cache, deletable at any time**: the bbox is in the catalog,
so regeneration is a re-crop, not a re-detect.

### CLI

```
imageharbor faces scan       --dest DEST     # stage 1
imageharbor faces cluster    --dest DEST     # stage 2  (--recluster)
imageharbor faces calibrate  --dest DEST     # measure the threshold
imageharbor faces status     --dest DEST
imageharbor faces models download
```

Bare `imageharbor faces` runs scan → cluster → attribute, which is what `watch`
invokes as its third pass.

## Invariants this work adds

- **Faces never rename or move a file.** No face code path calls
  `tiers.is_upgrade` or `relocate`.
- **No identity is written without human confirmation.**
- **Embeddings are never compared across `embed_model` values.** Attempting it
  raises rather than returning a plausible number.
- **Face failures never feed the circuit breaker.**
- **Name identity is exact.** No fuzzy or similarity-based name merging, ever —
  `Conrad Storz` and `Conrad Storz III` are different people.
- **Embeddings are L2-normalized where they are produced.**

## Failure modes

| Condition | Behaviour |
| --- | --- |
| `onnxruntime` not installed | `faces` errors clearly; `watch` warns once, skips the pass, keeps organizing |
| Model checksum mismatch | Refuse to run. A name match is not an artifact match |
| Corrupt image | `failed_files`, face-stage marker, never the breaker |
| Crop cache deleted | Regenerated from stored bbox |
| Model swapped | Old embeddings kept, never compared; work queue reopens on `detect_model` |
| Interrupted mid-cluster | Partial run discarded, rebuilt from stored embeddings. Minutes, not the overnight pass |
| Dashboard error | Logged; passes continue |

## Test plan

The pure core is testable with **zero model bytes**.

- `align.py` — warp geometry against known landmark sets, including collinear and
  degenerate ones.
- `cluster.py` — synthetic vectors with known groups must be recovered; threshold
  boundaries; determinism across two identical runs; **mixing `embed_model` must
  raise**.
- `attribute.py` — table-driven scoring: ties, no tagged photos, below threshold,
  single supporter.
- `calibrate.py` — synthetic anchors with known separation yield the expected
  threshold.
- **Idempotence property test**, mirroring
  `test_sidecar_schema.py::test_never_loses_a_value_over_a_random_merge_sequence`:
  the faces pass run twice leaves catalog and sidecars byte-identical.
- **Backward compatibility** — widening the `people` key leaves existing
  `google_photos_people` entries untouched.
- **Annotation registry** — assert every annotation key a face entry uses is in
  `_ANNOTATION_FIELDS`.
- **Breaker isolation** — a face failure does not move the circuit breaker.
- **Name normalization** — whitespace stripped and collapsed; case *never* folded
  automatically; `Conrad Storz` and `Conrad Storz III` stay distinct.

Integration tests needing real weights **skip when the weights are absent**, but
a **broken runtime must fail, not skip**. A version mismatch quietly passing as
"skipped" is how a model that never ran ships as tested.

## Order of work

One dependency is non-obvious and drives the sequencing:

1. `models.py`, `align.py`, `cluster.py`, `attribute.py`, `calibrate.py` — the
   pure core, fully tested, no weights needed.
2. Catalog schema and `store.py`.
3. `detect.py`, `embed.py`, `runner.py` — stage 1. Run it over the library.
4. **`calibrate` against real anchors, then choose the threshold.**
5. Stage 2 and 3.
6. Dashboard People section.
7. `watch` integration and the docker-compose model volume.

**The threshold cannot honestly be chosen before step 4**, because calibration
needs embeddings to exist. Picking it up front would be exactly the guess this
design exists to avoid.

## Deferred

- **Faces on video.** Follows whenever video becomes first-class.
- **Re-running with a different embedder.** The schema supports it; no UI does.
- **Face-based search in the dashboard** beyond the review queue — "show me every
  photo of X" as a browsable result set.
- **Exporting people to the organized tree** as symlink albums or an index.
- **Picasa roster join.** If a future export supplies a join key, those 1,496
  entries become evidence rather than vocabulary.

## Open questions

- `min_score` and `min_support` for a proposal are stated as configurable with no
  default chosen. Calibration should inform them, as it informs the threshold.
- Whether singleton clusters are hidden from review by default or merely sorted
  last. Hiding risks burying a genuine rare person; showing risks burying the
  operator.
