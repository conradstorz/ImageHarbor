# ImageHarbor invariants — full rationale

Verbatim detail behind the terse rules in `CLAUDE.md`.

- **PCS codes are strings, not ints, and are owned by the taxonomy, not the
  classifier.** A valid code matches `^\d+(~\d+)*$` — plain integers for the common
  case (`"330"`), or a `~N` suffix appended when a parent's decimal slots are used up
  or a leaf needs a child (**never a dot**, e.g. `"330~1"` not `"330.1"`). Codes are
  minted append-only by `taxonomy.Taxonomy` and persisted in the catalog `taxonomy`
  table, seeded once from `pcs.PCS_CATEGORIES` (now seed data only). The **9
  top-level classes are fixed**; all growth happens beneath them, and existing codes
  are never renumbered or reused.
- **The classifier perceives; it never classifies.** `AIClassifier.describe` returns
  a `ContentDescription` (subject/scene/objects/caption/tags/ocr_text) and nothing
  taxonomy-shaped. Deciding the top-level **class** is `concept_map.class_for`'s job
  (static seed + learned store); the classifier is only consulted as a **text-only
  fallback** (`pick_class`) on a concept-map miss, and that answer is immediately
  memoized via `concept_map.remember` so the same subject never asks the AI twice.
  Deciding the **code** for a (class, label) pair is `Taxonomy.resolve_or_create`'s
  job — by reusing an exact/alias match among the target parent's existing children,
  or, failing that, consulting `AIClassifier.adjudicate(label, candidates)` for a
  semantic synonym match before minting a new one. Do not add
  string-similarity/fuzzy-ratio gating in front of the adjudicator — it was tried and
  dropped because true synonyms (e.g. "festivities" vs. "holidays") are
  string-dissimilar, which made the adjudicator unreachable.
- **Placement comes from `date_resolver.ResolvedDate.folder`**, not from the PCS
  taxonomy. A file lands at `YYYY/YYYY-MM/` when a trustworthy capture date is
  known (EXIF, then a date pattern in the original filename), or `Undated/`
  otherwise — never a guessed year. PCS lives in the catalog and in the sidecar's
  `classification` block, not in the path or the filename. `taxonomy.folder_path
  (code)` still exists and is still called, but only to record a human-readable
  classification path *inside the sidecar* — it does not decide where any byte on
  disk lives. Do not reintroduce PCS-driven placement.
- **A file is renamed or moved only when `tiers.is_upgrade(old, new)` returns
  True** — the proposed `(date_tier, descriptor_tier)` must be strictly better in
  at least one dimension and worse in neither. Equal tiers are a no-op, which is
  what makes a repeated run idempotent. An AI subject (`DESC_AI_SUBJECT`, tier 20)
  can therefore never displace a human-authored filename (`DESC_HUMAN_FILENAME`,
  tier 30) — this is enforced structurally by the predicate, not by care at any
  call site. Never add a code path that renames or relocates a file
  unconditionally. **`--reclassify` does NOT bypass this predicate** — it only
  bypasses the *work queue* (`iter_all()` instead of `iter_unenriched()`, so
  already-enriched rows are revisited); `is_upgrade` still gates the rename for
  every row it walks, so re-running `--reclassify` on an already-enriched photo
  re-records classification (catalog + sidecar) but is a guaranteed rename
  no-op, because the existing row already ties the new answer at
  `DESC_AI_SUBJECT`. See `tests/test_monotonicity.py` for the pinned behavior.
- **File mtime must never enter the date ladder.** `date_resolver.py` deliberately
  never reads `stat().st_mtime`: mtime records when a file was copied, not when a
  photo was taken, and asserting a date we can't support is exactly the quiet
  corruption the project's SHA-256 discipline exists to prevent.
- **A sidecar may gain information; it may never lose any.** Every merge into a
  sidecar goes through `sidecar_schema.merge`, which is total (never raises)
  and relocates a superseded value into a `history[]` list rather than
  overwriting it — never drop a value on the floor to keep a sidecar smaller.
  This is only usable because history dedupes on the *value* with annotation
  fields stripped (`_core()`/`_ANNOTATION_FIELDS`), not on `observed_at` or any
  other timestamp: a repeated `process`/`enrich`/`watch`/`takeout ingest` run
  must leave every sidecar byte-identical
  (`merge(merge(B, U), U) == merge(B, U)`). Do not add a new annotation key to a
  history entry without adding it to `sidecar_schema._ANNOTATION_FIELDS` — an
  omission there means the entry can never match itself on a later merge, and
  the history list grows without bound on every run. This shipped once as a
  Critical bug and is why the registry comment on `_ANNOTATION_FIELDS` is as
  blunt as it is. One unbounded-growth path remains and is accepted, not a
  bug: `classification` is a `VERSIONED_BLOCKS` entry (no tier gate), so
  looping `enrich --reclassify` against a nondeterministic AI backend — one
  that answers a different caption/subject for the same image on every call
  — records a new `classification` history entry per distinct answer,
  forever. It is user-driven (an operator has to choose to loop
  `--reclassify`) and not exposed on `watch`, which never passes
  `--reclassify`.
- **Takeout pairing never guesses, and Takeout archives are never written to.**
  If no pairing rung yields exactly one sidecar match, `pairing.sidecar_for`
  returns `None` and the member is ingested from EXIF and its filename alone —
  a fully correct outcome. A *wrong* pairing writes another photo's capture date
  into this photo's name and folder, which is exactly the quiet corruption the
  SHA-256 discipline exists to prevent. Separately: archives are opened `'r'`
  only, and Google's `creationTime` must never reach `resolve_date`.
- **A pairing's confidence decides what it may contribute — never which engine
  produced it.** `own` may supply capture date, title, and face tags. `related`
  supplies the capture date only: its sidecar names a *different* file (usually
  this file's unedited original before an `-edited` copy), so the title and
  `people` inside it describe that other photo, not this one, and
  `_ingest_image`/`_merge_takeout_sidecar` must never apply them here. This
  holds identically whether the pairing came from a `Takeout_Inventory` index
  read by `index_reader.py` (`--takeout-index`, optional and auto-detected —
  without it, ingestion behaves exactly as it always has) or from the built-in
  six-rung ladder in `pairing.py` — `confidence` is the only thing either path
  is allowed to branch on, so a photo's title and face tags can never depend on
  whether a second tool was run. A `related` pairing's raw Google JSON is still
  preserved as provenance (never dropped — see the previous invariant) but
  labelled with its `confidence` and `pair_rule`, so the GPS coordinates and
  other fields inside it read as self-describing rather than silently
  authoritative.
- **The 43-char digest is located by counting back from the end of the stem, NOT by
  splitting on the last `_`.** Base64url legitimately contains `_`, so a naive rsplit
  corrupts parsing. This logic is duplicated in `hashing.extract_digest_from_stem`
  and `filename.parse_filename` — keep them in sync. `extract_digest_from_stem` no
  longer validates a PCS prefix (there is none in the current filename grammar); it
  validates the extracted 43 characters against the Base64url character class
  instead, and otherwise leaves the prefix unconstrained — which is also why
  legacy PCS-prefixed filenames from before this redesign still parse.
- **Non-AI failures must never feed the circuit breaker.** Only a failure raised
  by the classifier's backend-facing calls — `describe()` and the `pick_class`
  fallback — counts toward `--breaker-threshold` (`adjudicate` is caught inside
  `taxonomy.resolve_or_create` and degrades to minting a new code, so it never
  reaches the breaker); a filesystem fault (a missing organized file, a failed
  rename, a permissions error) must be handled and counted separately, never
  reported to the breaker as if it were backend-outage evidence. The facts pass
  makes no AI calls at all, so it never touches the breaker.
- **Content addressing must stay stable.** The digest is over raw file *bytes*
  (streamed in 64 KiB chunks). Changing the hash algorithm, the Base64url encoding,
  or the 43-char length assumption breaks every existing filename and catalog key.
- **Originals are read-only; the organized copy is verified before it is cataloged.**
  Preserve this copy → verify → catalog ordering when editing the pipeline. The
  one exception is `consume_source=True` (used only by Takeout ingestion, on a
  staging file the caller owns and created as disposable, never a real
  original): the ordering becomes rename → verify → catalog, with verification
  still reading the destination — nothing enters the catalog unverified either
  way. Written bytes are fsynced (`util.fsync_file`) *before* verification
  reads them back, so verification can never durably assert "copied, verified"
  for data that only reached the OS page cache and not the platter (a
  power-loss gap, not merely a process-crash one). The copy is deliberately
  `shutil.copyfile` → `fsync_file` → `shutil.copystat`, never `shutil.copy2` —
  `copy2` applies the source's mode bits as part of the copy itself, before
  `fsync_file`'s `"rb+"` handle would open, and a read-only source (the
  production RO NAS mount) then fails every copy with `PermissionError`.
- **Runtime output directories are git-ignored, not source** (`Photos-Organized/`,
  `Review/`, `Duplicates/`, `Logs/`, `catalog.db`, etc. in `.gitignore`).
- **Faces never rename or move a file.** No code path in `imageharbor/faces/`
  calls `tiers.is_upgrade` or `relocate` — a detected/clustered/confirmed face
  changes catalog rows and, once confirmed, a photo's sidecar `people` list,
  never its path or filename. This is the same "identity lives in the
  sidecar, not the name" posture PCS classification already has, applied to a
  fact that is far more personal to get wrong.
- **No identity is written without human confirmation.** `FaceStore.confirm`
  and `FaceStore.merge` are the *only* two methods that ever *assign a new*
  person to a cluster; `record_proposals` (what `faces cluster` calls) only
  ever writes to the `proposals` table and is guaranteed never to touch
  `person_id` — see `store.py`'s own mutation-tested guarantee.
  `replace_clusters` (a recluster) also writes `clusters.person_id`, but only
  to restore a person a human already confirmed onto its old cluster; it
  never invents one, so no identity is ever written without a human behind
  it. A rejected proposal is recorded as rejected, not deleted, so the same
  wrong guess isn't re-proposed every pass. This mirrors `pairing.sidecar_for`
  returning `None` rather than guessing a Takeout pairing, and
  `date_resolver` refusing mtime rather than assert a date it can't support —
  applied here to a person's name, which is both easier to get wrong
  unnoticed and more personal than a wrong date.
- **Embeddings are never compared across `embed_model` values.** A vector from
  one model shares a coordinate space with another only by coincidence, so a
  cross-model comparison is not merely wrong, it's a *plausible-looking*
  number that means nothing — worse than an error. `cluster.cluster_faces`
  raises `MixedModelError` rather than silently averaging or comparing across
  models, and `FaceStore.iter_face_vectors` filters to one `embed_model`
  before anything reaches clustering. Do not add a call path that pools
  vectors from two `embed_model`s to "get more data" — reprocessing under the
  new model is the only correct way to compare across a model swap.
- **Face failures never feed the circuit breaker.** That circuit is reserved
  for `AIClassifier.describe()` failures (AI-perception evidence about the
  backend); a face-scan failure (a corrupt image, a decode error) is a local
  filesystem/image fault with no bearing on the AI backend's health.
  `faces.runner.scan` catches per-photo exceptions itself and records them
  into `failed_files` via `catalog.record_file_failure` — the same table the
  enrichment pass's poison-quarantine bookkeeping uses — and never calls
  `breaker.record_failure`. `watch()`'s faces-pass block (`watcher.py`)
  mirrors this: it wraps the whole pass in its own `try/except` so a crash
  there can't take the loop down, and that block never references *breaker*
  at all.
- **Name identity is exact.** No fuzzy, similarity-based, or case-insensitive
  name merging, ever — `names.normalize` only strips and collapses
  whitespace; it deliberately leaves case alone. This library's own tagged
  vocabulary is why: `Conrad Storz` (3,309 photos) and `Conrad Storz III`
  (980 photos) are a father and son distinguished only by a suffix, and
  `pete storz`/`claire Storz`'s lowercase drift looks identical in *shape* to
  that real distinction. A scheme that can't tell those two cases apart must
  not auto-merge either one — case variants are only ever *reported*
  (`names.case_variants`) for a human to confirm.
- **Embeddings are L2-normalized where they are produced.** `Embedder.embed_
  batch` normalizes every vector before returning it, so cosine similarity
  and Euclidean distance stay equivalent for every downstream consumer —
  `cluster.py`'s centroid averaging, `calibrate.py`'s pairwise similarities —
  without each of them re-normalizing (or worse, one of them forgetting to).
  A stored embedding that somehow isn't unit-length is a bug upstream of
  storage, not something a consumer should silently correct for.
- **The `--json` document and exit codes are a stable contract.**
  `organize-my-life` consumes them as a subprocess tool. `to_dict()` keys
  may be added, never removed or retyped; the outcome constants and the
  0/1/2 exit mapping do not change. The same goes for every name in
  `imageharbor.__all__`.

## Known limitations

- **Poison-quarantine cannot fire when the poison IS the entire remaining
  queue.** `watcher._reconcile_poison` only counts a failure toward
  `--poison-max-fails` during a *healthy* enrichment pass (>=1 success,
  breaker not tripped this pass) — see `catalog.py`'s `failed_files`
  description above. `watch()`'s rotating probe offset (implemented in
  `watcher.py`, using `catalog.iter_unenriched`'s `offset` parameter — the
  breaker itself is unaware of it) lets a half-open probe skip a stuck head
  cluster and find a working file elsewhere in the queue, so quarantine can
  fire for poison files that have *some* describable file anywhere else in
  `iter_unenriched`. But if the files tripping the breaker are (or become)
  the *whole* unenriched queue — no describable file left to probe into —
  no pass can ever be simultaneously non-tripped and contain them, so they
  stay permanently un-quarantined in that state. This is accepted, not a
  bug: an all-poison remaining queue is information-theoretically
  indistinguishable from a real backend outage, and quarantining anyway
  would risk condemning an entire library during a genuine outage — exactly
  what the tripped/no-success discard rules exist to prevent. The cost is
  bounded (one half-open probe per backoff interval, capped at
  `--breaker-backoff-cap`, 900s by default) and self-resolving (any new
  describable file — a fresh photo, a poison file whose bytes change —
  lets normal quarantine accounting resume). `watch()` logs one diagnostic
  warning per occurrence via `CONSECUTIVE_ABORT_WARNING_THRESHOLD` if this
  persists, so it is visible rather than silent. See the "Known, deliberate
  limitation" note on `watcher._reconcile_poison` for the full reasoning,
  and `tests/test_poison.py::test_poison_at_the_head_does_not_halt_enrichment`
  for the test that exercises the boundary (it asserts quarantine only for
  poison files that DO have a good neighbour elsewhere in the queue).
