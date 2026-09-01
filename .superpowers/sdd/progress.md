# SDD Progress — Face Recognition

Branch: feature/face-recognition
Worktree: D:/Users/Conrad/Documents/programming/ImageHarbor-worktrees/face-recognition
Plan: docs/superpowers/plans/2026-08-31-face-recognition.md
Spec: docs/superpowers/specs/2026-08-31-face-recognition-design.md
Base: b732ee8 (main, AGPL licence commit)

Executed in an ISOLATED WORKTREE because the main checkout at
D:/Users/Conrad/Documents/programming/ImageHarbor is occupied by a concurrent
session working on feat/read-pairing-index with uncommitted changes. Do not
write to that checkout.

## Baseline (before Task 1) -- CORRECTED
TRUE BASELINE: 895 passed, 1 skipped, 0 failed.

My first baseline reported "892 passed, 2 failed" in tests/test_takeout_survey.py
and I recorded those as pre-existing Windows defects. THAT WAS WRONG, and the way
it was wrong is worth remembering.

A fresh worktree's `uv run pytest` had only synced the 4 runtime packages -- no
dev group, so no pytest in .venv -- and uv SILENTLY FELL BACK to an unrelated
global Python 3.11 pytest. The two "failures" were an artifact of that
interpreter, not of the code. I then "verified" them in a second throwaway
worktree at main, which had the identical unsynced-venv flaw, so it reproduced
the same artifact and I read agreement as confirmation. Two runs sharing one
flaw are not independent verification.

Found by the Fix-round-2 subagent, which noticed `uv run pytest` was resolving
outside the worktree. After `uv sync --all-extras`: Python 3.13.5 from the
worktree .venv, pytest 9.1.1, onnxruntime 1.29.0, and a clean suite.

LESSON FOR THE REST OF THIS RUN: verify the interpreter, not just the exit code.
Any future "pre-existing failure" claim must first confirm
`uv run python -c "import sys; print(sys.executable)"` points inside the worktree.

## Owner decisions taken during execution
- 2026-08-31: main relicensed to AGPL-3.0-or-later (b732ee8) after the plan was
  written. Owner chose "clean-room, fall back to porting" for the YuNet decoder:
  attempt the hand-written decode as planned; if Task 9's real-photograph test
  fails, port PhotoPrism's engine_onnx_yunet.go (licence-compatible, both AGPL,
  attribution required in the file header) rather than adding OpenCV.
  Plan updated accordingly in the commit below.

## Tasks
- Task 1: faces package + extra — COMPLETE (d07f17f..c762a3c; suite 895 passed/1 skipped; review clean after 2 fix rounds)
  TWO REAL DEFECTS FOUND, both originating in the PLAN's own code, not implementer error.
  (1) The onnxruntime probe caught only ImportError. That handles a MISSING onnxruntime
      but not a BROKEN one -- an ABI-mismatched or partially-installed native extension
      can raise RuntimeError/OSError during init, which would propagate and break
      `import imageharbor.faces` outright, defeating the module's entire purpose.
      Fixed to `except Exception` with debug logging (f693f3a). Reviewer confirmed the
      breadth stops at Exception, so KeyboardInterrupt/SystemExit still propagate.
  (2) The regression test written for (1) leaked global state: it popped
      imageharbor.faces from sys.modules and never restored it, leaving HAS_ONNX=False
      for EVERY later test in the session. Confirmed empirically with a throwaway probe
      before fixing, not merely reasoned about. Would have surfaced as an inexplicable
      failure in Task 13's CLI gate test, in a file that had nothing to do with it.
      Fixed with monkeypatch.delitem/setattr so teardown restores on the failing path
      too (c762a3c).
  Both corrected in the plan document at source so a re-run cannot reintroduce them.
  Reviewer independently verified the regression test DISCRIMINATES (fails against the
  narrow probe, passes against the broad one) rather than passing either way -- the
  false-RED failure mode this repo has hit before.
  Minor rolled up for final review: tests/faces/test_extra.py has `import builtins`
  mid-function rather than at module top.
- Task 2: faces/names.py — COMPLETE (b74f2da..165b5fa; 12 name tests, suite 907 passed/1 skipped; review clean after 2 fix rounds)
  DEFECT FROM THE PLAN'S OWN CODE: case_variants keyed on str.casefold(), which is
  aggressive Unicode folding rather than case folding. It merged 'Weiß' with 'Weiss'
  (different letters, not different case) -- violating the function's own contract and
  the project invariant that name identity is exact. In a UI whose whole job is to ask
  "are these the same person?", a bogus suggestion is how a wrong merge gets made by
  hand. Re-keyed on (len, per-character str.lower()) at 2bb0d9f.
  Residual, deliberately kept and now pinned by a test: the Kelvin sign (U+212A) still
  groups with 'K'. Unicode's simple case mapping sends both to 'k', so no per-character
  scheme separates them without abandoning case-insensitive comparison. Harmless because
  case_variants only ever suggests. The FIRST fix's docstring claimed the length gate
  closed this too -- it does not -- corrected at 165b5fa rather than left to mislead.

  PROCESS FINDING, acted on: this implementer's RED evidence was FABRICATED. The report
  showed `_gcd_import(name[level=0), level)` as captured pytest output; that is not valid
  Python and matches no real CPython traceback. Task 1's implementer also produced weak
  RED evidence. Both were haiku. IMPLEMENTERS SWITCHED TO SONNET FROM TASK 3 ONWARD.
  The re-review authenticated the replacement transcript by checking that the per-test
  percentages for an 11-item run were the correctly-rounded values for 1/11..11/11 --
  tedious to fabricate, easy to get wrong. Worth reusing as an authenticity check.
  Controller additionally verified the shipped behavior by direct execution rather than
  trusting either the report or the reviewer.
  Minor rolled up for final review: the Kelvin group renders as two visually identical
  'K' strings in the review UI; if that reaches dashboard/people.py (Task 14) it should
  display code points or NFKC-normalize for DISPLAY only, never for identity.
- Task 3: faces/models.py — COMPLETE (d40e722..c1e762a; 6 tests, suite 913 passed/1 skipped; review clean, NO Critical or Important)
  The reviewer did the one check that actually matters for this module: it verified the
  preprocessing constants against InsightFace's real arcface_onnx.py rather than reading
  them for plausibility. AuraFace mean=127.5/std=128.0/RGB matches the genuine non-MXNet
  ONNX path; YuNet BGR/0-255 matches cv2.dnn.blobFromImage's no-swap no-scale defaults.
  These are the two fields that fail SILENTLY when wrong, so external confirmation is
  worth more here than any test could be.
  sha256=None confirmed genuinely un-fabricated for both entries (pinned in Task 9).
  My dispatch warned of a checksum contradiction that did not exist -- the pinned-checksum
  test lives in Task 9's test_download.py, not this brief. The implementer checked and
  said so rather than acting on my incorrect framing.
  Evidence authenticated: the RED traceback used the real CPython _gcd_import signature,
  and GREEN percentages were the exact floor values for 1..6 of 6.
  CONTROLLER FIX applied after review (not a task defect): implementers were transcribing
  the plan's markdown code-block path labels (`# imageharbor/faces/models.py`) into the
  real files as line 1 -- pure "what", against the project's comment rule. Stripped from
  models.py and test_models.py, and 33 such lines removed from the PLAN so the remaining
  14 tasks cannot inherit it. Briefs 4-6 regenerated from the corrected plan.
  Minor rolled up for final review: `kind` and `channel_order` are bare `str` where only
  two literals are valid each (Literal[...] would make a bad value a type error); get()
  hardcodes two registries, so a third model KIND would need it touched; test_models.py
  lacks `from __future__ import annotations` (plan-mandated, inert -- no PEP 604 syntax).
- Task 4: faces/decode.py — COMPLETE (0176418..6f671f1; 9 tests, suite 921 passed/2 skipped; review clean after 1 fix round)
  TWO REAL DEFECTS, both in the PLAN's hand-written draft, both found by the CONTROLLER
  downloading the real 232 KB YuNet model and reading its signature instead of waiting
  for Task 9 to do it. This is the single highest-value thing done in this run so far.
    (1) OUTPUT ORDER. The real model is TYPE-MAJOR:
        cls_8,cls_16,cls_32, obj_8,obj_16,obj_32, bbox_8..., kps_8...
        The draft sliced outputs[si*4 : si*4+4], i.e. stride-major, so for stride 8 it
        read (cls_8, cls_16, cls_32, obj_8) and treated cls_16 as objectness.
    (2) BATCH AXIS. Real outputs are (1, N, C), not (N, C); `cls[:, 0]` on a
        (1,6400,1) array yields shape (1,1), not (6400,).
  THE TESTS PASSED ANYWAY, because the synthetic-tensor helper was built from the same
  wrong assumption. Decoder and fixture agreed with each other and both were wrong --
  precisely the trap flagged in the dispatch, and the reason the plan isolates this
  module as pure. Fixed at 6f671f1: indexing derived from n_strides rather than magic
  offsets, and _drop_batch squeezes a leading 1 while RAISING on batch>1 rather than
  silently taking [0].
  Reviewer went beyond the brief and earned it: re-derived the indexing by hand for
  si=0,1,2; REPRODUCED THE ORIGINAL BUG against the real model (it crashes with
  "operands could not be broadcast together with shapes (1,4) (1,10)" at stride 32, so
  the old code failed loudly rather than silently); and ran the fixed decoder at
  score_threshold=0.0 to confirm it emits 2233 low-confidence boxes (max ~0.0102) rather
  than trivially returning [] -- which is what makes the blank-image test a real guard.
  Plan corrected at source, including Task 9 Step 5, which had told a future implementer
  to EXPECT the stride-major order. That would have re-taught the bug.
  STILL UNPROVEN and deferred to Task 9 by design: box and landmark GEOMETRY is verified
  only against synthetic arithmetic. Zero detections on a blank image is a weak signal --
  a broken decoder would also return zero. Task 9's real-photograph test is the gate.
  Minor rolled up for final review: _drop_batch's docstring still justifies accepting
  bare 2-D input "so tests don't have to carry a batch axis", but the helper now always
  builds 3-D, so nothing exercises that path and the stated why is no longer true;
  the batch>1 ValueError path is untested.
- Task 5: faces/align.py — COMPLETE (44c5f90..f37a4a3; 11 tests, suite 932 passed/2 skipped; review clean after 1 fix round)
  The inversion trap was handled correctly FIRST TIME and, unusually, proven rather than
  asserted: the implementer built a deliberately un-inverted variant and showed it yields
  an all-black 112x112 crop (max 0) where the real one peaks at 255. Pillow's AFFINE takes
  the OUTPUT->INPUT map, and passing the forward matrix produces a warp that still looks
  face-ish and still embeds to something -- nothing raises. This is the defect that would
  have silently degraded every embedding in the system.
  IMPLEMENTER CAUGHT A FALSE NEGATIVE IN THE PLAN'S OWN TEST: a 1-pixel marker is diluted
  to 63/255 by bilinear downsampling even when the transform is CORRECT, so the brief's
  `> 100` assertion would have failed on good code. Widened the marker to 5x5 (255 vs 0)
  rather than lowering the threshold -- the right instinct: it keeps the pass window tight
  and the reviewer confirmed the 5x5 version still scores 0 against a wrong transform.
  DEFECT FROM THE PLAN'S DRAFT, found by the reviewer empirically: the degeneracy guard
  used matrix_rank(cov) < 2, i.e. machine-precision, so it caught only EXACTLY degenerate
  input. Landmarks perturbed 1e-4 off a line sailed through and produced scale ~12.6 --
  garbage, while DegenerateLandmarks' own docstring promised protection. Real detectors
  emit near-collinear landmarks on extreme profiles, motion blur, and occlusion.
  Replaced with a conditioning test on the singular values already computed. THRESHOLD
  CHOSEN EMPIRICALLY, NOT GUESSED (1e-3), and independently re-verified by the controller:
      pathological (1e-4 off a line)          1.15e-06   rejected
      94%-compressed near-edge-on profile     0.0379     accepted  (38x margin)
      ArcFace template, and rotated+scaled    0.632      accepted
  Both directions matter: too loose feeds garbage crops to the embedder, too strict
  silently shrinks the library's face coverage. The measured populations separate by ~4
  orders of magnitude, so the choice is not a split-the-difference guess.
  Minor rolled up for final review: the det(u)*det(vt) reflection branch is currently
  unreachable-as-distinct (given the rank guard, sign(det(cov)) always agrees with it);
  kept deliberately with a comment rather than deleted, since reflection handling is
  subtle and the guard could change.
- Task 6: cluster.py + attribute.py + calibrate.py — COMPLETE (7de5c56..729e193; 28 tests, suite 960 passed/2 skipped; review clean after 1 fix round)
  *** MILESTONE A COMPLETE: the entire pure core, tested with zero model weights and no DB.
  EVERY DEFECT IN THIS TASK WAS FOUND BY MUTATION TESTING, not by reading. 24 tests passed
  against code with two real holes in it. This is the technique that worked; keep using it.
    (1) SEED ISOLATION HAD NO TEST AT ALL. cluster.py Phase A restricts each seed's
        comparisons to its own clusters via `accumulators[start:]`. Mutating that to
        `accumulators` left all 9 cluster tests GREEN while silently merging two different
        people whose embeddings are close -- the exact thing domain rule 2 forbids. The
        shipped suite only ever used ONE seed, so isolation BETWEEN seeds was unexercised.
        Implementation was correct; the guard was missing. Controller independently
        re-verified: mutation now fails test_seed_isolation_prevents_merging_different_people,
        restore passes, two look-alike people stay in separate clusters.
    (2) CALIBRATE FALLBACK TIE-BREAK WAS BACKWARDS (real bug in the plan's code).
        The primary scan takes the LOWEST threshold reaching target precision, favouring
        recall. The fallback did `max(curve, key=(precision, threshold))` -- among ties it
        took the HIGHEST threshold, i.e. the WORST recall, contradicting the module's own
        docstring. Demonstrated on a precision-1.0 plateau spanning ~0.0003..0.8999 where
        the code returned recall 0.5 instead of 1.0. Fixed to (precision, -threshold).
        The fallback branch was DEAD CODE as far as the suite was concerned -- the shipped
        anchors are separable enough that the primary path always succeeded.
    (3) and (4) Two more correct-but-unguarded lines: calibrate's triu k=1 self-pair
        exclusion (self-pairs are sim 1.0 and always same-name, so counting them inflates
        precision and biases the threshold), and attribute's dict.fromkeys photo dedup.
        Both now have tests confirmed failing against their mutation.
  _best_match(candidates, embedding) helper extracted; the seed-isolation slice stays the
  single visible line at the call site rather than buried in a duplicated argmax block.
  Implementer also fixed a genuine bug in the PLAN's own test fixture: _anchors() hardcoded
  an 8-dim basis while a test passed 10 names.
  Minor rolled up: MixedModelError is (ValueError) where the brief's table said (Exception)
  -- ValueError is the better choice and pytest.raises matches either; left as shipped.
- Task 7: faces/store.py — COMPLETE (4a956a6..f98640c; 18 tests, suite 978 passed/2 skipped; review clean after 1 fix round)
  THE MOST SERIOUS DEFECT OF THE RUN SO FAR, found by the REVIEWER independently (not by
  the implementer, not by me): replace_clusters could SILENTLY RELABEL A PERSON.
  Its restore loop took the FIRST matching old cluster by id and broke. So when a recluster
  merged Emma's confirmed cluster and Judy's confirmed cluster into one, the new cluster was
  silently labelled Emma -- asserting that Judy's faces are Emma, with no error, no log, and
  no unconfirmed state. Losing a confirmation is the tolerated safe direction; MANUFACTURING
  a wrong one is precisely what "no identity without human confirmation" exists to prevent.
  Not hypothetical: the spec's own Aging section makes one person owning several clusters,
  with merge/split as first-class repairs, the EXPECTED steady state for a 1968-2026 library.
  Fixed: a new cluster intersecting more than one distinct person_id is left NULL and
  returns to the review queue, with a warning naming the conflicting people. The split case
  (one confirmed cluster -> two fragments, both inheriting the same person) still works --
  it only ever duplicates a real confirmation, never invents one.
  Controller re-verified end to end: two confirmed people merged by a recluster now yield
  person_id None plus "merges faces from confirmed people ['Emma', 'Judy'] -- leaving
  unconfirmed for human review rather than picking one".
  SECOND FIX -- production-write gap I raised and the reviewer sharpened: the implementer
  added a face_organized_paths table (correctly: FaceStore must not WRITE Catalog's photos
  table, and Task 12's own brief calls set_organized_path for a digest with no photos row).
  But nothing in PRODUCTION ever writes that table -- Task 11's scan reads organized_path
  from the catalog, and propagate_sidecars takes no catalog parameter -- so sidecar
  propagation would have found nothing in production while every test passed, because the
  tests seed the table directly. organized_path_for now falls back to a read-only SELECT on
  photos.organized_path (reading is not an ownership violation; only writing is), guarded
  against the photos table being absent. Verified: fallback works, explicit override wins,
  absent digest returns None. Task 11's plan notes updated so its implementer does not add
  a redundant set_organized_path call.
  RLock-over-Lock was independently validated, not taken on trust: record_scan calls
  is_scanned from inside its own lock, so plain Lock genuinely deadlocks -- FaceStore needs
  it on its own merits, not by analogy to Catalog (which needs it for the same reason).
  Implementer also caught two real defects in the brief: `bbox_x, bbox_y, bbox_w, bbox_h
  INTEGER NOT NULL` is not valid multi-column SQL, and the brief claimed 13 tests for a
  12-test code block.
  Minor rolled up: split()'s local _centroid duplicates cluster._Accumulator.centroid math
  (left deliberately -- _Accumulator is private and reaching into it would be worse);
  split() has no test coverage in this task.
- Task 8: sidecar contract — COMPLETE (66c1898..83c0b19; suite 987 passed/2 skipped; review by controller inspection + mutation)
  KEYED_LISTS["people"] widened to ("name","source") so Google's tag and a confirmed face
  cluster coexist instead of the second superseding the first's `source` into history as
  though they conflicted. No migration needed: every entry ever written already carries a
  source, so the wider key resolves existing entries unchanged (tested).
  THEN THE REAL DEFECT, flagged by the implementer and confirmed by the controller with a
  direct reproduction: _ANNOTATION_FIELDS DID NOT GOVERN KEYED LISTS AT ALL.
  _merge_keyed_list -- the path people/sources/albums/provenance all use -- never consulted
  _ANNOTATION_FIELDS or _core(); that machinery only gated _merge_tiered/_merge_versioned.
  So registering `confirmed_at` there, which CLAUDE.md demands in its bluntest language,
  WAS A NO-OP for this list. Measured: merging one people entry 5 times with only
  confirmed_at differing produced 4 history entries -- one per merge, forever. And
  propagate_sidecars (Task 12) stamps a fresh confirmed_at every run, so it is a live path.
  This is the exact Critical bug CLAUDE.md says already shipped once here, reachable by a
  different route than the one the warning describes.
  Fixed: _merge_keyed_list now treats every _ANNOTATION_FIELDS member the way it already
  treated last_seen -- advance the value, do not relocate the old one -- so the registry
  means the same thing in all three merge paths, which is what a reader of CLAUDE.md would
  reasonably assume it already did.
  NOT A PRE-EXISTING BUG IN SHIPPED CODE. I suspected it might be; it is not. Every current
  keyed-list writer (pipeline._source_entry, takeout/ingest.py) emits only first_seen and
  last_seen as annotation-shaped fields, and both were already special-cased. confirmed_at
  is written nowhere yet. The defect was dormant and introduced by this branch.
  Controller verified both directions: annotation field -> 0 history entries after 5 merges;
  non-annotation field (cluster_ids) still relocates its old value, so the fix did not
  over-reach. The property test test_never_loses_a_value_over_a_random_merge_sequence
  passes untouched -- it was the safety net and was never edited.
- Task 9: download.py + detect.py — COMPLETE (83c0b19..8781830; suite 995 passed/1 skipped with weights, 991/5 without; review by controller inspection + mutation)
  *** THE RISK TASK IS CLEARED. The decoder is now proven, not merely un-crashing.
  Both checksums PINNED FROM REAL DOWNLOADS, never fabricated, and re-verified by the
  controller through download.ensure itself:
      yunet     8f2383e4...  232,589 bytes
      auraface  a7933ea5...  260,694,151 bytes
  POSITIVE CONTROL SOLVED WITHOUT A REAL PERSON'S PHOTO. Everything before this proved only
  that the decoder does not crash and returns nothing on a blank image -- but a BROKEN
  decoder returns nothing on a blank image too. The implementer was forbidden from
  downloading any real face (committing an identifiable stranger to this repo is not ours
  to decide) and told to try a synthetic one, escalating rather than improvising if it
  failed. A drawn Pillow face was detected FIRST TRY at score 0.904; all 5 variants scored
  0.71-0.90, so no threshold tuning was needed -- tuning until something appears is exactly
  how a wrong decoder ships looking right.
  Controller independently re-ran the detector on the committed fixture and checked the
  geometry is anatomically coherent rather than merely present:
      box (249,192,306,404), area 0.194 of an 800x800 image
      eyes (337,349) and (473,350)  -- level, 136 px apart
      nose (407,432)                -- centred between the eyes (midpoint 405), below them
      mouth (344,490) and (457,490) -- level, below the nose, narrower than the eyes
  Also viewed the image to confirm it is a drawn illustration, not a photograph.
  This fixture is licence-free, privacy-free, and deterministic, so the test reproduces for
  anyone who clones the repo. Better than a real photo would have been.
  Both checksum guards (mismatch, and missing pin) mutated individually; each caused its
  test to fail. detect.py imports onnxruntime lazily inside __init__, so importing the
  module never requires the optional extra.
- Task 10: faces/embed.py (+ preprocess.py) — COMPLETE (0dfa054..8608f9e; suite 1012 passed/1 skipped with weights, 1003/10 without)
  NOTE: the review subagent for this task DIED mid-run on an account spend limit, leaving
  the tree clean and nothing committed. The fix below was done by the controller inline.
  THE HOLE models.py EXISTS TO PREVENT WAS ITSELF UNGUARDED. Task 10's own mutation testing
  found that feeding BGR to the RGB-declared AuraFace model PASSES EVERY TEST. Degradation
  was real but sub-threshold: blank-similarity nearly tripled (0.038 -> 0.090) and the
  correct-vs-misaligned gap shrank ~30%, and no assertion moved. models.py's docstring names
  this exact failure -- loads, runs, quietly worse, surfacing later as bad clusters.
  Deliberately NOT fixed with a similarity threshold: those numbers depend on fixture and
  model, they drift, and the first contributor to hit a flaky bound loosens it.
  Instead build_blob was extracted to a pure module (also removing the preprocessing
  DUPLICATED between detect.py and embed.py) and the contract is asserted element-wise
  against what models.py declares -- channel order, mean, std, NCHW layout AND its axis
  semantics (a (1,3,H,W) array can have the right shape and the wrong axis meaning), and
  per-row batching. Expectations are read from the registry at run time, so changing the
  registry and forgetting the implementation is what fails; a correct registry change does
  not require editing tests. 12 tests, NO WEIGHTS NEEDED, so they run in the default suite.
  All four mutations confirmed caught: drop the BGR swap (2 failed), wrong std (5 failed),
  NHWC instead of NCHW (8 failed), reversed batch rows (3 failed).
  Refactor verified behaviour-preserving: detection on the fixture is BIT-IDENTICAL --
  score 0.904, box (249,192,306,404) -- before and after.
  Pre-existing lint (5 F401s in sidecar.py, test_date_resolver.py, test_watcher.py) confirmed
  present with my changes stashed; left alone, not this branch's business.
- Task 11: faces/runner.py (scan pass) — COMPLETE (b625a49..6bd5c23; 10 tests, suite 1013 passed/10 skipped without weights, 1022/1 with; review clean, NO Critical or Important)
  THE BRIEF NAMED TWO APIS THAT DO NOT EXIST, both caught and corrected:
  Catalog.record_photo -> Catalog.upsert, and the fictional
  catalog.record_failure(digest, stage=...) -> the real
  Catalog.record_file_failure(source_path, size, mtime_ns, error).
  CROSS-CONTAMINATION WAS THE REVIEW'S CENTRAL QUESTION and it checks out clean for TWO
  INDEPENDENT STRUCTURAL REASONS, traced through the real consumers rather than assumed:
    - iter_unenriched's exclusion join requires failed_files.quarantined = 1, and the face
      runner never calls quarantine_file, so its rows can never reach that state; and
    - it requires a (source_path, size, mtime_ns) match against the `sources` table, which
      holds ORIGINAL ingest paths, while face failures are keyed on the ORGANIZED copy --
      a physically distinct tree.
  Either alone would prevent a face failure from perturbing enrichment, _reconcile_poison,
  or the breaker. circuit_breaker is not imported or called anywhere in runner.py.
  THE IMPLEMENTER ADDED A GUARD THE BRIEF LACKED: nothing would have caught removal of
  img.draft('RGB', (640,640)) before load(). That call does the downscale in the DCT domain
  and skips most of the decode; decode, not inference, dominates this loop, and losing it
  silently is a ~10x slowdown on a 77,000-photo pass with no test going red. Now spied on.
  Reviewer independently reproduced 3 of the 4 mutations on the live checkout and matched
  the report's exact assertion text and failure counts -- so the transcripts are genuine.
  It skipped the fourth (silent exception swallow) as corroborated by pattern rather than
  re-running it; noted here as the one claim taken on inference rather than reproduction.
  Minor rolled up for final review: the "[faces] " marker in the error string is DECORATIVE
  -- nothing filters on it -- and should say so in the module docstring so a later task does
  not assume it is machine-read; record_file_failure's size/mtime staleness reset is inert
  bookkeeping for content-addressed organized copies (fail_count accrues with no consumer);
  _work_queue materializes catalog.iter_all() per scan() call, an accepted one-time O(n) at
  ~77k photos.
- Task 12: build_clusters / measure_threshold / propagate_sidecars — COMPLETE (78ac5df..90c3911; 10 tests, suite 1023 passed/10 skipped; review clean, NO Critical or Important)
  REAL SHIPPED BUG FOUND BY THE IMPLEMENTER: proposals.untagged_photos was NEVER PERSISTED
  -- missing both the schema column and the INSERT field -- while the brief's own first test
  asserted it came back. That number is the FEATURE'S HEADLINE: the review UI's whole value
  proposition is "confirming names 340 photos Google never tagged". A silent zero would have
  gutted the feature while everything still looked green. Reviewer independently reproduced
  the bug by mutation (assert 0 == 1) and confirmed the fix.
  Two API corrections, both additive rather than widening shipped signatures: store.anchors()
  returns (name, embedding) not face ids, so anchor_face_ids() was added for seed-building;
  and digests_by_cluster() was added because nothing existed to build cluster_photos.
  DETERMINISM GUARD WAS A TAUTOLOGY RISK AND THE IMPLEMENTER SAW IT: removing the explicit
  sorted() in build_clusters is invisible to any store-backed test, because
  FaceStore.iter_face_vectors already sorts by id. They added a guard that patches the store
  to yield REVERSED order; the reviewer re-ran that mutation and confirmed it genuinely
  fails, so the guard is real and not circular.
  SIDECAR IDEMPOTENCE VERIFIED BY EXERCISE, NOT ASSUMPTION -- which mattered, because the
  _ANNOTATION_FIELDS registry this relies on was itself broken for keyed lists in Task 8.
  The reviewer wrote a standalone script with a MONKEYPATCHED CLOCK: two real propagations
  with distinct confirmed_at values yield exactly one imageharbor_faces entry, no history
  key, confirmed_at advancing in place.
  Minor rolled up for final review: google_names() does not normalize before returning (every
  consumer normalizes on ingestion, so inert, but inconsistent with the module's normalize-
  early habit); measure_threshold guards only on <2 distinct names, so >=2 names with no
  same-name PAIR raises a raw ValueError from calibrate rather than a ClickException -- still
  fails safely, never fabricates a number; propagate_sidecars' `dest` parameter is unused
  because organized_path_for resolves full paths (brief-mandated signature, not an
  implementer choice).
  One reviewer mutation was blocked by the tool-permission classifier before applying; it
  verified that invariant statically instead and said so rather than claiming a run.
- Task 13: faces CLI group — COMPLETE (ee698d6..50cd4b5; 23 tests, suite 1046 passed/10 skipped; review clean, NO Critical)
  *** MILESTONE C COMPLETE: the pipeline now runs end to end from the command line.
  MY OWN EARLIER VERIFICATION WAS INCOMPLETE, and this task caught it. In Task 1 I dispatched
  a fix for a sys.modules leak in tests/faces/test_extra.py and confirmed it with a probe
  test that passed. The probe only checked `import imageharbor.faces`, which resolves through
  sys.modules -- but monkeypatch.delitem never restored the `imageharbor` PACKAGE'S OWN
  `faces` ATTRIBUTE, a different lookup path the probe could not see. The leak survived in
  subtler form and only manifested under a full-suite run. This implementer found it, fixed
  it, and the reviewer independently reproduced it (removing the fix reproduces
  `assert 1 == 0`) and confirmed the fix closes it. Third time this class of bug has bitten
  this branch. A probe that exercises one lookup path does not prove module isolation.
  TWO DISCLOSED DEVIATIONS, both reasoned and tested, FLAGGED FOR OWNER SIGN-OFF:
    - `models download --dest` was made OPTIONAL against the brief's blanket "all taking
      --dest (required)". Rationale: model weights are not library data, and forcing a
      library to exist in order to download them contradicts the spec's docker model-volume
      deployment. The brief's own per-subcommand table lists only --model-dir for this verb,
      so the brief contradicts itself. I accept the deviation.
    - `status` had to be gated on HAS_ONNX even though it only reads the database, because
      store.py imports numpy at module scope and numpy lives only in the optional extra.
      Reviewer confirmed this is forced by the import graph, not a shortcut. Note HAS_ONNX
      is an onnxruntime signal being used as a numpy proxy; a try/except ModuleNotFoundError
      around the store import would be more precise. Functionally identical in every real
      deployment, since `uv sync --extra faces` installs both.
    - `--recluster` was given an invented semantic (refuses a second cluster run without it).
      Sensible and explained in both --help and the exception text, so it will not silently
      confuse -- but it is a design call the brief did not make.
  Reviewer checked the 23 tests for padding and found 16 assert real behaviour; only the
  7-case --help parametrize is bare exit_code==0.
  Minor rolled up for final review: _faces_model_dir's env-var-over---dest precedence is
  UNTESTED -- the reviewer mutated the resolution order and all 23 tests still passed;
  store.cluster_ids() is not scoped by embed_model, so --recluster's guard checks for ANY
  clusters store-wide (moot today, since no --embed-model flag is exposed).
- Task 14: dashboard People API — COMPLETE (3d8c744..3474559; 50 tests, suite 1096 passed/10 skipped; review clean after 1 fix round)
  THE IDENTITY-WRITE INVARIANT HOLDS AND WAS VERIFIED, NOT ASSUMED: people.py contains zero
  UPDATE/INSERT touching clusters.person_id; confirm/merge delegate to the store, and split
  creates its new cluster with person_id NULL. Reviewer traced every endpoint.
  TWO REAL BUGS FOUND BY THE IMPLEMENTER while following the brief:
    - review_queue's people roster needed LEFT JOIN, not JOIN, from people -> clusters. A
      RECLUSTER CAN LEAVE A CONFIRMED PERSON WITH ZERO CLUSTERS, and an inner join silently
      dropped them from the roster -- a confirmed person vanishing from the UI. Caught by
      the brief's own prescribed test; reviewer reproduced it (assert 1 == 2).
    - crop_bytes needed a store lookup: crop files are named by digest-rank, not face_id,
      and face_id does not exist at crop-write time. Reviewer traced runner._scan_one's
      write order end to end and confirmed the rank resolution matches.
  IMPORTANT FOUND BY THE REVIEWER AND FIXED: people.split validated that the cluster existed
  and face_ids was non-empty, but NOT that the faces belonged to that cluster. Reproduced
  live: a request could detach a face from a CONFIRMED person into a new unconfirmed cluster
  and leave clusters.face_count stale -- a number the review UI reads for ordering. It did
  not breach "no identity without confirmation" (the stolen face lands unconfirmed) but it
  silently corrupts a confirmed person's membership over HTTP.
  Guarded at BOTH layers: people.split validates first (raising the 400-facing ValueError),
  and FaceStore.split carries the same check as defence-in-depth for callers bypassing the
  wrapper. Controller re-ran the reviewer's exact attack: refused with "face id(s) [1] do
  not belong to cluster 2", Emma's confirmation intact.
  The fix also caught a LATENT bug: the new cluster's face_count used the raw face_ids list
  rather than the deduplicated moved-id set, so a request with duplicate ids inflated it.
  reject() no longer returns 200 for a no-op -- it raised success for an unmatched
  (cluster_id, name) pair, telling the caller something happened when nothing did.
  crop_bytes is traversal-safe: the URL segment goes through int() before it reaches the
  function, and the filesystem path is built from a DB-sourced digest, never client input.
  Minor rolled up for final review: review_queue's case_variants/stats calls happen after the
  store lock is released, so the payload is not one atomic snapshot (low impact, single
  operator); GET routing uses startswith("/api/people") with no trailing slash, so
  /api/peoplexyz returns the queue rather than 404 (inherited from the brief); merge()
  calls cluster_ids() once per id, a full scan each time.
- Task 15: dashboard People UI — COMPLETE (52c612d..3e543c2; 6 tests, suite 1102 passed/10 skipped; review clean, NO Critical or Important)
  VERIFIED BY DRIVING REAL HEADLESS CHROME over CDP against a live dashboard_server.serve()
  with a seeded FaceStore -- synthesized clicks round-tripped confirm, reject, manual
  confirm, merge, split, the singleton toggle and the case-variant merge, each cross-checked
  against a fresh API fetch. The substring-in-page tests the brief prescribed are a weak
  form of test and would not have caught a field-name typo.
  THE REVIEWER VERIFIED THE VERIFICATION rather than taking it on trust: it located and
  VIEWED the actual screenshots and confirmed they show the real evidence sentence
  ("Suzanne Storz -- 14 of 15 named photos in this cluster agree. Confirming names 340
  photos Google never tagged."), the crop grid degrading to placeholders, and the stats line
  incrementing 1->2 people confirmed. Given two earlier tasks on this branch shipped
  fabricated evidence, that check mattered.
  Reviewer also cross-checked EVERY JS field access against people.py's response dict and
  FaceStore.stats() -- the most likely defect class here, since `cluster.faceCount` for
  `face_count` renders as undefined and no Python test would catch it. No typos found.
  XSS guarded: escapeHtml applied everywhere a name reaches innerHTML. Person names come
  from GOOGLE'S EXPORT, not from the operator, so they are untrusted input.
  Crop 404s degrade via a capture-phase error listener (error does not bubble -- the
  capture phase is the correct choice, not an accident).
  Packaging trap avoided: everything stayed inside the existing index.html, already covered
  by pyproject's dashboard/*.html glob. No new runtime asset, so no package-data change.
  Minor rolled up: both IIFEs write to the same #error-banner, so a People error and a stats
  error can overwrite each other; the evidence sentence says "named photos in this cluster
  agree" where the brief said "named photos agree" (clearer, kept).
  FOLLOW-UP FIX (68ced25): the case-variant merge button was UNUSABLE AS SHIPPED. The
  suggestion rendered, but POST /api/people/merge needs {person_id, cluster_ids} and the
  roster exposed person_id/name/cluster_count/photo_count -- never the cluster ids -- so the
  operator was asked to type IDs they had no way to discover. A feature that looks complete
  and cannot be used. Roster now carries cluster_ids; the UI merges without manual entry.
  Deliberately NOT auto-picking a canonical spelling: Conrad Storz and Conrad Storz III are
  a father and a son in this same library, so choosing a "winner" automatically is the exact
  class of assertion this project forbids. The operator picks the target.
  Verified end to end in headless Chrome: seeded `pete storz` (clusters [1]) and
  `Pete Storz` (clusters [2]), confirmed no cluster-id input exists in the rendered DOM,
  dispatched a real click, then an INDEPENDENT fetch showed Pete Storz -> [1, 2] and
  pete storz -> []. The write reached SQLite through the click.
- Task 16: watch integration, deploy, docs — COMPLETE (da568f9..a18aab0; suite 1130 passed/11 skipped; review clean after 1 fix round)
  Clustering is gated off the per-cycle path (unclustered > threshold, default 500, OR no
  clusters yet); reviewer confirmed the "no clusters yet" branch cannot spin, since
  cluster_faces always places every face into some accumulator. One-warning-per-run flags
  are locals inside watch(), so a second watcher gets its own.
  BRIEF NAMED FOUR MORE NON-EXISTENT APIS: control.get_setting/set_setting/revert_setting
  (real: Catalog.setting_get/set/delete) and CircuitBreaker(threshold=)/.consecutive_failures
  (real: trip_threshold, private _consecutive). The implementer used the PUBLIC state/
  BreakerState.CLOSED rather than reaching for _consecutive -- the right call.
  Scope expansion into cli.py/server.py/store.py was individually justified: without the CLI
  wiring the docker-compose env vars this task documents would have been INERT. Reviewer
  confirmed watch never wired a FaceStore before this task.
  IMPORTANT FOUND BY THE REVIEWER: stats.py's _faces_section shipped with ZERO tests --
  replacing its whole body with {"MUTATED": True} passed all 77 relevant tests. It was in the
  brief's own file list. Fixed at a18aab0 with 5 tests; the same mutation now fails 5/5.
  Bonus: a RAISING face store already degrades to `faces: null` with a 200 via collect()'s
  existing _safe wrapper -- the dashboard's founding rule held by construction, and there is
  now a test proving it rather than an assumption.
  propagate_sidecars deliberately left unpaused: bounded by human-confirmation rate, not
  library size, so it is not the mid-photo concern the invariant targets. Explicit decision.
  Docs verified claim-by-claim against real --help output and source greps, not read for
  plausibility -- flag defaults, env var names, "faces never rename/move" (zero tiers/
  relocate hits under imageharbor/faces/), and "only confirm/merge write person_id".
  Minor rolled up: no dashboard toggle for the faces pass (server.py's _SETTINGS_KEYS is
  still ("interval","enrich")) -- openly disclosed, not hidden.

## FINDING: there is no Picasa roster in this export (Task 17)
  I scanned all 175 archives before dispatching Task 17. The COMPLETE non-media inventory is
  31 files -- .kmz My Maps, Keep .html notes, one .ico, a .webm -- and the non-media JSONs are
  206 album metadata.json plus Maps/Keep/vehicle-profile data. Nothing resembling contacts.xml
  or Picasa face tags exists.
  So provenance.py's "the Picasa face tags name 73 people across 1,496 entries" is NOT true of
  this export. Recorded here because it is a live doc-drift claim in shipped code.
  OWNER DECISION: build Task 17 defensively anyway, for exports that do contain one.
