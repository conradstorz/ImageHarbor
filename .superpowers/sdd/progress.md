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
