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
