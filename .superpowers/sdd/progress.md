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
