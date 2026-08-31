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

## Baseline (before Task 1)
892 passed, 2 failed. Both failures are PRE-EXISTING and unrelated:
  tests/test_takeout_survey.py::test_an_unstattable_loose_file_is_counted_not_raised
  tests/test_takeout_survey.py::test_a_zip_whose_stat_fails_does_not_abort_the_run
Verified identical at main (b732ee8) in a throwaway worktree, so they are not
caused by this branch. They are Windows-specific: a monkeypatched stat raising
PermissionError escapes a guard the tests say should catch it. Out of scope for
this feature (takeout survey area, another session's work) -- flagged here so
the final review can decide whether to report them to the owner.

## Owner decisions taken during execution
- 2026-08-31: main relicensed to AGPL-3.0-or-later (b732ee8) after the plan was
  written. Owner chose "clean-room, fall back to porting" for the YuNet decoder:
  attempt the hand-written decode as planned; if Task 9's real-photograph test
  fails, port PhotoPrism's engine_onnx_yunet.go (licence-compatible, both AGPL,
  attribution required in the file header) rather than adding OpenCV.
  Plan updated accordingly in the commit below.

## Tasks
