# Accuracy Module Regression Report

## Problem statement

Accuracy must be added without changing the fact output contract of the already implemented Fluency and Range paths. It must emit observations only; correctness, scores, CEFR, and LLM conclusions remain downstream responsibilities.

## Reproduction / RED evidence

1. Run `uv run python -m unittest tests.test_accuracy tests.test_evidence`.
2. Observe `ModuleNotFoundError: No module named 'jgrade_eval.accuracy'` from the new Accuracy contract test.
3. Observe the seven pre-existing Evidence/Range/Fluency compatibility tests still execute successfully in the same run.

## Timeline

- Existing Evidence v1 implementation committed as `eeecf7c`.
- Added Accuracy contract tests and Fluency compatibility snapshot before production code.
- Reproduced the intended missing-module RED state.

## 5 Whys

1. Why is the new test red? `AccuracyModule` does not exist.
2. Why does it not exist? Accuracy was deliberately designed but not implemented.
3. Why test before code? To fix the fact-only contract and prevent accidental scoring semantics.
4. Why include Fluency/Range tests? Shared Evidence changes can silently alter established fact output or cause duplicate tokenisation.
5. Why retain these tests? Every future capability extension must prove prior modules remain contract-compatible.

## Fishbone

- Code: Evidence v1 lacks an Accuracy collector and fact packet.
- Data: existing Wav2Vec2 handoff exposes mora timings but not compact posterior summaries.
- Integration: API currently has no selected-module route for Accuracy.
- Regression risk: changing Evidence schemas can affect Fluency adapter serialization and Range token consumption.

## Repair and prevention plan

- Implement an independent collector that emits only ASR timing observations, morphology observations, optional reference differences, and explicit unavailable capabilities.
- Keep ASR posterior confidence unavailable until a compact provider is implemented and calibrated.
- Add API module selection only after unit-level contracts are green.
- Run focused Accuracy/Evidence tests, then the full suite and output-contract regression checks before completion.

## Repair and verification

- Added `jgrade_eval/accuracy.py`. It emits mora-timing ASR observations, shared morphology observations, optional explicit-reference edit operations, and unavailable-capability names only.
- Added optional API `fact_modules`; the default remains exactly `fluency` plus `range`, so Accuracy is not run or sent to Judges unless selected.
- Added the Judge prompt boundary: `accuracy_data` is observation data; unavailable/uncalibrated ASR information and reference differences are not learner-error instructions.
- Verified focused tests (17), the complete suite (76), compile checks, and `git diff --check`.
- Compared the Accuracy implementation against the preceding common-Evidence checkpoint `eeecf7c`: `jgrade_eval/audio_pipeline.py`, `jgrade_eval/range.py`, and `jgrade_eval/evidence/` have no diff. Accuracy is additive and runs only when selected.
- Inspected the existing real-audio objective baseline at `outputs/jgrade_objective_data.json`; it records the same Wav2Vec2/Silero source pipeline. No new live model inference was needed to establish that Accuracy did not alter the pre-existing extraction code.
- Coverage tooling is not installed in the current `uv` environment (`No module named coverage`); this is recorded rather than treated as a passing coverage measurement.
