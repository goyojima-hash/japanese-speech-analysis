# Common Evidence Layer Integration Report

## Problem

The test-first evidence-layer test initially failed because `jgrade_eval.evidence` did not yet exist. This is the expected red phase of the implementation, but it establishes the integration contract to preserve during the repair.

## Reproduction

1. Run `uv run python -m unittest tests.test_evidence`.
2. Observe `ModuleNotFoundError: No module named 'jgrade_eval.evidence'`.

## Timeline

- Added requirements, design, task plan, and ADR.
- Added focused evidence tests.
- Reproduced the missing-package failure.
- Added the evidence package; focused tests will be rerun after Range and API integration.
- Reproduced a fixture-contract mismatch: the strict pipeline rejected a legacy dictionary supplied directly by test fakes.
- Reproduced an API integration import failure because the public evidence package did not export its compatibility adapter.
- Added a public export for the adapter, added model-provenance cache-key coverage, and verified the full suite: 68 tests passed.

## 5 Whys

1. Why did the test fail? The evidence package could not be imported.
2. Why could it not be imported? The test was deliberately written before the package.
3. Why write the test first? To define a stable, fact-only contract before implementation.
4. Why is the contract important? Five selectable modules must share source facts without sharing evaluation logic.
5. Why record this as an incident? Future refactors must keep the focused contract and not silently reintroduce per-module extraction or scoring.

## Fishbone

- Code: package and integration points are new.
- Dependencies: Range currently owns Sudachi tokenisation.
- Data: current Fluency output discards some facts needed by evidence.
- Process: existing API tests use legacy fakes without evidence methods.
- Packaging: the public `evidence` namespace did not list all collaborators used by the API.

## Repair and prevention

- Add the evidence package and test it before API integration.
- Keep adapters for legacy Fluency and Range fakes.
- Require the explicit Fluency compatibility adapter at the strict pipeline boundary; tests exercise the adapter.
- Add regression tests for one-tokenisation Range consumption and opt-in cache reuse.
- Run focused and complete test suites before completion.
- Keep API integration tests importing through the public package boundary.
- The environment does not provide `ruff`; retain compile, full-test, and `git diff --check` verification until linting is added to the project toolchain.
