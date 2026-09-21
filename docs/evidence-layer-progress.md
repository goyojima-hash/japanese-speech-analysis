# Common Evidence Layer Progress

## 2026-09-14

- Requirements, design, tasks, and ADR created before implementation.
- Added `tests/test_evidence.py` before production code.
- Expected initial test result: `ModuleNotFoundError: No module named 'jgrade_eval.evidence'`.
- Next: add the evidence package and make the focused tests pass.
- Focused test rerun found a fixture mismatch: `EvidencePipeline` correctly requires `SpeechEvidence`, but the fake supplied legacy output directly. Tests now use the explicit compatibility adapter.
- API integration initially failed to import the compatibility adapter from the evidence package. The package now exports its public pipeline collaborators explicitly; this is covered by `tests.test_jgrade_api`.
- Added cache-key regression coverage for changed model provenance; the focused Evidence/API suite passed 8 tests.
- The interactive CLI now also composes the shared evidence before Range consumes it.
- Verification: `uv run python -m compileall -q jgrade_eval tests fluency.py`, `uv run python -m unittest discover -s tests`, and `git diff --check` passed. The full suite ran 68 tests. `ruff` is not installed in this environment.
