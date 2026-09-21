# Accuracy Module Plan

## Goal

Define the fact-only Accuracy module before implementation. It must remain independently selectable, consume the shared EvidenceBundle, and leave CEFR/JFS evaluation to the downstream AI Judge.

## Phases

1. **Requirements** — complete: scope, non-goals, contracts, and acceptance criteria documented.
2. **Research and design** — complete: CTC timestamp/logit and forced-alignment boundaries, provider protocol, and schema documented.
3. **Implementation plan** — complete: Accuracy timing/morphology/reference facts, independent selection, and regression verification implemented.

## Decisions already fixed

- Common layer and modules must produce facts, not grades, CEFR labels, correct/incorrect labels, or LLM conclusions.
- Accuracy must not depend on Range or Fluency module output; it consumes shared evidence directly.
- Current Evidence v1 has transcript, timings, and morphology but lacks calibrated ASR confidence and token-to-audio alignment.

## Errors encountered

| Error | Resolution |
| --- | --- |
| Initial web-result handling expected structured content, but the browser returned a string | Recorded the tool-shape mismatch and used the returned primary-source URLs without repeating the same handling approach. |
| Mermaid file replacement attempted delete and add for the same file in one patch | Applied the delete and add as separate patches, then replaced the Markdown copy with the sequence form. |
| New Accuracy contract test cannot import `jgrade_eval.accuracy` | Intended RED state; production module has not yet been created. Existing Evidence regression tests pass. |
| `python -m coverage` could not run because the package is absent | Recorded as an unavailable verification, not as coverage success; focused and complete unit suites remain the executed evidence. |
