# Coherence Module Implementation Plan

## Phase 0 — Regression baseline and Bug Hunter pre-mortem

1. Freeze existing Fluency, Range, and Accuracy fact-packet/API snapshots with a deterministic Evidence fixture.
2. Reproduce the two relevant failure modes before implementation: duplicate tokenisation when three modules are selected, and a default payload changed by an unselected module.
3. Record the 5 Whys and Fishbone in the bilingual regression report.

### Pre-mortem

| Risk | Early detection | Prevention |
| --- | --- | --- |
| Coherence changes Range tokenisation | tokenizer call count exceeds one | inject one shared `LinguisticEvidence` fixture and assert identity/value |
| Default API contract changes | existing API snapshot differs | leave `DEFAULT_FACT_MODULES` unchanged |
| Rules become hidden scoring | prohibited-key test passes an evaluation word | strict packet schema and prompt-boundary test |
| ASR without punctuation yields fabricated sentences | candidate unit lacks derivation | require every boundary derivation or fallback one-unit result |
| Future model invalidates common cache | Evidence schema/cache key changes | keep model output outside EvidenceBundle |

## Phase 1 — Requirements and RED tests

1. Add tests for connective categories, deterministic candidate-unit fallback, repetition observations, pause copying, empty input, and prohibited evaluation terms.
2. Add API-selection tests for Coherence alone, all three text modules, and default compatibility.
3. Add regression tests for no Range/Accuracy invocation and single shared tokenisation.
4. Run the focused tests red and create the required TDD checkpoint commit.

## Phase 2 — Core collector

1. Add `jgrade_eval/coherence.py` with packet dataclasses, versioned lexicon, and deterministic collectors.
2. Preserve `EvidenceBundle v1`; do not alter existing module files.
3. Run focused tests green and create the GREEN checkpoint commit.

## Phase 3 — API and Judge boundary

1. Add explicit `coherence` selection in the API module registry only.
2. Add `coherence_data` only for selected requests and update the Judge prompt boundary.
3. Re-run API/default regression tests and create a checkpoint commit.

## Phase 4 — Verification and delivery

1. Run focused tests, the full suite, compile checks, and `git diff --check`.
2. Compare touched paths against the pre-Coherence commit; report any unexpected change to Fluency, Range, Accuracy, or Evidence as a blocker.
3. Perform Bug Hunter 5 Whys, Fishbone, timeline, bilingual report, and recurrence-prevention verification.
4. Do not install KWJA, GiNZA, embeddings, pyannote, or alter interactive defaults in this implementation.

## Deferred work

- Teacher-reviewed connective/boundary lexicon.
- Validated token-to-time alignment for attaching pauses to units.
- KWJA provider experiment against a Japanese learner-speech test set.
- Explicit console module-selection UX and Interaction input contract.
