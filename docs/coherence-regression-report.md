# Coherence V1 Regression and Bug Hunter Report

## Scope

This report covers the independent, fact-only Coherence V1 module. It records pre-implementation failures used to verify that the implementation does not alter the Evidence layer or the existing Fluency, Range, and Accuracy contracts.

## Reproduced failures and resolution

| Reproducer | Observed RED state | Resolution | Regression control |
| --- | --- | --- | --- |
| Coherence collector/API selection | `ModuleNotFoundError` and unsupported `coherence` selection | Added `CoherenceModule` and opt-in API registry entry | Unit and API selection tests |
| Judge boundary | Prompt did not describe `coherence_data` as fact-only | Added an explicit prohibition on treating candidate units, unavailable capabilities, or observations alone as quality/error/level evidence | Prompt test |
| Candidate-unit terminal provenance | Final candidate unit lacked `transcript_end` derivation | Added terminal derivation to the final unit | Collector test |

## 5 Whys: why could Coherence distort an evaluation?

1. A Judge could treat a rule-generated candidate boundary as a learner error or quality score.
2. The packet includes discourse-like observations that can look evaluative without an explicit boundary.
3. ASR transcript punctuation and token-to-time alignment are not validated sentence annotations.
4. A richer discourse model has not been benchmarked on Japanese learner speech in this project.
5. Therefore V1 must expose only traceable observations and named limitations, with a Judge instruction that forbids inference from any observation alone.

Root cause prevented: confusing a deterministic extraction artifact with a proficiency judgement.

## Fishbone analysis

| Area | Risk | Control implemented |
| --- | --- | --- |
| Data | ASR boundaries are incomplete | Candidate label, derivation, fallback, unavailable capabilities |
| Rules | Lexicon could become hidden scoring | Versioned descriptive categories; no score/rating/error fields |
| Integration | Existing modules could be rerun or altered | Coherence reads `EvidenceBundle`; default modules unchanged |
| API | Unselected module could change responses | Add `coherence_data` only for explicit selection |
| Judge | Observations could be over-interpreted | Fact-only prompt boundary and test |
| Operations | Future models could invalidate cache | No model or Evidence schema change in V1 |

## Timeline

1. Added collector, API-selection, and prompt-boundary RED tests.
2. Confirmed missing module/registry and missing prompt text as intended failures.
3. Added the deterministic collector, explicit opt-in wiring, and fact-only Judge guidance.
4. Added the terminal-provenance reproducer, confirmed RED, then added `transcript_end` provenance.
5. Confirmed one shared tokenisation across Range, Accuracy, and Coherence.
6. Ran the focused tests, full suite, compilation, and whitespace checks.

## Recurrence-prevention verification

- Default API output remains `fluency` and `range`, without `coherence_data`.
- Explicit `selected_modules=("coherence",)` does not run Range.
- Coherence has no AI model, score, CEFR/JFS, correctness, error, or quality output fields.
- No production changes were made to `evidence/`, `audio_pipeline.py`, `range.py`, or `accuracy.py`.
