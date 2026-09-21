# Accuracy Module Requirements

## Goal

Add an independently selectable, fact-only Accuracy module for the CEFR speaking axes. It consumes a shared `EvidenceBundle` and emits an `AccuracyFactPacket`; it never emits a score, CEFR/JFS level, correct/incorrect verdict, grammar correction, or LLM conclusion.

## Scope of implemented v1

- Emit CTC-ASR timing observations for each decoded mora/character: time span, CTC argmax derivation, model provenance, and calibration status `unavailable`.
- Emit morphology observations from the existing shared Sudachi tokens: surface, lemma, reading, POS, character offsets, and deterministic local-pattern identifiers (particle, conjugation, and lexical-form patterns).
- If a caller supplies an explicit task/reference transcript, emit a normalized token alignment with insertions, deletions, substitutions, and unchanged spans. It remains a transcript difference, not an error judgement.
- Mark unavailable capabilities explicitly. The first implementation does not synthesize posterior summaries, calibrated confidence, phone-level alignment, pronunciation diagnosis, or a reference transcript.
- Remain usable alone or alongside any subset of Range, Fluency, Interaction, and Coherence.

## Non-goals

- No LLM, grammar correction model, CEFR mapping, error rate, pronunciation score, or `correct` boolean.
- No comparison with a task prompt, expected keyword list, or reference transcript unless the caller explicitly selects and provides a reference.
- No persistent cache by default and no storage of frame-level logits. Compact derived observations are serializable only through the existing opt-in evidence cache.
- No mandatory Montreal Forced Aligner dependency in v1. It remains an optional future provider for teacher/reference-script alignment.

## Inputs

| Input | Required | Source |
| --- | --- | --- |
| `EvidenceBundle.speech` transcript, mora times, provenance | Yes | common layer |
| `EvidenceBundle.linguistic` tokens | Yes | common layer |
| `AsrAlignmentEvidence` | Optional capability | CTC adapter |
| `reference_transcript` and normalization profile | Optional | explicit caller input |

## Output contract

`AccuracyFactPacket` has a schema version, module id, input provenance, capability status, and only these fact collections:

- `asr_observations`: token/mora spans derived from collapsed CTC argmax timings; posterior fields are null and marked unavailable.
- `morphology_observations`: token facts plus deterministic local pattern ids.
- `reference_differences`: optional normalized alignment operations.
- `unavailable_capabilities`: named reasons, never fabricated fallbacks.

The downstream AI Judge receives the packet together with other axis FactPackets and decides whether an observation is meaningful for accuracy.

## Acceptance criteria

- The default API/CLI path remains free of Accuracy work unless the module is selected.
- A selected Accuracy module uses shared tokens and does not invoke Range or Fluency modules.
- Tests prove unavailable posterior values cannot be exposed as a correctness flag or score.
- Tests prove reference alignment is absent without an explicit reference and records only edit operations when supplied.
- Tests prove unavailable alignment/phoneme capabilities are explicit.
- API tests prove module selection can request Accuracy without changing unselected module output.
