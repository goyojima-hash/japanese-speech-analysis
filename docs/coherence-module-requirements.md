# Coherence Module Requirements

## Goal

Add an independently selectable, fact-only Coherence module for the CEFR qualitative speaking axes. It consumes the existing `EvidenceBundle v1` and emits a `CoherenceFactPacket`. It never assigns a coherence score, CEFR/JF level, “good/bad” judgement, correction, or LLM conclusion.

## Scope: v1

- Reuse the common transcript, shared Sudachi tokens, token character offsets, and pause spans without re-running ASR, VAD, Sudachi, Range, Fluency, or Accuracy.
- Emit explicit discourse-connective observations from a versioned local lexicon. Each observation records its token position, surface/lemma, and a descriptive category such as `causal`, `contrast`, `additive`, `sequence`, or `conclusion`.
- Emit deterministic **candidate** discourse units. A candidate boundary may come from an explicit connective or a finite/polite predicate pattern; its derivation is recorded. It is not a sentence-correctness judgement.
- Emit lexical repetition observations across candidate units, including lemma, unit indexes, and distances.
- Copy pause spans as independent temporal observations. V1 does not claim a pause belongs to a specific token boundary because Evidence v1 has no validated token-to-time alignment.
- Explicitly list unavailable capabilities: dependency parse, coreference, implicit discourse relation, semantic continuity embedding, and token-to-pause alignment.
- Make Coherence available only when explicitly selected. The existing API default (`fluency`, `range`) and all existing Fluency/Range/Accuracy output contracts remain unchanged.

## Non-goals

- No LLM, KWJA, GiNZA, embedding model, semantic-similarity score, grammar check, discourse quality score, CEFR mapping, or “coherent/incoherent” label in v1.
- No new field in `EvidenceBundle v1`; no schema bump; no second tokenisation.
- No use of `range_data` or `accuracy_data`; Coherence depends directly on common evidence only.
- No implicit claim that an ASR transcript boundary, connective, repetition, or pause is a learner mistake.
- No automatic change to existing interactive-console defaults. Console module-selection UX is a later, separately tested integration step.

## Inputs

| Input | Required | Use |
| --- | --- | --- |
| `EvidenceBundle.speech.raw_transcript_hiragana` | Yes | candidate-unit text slices and provenance |
| `EvidenceBundle.speech.pause_segments` | Yes | independent temporal pause observations |
| `EvidenceBundle.linguistic.tokens` | Yes | connective, predicate-pattern, and repetition observations |
| `EvidenceBundle.linguistic` tokenizer metadata | Yes | reproducibility |

## Output contract

`CoherenceFactPacket` must contain only:

- `module_id = "coherence"`;
- `input_provenance`: Evidence schema, STT model, tokenizer version/split mode, unitizer-policy version, and connective-lexicon version;
- `connective_observations`: observed tokens and categories;
- `candidate_units`: character/token spans plus boundary derivations;
- `repetition_observations`: repeated lemmas and their unit positions;
- `pause_observations`: copied timed pause spans;
- `unavailable_capabilities`: named limitations.

No output key may contain `score`, `rating`, `cefr`, `jfs`, `correct`, `error`, `quality`, or a boolean verdict.

## Acceptance criteria

1. Default API payloads remain byte-for-byte compatible in their Fluency/Range facts and continue to report only `fluency` and `range` as default fact modules.
2. `selected_modules=["coherence"]` invokes Coherence but not Range or Accuracy.
3. `selected_modules=["range", "accuracy", "coherence"]` tokenizes once in the common layer and gives all three modules the same `LinguisticEvidence` instance/value.
4. Coherence never modifies `EvidenceBundle`, `audio_pipeline.py`, `range.py`, or `accuracy.py`.
5. Empty transcript, empty tokens, no pauses, unknown connective, and incomplete timing data return explicit empty/unavailable facts rather than an inferred rating or exception.
6. Every output is deterministic for the same EvidenceBundle and provider versions.
7. The Judge prompt describes Coherence as observations only and prohibits treating a candidate boundary or unavailable capability as a learner error.
