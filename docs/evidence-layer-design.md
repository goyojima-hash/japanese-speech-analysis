# Common Evidence Layer Design

## Decision

Use a thin shared evidence layer for objective, versioned source facts. Keep all axis-specific interpretation in independently selectable fact modules and all CEFR/JFS evaluation in the downstream AI Judge.

## Components

- `jgrade_eval/evidence/models.py`: immutable, serialisable evidence records.
- `jgrade_eval/evidence/speech.py`: adapter from the existing `FluencyExtractor` output to `SpeechEvidence`.
- `jgrade_eval/evidence/linguistic.py`: shared SudachiPy tokenisation and token records.
- `jgrade_eval/evidence/cache.py`: optional JSON cache keyed by source content and provenance.
- `jgrade_eval/evidence/pipeline.py`: composes evidence and uses the optional cache.
- `jgrade_eval/range.py`: retains its public API and gains an evidence-consuming path.

## Data boundaries

`EvidenceBundle` contains facts and provenance only. It may contain transcript text, timings, morphology, model versions, dictionary versions, and capability availability.

It must not contain a CEFR level, numerical competence score, grammatical-error label, fluency grade, or AI/Judge conclusion.

## Migration

`FluencyExtractor` remains the compatibility adapter for current callers. Its output gains `mora_timings`, an additive factual field. `api_service.py` builds one bundle, derives its legacy objective data, and passes its linguistic evidence to Range.

The v1 cache is opt-in. Its key includes source bytes, schema version, speech-model provenance, VAD-model provenance, tokenizer provenance, and preprocessing configuration.

## Deferred capabilities

- ASR confidence and alignment calibrated for Accuracy.
- Speaker diarisation and turn alignment for Interaction.
- Utterance/clause segmentation for Coherence.

They will extend the evidence schema through explicit capability fields rather than change the meaning of v1 fields.
