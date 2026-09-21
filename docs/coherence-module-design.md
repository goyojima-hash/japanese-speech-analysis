# Coherence Module Design

## Decision

Implement `jgrade_eval/coherence.py` as a pure, independent collector over `EvidenceBundle v1`. It returns a `CoherenceFactPacket` and does not change common preprocessing or existing module code. V1 deliberately uses no external model; its job is to establish an auditable fact baseline for Japanese learner speech before optional model providers are evaluated.

## Why this protects the existing modules

The shared pipeline already performs the expensive and cross-cutting work exactly once: audio extraction, transcript generation, VAD/pause extraction, mora timings, and Sudachi tokenisation. Coherence reads those immutable facts. It does not call `RangeExtractor`, `AccuracyModule`, or `FluencyExtractor`; it therefore cannot alter their algorithms, output shape, dependency versions, or cache key.

`EvidenceBundle v1` remains unchanged. A future model-specific result is an optional field in the *Coherence packet*, not a field added to common evidence. This prevents a KWJA/GiNZA/embedding experiment from invalidating cached Evidence or changing existing output contracts.

## Data types

```text
CoherenceFactPacket
├── module_id: "coherence"
├── input_provenance: map[str, str]
├── connective_observations: ConnectiveObservation[]
│   ├── token_index, surface, dictionary_form
│   ├── category: causal | contrast | additive | sequence | condition | conclusion | other
│   └── lexicon_version
├── candidate_units: CandidateUnit[]
│   ├── unit_index, token_start_index, token_end_index
│   ├── text_start_offset, text_end_offset
│   └── boundary_derivations: [connective:<category> | terminal_pattern:<id> | transcript_end]
├── repetition_observations: RepetitionObservation[]
│   ├── dictionary_form, unit_indexes, occurrence_count, unit_distances
├── pause_observations: TimedSpan[]
└── unavailable_capabilities: string[]
```

`CandidateUnit` is intentionally named a candidate. ASR text normally lacks dependable punctuation, and a rule-generated boundary is not a verified sentence boundary. A unit has no score or quality field.

## Deterministic v1 algorithms

### 1. Connective lexicon

`CoherenceLexicon v1` is a small, versioned mapping from token lemma/surface to descriptive categories. Examples include `だから`/`ので` (causal), `しかし`/`けど` (contrast), `そして`/`また` (additive), and `まず`/`次に` (sequence). The mapping records observed language; it does not judge whether use is appropriate.

### 2. Candidate units

Start at token zero and create a boundary:

1. immediately **before** a connective token that begins a new relation; or
2. immediately **after** a token matching a conservative terminal/polite pattern, unless that next token is itself a connective boundary; or
3. at the end of the transcript.

The connective therefore remains in the unit it introduces. The algorithm keeps the source token and character offsets. If no conservative pattern is found, it emits exactly one transcript-wide candidate unit. This fail-safe is preferable to fabricating sentence segmentation.

### 3. Repetition

For lexical tokens only, group normalised dictionary forms that appear in more than one candidate unit. Record appearances and unit distances. Do not emit repetition inside one unit, stopword-only repetition, a diversity ratio, or a “redundant” judgement.

### 4. Pauses

Copy `EvidenceBundle.speech.pause_segments` into the packet and include their existing timings. Do not attach pauses to units in V1. Reliable attachment requires an explicit, validated token-to-time alignment capability, which is not present in Evidence v1.

## Model-provider boundary: deferred v2

`CoherenceObservationProvider` will be a protocol:

```python
class CoherenceObservationProvider(Protocol):
    provider_id: str
    version: str
    def observe(self, evidence: EvidenceBundle) -> CoherenceModelObservations: ...
```

KWJA is the first research candidate, because it can provide Japanese dependency, coreference, and discourse-relation predictions. Its outputs must remain separately tagged `derivation="kwja"`, include model revision/runtime provenance, and be marked model-derived observations. It cannot replace the deterministic v1 facts or make an evaluation decision.

## API and Judge integration

Add `"coherence"` to `SUPPORTED_FACT_MODULES`, but leave `DEFAULT_FACT_MODULES` untouched. When selected, `evaluate_speech_level` adds `coherence_data` to `objective_data` and `roleplay_input`; otherwise neither key exists. The Judge prompt must say that candidate units, connective categories, repetitions, and missing capabilities are observations, not evidence of error or level by themselves.

The interactive console uses the same fact-module runner as the API. Its explicit console profile selects every implemented fact module and prints `=== Coherence客観データ ===`; the API retains its historical default selection for response compatibility.

## Failure behavior

| Condition | Packet behavior |
| --- | --- |
| Empty transcript/tokens | empty collections; `no_linguistic_tokens` unavailable capability |
| No connective in lexicon | empty `connective_observations`; not an error |
| No conservative boundary | one transcript-wide candidate unit |
| No pauses | empty pause collection; not an error |
| Missing offsets | omit affected candidate text span and add `token_offsets_unavailable` |
| Future provider fails | retain deterministic v1 facts; add provider-specific unavailable capability |

## Regression safeguards

- Keep a frozen fixture asserting the serialized Fluency/Range/Accuracy facts before and after Coherence selection support.
- Use an instrumented tokenizer fixture to prove one shared tokenisation for all selected modules.
- Assert no production diff in `audio_pipeline.py`, `range.py`, `accuracy.py`, or `evidence/` for the V1 implementation.
- Test module isolation and both default/non-default API payload shapes.
- Reject prohibited evaluation terms from Coherence serialization and validate exact provider/lexicon versions.
