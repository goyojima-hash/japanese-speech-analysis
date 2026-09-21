# Common Evidence Layer Requirements

## Scope

Introduce a versioned, immutable `EvidenceBundle` that represents shared, objective evidence for the five future CEFR speaking-axis modules. This first implementation supports the existing Fluency and lexical Range paths without producing scores, CEFR levels, correctness judgements, or LLM output.

## Requirements

- Build the shared evidence once from an audio path: source metadata, raw hiragana transcript, romaji transcript, VAD speech and pause spans, mora timings, and extraction provenance.
- Build linguistic evidence once from the raw transcript: tokens, source character offsets, dictionary form, reading, POS, and tokenizer provenance.
- Preserve the existing API response facts and existing Range analysis output; intentionally remove the former interpreted `fluency_grade` field.
- Make Range consume the common linguistic tokens in the API path, rather than tokenising a second time.
- Offer an opt-in, version-keyed JSON cache. Persistent caching is disabled by default because evidence may contain learner speech data.
- Provide serialisable evidence and fact-only module interfaces; the common layer and modules must not emit a grade, CEFR level, correct/incorrect label, or LLM judgement.
- Keep missing future capabilities explicit. Accuracy confidence/alignment, interaction turns, and discourse segmentation are not fabricated by v1.

## Acceptance criteria

- Existing API tests continue to pass unchanged.
- A focused test proves that one common linguistic tokenisation feeds Range.
- A focused test proves cache keys vary with audio content, preprocessing version, and model provenance.
- A focused test proves a cache hit avoids re-extracting evidence.
- The default runtime writes no cache file.
