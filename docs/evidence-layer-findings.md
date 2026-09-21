# Common Evidence Layer Findings

- `FluencyExtractor.extract()` already creates raw hiragana, VAD segments, pauses, and mora timings internally, but does not currently expose mora timings in its returned dictionary.
- `RangeExtractor` owns `SudachiTokenizer` today. It needs an evidence-consuming entry point so the API path does not tokenise twice.
- `api_service.evaluate_speech_level()` is the single current composition point for Fluency and Range, making it the safe integration boundary.
- The persistent evidence cache must be opt-in because serialized bundles contain learner transcript data.
- Existing `fluency_grade` was an interpretation rather than a source fact. It is intentionally excluded from the common bundle and the derived API/CLI facts; the downstream Judge remains responsible for evaluation.
- Cache provenance must describe the configured extractor before extraction. The compatibility adapter therefore forwards the legacy extractor provenance, and the production Fluency extractor declares its STT/VAD identities.
- `ruff` is unavailable in the current `uv` environment; compile, full-unit-test, and diff-whitespace checks passed instead.
- The standalone `fluency.py` display also no longer creates an S–D grade, so both supported paths keep evaluation downstream.
