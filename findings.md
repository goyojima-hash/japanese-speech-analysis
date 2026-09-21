# Accuracy Module Findings

- The checked-in Evidence v1 supports Fluency and lexical Range. Its documented deferred capability for Accuracy is ASR confidence and alignment.
- `fluency.py`/`FluencyExtractor` already has Wav2Vec2 logits internally, but the shared bundle does not expose confidence or an alignment contract.
- The current worktree was clean before planning began.
- Hugging Face's CTC ASR pipeline documents character and word timestamps for pure CTC models such as Wav2Vec2. It describes them as model predictions, so timestamps/confidence must be retained as evidence rather than treated as correctness. Source: https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/automatic_speech_recognition.py
- Wav2Vec2 CTC exposes logits representing token log probabilities. A confidence-derived value must carry its derivation and calibration status; raw logits alone do not establish learner accuracy. Source: https://huggingface.co/docs/transformers/v4.42.4/en/model_doc/wav2vec2
- Montreal Forced Aligner requires audio, an orthographic transcript, a pronunciation dictionary, and an acoustic model. It is a future optional refinement for externally supplied/reference transcripts, not a dependency for v1 learner-ASR evidence. Source: https://montreal-forced-aligner.readthedocs.io/en/v3.3.4/user_guide/workflows/alignment.html
