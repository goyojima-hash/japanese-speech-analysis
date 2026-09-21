# Model Research for Coherence and Interaction Modules

Date: 2026-09-14  
Scope: model candidates for future fact-only Coherence and Interaction modules. No model is installed or selected by this report.

## Decision summary

1. **Implement Coherence v1 without a new model.** Use the existing shared Sudachi evidence to emit connective counts, sentence-like units, unit lengths, and repetitions as facts.
2. **Keep KWJA as the leading Coherence v2 research candidate.** It is a Japanese foundation-model analyzer that includes dependency, coreference, bridging, and discourse-relation analysis, but it must be integrated as an optional, separately versioned provider rather than replacing the shared tokenizer.
3. **Use local pyannote Community-1 as the first Interaction prototype candidate.** It produces speaker turns, supports offline use after download, and exposes an exclusive diarization timeline intended to reconcile with ASR timestamps.
4. **Do not introduce a turn-taking prediction model yet.** Turn timing, overlap, and speaker-turn facts are sufficient for the first Interaction packet. A predictor would need Japanese learner-dialogue labels and could blur the required fact/evaluation boundary.
5. **Do not replace the current Japanese Wav2Vec2 ASR merely for these modules.** It already produces the transcript/timing input; model replacement needs a held-out Japanese-learner benchmark.

## Candidate matrix

| Area | Candidate | Useful factual output | Advantages | Important limits | Recommendation |
| --- | --- | --- | --- | --- | --- |
| Coherence v1 | Existing Sudachi evidence | connective inventory, POS sequences, unit-length distribution, lexical repetition | no new dependency; preserves one shared tokenisation | cannot resolve implicit discourse/coreference | adopt first |
| Coherence v2 | KWJA | dependency/PAS, bridging, coreference, discourse-relation predictions | Japanese-specific unified analyzer; MIT licensed | different analysis stack and model predictions are uncertain observations, not ground truth | evaluate as optional provider |
| Coherence v2 | GiNZA `ja_ginza_electra` | bunsetsu, dependency and syntax facts | MIT; Sudachi-compatible ecosystem | not a complete discourse/cohesion analyzer | lighter syntax alternative |
| Coherence semantic extension | `multilingual-e5-large-instruct` | adjacent-unit embedding vectors and cosine similarities | 94 languages, MIT, standard Transformers/SentenceTransformers path | 0.6B parameters; inputs truncate at 512 tokens; similarity is not a coherence score | optional experiment only |
| Interaction v1 | `pyannote/speaker-diarization-community-1` | speaker turns, overlap, turn gaps, diarization provenance | local/offline after download; exclusive diarization helps align transcript timestamps | HF access token/user conditions; CC-BY-4.0; must validate Japanese learner conversations | first prototype |
| Interaction alternative | NVIDIA NeMo Sortformer / MSDD | speaker activity/labels and turns | end-to-end Sortformer or modular VAD + embeddings + MSDD choices | heavier NeMo stack and GPU-oriented operational cost; no reason yet to prefer it on this macOS project | benchmark alternative, not initial implementation |

## Evidence

### Coherence

The project already depends on SudachiPy and uses it in the shared Evidence layer. It is therefore enough for a first fact packet: extract explicit connectives, unit boundaries derived from pauses/transcript boundaries, unit token counts, and repeats. These are transparent observations and add no new model risk.

For a richer optional provider, [KWJA](https://github.com/ku-nlp/kwja) is the strongest Japanese-specific candidate found. Its ACL system description reports one unified foundation-model analyzer covering morphological analysis, dependency parsing, predicate-argument structure, bridging reference, coreference, and discourse-relation analysis; the package is MIT licensed. The related [cohesion analyzer](https://github.com/nobu-g/cohesion-analysis) publishes base and large checkpoints, but has a materially heavier Juman++/KNP or KWJA preprocessing chain. Therefore neither must replace `EvidenceBundle` tokenisation; their outputs should be a separately versioned `coherence_model_observations` extension. [KWJA source](https://github.com/ku-nlp/kwja), [cohesion analyzer source](https://github.com/nobu-g/cohesion-analysis), [ACL paper](https://aclanthology.org/2023.acl-demo.52)

[GiNZA](https://github.com/megagonlabs/ginza) is a lower-scope Japanese Universal Dependencies parser. It has an MIT license and a transformer model (`ja_ginza_electra`), making it a reasonable syntax-only comparison; it does not by itself supply the full cohesion/discourse capability of KWJA.

`multilingual-e5-large-instruct` is an optional semantic-continuity provider: it supports 94 languages, has an MIT license, is available through Transformers/SentenceTransformers, has 0.6B parameters, and truncates long inputs at 512 tokens. It can provide vectors and cosine similarities between adjacent units, but neither a similarity value nor a model output may be labeled “coherent” by this module. [Model card](https://huggingface.co/intfloat/multilingual-e5-large-instruct), [technical report](https://arxiv.org/abs/2402.05672)

### Interaction

The present single-speaker audio and Silero VAD can measure pauses but cannot establish who spoke when. Interaction requires a multi-speaker input contract plus diarization before meaningful response-lag or turn-taking facts can exist.

`pyannote/speaker-diarization-community-1` accepts mono 16kHz audio (resampling when needed), runs locally, can later be loaded from disk offline, and returns an exclusive speaker timeline specifically intended to reconcile with ASR timestamps. It is CC-BY-4.0 and requires accepting model conditions plus an HF token for the initial download. Its published benchmarks include Mandarin meeting data, but no Japanese learner-dialogue guarantee was found; local Japanese validation remains mandatory. [Model card](https://huggingface.co/pyannote/speaker-diarization-community-1), [multilingual diarization benchmark](https://arxiv.org/abs/2509.26177)

NeMo offers two supported families: Transformer-based Sortformer and a cascaded VAD + TitaNet speaker embedding + MSDD pipeline. This makes it a serious benchmark alternative, particularly for a CUDA deployment, but it adds a large stack to a current macOS-oriented project. [NeMo model guide](https://docs.nvidia.com/nemo-framework/user-guide/25.07/nemotoolkit/asr/speaker_diarization/models.html)

### Existing ASR and Accuracy

The current `vumichien/wav2vec2-large-xlsr-japanese-hiragana` remains usable as the shared transcription/timing source: its card identifies Japanese Common Voice fine-tuning, Apache-2.0 licensing, and approximately 2.52GB repository size. Accuracy should first expose calibrated information from this same CTC path only after a learner-speech benchmark validates it; it should not add a second ASR model merely to create a confidence-like number. [Model card](https://huggingface.co/vumichien/wav2vec2-large-xlsr-japanese-hiragana)

## Recommended validation before adoption

1. Create a held-out Japanese learner set with manual transcript, speaker-turn, and overlap boundaries; include clean, noisy, and overlapping conversations.
2. For Coherence, compare deterministic Sudachi facts with KWJA/GiNZA observations and record output provenance, runtime, memory, and failure cases. Do not compare them through CEFR labels initially.
3. For Interaction, measure diarization error rate, missed speech, and speaker confusion on the held-out set before calculating response-lag facts. The 2025 multilingual study identifies missed speech and speaker confusion as major error sources.
4. Establish provider interfaces so every output includes model ID, revision, device, input schema, and unavailable-capability flags.
5. Only after fact extraction is stable, give those fact packets to the downstream Judge for separate level evaluation.
