# J-GRADE Model Selection Decision for Japanese Learner Speech

Date: 2026-09-14  
Decision status: research conclusion; no new model is installed or adopted.

## Decision

For J-GRADE, **there is no defensible single “best model” that should directly assign a Japanese learner's CEFR/JF level.** The best architecture is a calibrated evidence pipeline: independent fact extractors produce explicitly versioned observations, then multiple downstream Judges apply JF Standard Can-do criteria and expose disagreement for human review.

This follows the target construct. The Japan Foundation defines JF Standard levels through what a learner can do, not through grammar or vocabulary quantity alone. Its qualitative spoken-language reference explicitly lists Range, Accuracy, Fluency, Interaction, and Coherence as distinct dimensions. [JF Standard overview](https://www.jfstandard.jpf.go.jp/summary/ja/render.do), [JF Standard guidebook, p.86](https://www.jfstandard.jpf.go.jp/pdf/web_whole_en.pdf)

## Concrete model decisions

| J-GRADE responsibility | Best choice now | Why this is the J-GRADE choice | Do not do |
| --- | --- | --- | --- |
| Shared transcript/timing | Keep `vumichien/wav2vec2-large-xlsr-japanese-hiragana` plus Silero VAD | It already supplies the common factual substrate. Replacing it without Japanese-learner ASR evidence would merely move risk upstream. | Treat ASR output or confidence as learner correctness. |
| Coherence v1 | **No additional model**; shared Sudachi facts | Transparent connectives, unit lengths, repetition, and boundary observations are enough to establish a benchmarkable baseline and preserve independent modules. | Start with a general LLM or call a similarity value a proficiency score. |
| Coherence v2 | **KWJA, optional provider after benchmark** | It is the strongest found Japanese-specific model candidate for discourse relation, dependency, PAS, bridging, and coreference observations. It is closer to the construct than a generic embedding model. | Replace shared tokenisation; present predicted discourse relations as ground truth. |
| Interaction v1 | **pyannote Community-1, only after dialogue input exists** | Speaker turns and overlap are indispensable facts for interaction; the local pipeline has exclusive diarization designed for ASR timestamp reconciliation. | Calculate response lags from a one-speaker recording or use a turn-prediction model without learner-dialogue labels. |
| Level decision | **Multi-Judge panel, teacher-calibrated** | CEFR/JF assessment is a construct decision across task performance and the five evidence axes. The evaluator must be validated against trained human raters, rather than selected from a generic benchmark. | Let one model's uncalibrated answer become the ground truth. |

`multilingual-e5-large-instruct` remains an experiment, not the primary choice: it can provide multilingual sentence-vector similarities, but a similarity value does not represent the JF Standard's task accomplishment, interaction, or coherence construct. [Model card](https://huggingface.co/intfloat/multilingual-e5-large-instruct)

## Why KWJA, not a general LLM, for Coherence observations

KWJA is a Japanese unified analyzer that supports discourse relation analysis alongside morphology, syntax, predicate-argument structure, bridging, and coreference. That makes its output suitable as *model-derived evidence* for an optional Coherence packet. It is not a level classifier. The existing Japanese cohesion analyzer shows that this family requires a heavier preprocessing/model stack, so it should be isolated behind a provider interface and benchmarked before inclusion. [KWJA ACL system paper](https://aclanthology.org/2023.acl-demo.52), [cohesion analyzer](https://github.com/nobu-g/cohesion-analysis)

## Why pyannote, not NeMo, for the first Interaction prototype

Interaction facts require speaker identity over time. `pyannote/speaker-diarization-community-1` supports local execution, later offline loading, 16kHz input/resampling, and an exclusive speaker timeline intended to reconcile speaker turns with ASR timestamps. It is therefore the smallest useful extension to the present pipeline. Its initial download requires an HF token and acceptance of terms, and the model is CC-BY-4.0. [Model card](https://huggingface.co/pyannote/speaker-diarization-community-1)

NeMo provides both end-to-end Sortformer and a VAD + TitaNet + MSDD pipeline. It is worth a later benchmark on a CUDA deployment, but carries a larger operational stack and is not the default fit for this macOS-oriented local project. [NeMo guide](https://docs.nvidia.com/nemo-framework/user-guide/25.07/nemotoolkit/asr/speaker_diarization/models.html)

## The required evidence before any model becomes “best”

Generic Japanese or multilingual benchmarks are insufficient because J-GRADE evaluates L2 Japanese speech. I-JAS contains speech and writing from 1,000 Japanese learners across 12 first-language groups, and C-JAS contains roughly 46.5 hours of natural learner/native conversations. These are valuable evaluation sources, but their access and publication conditions must be reviewed before use; C-JAS transcripts are CC BY-NC-ND 4.0. [I-JAS](https://www2.ninjal.ac.jp/jll/lsaj/ijas-document.html), [C-JAS](https://mmsrv.ninjal.ac.jp/c-jas/en/index.html)

Before adoption, build a held-out evaluation set with task prompts, audio, reference transcription, speaker-turn/overlap boundaries where applicable, and independent teacher CEFR/JF labels. Select providers by:

1. ASR error and timestamp error on learner speech, not native-only speech;
2. diarization error, missed speech, and speaker confusion on learner dialogues;
3. module-fact stability, provenance, runtime, and failure behavior;
4. downstream agreement with human labels, confidence calibration, and fairness slices by L1, task, recording condition, and level;
5. mandatory human review when Judges disagree or evidence is unavailable.

## Implementation order

1. Coherence v1 from the existing EvidenceBundle — no new model.
2. Collect and label a learner-speech validation set.
3. Compare KWJA against the deterministic Coherence baseline as an optional provider.
4. Add a multi-speaker input contract, then prototype pyannote Interaction facts.
5. Compare pyannote with NeMo only if a CUDA deployment or latency requirement justifies it.
6. Calibrate the existing multi-Judge panel to human raters. Only then can J-GRADE make a defensible proficiency claim.
