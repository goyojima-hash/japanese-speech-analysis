from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from jgrade_eval.accuracy import AccuracyModule
from jgrade_eval.coherence import CoherenceModule
from jgrade_eval.evidence.cache import EvidenceCache
from jgrade_eval.evidence.linguistic import LinguisticEvidenceExtractor, TokenEvidence
from jgrade_eval.evidence.pipeline import EvidencePipeline
from jgrade_eval.evidence.speech import FluencySpeechEvidenceExtractor
from jgrade_eval.range import JLPTVocabulary, RangeExtractor, TokenizedWord


class FakeSpeechExtractor:
    provenance = {
        "stt_model": "fake-stt-v1",
        "vad_model": "fake-vad-v1",
    }

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, audio_path: Path) -> dict:
        self.calls += 1
        return {
            "audio_path": str(audio_path),
            "raw_transcript_hiragana": "わたしはすしがすきです",
            "raw_transcript_romaji": "watashiha",
            "speech_segments": [{"start": 0.0, "end": 1.2, "duration": 1.2}],
            "top_pauses": [],
            "mora_timings": [{"mora": "わ", "start": 0.0, "end": 0.1}],
            "stt_model": "fake-stt-v1",
            "vad_model": "fake-vad-v1",
        }


class FakeTokenizer:
    version = "fake-tokenizer-v1"
    split_mode = "A"

    def tokenize(self, text: str) -> list[TokenEvidence]:
        self.calls = getattr(self, "calls", 0) + 1
        self.last_text = text
        return [
            TokenEvidence("わたし", "私", "わたし", ("名詞",), 0, 3),
            TokenEvidence("は", "は", "は", ("助詞",), 3, 4),
            TokenEvidence("すし", "寿司", "すし", ("名詞",), 4, 6),
        ]


class FakeSpeechExtractorV2(FakeSpeechExtractor):
    provenance = {
        "stt_model": "fake-stt-v2",
        "vad_model": "fake-vad-v1",
    }


class EvidencePipelineTests(unittest.TestCase):
    def test_pipeline_creates_factual_bundle_without_scores(self) -> None:
        speech = FakeSpeechExtractor()
        pipeline = EvidencePipeline(
            speech_extractor=FluencySpeechEvidenceExtractor(speech),
            linguistic_extractor=LinguisticEvidenceExtractor(FakeTokenizer()),
        )

        bundle = pipeline.build(Path("sample.mp3"))

        self.assertEqual(speech.calls, 1)
        self.assertEqual(bundle.schema_version, "evidence.v1")
        self.assertEqual(bundle.speech.raw_transcript_hiragana, "わたしはすしがすきです")
        self.assertEqual(bundle.linguistic.tokens[0].dictionary_form, "私")
        self.assertFalse(any(key in bundle.to_dict() for key in {"score", "grade", "cefr_level"}))
        self.assertNotIn("fluency_grade", bundle.to_dict())

    def test_opt_in_cache_reuses_bundle_for_the_same_source_and_provenance(self) -> None:
        speech = FakeSpeechExtractor()
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.mp3"
            audio_path.write_bytes(b"audio-v1")
            cache = EvidenceCache(Path(directory) / "cache")
            pipeline = EvidencePipeline(
                speech_extractor=FluencySpeechEvidenceExtractor(speech),
                linguistic_extractor=LinguisticEvidenceExtractor(FakeTokenizer()),
                cache=cache,
            )

            first = pipeline.build(audio_path)
            second = pipeline.build(audio_path)

        self.assertEqual(speech.calls, 1)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_cache_key_changes_when_speech_model_provenance_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.mp3"
            audio_path.write_bytes(b"audio-v1")
            cache = EvidenceCache(Path(directory) / "cache")
            first_speech = FakeSpeechExtractor()
            first_pipeline = EvidencePipeline(
                speech_extractor=FluencySpeechEvidenceExtractor(first_speech),
                linguistic_extractor=LinguisticEvidenceExtractor(FakeTokenizer()),
                cache=cache,
            )
            first_pipeline.build(audio_path)
            second_speech = FakeSpeechExtractorV2()
            second_pipeline = EvidencePipeline(
                speech_extractor=FluencySpeechEvidenceExtractor(second_speech),
                linguistic_extractor=LinguisticEvidenceExtractor(FakeTokenizer()),
                cache=cache,
            )
            second_pipeline.build(audio_path)

        self.assertEqual(first_speech.calls, 1)
        self.assertEqual(second_speech.calls, 1)

    def test_range_can_consume_common_linguistic_evidence_without_tokenising_again(self) -> None:
        bundle = EvidencePipeline(
            speech_extractor=FluencySpeechEvidenceExtractor(FakeSpeechExtractor()),
            linguistic_extractor=LinguisticEvidenceExtractor(FakeTokenizer()),
        ).build(Path("sample.mp3"))
        vocabulary = JLPTVocabulary.from_entries(
            [
                {"surface": "私", "reading": "わたし", "level": 5},
                {"surface": "寿司", "reading": "すし", "level": 5},
            ],
            version="test-dictionary",
        )
        extractor = RangeExtractor(_FailingTokenizer(), vocabulary)

        analysis = extractor.analyze_linguistic_evidence(bundle.linguistic)
        direct_analysis = RangeExtractor(FakeTokenizer(), vocabulary).analyze(
            bundle.speech.raw_transcript_hiragana
        )

        self.assertEqual(analysis["statistics"]["token_count"], 3)
        self.assertEqual(analysis["statistics"]["known_token_count"], 2)
        self.assertEqual(analysis, direct_analysis)

    def test_fact_modules_read_one_shared_tokenisation(self) -> None:
        tokenizer = FakeTokenizer()
        bundle = EvidencePipeline(
            speech_extractor=FluencySpeechEvidenceExtractor(FakeSpeechExtractor()),
            linguistic_extractor=LinguisticEvidenceExtractor(tokenizer),
        ).build(Path("sample.mp3"))
        vocabulary = JLPTVocabulary.from_entries([], version="test-dictionary")

        RangeExtractor(_FailingTokenizer(), vocabulary).analyze_linguistic_evidence(bundle.linguistic)
        AccuracyModule().collect(bundle)
        CoherenceModule().collect(bundle)

        self.assertEqual(tokenizer.calls, 1)

    def test_range_tokenized_word_keeps_its_public_lexical_property(self) -> None:
        self.assertTrue(TokenizedWord("寿司", "寿司", "すし", ("名詞",)).is_lexical)
        self.assertFalse(TokenizedWord("は", "は", "は", ("助詞",)).is_lexical)

    def test_fluency_compatibility_facts_remain_stable(self) -> None:
        from jgrade_eval.evidence.speech import objective_data_from_evidence

        evidence = FluencySpeechEvidenceExtractor(FakeSpeechExtractor()).extract(Path("sample.mp3"))

        self.assertEqual(
            objective_data_from_evidence(evidence, audio_path="sample.mp3"),
            {
                "audio_path": "sample.mp3",
                "raw_transcript_hiragana": "わたしはすしがすきです",
                "raw_transcript_romaji": "watashiha",
                "fluency_metrics": {},
                "top_pauses": [],
                "pause_segments": [],
                "speech_segments": [{"start": 0.0, "end": 1.2, "duration": 1.2}],
                "mora_timings": [{"mora": "わ", "start": 0.0, "end": 0.1}],
                "stt_model": "fake-stt-v1",
                "vad_model": "fake-vad-v1",
            },
        )


class _FailingTokenizer:
    version = "unused"
    split_mode = "A"

    def tokenize(self, text: str) -> list[TokenEvidence]:
        raise AssertionError("Range must consume the shared linguistic evidence.")


if __name__ == "__main__":
    unittest.main()
