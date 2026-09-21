from __future__ import annotations

from pathlib import Path
import unittest

from jgrade_eval.accuracy import AccuracyModule
from jgrade_eval.evidence.models import (
    EvidenceBundle,
    LinguisticEvidence,
    MoraTiming,
    SourceEvidence,
    SpeechEvidence,
    TimedSpan,
    TokenEvidence,
)


class AccuracyModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = EvidenceBundle(
            source=SourceEvidence(audio_path="sample.wav", audio_sha256="a" * 64),
            speech=SpeechEvidence(
                raw_transcript_hiragana="わたしはすしです",
                raw_transcript_romaji="watashihasushidesu",
                duration_sec=2.0,
                speech_segments=(TimedSpan(0.0, 1.6),),
                pause_segments=(TimedSpan(1.6, 2.0),),
                mora_timings=(
                    MoraTiming("わ", 0.0, 0.1),
                    MoraTiming("た", 0.1, 0.2),
                    MoraTiming("し", 0.2, 0.3),
                ),
                factual_metrics=(("mora_count", 3),),
                provenance=(("stt_model", "fake-stt"), ("vad_model", "fake-vad")),
            ),
            linguistic=LinguisticEvidence(
                tokens=(
                    TokenEvidence("わたし", "私", "わたし", ("名詞",), 0, 3),
                    TokenEvidence("は", "は", "は", ("助詞",), 3, 4),
                    TokenEvidence("すし", "寿司", "すし", ("名詞",), 4, 6),
                    TokenEvidence("です", "です", "です", ("助動詞",), 6, 8),
                ),
                tokenizer_version="fake-tokenizer",
                split_mode="A",
            ),
        )

    def test_collects_only_asr_and_morphology_facts(self) -> None:
        packet = AccuracyModule().collect(self.bundle)

        data = packet.to_dict()
        self.assertEqual(data["module_id"], "accuracy")
        self.assertEqual(data["asr_observations"][0]["unit"], "わ")
        self.assertEqual(data["asr_observations"][0]["calibration_status"], "unavailable")
        self.assertIn("particle_after_noun", data["morphology_observations"][1]["pattern_ids"])
        self.assertIn("auxiliary_after_noun", data["morphology_observations"][3]["pattern_ids"])
        self.assertIn("asr_posterior_confidence", data["unavailable_capabilities"])
        self.assertNotIn("score", data)
        self.assertNotIn("correct", data)
        self.assertNotIn("error", data)
        self.assertNotIn("cefr", data)

    def test_reference_difference_requires_explicit_reference(self) -> None:
        module = AccuracyModule(reference_tokenizer=_ReferenceTokenizer())

        without_reference = module.collect(self.bundle)
        with_reference = module.collect(self.bundle, reference_transcript="わたしはさしみです")

        self.assertEqual(without_reference.to_dict()["reference_differences"], [])
        differences = with_reference.to_dict()["reference_differences"]
        self.assertEqual([item["operation"] for item in differences], ["equal", "equal", "replace", "equal"])
        self.assertEqual(differences[2]["observed"], "すし")
        self.assertEqual(differences[2]["reference"], "さしみ")

    def test_reference_difference_records_extra_observed_tokens_without_an_error_label(self) -> None:
        module = AccuracyModule(
            reference_tokenizer=_StaticReferenceTokenizer(
                [
                    TokenEvidence("わたし", "私", "わたし", ("名詞",), 0, 3),
                    TokenEvidence("です", "です", "です", ("助動詞",), 3, 5),
                ]
            )
        )

        differences = module.collect(
            self.bundle,
            reference_transcript="ignored-by-static-tokenizer",
        ).to_dict()["reference_differences"]

        self.assertEqual([item["operation"] for item in differences], ["equal", "delete", "delete", "equal"])
        self.assertNotIn("error", differences[1])


class _ReferenceTokenizer:
    version = "reference-tokenizer"
    split_mode = "A"

    def tokenize(self, text: str) -> list[TokenEvidence]:
        if text != "わたしはさしみです":
            raise AssertionError(f"unexpected reference text: {text}")
        return [
            TokenEvidence("わたし", "私", "わたし", ("名詞",), 0, 3),
            TokenEvidence("は", "は", "は", ("助詞",), 3, 4),
            TokenEvidence("さしみ", "刺身", "さしみ", ("名詞",), 4, 7),
            TokenEvidence("です", "です", "です", ("助動詞",), 7, 9),
        ]


class _StaticReferenceTokenizer:
    version = "static-reference-tokenizer"
    split_mode = "A"

    def __init__(self, tokens: list[TokenEvidence]) -> None:
        self._tokens = tokens

    def tokenize(self, text: str) -> list[TokenEvidence]:
        return self._tokens
