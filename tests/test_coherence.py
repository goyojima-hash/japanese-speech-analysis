from __future__ import annotations

import unittest

from jgrade_eval.evidence.models import (
    EvidenceBundle,
    LinguisticEvidence,
    SourceEvidence,
    SpeechEvidence,
    TimedSpan,
    TokenEvidence,
)


class CoherenceModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = EvidenceBundle(
            source=SourceEvidence(audio_path="sample.wav", audio_sha256="a" * 64),
            speech=SpeechEvidence(
                raw_transcript_hiragana="わたしはすしがすきですだからすしをたべます",
                raw_transcript_romaji="watashihasushigasukidesukarasushiwotabemasu",
                duration_sec=4.0,
                speech_segments=(TimedSpan(0.0, 3.0),),
                pause_segments=(TimedSpan(1.2, 1.7),),
                mora_timings=(),
                factual_metrics=(),
                provenance=(("stt_model", "fake-stt"),),
            ),
            linguistic=LinguisticEvidence(
                tokens=(
                    TokenEvidence("わたし", "私", "わたし", ("名詞",), 0, 3),
                    TokenEvidence("は", "は", "は", ("助詞",), 3, 4),
                    TokenEvidence("すし", "寿司", "すし", ("名詞",), 4, 6),
                    TokenEvidence("が", "が", "が", ("助詞",), 6, 7),
                    TokenEvidence("すき", "好き", "すき", ("形容詞",), 7, 9),
                    TokenEvidence("です", "です", "です", ("助動詞",), 9, 11),
                    TokenEvidence("だから", "だから", "だから", ("接続詞",), 11, 14),
                    TokenEvidence("すし", "寿司", "すし", ("名詞",), 14, 16),
                    TokenEvidence("を", "を", "を", ("助詞",), 16, 17),
                    TokenEvidence("たべます", "食べる", "たべます", ("動詞",), 17, 21),
                ),
                tokenizer_version="fake-tokenizer",
                split_mode="A",
            ),
        )

    def test_collects_auditable_facts_without_a_proficiency_judgement(self) -> None:
        from jgrade_eval.coherence import CoherenceModule

        data = CoherenceModule().collect(self.bundle).to_dict()

        self.assertEqual(data["module_id"], "coherence")
        self.assertEqual(data["input_provenance"]["tokenizer_version"], "fake-tokenizer")
        self.assertEqual(
            data["connective_observations"],
            [
                {
                    "token_index": 6,
                    "surface": "だから",
                    "dictionary_form": "だから",
                    "category": "causal",
                    "lexicon_version": "coherence-lexicon.v1",
                }
            ],
        )
        self.assertEqual(
            [(unit["token_start_index"], unit["token_end_index"]) for unit in data["candidate_units"]],
            [(0, 6), (6, 10)],
        )
        self.assertIn("connective:causal", data["candidate_units"][1]["boundary_derivations"])
        self.assertIn("transcript_end", data["candidate_units"][-1]["boundary_derivations"])
        self.assertEqual(data["repetition_observations"][0]["dictionary_form"], "寿司")
        self.assertEqual(data["repetition_observations"][0]["unit_indexes"], [0, 1])
        self.assertEqual(data["pause_observations"], [{"start": 1.2, "end": 1.7, "duration": 0.5}])
        for forbidden in ("score", "rating", "cefr", "jfs", "correct", "error", "quality"):
            self.assertNotIn(forbidden, data)

    def test_empty_linguistic_evidence_is_a_named_limitation_not_an_exception(self) -> None:
        from jgrade_eval.coherence import CoherenceModule

        empty_bundle = EvidenceBundle(
            source=self.bundle.source,
            speech=self.bundle.speech,
            linguistic=LinguisticEvidence(tokens=(), tokenizer_version="fake-tokenizer", split_mode="A"),
        )

        data = CoherenceModule().collect(empty_bundle).to_dict()

        self.assertEqual(data["connective_observations"], [])
        self.assertEqual(data["candidate_units"], [])
        self.assertIn("no_linguistic_tokens", data["unavailable_capabilities"])


if __name__ == "__main__":
    unittest.main()
