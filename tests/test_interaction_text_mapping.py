from __future__ import annotations

import unittest

from jgrade_eval.evidence.models import (
    EvidenceBundle, LinguisticEvidence, SourceEvidence, SpeechEvidence, TimedSpan,
)
from jgrade_eval.interaction import InteractionModule


class InteractionTextMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transcript = "きんようびですかりゆうはこうです"
        self.bundle = EvidenceBundle(
            source=SourceEvidence("sample.wav", None),
            speech=SpeechEvidence(
                raw_transcript_hiragana=self.transcript,
                raw_transcript_romaji="", duration_sec=8.0,
                speech_segments=(TimedSpan(0, 7),), pause_segments=(),
                mora_timings=(), factual_metrics=(), provenance=(("stt_model", "fake"),),
            ),
            linguistic=LinguisticEvidence(tokens=(), tokenizer_version="fake", split_mode="A"),
        )
        self.prompts = [{"prompt_id": "q1", "text": "日付を確認"},
                        {"prompt_id": "q2", "text": "理由を説明"}]

    def test_time_only_mapping_remains_without_transcript_confirmation(self) -> None:
        packet = InteractionModule().collect(self.bundle, interaction_context={
            "prompts": self.prompts,
            "answer_segments": [{"prompt_id": "q1", "start_sec": 0, "end_sec": 3}],
        }).to_dict()
        segment = packet["candidate_answer_segments"][0]
        self.assertEqual(segment["mapping_status"], "confirmed")
        self.assertNotIn("transcript_text", segment)

    def test_human_confirmed_text_span_is_returned_with_provenance(self) -> None:
        packet = InteractionModule().collect(self.bundle, interaction_context={
            "prompts": self.prompts,
            "answer_segments": [
                {"prompt_id": "q1", "start_sec": 0, "end_sec": 3,
                 "start_offset": 0, "end_offset": 8,
                 "transcript_text": self.transcript[:8], "confirmed_by": "teacher-1"},
                {"prompt_id": "q2", "start_sec": 3, "end_sec": 8,
                 "start_offset": 8, "end_offset": len(self.transcript),
                 "transcript_text": self.transcript[8:], "confirmed_by": "teacher-1"},
            ],
        }).to_dict()
        first, second = packet["candidate_answer_segments"]
        self.assertEqual(first["transcript_text"], self.transcript[:8])
        self.assertEqual(second["start_offset"], 8)
        self.assertEqual(first["confirmed_by"], "teacher-1")

    def test_rejects_text_span_that_does_not_match_shared_transcript(self) -> None:
        with self.assertRaisesRegex(ValueError, "transcript span"):
            InteractionModule().collect(self.bundle, interaction_context={
                "prompts": self.prompts,
                "answer_segments": [{"prompt_id": "q1", "start_sec": 0, "end_sec": 3,
                                     "start_offset": 0, "end_offset": 4,
                                     "transcript_text": "ちがう", "confirmed_by": "teacher-1"}],
            })


if __name__ == "__main__":
    unittest.main()
