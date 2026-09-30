from __future__ import annotations

import unittest

from jgrade_eval.evidence.alignment import build_transcript_alignment
from jgrade_eval.evidence.models import MoraTiming, SpeechEvidence, TimedSpan
from jgrade_eval.evidence.speech import objective_data_from_evidence


def speech(transcript: str, timings: tuple[MoraTiming, ...]) -> SpeechEvidence:
    return SpeechEvidence(
        raw_transcript_hiragana=transcript, raw_transcript_romaji="",
        duration_sec=4.0, speech_segments=(TimedSpan(0.0, 0.4), TimedSpan(2.0, 2.6)),
        pause_segments=(TimedSpan(0.4, 2.0),), mora_timings=timings,
        factual_metrics=(), provenance=(("stt_model", "fake-ctc"),),
    )


class TranscriptAlignmentTests(unittest.TestCase):
    def test_exact_labels_produce_candidate_time_and_text_links(self) -> None:
        evidence = speech("あいうえお", (
            MoraTiming("あ", 0.0, 0.2), MoraTiming("い", 0.2, 0.4),
            MoraTiming("う", 2.0, 2.2), MoraTiming("え", 2.2, 2.4),
            MoraTiming("お", 2.4, 2.6),
        ))
        alignment = build_transcript_alignment(evidence)
        self.assertEqual(alignment.status, "complete")
        self.assertEqual(alignment.aligned_chars, 5)
        self.assertEqual(alignment.units[2].to_dict(), {
            "text": "う", "start_offset": 2, "end_offset": 3,
            "start_sec": 2.0, "end_sec": 2.2,
        })
        self.assertEqual(alignment.candidate_for_interval(0.0, 2.0), {
            "start_offset": 0, "end_offset": 2, "transcript_text": "あい",
            "alignment_status": "complete",
        })
        self.assertEqual(alignment.candidate_for_interval(2.0, 4.0)["transcript_text"], "うえお")
        self.assertEqual(objective_data_from_evidence(evidence, audio_path="sample.mp3")
                         ["transcript_alignment"]["status"], "complete")

    def test_trailing_unaligned_text_is_partial_never_assumed_timed(self) -> None:
        alignment = build_transcript_alignment(speech("あいう", (
            MoraTiming("あ", 0.0, 0.2), MoraTiming("い", 0.2, 0.4),
        )))
        self.assertEqual(alignment.status, "partial")
        self.assertEqual(alignment.aligned_chars, 2)
        self.assertEqual(alignment.candidate_for_interval(0.0, 4.0)["transcript_text"], "あい")
        self.assertEqual(alignment.to_dict()["reason"], "trailing_transcript_unaligned")

    def test_mismatched_or_invalid_timing_never_fabricates_links(self) -> None:
        cases = (
            speech("あいう", (MoraTiming("あ", 0.0, 0.2), MoraTiming("え", 0.2, 0.4))),
            speech("あ", (MoraTiming("あ", 0.5, 0.4),)),
            speech("あ", (MoraTiming("あ", float("nan"), 0.2),)),
            speech("あ", (MoraTiming("あ", 4.1, 4.2),)),
            speech("あ", ()),
            speech("", (MoraTiming("あ", 0.0, 0.2),)),
        )
        for evidence in cases:
            with self.subTest(transcript=evidence.raw_transcript_hiragana,
                              timings=evidence.mora_timings):
                alignment = build_transcript_alignment(evidence)
                self.assertEqual(alignment.status, "unavailable")
                self.assertEqual(alignment.units, ())
                self.assertIsNone(alignment.candidate_for_interval(0.0, 4.0))


if __name__ == "__main__":
    unittest.main()
