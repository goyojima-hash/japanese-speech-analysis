from __future__ import annotations

import unittest

import numpy as np

from jgrade_eval.ctc_forced_alignment import CTCLogitChunk, align_ctc_chunks


def logits(*frames: int, classes: int = 3) -> np.ndarray:
    scores = np.full((len(frames), classes), -8.0, dtype=np.float32)
    for index, label in enumerate(frames):
        scores[index, label] = 8.0
    return scores


class CTCForcedAlignmentTests(unittest.TestCase):
    def test_exact_kana_alignment_uses_audio_samples_not_fixed_frame_duration(self) -> None:
        result = align_ctc_chunks(
            (CTCLogitChunk(0, 1100, logits(0, 1, 0, 2, 0)),),
            "あい", {"あ": 1, "い": 2}, blank_id=0, sample_rate=1000,
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual([(unit.text, unit.start_sec, unit.end_sec) for unit in result.units], [
            ("あ", 0.22, 0.44), ("い", 0.66, 0.88),
        ])
        self.assertEqual(result.to_dict()["schema_version"], "forced-alignment.v2")
        self.assertEqual(result.to_dict()["timing_quality"], "unverified_ctc_forced_alignment")

    def test_repeated_kana_require_a_ctc_blank_between_them(self) -> None:
        result = align_ctc_chunks(
            (CTCLogitChunk(0, 500, logits(1, 1, 0, 1, 1, classes=2)),),
            "ああ", {"あ": 1}, blank_id=0, sample_rate=1000,
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual([(u.start_offset, u.end_offset, u.start_sec, u.end_sec) for u in result.units], [
            (0, 1, 0.0, 0.2), (1, 2, 0.3, 0.5),
        ])

    def test_chunk_start_uses_sample_offset_even_when_frame_count_rounds(self) -> None:
        result = align_ctc_chunks(
            (CTCLogitChunk(0, 1001, logits(0, 1, 0)),
             CTCLogitChunk(1001, 999, logits(0, 2, 0))),
            "あい", {"あ": 1, "い": 2}, blank_id=0, sample_rate=1000,
        )
        self.assertEqual(result.status, "complete")
        self.assertAlmostEqual(result.units[1].start_sec, (1001 + 999 // 3) / 1000)
        self.assertEqual(result.units[1].end_sec, (1001 + 2 * 999 // 3) / 1000)

    def test_missing_kana_and_impossible_or_bad_logits_are_unavailable(self) -> None:
        cases = (
            ("あえ", (CTCLogitChunk(0, 400, logits(0, 1, 0, 2)),), "unsupported_character"),
            ("ああ", (CTCLogitChunk(0, 200, logits(1, 1, classes=2)),), "ctc_path_unavailable"),
            ("あ", (CTCLogitChunk(0, 200, np.array([[np.nan, 1.0]])),), "invalid_logits"),
            ("", (CTCLogitChunk(0, 200, logits(0, 1)),), "empty_transcript"),
        )
        for transcript, chunks, reason in cases:
            with self.subTest(transcript=transcript, reason=reason):
                result = align_ctc_chunks(chunks, transcript, {"あ": 1}, blank_id=0,
                                          sample_rate=1000)
                self.assertEqual(result.status, "unavailable")
                self.assertEqual(result.reason, reason)
                self.assertEqual(result.units, ())


if __name__ == "__main__":
    unittest.main()
