from __future__ import annotations

from pathlib import Path
import unittest

from jgrade_eval.api_service import evaluate_speech_level


class AlignedExtractor:
    def extract(self, audio_path: Path) -> dict:
        return {
            "audio_path": str(audio_path),
            "raw_transcript_hiragana": "あい", "raw_transcript_romaji": "ai",
            "fluency_metrics": {"audio_duration_sec": 1.0, "speech_sec": 0.4,
                                "speech_ratio_pct": 40.0, "pause_total_sec": 0.6,
                                "pause_count": 0, "avg_pause_sec": 0.0,
                                "max_pause_sec": 0.0, "mora_count": 2,
                                "mora_per_sec": 2.0},
            "speech_segments": [{"start": 0.0, "end": 0.4}],
            "pause_segments": [], "top_pauses": [],
            "mora_timings": [
                {"mora": "あ", "start": 0.0, "end": 0.2},
                {"mora": "い", "start": 0.2, "end": 0.4},
            ],
            "stt_model": "fake-ctc", "vad_model": "fake-vad",
        }


class TranscriptAlignmentApiTests(unittest.TestCase):
    def test_alignment_is_separate_from_legacy_objective_data(self) -> None:
        result = evaluate_speech_level(Path("sample.wav"), extractor=AlignedExtractor(),
                                       judge_mode="mock", selected_modules=())
        self.assertNotIn("transcript_alignment", result["objective_data"])
        self.assertEqual(result["transcript_alignment"]["status"], "complete")
        self.assertEqual(result["transcript_alignment"]["units"][0]["start_offset"], 0)


if __name__ == "__main__":
    unittest.main()
