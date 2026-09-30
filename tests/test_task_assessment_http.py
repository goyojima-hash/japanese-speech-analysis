from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient

from jgrade_eval.api import app


class FakeExtractor:
    def extract(self, audio_path: Path) -> dict:
        return {
            "audio_path": str(audio_path),
            "raw_transcript_hiragana": "きんようびでいいですか",
            "raw_transcript_romaji": "",
            "fluency_metrics": {
                "audio_duration_sec": 3.0, "speech_sec": 2.0,
                "speech_ratio_pct": 66.67, "pause_total_sec": 1.0,
                "pause_count": 1, "avg_pause_sec": 1.0,
                "max_pause_sec": 1.0, "mora_count": 11,
                "mora_per_sec": 3.67,
            },
            "top_pauses": [], "speech_segments": [], "extraction_time_sec": 0.01,
            "stt_model": "fake", "vad_model": "fake",
        }


class TaskAssessmentHttpTests(unittest.TestCase):
    def test_http_shadow_accepts_independent_task_context(self) -> None:
        with patch("jgrade_eval.api_service.FluencyExtractor", return_value=FakeExtractor()):
            response = TestClient(app).post(
                "/api/v1/speech-level-evaluations",
                data={
                    "judge_mode": "mock", "assessment_mode": "shadow",
                    "fact_modules": "accuracy",
                    "task_context": ('{"prompts":[{"prompt_id":"q1","text":"日付を確認してください",'
                                     '"task_type":"date_confirmation","rubric_id":"date-test"}],'
                                     '"whole_recording_answer_prompt_id":"q1"}'),
                },
                files={"audio": ("sample.mp3", b"fake audio", "audio/mpeg")},
            )
        self.assertEqual(response.status_code, 201)
        data = response.json()["data"]
        self.assertEqual(data["task_assessment_shadow"]["status"], "mock_unavailable")
        self.assertNotIn("interaction_data", data["objective_data"])

    def test_http_shadow_rejects_invalid_task_context(self) -> None:
        with patch("jgrade_eval.api_service.FluencyExtractor", return_value=FakeExtractor()):
            response = TestClient(app).post(
                "/api/v1/speech-level-evaluations",
                data={"judge_mode": "mock", "assessment_mode": "shadow",
                      "task_context": '{"prompts":[{"prompt_id":"q1","text":"A"},'
                                      '{"prompt_id":"q1","text":"B"}]}'},
                files={"audio": ("sample.mp3", b"fake audio", "audio/mpeg")},
            )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
