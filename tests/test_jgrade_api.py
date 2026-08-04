from pathlib import Path
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient

from jgrade_eval.api import EVALUATIONS, app


class FakeExtractor:
    def extract(self, audio_path: Path) -> dict:
        return {
            "audio_path": str(audio_path),
            "raw_transcript_hiragana": "わたしはきのうともだちとえきにいきました" * 14,
            "raw_transcript_romaji": "watashi",
            "fluency_metrics": {
                "audio_duration_sec": 60.0,
                "speech_sec": 42.0,
                "speech_ratio_pct": 70.0,
                "pause_total_sec": 3.0,
                "pause_count": 3,
                "avg_pause_sec": 1.0,
                "max_pause_sec": 1.5,
                "mora_count": 270,
                "mora_per_sec": 4.5,
                "fluency_grade": "B",
            },
            "top_pauses": [],
            "speech_segments": [],
            "extraction_time_sec": 0.01,
            "stt_model": "fake",
            "vad_model": "fake",
        }


class FakeUnusableExtractor:
    def __init__(self, *, duration: float = 10.0, transcript: str = "", speech_sec: float = 0.0):
        self.duration = duration
        self.transcript = transcript
        self.speech_sec = speech_sec

    def extract(self, audio_path: Path) -> dict:
        ratio = self.speech_sec / self.duration * 100 if self.duration else 0.0
        return {
            "audio_path": str(audio_path),
            "raw_transcript_hiragana": self.transcript,
            "raw_transcript_romaji": "",
            "fluency_metrics": {
                "audio_duration_sec": self.duration,
                "speech_sec": self.speech_sec,
                "speech_ratio_pct": ratio,
                "pause_total_sec": self.duration - self.speech_sec,
                "pause_count": 1,
                "avg_pause_sec": self.duration - self.speech_sec,
                "max_pause_sec": self.duration - self.speech_sec,
                "mora_count": len(self.transcript),
                "mora_per_sec": 0.0,
                "fluency_grade": "D",
            },
            "top_pauses": [],
            "speech_segments": [],
            "extraction_time_sec": 0.01,
            "stt_model": "fake",
            "vad_model": "fake",
        }


class FakeDownloadResponse:
    content = b"fake audio"

    def raise_for_status(self) -> None:
        return None


class JGradeApiTests(unittest.TestCase):
    def test_create_speech_level_evaluation_returns_completed_result(self) -> None:
        EVALUATIONS.clear()
        client = TestClient(app)

        with patch("jgrade_eval.api_service.FluencyExtractor", return_value=FakeExtractor()):
            response = client.post(
                "/api/v1/speech-level-evaluations",
                data={
                    "external_id": "sample-1",
                    "language": "ja",
                    "judge_mode": "mock",
                    "roleplay_task": "駅で友達との出来事を説明する。",
                },
                files={"audio": ("sample.mp3", b"fake audio", "audio/mpeg")},
            )

        self.assertEqual(response.status_code, 201)
        data = response.json()["data"]
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["external_id"], "sample-1")
        self.assertEqual(data["final_cefr_level"], "B1")
        self.assertEqual(data["consensus"]["judge_count"], 3)
        self.assertTrue(data["objective_data"]["raw_transcript_hiragana"].startswith("わたし"))
        self.assertEqual(data["reasons"][0]["type"], "transcript")

        fetched = client.get(f"/api/v1/speech-level-evaluations/{data['id']}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["data"]["id"], data["id"])

    def test_create_speech_level_evaluation_rejects_non_japanese_language(self) -> None:
        client = TestClient(app)

        response = client.post(
            "/api/v1/speech-level-evaluations",
            data={"language": "en", "judge_mode": "mock"},
            files={"audio": ("sample.mp3", b"fake audio", "audio/mpeg")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "validation_error")

    def test_create_speech_level_evaluation_rejects_silence_before_judging(self) -> None:
        client = TestClient(app)

        with patch(
            "jgrade_eval.api_service.FluencyExtractor",
            return_value=FakeUnusableExtractor(),
        ):
            response = client.post(
                "/api/v1/speech-level-evaluations",
                data={"language": "ja", "judge_mode": "mock"},
                files={"audio": ("silence.wav", b"fake silence", "audio/wav")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "unusable_audio")
        self.assertIn("評価できない音声", response.json()["error"]["message"])

    def test_create_speech_level_evaluation_rejects_too_short_audio(self) -> None:
        client = TestClient(app)

        with patch(
            "jgrade_eval.api_service.FluencyExtractor",
            return_value=FakeUnusableExtractor(
                duration=0.25,
                transcript="あ",
                speech_sec=0.25,
            ),
        ):
            response = client.post(
                "/api/v1/speech-level-evaluations",
                data={"language": "ja", "judge_mode": "mock"},
                files={"audio": ("short.wav", b"fake short audio", "audio/wav")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "unusable_audio")
        self.assertIn("短すぎます", response.json()["error"]["message"])

    def test_create_speech_level_evaluation_accepts_json_audio_url(self) -> None:
        EVALUATIONS.clear()
        client = TestClient(app)

        with (
            patch("jgrade_eval.api.requests.get", return_value=FakeDownloadResponse()),
            patch("jgrade_eval.api_service.FluencyExtractor", return_value=FakeExtractor()),
        ):
            response = client.post(
                "/api/v1/speech-level-evaluations",
                json={
                    "audio_url": "https://example.com/sample.mp3",
                    "external_id": "json-sample",
                    "judge_mode": "mock",
                    "include_objective_data": False,
                },
            )

        self.assertEqual(response.status_code, 201)
        data = response.json()["data"]
        self.assertEqual(data["external_id"], "json-sample")
        self.assertEqual(data["final_cefr_level"], "B1")
        self.assertNotIn("objective_data", data)


if __name__ == "__main__":
    unittest.main()
